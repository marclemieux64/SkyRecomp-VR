// The VR controllers as the game's Joy-Con pair: buttons and sticks, and the motion sensors (vr_input.c feeds the
// translator; runtime/hle/input.c and runtime/hle/motion.c read these on the game thread).
#include "vr_internal.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <SDL3/SDL_timer.h>
#include <stdlib.h>
#include <string.h>

#if SKYRECOMP_HAS_OPENXR
#include "hle.h"
#include "nvn/vulkan/JoyConTranslator.h"
#include "vr_settings.h"

static bool controllers_off(void) {  // (test) SKYRECOMP_VR_NO_INPUT=1: controllers ignored, for scripted runs
    static int off = -1;
    if (off < 0) off = getenv("SKYRECOMP_VR_NO_INPUT") != NULL;
    return off;
}

_Static_assert(sizeof(VrJoyConState) == sizeof(SkyJoyConState), "the same layout as JoyConTranslator.h");
_Static_assert(sizeof(VrJoyConSensor) == sizeof(SkyJoyConSensor), "the same layout as JoyConTranslator.h");

// ---- the menu pointer: on the title / file select screens (no Link in the world) and while a menu pauses the game
// (the gear menu, the map...), the game's cursor goes where the right controller's ray meets the floating HUD panel.
// The game turns the right Joy-Con's orientation, relative to its canonical centre, into a cursor like a flat screen:
// offset = K tan(angle), with K = 1.298 screen widths horizontally (yaw about the Joy-Con's z) and 2.316 screen
// heights vertically (pitch about x; down positive) - measured 2026-10-26 with SKYRECOMP_JOYCON_TEST=point. The
// panel's hit (u, v) gives those angles back, clamped to the panel's edges (panel_uv). Outside the title / file
// select and paused menus the real orientation goes to the game (the sword follows it).
// SKYRECOMP_VR_PANEL_POINTER=0 turns it off.
#define POINTER_KX 1.298f
#define POINTER_KY 2.316f
static pthread_mutex_t g_ptr_lock = PTHREAD_MUTEX_INITIALIZER;
static float g_ptr_aim[16], g_ptr_hud[16];  // (row-major world-from-local: the right aim pose; the panel, half extents)
static bool g_ptr_aim_ok, g_ptr_hud_ok;

// The Joy-Con direction (rows: its axes in the player's frame) that puts the game's cursor at (u, v) of the screen
// (0..1, top left): a turn about its own z then x from the canonical centre.
void vr_pointer_direction(float u, float v, float out[9]);  // (also motion.c tests)
void vr_pointer_direction(float u, float v, float out[9]) {
    const float a = atanf((u - 0.5f) / POINTER_KX), b = atanf((v - 0.5f) / POINTER_KY);
    const float ca = cosf(a), sa = sinf(a), cb = cosf(b), sb = sinf(b);
    const float yaw[9] = {ca, -sa, 0, sa, ca, 0, 0, 0, 1}, pitch[9] = {1, 0, 0, 0, cb, -sb, 0, sb, cb};
    for (int r = 0; r < 3; r++)
        for (int k = 0; k < 3; k++)
            out[r * 3 + k] = pitch[r * 3] * yaw[k] + pitch[r * 3 + 1] * yaw[3 + k] + pitch[r * 3 + 2] * yaw[6 + k];
}

// (vr_input.c, once per frame) The right controller's aim pose in the local space.
void vr_pointer_set_aim(bool tracked, const float q[4], const float p[3]) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float m[16] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p[0],
                         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p[1],
                         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p[2],
                         0, 0, 0, 1};
    pthread_mutex_lock(&g_ptr_lock);
    g_ptr_aim_ok = tracked;
    if (tracked) memcpy(g_ptr_aim, m, sizeof m);
    pthread_mutex_unlock(&g_ptr_lock);
}

// (vr.c, each frame the HUD panel is placed) The panel the menu pointer aims at (world-from-panel, half extents).
void vr_controls_set_hud_panel(const float m[16]) {
    pthread_mutex_lock(&g_ptr_lock);
    memcpy(g_ptr_hud, m, sizeof g_ptr_hud);
    g_ptr_hud_ok = true;
    pthread_mutex_unlock(&g_ptr_lock);
}

bool vr_game_paused(void) { return *(volatile uint32_t *)(uintptr_t)(MAIN_BASE + 0x1827370) != 0; }  // (vr.c)
// (vr.c) Paused, or within half a second after: a closing menu keeps fading out once the game runs again - the panel
// holds still for it (it swung back to follow the head mid-fade: a flash of the menu at another angle).
bool vr_game_paused_recently(void) {
    extern uint64_t renderer_present_count(void);
    static uint64_t last = ~0ull;
    const uint64_t now = renderer_present_count();
    if (vr_game_paused()) last = now;
    return last != ~0ull && now - last < 30;
}
static bool panel_pointer_on(void) {
    static int env = -1;
    if (env < 0) env = getenv("SKYRECOMP_VR_PANEL_POINTER") ? atoi(getenv("SKYRECOMP_VR_PANEL_POINTER")) != 0 : 1;
    extern bool vk_hud_layer_enabled(void);
    extern bool player_eye(float eye[3]);
    if (!env || !vk_hud_layer_enabled()) return false;
    return *(volatile uint32_t *)(uintptr_t)(MAIN_BASE + 0x1827370) != 0  // (a menu pauses the game)
           || !player_eye(NULL);  // (no Link: the title, file select, loading)
}

bool vr_menu_pointer_active(void);
bool vr_menu_pointer_active(void) { return panel_pointer_on(); }  // (vr.c: the HUD panel holds still meanwhile)

// Where the ray points on the panel, (u, v) in 0..1 (top left), clamped to its edges: the panel follows the head's
// yaw, so a turned head can leave the controller pointing past it - the cursor then stays on the nearest edge
// instead of leaving the screen (the game hides it off screen). A ray pointing away from the panel's plane goes to
// the edge in its direction along the panel. False only for unusable matrices.
static bool panel_uv_ex(const float aim[16], const float hud[16], float *u, float *v, bool *inside) {
    const float o[3] = {aim[3], aim[7], aim[11]}, d[3] = {-aim[2], -aim[6], -aim[10]};
    const float c[3] = {hud[3], hud[7], hud[11]}, x[3] = {hud[0], hud[4], hud[8]}, y[3] = {hud[1], hud[5], hud[9]};
    float n[3] = {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
    const float nl = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    const float xx = x[0] * x[0] + x[1] * x[1] + x[2] * x[2], yy = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
    if (!(nl > 1e-6f) || !(xx > 1e-9f) || !(yy > 1e-9f)) return false;
    for (int i = 0; i < 3; i++) n[i] /= nl;
    const float dn = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    const float co[3] = {c[0] - o[0], c[1] - o[1], c[2] - o[2]};
    const float t = fabsf(dn) > 1e-6f ? (co[0] * n[0] + co[1] * n[1] + co[2] * n[2]) / dn : -1.0f;
    float lx, ly;
    if (t > 0) {  // (the plane is ahead: the hit, in panel units)
        const float r[3] = {o[0] + d[0] * t - c[0], o[1] + d[1] * t - c[1], o[2] + d[2] * t - c[2]};
        lx = (r[0] * x[0] + r[1] * x[1] + r[2] * x[2]) / xx;
        ly = (r[0] * y[0] + r[1] * y[1] + r[2] * y[2]) / yy;
    } else {  // (pointing away from it: the direction along the panel, pushed to the edge)
        lx = (d[0] * x[0] + d[1] * x[1] + d[2] * x[2]) / sqrtf(xx) * 1e3f;
        ly = (d[0] * y[0] + d[1] * y[1] + d[2] * y[2]) / sqrtf(yy) * 1e3f;
    }
    if (!isfinite(lx) || !isfinite(ly)) return false;
    if (inside) *inside = t > 0 && fabsf(lx) <= 1.0f && fabsf(ly) <= 1.0f;
    lx = fminf(1.0f, fmaxf(-1.0f, lx));
    ly = fminf(1.0f, fmaxf(-1.0f, ly));
    *u = lx * 0.5f + 0.5f;
    *v = 0.5f - ly * 0.5f;  // (+y: the top of the screen)
    return true;
}

static bool panel_uv(const float aim[16], const float hud[16], float *u, float *v) {
    return panel_uv_ex(aim, hud, u, v, NULL);
}

// (menu.mm, the settings menu on the HUD panel) Where the right controller's ray meets the panel, (u, v) in 0..1 from
// the top left; false when it misses the panel. click: the right trigger is held.
_Atomic bool g_vr_right_trigger;  // (vr_input.c)
_Atomic float g_vr_right_stick_y;  // (vr_input.c: up +)
float vr_menu_scroll(void) { return atomic_load(&g_vr_right_stick_y); }  // (menu.mm: the stick scrolls the menu)
// (vr.c, the menu's beam) The pointing controller's ray: origin, direction (unit), and its length to the panel
// (1.5 m when it misses).
bool vr_menu_beam(float o[3], float d[3], float *len);
bool vr_menu_beam(float o[3], float d[3], float *len) {
    pthread_mutex_lock(&g_ptr_lock);
    const bool ok = g_ptr_aim_ok && g_ptr_hud_ok;
    float aim[16], hud[16];
    memcpy(aim, g_ptr_aim, sizeof aim);
    memcpy(hud, g_ptr_hud, sizeof hud);
    pthread_mutex_unlock(&g_ptr_lock);
    if (!ok) return false;
    o[0] = aim[3], o[1] = aim[7], o[2] = aim[11];
    d[0] = -aim[2], d[1] = -aim[6], d[2] = -aim[10];
    *len = 1.5f;
    float u, v;
    bool inside = false;
    if (panel_uv_ex(aim, hud, &u, &v, &inside) && inside) {  // (the hit: back to metres along the ray)
        const float c[3] = {hud[3], hud[7], hud[11]}, x[3] = {hud[0], hud[4], hud[8]}, y[3] = {hud[1], hud[5], hud[9]};
        const float n[3] = {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
        const float dn = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
        if (fabsf(dn) > 1e-9f) *len = ((c[0] - o[0]) * n[0] + (c[1] - o[1]) * n[1] + (c[2] - o[2]) * n[2]) / dn;
    }
    return *len > 0.01f;
}

bool vr_menu_ray(float *u, float *v, bool *click) {
    pthread_mutex_lock(&g_ptr_lock);
    const bool ok = g_ptr_aim_ok && g_ptr_hud_ok;
    float aim[16], hud[16];
    memcpy(aim, g_ptr_aim, sizeof aim);
    memcpy(hud, g_ptr_hud, sizeof hud);
    pthread_mutex_unlock(&g_ptr_lock);
    bool inside = false;
    if (click) *click = atomic_load(&g_vr_right_trigger);
    return ok && panel_uv_ex(aim, hud, u, v, &inside) && inside;
}

static bool panel_pointer(SkyJoyConSensor *s) {
    pthread_mutex_lock(&g_ptr_lock);
    const bool on = panel_pointer_on() && g_ptr_aim_ok && g_ptr_hud_ok;
    float aim[16], hud[16];
    memcpy(aim, g_ptr_aim, sizeof aim);
    memcpy(hud, g_ptr_hud, sizeof hud);
    pthread_mutex_unlock(&g_ptr_lock);
    if (!on) return false;
    {   // The controller where it is when the game reads it (once per game frame, evenly spaced), with the runtime's
        // display lead - not the render thread's latest sample: the game thread reads that on its own schedule, so
        // some game frames got the same pose twice and the next one a double step (the cursor juddered).
        extern int64_t vr_display_time_now(void);
        extern bool vr_input_pointer_at(int64_t time, float q[4], float p[3]);
        extern bool nvn_async_on(void);
        extern uint64_t nvn_async_game_frame(void), renderer_present_count(void);
        const int64_t when = vr_display_time_now();
        const uint64_t g = (nvn_async_on() ? nvn_async_game_frame() : renderer_present_count()) + 1;
        if (when > 0) {
            float q[4], p[3];
            if (vr_input_pointer_at(when, q, p)) {
                // Tremor (and the prediction's noise): small turns eased in, once per game frame (the same frame
                // read twice gets the same pose); larger, deliberate turns followed at once.
                static float sq[4];
                static uint64_t sg;
                static bool sq_ok;
                static float out[4];
                if (!sq_ok || g < sg || g > sg + 8) memcpy(sq, q, sizeof sq), sq_ok = true;
                else if (g != sg) {
                    float dot = sq[0] * q[0] + sq[1] * q[1] + sq[2] * q[2] + sq[3] * q[3];
                    const float sign = dot < 0 ? -1.0f : 1.0f;
                    const float angle = 2 * acosf(fminf(1.0f, fabsf(dot)));
                    const float a = fminf(1.0f, fmaxf(0.25f, angle / 0.02f));  // (fully eased below ~1 degree)
                    float l = 0;
                    for (int i = 0; i < 4; i++) sq[i] += (q[i] * sign - sq[i]) * a, l += sq[i] * sq[i];
                    l = sqrtf(l);
                    for (int i = 0; i < 4; i++) sq[i] /= l;
                }
                sg = g;
                memcpy(out, sq, sizeof out);
                const float x = out[0], y = out[1], z = out[2], w = out[3];
                const float m[16] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p[0],
                                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p[1],
                                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p[2],
                                     0, 0, 0, 1};
                memcpy(aim, m, sizeof aim);
            }
        }
    }
    float u, v;
    if (!panel_uv(aim, hud, &u, &v)) return false;
    {   // (diagnostics) every half second while the pointer is active: the panel point, the ray, the panel
        static uint64_t next;
        const uint64_t now = SDL_GetTicksNS();
        if (now >= next) {
            next = now + 500000000ull;
            LOG("[vr-pointer] u %.2f v %.2f | ray from (%.2f %.2f %.2f) dir (%.2f %.2f %.2f) | panel centre (%.2f %.2f %.2f) "
                "x (%.2f %.2f %.2f) y (%.2f %.2f %.2f)\n", u, v, aim[3], aim[7], aim[11], -aim[2], -aim[6], -aim[10],
                hud[3], hud[7], hud[11], hud[0], hud[4], hud[8], hud[1], hud[5], hud[9]);
        }
    }
    vr_pointer_direction(u, v, s->direction);
    // Gravity as the Joy-Con would feel it held that way (its axes' z, the player's up, negated: (0, 0, -1) level),
    // not as the controller is really held: the game also reads the tilt from it, and the real controller is held
    // lower and pitched down to aim at the panel - the cursor could not rise past the middle of the screen.
    for (int i = 0; i < 3; i++) s->accel[i] = -s->direction[i * 3 + 2];
    static int logged;
    if (!logged++) LOG("[vr] menu pointer: the game's cursor follows the right controller's ray on the HUD panel\n");
    return true;
}

// Aiming an item (a reticle on screen): the Joy-Con orientation that puts the game's reticle where the right
// controller points (runtime/hle/vr_aim.c vr_aim_pointer), with the menu pointer's mapping.
static bool aim_pointer(SkyJoyConSensor *s) {
    extern bool vr_aim_pointer(float *u, float *v);
    float u, v;
    if (!vr_aim_pointer(&u, &v)) return false;
    vr_pointer_direction(u, v, s->direction);
    for (int i = 0; i < 3; i++) s->accel[i] = -s->direction[i * 3 + 2];  // (gravity as it would be held: see above)
    {   // (the Slingshot aimed with the controller: the game's pointer held still - no turning either)
        extern bool vr_aim_slingshot(void);
        if (vr_aim_slingshot()) for (int i = 0; i < 3; i++) s->gyro[i] = 0;
    }
    static int logged;
    if (!logged++) LOG("[vr] aiming: the item aims where the right controller points\n");
    return true;
}

bool vr_joycon_state(VrJoyConState *state) {
    SkyJoyConState s;
    extern bool motion_joycon_mode(void);
    if (!state || controllers_off() || !motion_joycon_mode() || !sky_joycon_state(&s)) return false;
    memcpy(state, &s, sizeof s);
    return true;
}

// Skyward Strike asks for a nearly upright, steady Joy-Con. Real VR grips have controller-specific pitch offsets
// and considerably more resting jitter than a calibrated Joy-Con. Once the sword is already raised mostly upward,
// report the intended cardinal pose and remove only low-rate jitter; swings below this cone remain completely raw.
// Only with the sword out (vr_sword_sheathed() == 0, hle/vr_sword.c): anything else held up - a bomb, before an
// overhand throw - was sent as that still, upright pose with gravity only, and the game's throw / roll gesture lost its
// wind-up (throws taken as rolls and back, swings ignored - user 2026-11-21, SKYRECOMP_BOMB_LOG).
static void skyward_strike_assist(SkyJoyConSensor *s) {
    extern int vr_sword_sheathed(void);
    if (vr_sword_sheathed() != 0) return;
    if (s->direction[5] < 0.72f) return;  // Joy-Con +Y follows the controller's pointing axis.
    float x = s->direction[0], y = s->direction[1];
    const float n = sqrtf(x * x + y * y);
    if (n > 1e-4f) x /= n, y /= n; else x = 1.0f, y = 0.0f;
    // Rows are the Joy-Con x/y/z axes in player space; x cross blade gives the remaining orthogonal axis.
    s->direction[0] = x;    s->direction[1] = y;    s->direction[2] = 0.0f;
    s->direction[3] = 0.0f; s->direction[4] = 0.0f; s->direction[5] = 1.0f;
    s->direction[6] = y;    s->direction[7] = -x;   s->direction[8] = 0.0f;
    s->accel[0] = 0.0f; s->accel[1] = -1.0f; s->accel[2] = 0.0f;
    const float jitter = 0.08f;  // revolutions/s; preserve deliberate rotations and every real swing.
    for (int i = 0; i < 3; i++) if (fabsf(s->gyro[i]) < jitter) s->gyro[i] = 0.0f;
}

// Bombs, overhand or underhand by where the hand is. The game tells a throw from a roll by which way the Joy-Con points
// as the swing starts: up throws, down rolls (SKYRECOMP_BOMB_LOG dumps, user 2026-11-25: five overhand throws pointing
// up, 0.98 of the way, and five underhand rolls pointing down, -0.95, all as meant). A swing begun with the controller
// level was the game's guess. So while Link holds a bomb, the Joy-Con's reported pitch is set from the hand's height
// instead: less than BOMB_OVERHAND_DROP below the eyes, BOMB_PITCH up; lower, BOMB_PITCH down. It is turned about the
// horizontal axis across its heading (its yaw and roll kept), gravity in the accelerometer turned with it (the hand's
// own acceleration kept). Once the swing starts (BOMB_SWING_GYRO) the offset is frozen until the throw (0x41) ends -
// or, a swing that threw nothing, the hand has been still for BOMB_SETTLE_NS - so the game sees the swing as made, from
// that pitch. SKYRECOMP_VR_BOMB_RAW=1: as the hand points.
// A bomb in his hand: his action is 0x43 when he takes it out or stops, but back to 0 standing and 3 walking with it
// still held (user's log 2026-11-25: throws from 3 too) - so from 0x43 until he throws or rolls it (0x41, then any
// other action), puts it away (0x42, 0x44), is hurt (0x36, 0x37), the sword comes out, or BOMB_HELD_NS.
#define BOMB_OVERHAND_DROP 0.40f  // metres below the eyes
#define BOMB_PITCH 70.0f          // degrees
#define BOMB_SWING_GYRO 0.6f      // rev/s
#define BOMB_SETTLE_NS 400000000ull
#define BOMB_HELD_NS 12000000000ull
static bool g_bomb_sword_pull;  // (the sword came out with a bomb in hand: the log's motion dump - vr_joycon_sensor)
static void bomb_gesture(SkyJoyConSensor *s) {
    static int raw = -1, log = -1;
    if (raw < 0) raw = getenv("SKYRECOMP_VR_BOMB_RAW") != NULL;
    if (log < 0) log = getenv("SKYRECOMP_BOMB_LOG") != NULL;
    extern int player_link_action(void), vr_sword_sheathed(void);
    extern float vr_input_right_hand_drop(void);
    extern bool vr_running(void);
    static bool active, frozen, held, thrown;
    static uint64_t held_since, still_since;
    static float angle, hx = 0.0f, hy = 1.0f;  // (the offset; the pointing's last heading in the player's frame)
    const int action = player_link_action();
    const uint64_t now = SDL_GetTicksNS();
    const float g = sqrtf(s->gyro[0] * s->gyro[0] + s->gyro[1] * s->gyro[1] + s->gyro[2] * s->gyro[2]);
    if (action == 0x43 && !held) held = true, thrown = false, held_since = now;
    if (held) {
        const char *why = NULL;
        if (action == 0x41) thrown = true;
        else if (thrown) why = "thrown";
        if (action == 0x42 || action == 0x44) why = "put away";
        if (action == 0x36 || action == 0x37) why = "hurt";
        // (the sword's state settles over 10 updates - hle/vr_sword.c: not in the first half second, while it's put away)
        if (vr_sword_sheathed() == 0 && now - held_since > 500000000ull) why = "the sword came out", g_bomb_sword_pull = true;
        if (now - held_since > BOMB_HELD_NS) why = "held too long";
        if (why) {
            held = active = frozen = false;
            if (log) LOG("[bomb] no longer in hand: %s (action %#x)\n", why, action);
        }
    }
    if (raw || !vr_running() || g_vr_config.view_mode != 1 || !held) {
        active = frozen = false;
        return;
    }
    const float deg = (float)(M_PI / 180.0);
    if (frozen && action != 0x41) {  // (a swing that threw nothing: free again once the hand is still)
        if (g >= BOMB_SWING_GYRO) still_since = now;
        else if (now - still_since > BOMB_SETTLE_NS) frozen = false;
    }
    if (!frozen) {
        const float drop = vr_input_right_hand_drop();
        if (!isfinite(drop)) { active = false; return; }
        const bool over = drop < BOMB_OVERHAND_DROP;
        const float fx = s->direction[3], fy = s->direction[4], n = sqrtf(fx * fx + fy * fy);
        if (n > 0.2f) hx = fx / n, hy = fy / n;  // (pointing nearly straight up or down: the last heading)
        const float p = asinf(fmaxf(-1.0f, fminf(1.0f, s->direction[5])));
        angle = (over ? BOMB_PITCH : -BOMB_PITCH) * deg - p;
        active = true;
        if (g >= BOMB_SWING_GYRO) {  // (the swing starts: this offset until the throw ends)
            frozen = true, still_since = now;
            if (log) LOG("[bomb] swing from %.2f m below the eyes: %s (pointing %.0f degrees, sent as %+.0f)\n", drop,
                         over ? "overhand, a throw" : "underhand, a roll", p / deg, over ? BOMB_PITCH : -BOMB_PITCH);
        }
    }
    if (!active) return;
    // (about a = heading x up = (hy, -hx, 0): a positive angle raises the pointing - Rodrigues, a perpendicular to up)
    const float ax = hy, ay = -hx, c = cosf(angle), sn = sinf(angle);
    float down_old[3], down_new[3];
    for (int i = 0; i < 3; i++) down_old[i] = -s->direction[i * 3 + 2];
    for (int r = 0; r < 3; r++) {
        float *v = s->direction + r * 3;
        const float dot = ax * v[0] + ay * v[1];
        const float cx = ay * v[2], cy = -ax * v[2], cz = ax * v[1] - ay * v[0];  // a x v
        const float o[3] = {v[0] * c + cx * sn + ax * dot * (1 - c), v[1] * c + cy * sn + ay * dot * (1 - c), v[2] * c + cz * sn};
        memcpy(v, o, sizeof o);
    }
    for (int i = 0; i < 3; i++) down_new[i] = -s->direction[i * 3 + 2];
    for (int i = 0; i < 3; i++) s->accel[i] += down_new[i] - down_old[i];  // (accel = down - motion: the motion kept)
}

bool vr_joycon_sensor(int device, VrJoyConSensor *sensor) {
    SkyJoyConSensor s;
    extern bool motion_joycon_mode(void);
    if (!sensor || controllers_off() || !motion_joycon_mode() || !sky_joycon_sensor(device, &s)) return false;
    // (the right Joy-Con: an aimed item, the menu pointer, the sword)
    if (device == 1 && !aim_pointer(&s) && !panel_pointer(&s)) {
        // Physical Sheathing (vr_input.c): the sword comes out only from over the shoulder, so a swing of the right
        // Joy-Con doesn't draw it - while it is sheathed and Link is on his feet with nothing in use
        // (player_bare_on_foot, hle/player.c), the game sees the Joy-Con held still where it points: no turning (gyro)
        // and gravity only. Its pointing stays, and so does everything with an item out, in water, climbing, flying
        // (the Loftwing steers by tilt) and with the sword drawn.
        extern bool player_bare_on_foot(void);
        extern int vr_sword_sheathed(void);
        extern bool vr_running(void);
        if (g_vr_config.physical_sheathing && g_vr_config.view_mode == 1 && vr_running() && vr_sword_sheathed() == 1 &&
            player_bare_on_foot()) {
            for (int i = 0; i < 3; i++) s.gyro[i] = 0, s.accel[i] = -s.direction[i * 3 + 2];
            static int said;
            if (!said++) LOG("[vr] physical sheathing: a swing of the right controller doesn't draw the sword\n");
        }
        skyward_strike_assist(&s);
        bomb_gesture(&s);  // (a bomb held: thrown or rolled by where the hand is)
    }
    {   // (physical sheathing, vr_input.c: a grab over the shoulder draws the sword with a swing of the Joy-Con)
        extern bool vr_input_draw_swing(float gyro[3], float accel[3]);
        if (device == 1) vr_input_draw_swing(s.gyro, s.accel);
        extern bool vr_input_shield_flick(float gyro[3], float accel[3]);  // (the left grip raising the shield)
        if (device == 0) vr_input_shield_flick(s.gyro, s.accel);
    }
    {   // (diagnostics: SKYRECOMP_BOMB_LOG=1 - Link's action and flags as they change, and the right Joy-Con's motion
        // sent to the game: its peaks since the last line, and 10 lines a second while it turns fast)
        static int log = -1;
        if (log < 0) log = getenv("SKYRECOMP_BOMB_LOG") != NULL;
        if (log && device == 1) {
            extern int player_link_action(void);
            extern uint32_t player_link_flags(void);
            static int last_action = -2;
            static uint32_t last_flags;
            static float peak_g, peak_a;
            static uint64_t next;
            const float g = sqrtf(s.gyro[0] * s.gyro[0] + s.gyro[1] * s.gyro[1] + s.gyro[2] * s.gyro[2]);
            const float a = sqrtf(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] + s.accel[2] * s.accel[2]);
            peak_g = fmaxf(peak_g, g), peak_a = fmaxf(peak_a, a);
            const int action = player_link_action();
            const uint32_t flags = player_link_flags();
            const uint64_t now = SDL_GetTicksNS();
            // (the motion the game was given, sample by sample: dumped when Link starts a new kind of action - a throw,
            // a roll - with the last BOMB_RING samples before it, to see which motion the game takes as which)
            enum { BOMB_RING = 48 };
            static struct { uint64_t t; float g[3], a[3], y[3], z[3]; } ring[BOMB_RING];
            static int head;
            ring[head % BOMB_RING].t = now;
            memcpy(ring[head % BOMB_RING].g, s.gyro, sizeof s.gyro);
            memcpy(ring[head % BOMB_RING].a, s.accel, sizeof s.accel);
            memcpy(ring[head % BOMB_RING].y, s.direction + 3, 3 * sizeof(float));
            memcpy(ring[head % BOMB_RING].z, s.direction + 6, 3 * sizeof(float));
            head++;
            const bool pull = g_bomb_sword_pull;  // (the sword came out with a bomb in hand - bomb_gesture)
            g_bomb_sword_pull = false;
            if (action != last_action || flags != last_flags || pull) {
                LOG("[bomb-log] action %#x flags %08x (was %#x %08x) | since: gyro peak %.2f accel peak %.2f\n", action,
                    flags, last_action, last_flags, peak_g, peak_a);
                const bool routine = action == 0 || action == 3 || action == 0x3f || action == 0x40 || action == 0x43 ||
                                     action == -1 || action == 8 || action == 0x10 || action == 7;
                if ((action != last_action && !routine) || pull) {
                    static int n;
                    LOG("[bomb-dump] #%d %s %#x from %#x: the motion given before it (ms before; gyro rev/s; accel g; "
                        "pointing y-axis; z-axis, all in the player's frame: z up)\n", ++n,
                        pull ? "THE SWORD CAME OUT WITH A BOMB IN HAND, action" : "action", action, last_action);
                    for (int i = head - BOMB_RING; i < head; i++) {
                        if (i < 0) continue;
                        const typeof(ring[0]) *r = &ring[i % BOMB_RING];
                        LOG("[bomb-dump] #%d %5.0f | g %6.2f %6.2f %6.2f | a %6.2f %6.2f %6.2f | y %5.2f %5.2f %5.2f | z %5.2f %5.2f %5.2f\n",
                            n, (double)(now - r->t) / 1e6, r->g[0], r->g[1], r->g[2], r->a[0], r->a[1], r->a[2], r->y[0],
                            r->y[1], r->y[2], r->z[0], r->z[1], r->z[2]);
                    }
                }
                last_action = action, last_flags = flags, peak_g = peak_a = 0;
            }
            if (g > 0.5f && now >= next) {
                next = now + 100000000ull;
                LOG("[bomb-log] motion gyro %.2f %.2f %.2f accel %.2f %.2f %.2f dir-y %.2f %.2f %.2f\n", s.gyro[0], s.gyro[1],
                    s.gyro[2], s.accel[0], s.accel[1], s.accel[2], s.direction[3], s.direction[4], s.direction[5]);
            }
        }
    }
    memcpy(sensor, &s, sizeof s);
    return true;
}

#else
bool vr_menu_beam(float o[3], float d[3], float *len) { (void)o; (void)d; (void)len; return false; }
float vr_menu_scroll(void) { return 0; }
bool vr_menu_ray(float *u, float *v, bool *click) { (void)u; (void)v; if (click) *click = false; return false; }
bool vr_menu_pointer_active(void) { return false; }
void vr_pointer_direction(float u, float v, float out[9]) {
    (void)u; (void)v;
    for (int i = 0; i < 9; i++) out[i] = i % 4 == 0;
}
void vr_controls_set_hud_panel(const float m[16]) { (void)m; }
bool vr_joycon_state(VrJoyConState *state) { (void)state; return false; }
bool vr_joycon_sensor(int device, VrJoyConSensor *sensor) { (void)device; (void)sensor; return false; }
#endif
