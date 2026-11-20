// VR: the Beetle launches from the right controller, along where it points (VR Options > Aim Items with the Controller).
//
// The Beetle is an actor of its own (vtable BEETLE_VTABLE) with its own update (BEETLE_UPDATE, vtable slot 20, called
// through the vtable: hle_override). Its fields, measured headset-free (SKYRECOMP_BEETLE_SNAP, in a test build) over a
// launch and its first second of flight:
//  - position +0x144 (the usual actor position), with copies of past positions at +0x1b0, +0x1bc, +0x1c8;
//  - heading +0x13c pitch and +0x13e yaw, 16-bit angles (0x10000 a turn; pitch negative upward). Checked against its
//    path: frames 3170..3260 it flew at yaw -26.1 deg (0xed76) and about 17 deg up (0xf30a..0xf4bf); +0x1d4 / +0x1d6
//    hold -pitch / yaw;
//  - velocity +0x1e8 (vec3, about 9 units an update), recomputed from the heading every update, so the heading steers.
// Link's actor says when it flies (runtime/hle/player.c player_beetle_flying: +0x74 0x0f, launch to catch).
//
// On the Beetle's first update in flight its position becomes the right controller's (a little ahead along it) and
// its heading and velocity where the controller points, its speed kept. Straight from the controller's tracked pose,
// nothing from the game's Joy-Con simulation.
//
// Steering: after each of its updates while you steer it, its pitch and yaw are the controller's and its roll (+0x140,
// the third of the actor's angles: the game's own flight writes it, and zeroes it at the launch) the controller's twist
// about where it points. Its velocity goes along the new heading at the game's speed. The controller is read against
// the view as it was at the launch (vr_rig_controller_aim_frozen): the game's camera, which the eyes follow in flight,
// turns to chase the Beetle, and read against it the controller would turn with it and keep steering. So the Beetle
// points where the controller points, as if held: turn it 30 degrees right and it flies 30 degrees right of where it
// was launched.
// The game's return is left to it: steering ends for the flight when the game's own update turns it more than
// STEER_TAKEN degrees from where it was steered (a bounce, a recall), or toward Link HOMING_UPDATES updates running
// while the controller doesn't (its return). (Not when the game's camera stops following it - main+0x1829ea8 bit 16:
// in VR that came ~205 updates in, and the flight went on under the Joy-Con simulation, 2026-11-19.)
// SKYRECOMP_VR_BEETLE_JOYCON=1: the launch only, steered by the game's Joy-Con simulation as before.
// SKYRECOMP_VR_BEETLE_ROLL_SIGN=-1 turns its roll the other way. SKYRECOMP_BEETLE_LOG logs every steered update.
// Test (no headset): SKYRECOMP_BEETLE_TEST_DIR="x y z" launches every Beetle along that world direction from Link's eye.
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"
#include "../vr/vr_hud.h"

#define BEETLE_VTABLE (MAIN_BASE + 0x1570498)
#define BEETLE_UPDATE (MAIN_BASE + 0x27e688)
#define BEETLE_POS 0x144
#define BEETLE_PITCH 0x13c
#define BEETLE_YAW 0x13e
#define BEETLE_ROLL 0x140
#define STEER_TAKEN 30.0f      // (degrees the game's own update may turn it before steering lets go)
#define HOMING_DEG 1.0f        // (degrees an update of the game's turns it toward Link that count as its return..)
#define HOMING_UPDATES 3       // (..this many updates running)
#define STEER_PITCH_MAX 75.0f  // (degrees up or down)
#define STEER_MAX_UPDATES 1200 // (a flight steered no longer than this, whatever the signs say)
#define BEETLE_PITCH2 0x1d4
#define BEETLE_YAW2 0x1d6
#define BEETLE_VEL 0x1e8
static const int k_past_pos[] = {0x1b0, 0x1bc, 0x1c8};
#define FROM_HAND_AHEAD 8.0f  // (units along the controller: out of the hand)

bool vr_running(void);
bool vr_rig_controller_aim(float o[3], float d[3]);
bool vr_rig_controller_aim_frozen(bool freeze, float d[3], float up[3]);
bool player_eye(float eye[3]);
bool player_beetle_flying(void);

static GuestFn g_update;
static float g_test_dir[3];
static bool g_test_on;

static bool launch_ray(float o[3], float d[3]) {
    if (g_test_on) {
        if (!player_eye(o)) return false;
        memcpy(d, g_test_dir, sizeof g_test_dir);
        for (int k = 0; k < 3; k++) o[k] += d[k] * 30.0f;
        return true;
    }
    if (!(g_vr_hud_config.aim_with_controller && vr_running() && vr_rig_controller_aim(o, d))) return false;
    for (int k = 0; k < 3; k++) o[k] += d[k] * FROM_HAND_AHEAD;
    return true;
}

static int16_t angle16(float radians) { return (int16_t)lrintf(radians * (32768.0f / (float)M_PI)); }

// Pitch, yaw and roll (16-bit, the Beetle's conventions) of a direction d and an up axis u (both unit, game world)
static void heading_of(const float d[3], const float u[3], int16_t *pitch, int16_t *yaw, int16_t *roll) {
    const float flat = sqrtf(d[0] * d[0] + d[2] * d[2]);
    const float lim = STEER_PITCH_MAX * (float)M_PI / 180.0f;
    *yaw = angle16(atan2f(d[0], d[2]));
    *pitch = angle16(-fmaxf(-lim, fminf(lim, atan2f(d[1], flat))));
    // (the twist about d: u against the level up through d, u0 = Y - d d.y, and its right, d x u0)
    float u0[3] = {-d[0] * d[1], 1.0f - d[1] * d[1], -d[2] * d[1]};
    const float n = sqrtf(u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
    *roll = 0;
    if (n < 1e-3f) return;  // (straight up or down: no level to roll from)
    for (int k = 0; k < 3; k++) u0[k] /= n;
    const float r0[3] = {d[1] * u0[2] - d[2] * u0[1], d[2] * u0[0] - d[0] * u0[2], d[0] * u0[1] - d[1] * u0[0]};
    static float sign;
    if (sign == 0) { const char *e = getenv("SKYRECOMP_VR_BEETLE_ROLL_SIGN"); sign = e && atof(e) < 0 ? -1.0f : 1.0f; }
    *roll = angle16(sign * atan2f(u[0] * r0[0] + u[1] * r0[1] + u[2] * r0[2], u[0] * u0[0] + u[1] * u0[1] + u[2] * u0[2]));
}

static float angle_between16(int16_t a, int16_t b) { return fabsf((float)(int16_t)(a - b)) * (180.0f / 32768.0f); }

static void dir_of(int16_t pitch, int16_t yaw, float d[3]) {  // (pitch negative upward)
    const float p = -(float)pitch * ((float)M_PI / 32768.0f), y = (float)yaw * ((float)M_PI / 32768.0f);
    d[0] = cosf(p) * sinf(y), d[1] = sinf(p), d[2] = cosf(p) * cosf(y);
}
static float degrees_between(const float a[3], const float b[3]) {  // (unit vectors)
    const float c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return acosf(fmaxf(-1.0f, fminf(1.0f, c))) * (180.0f / (float)M_PI);
}

// (RE, SKYRECOMP_BEETLE_LOG=1) Each 32-bit word of the Beetle's first 0x1000 bytes that changes after holding still for
// 30 updates - its state changes, among them its return - logged with the update it changed in.
#define WATCH_SIZE 0x1000
static void beetle_watch(uint64_t a, int update, bool first) {
    static int on = -1;
    if (on < 0) on = getenv("SKYRECOMP_BEETLE_LOG") != NULL;
    if (!on) return;
    static uint32_t last[WATCH_SIZE / 4];
    static uint16_t still[WATCH_SIZE / 4];
    const uint32_t *w = (const uint32_t *)(uintptr_t)a;
    int said = 0;
    for (int i = 0; i < WATCH_SIZE / 4; i++) {
        const uint32_t now = w[i];
        if (!first && now != last[i]) {
            if (still[i] >= 30 && said++ < 24)
                LOG("[beetle-watch] update %d: +0x%03x %08x -> %08x (still %d)\n", update, i * 4, last[i], now, still[i]);
            still[i] = 0;
        } else if (still[i] < 60000)
            still[i]++;
        last[i] = now;
    }
}

// After the launch: each update while you steer it, the Beetle's heading is the controller's (above)
static void beetle_steer(uint64_t a, bool first) {
    static bool steering;
    static int off = -1;
    static int16_t last_pitch, last_yaw;
    static int updates, homing;
    if (off < 0) off = getenv("SKYRECOMP_VR_BEETLE_JOYCON") != NULL;
    if (first) steering = !off && !g_test_on, updates = 0, homing = 0;
    beetle_watch(a, updates, first);
    if (!steering) { updates++; return; }
    const char *why = NULL;
    int16_t *ang = (int16_t *)(uintptr_t)(a + BEETLE_PITCH);
    float d[3], u[3];
    const bool have = g_vr_hud_config.aim_with_controller && vr_running() && vr_rig_controller_aim_frozen(first, d, u);
    int16_t pitch = 0, yaw = 0, roll = 0;
    if (have) heading_of(d, u, &pitch, &yaw, &roll);
    if (!first && (angle_between16(ang[0], last_pitch) > STEER_TAKEN || angle_between16(ang[1], last_yaw) > STEER_TAKEN))
        why = "the game turned it sharply (a bounce or a recall)";
    else if (++updates > STEER_MAX_UPDATES) why = "a long flight";
    else if (!have) why = "no controller";
    else if (!first) {
        // (its return: the game's own update turning it toward Link, HOMING_UPDATES running, while the controller
        // isn't turning it toward him)
        float eye[3];
        const float *p = (const float *)(uintptr_t)(a + BEETLE_POS);
        if (player_eye(eye)) {
            float to[3] = {eye[0] - p[0], eye[1] - p[1], eye[2] - p[2]};
            const float n = sqrtf(to[0] * to[0] + to[1] * to[1] + to[2] * to[2]);
            if (n > 1.0f) {
                for (int k = 0; k < 3; k++) to[k] /= n;
                float game[3], ours[3], want[3];
                dir_of(ang[0], ang[1], game), dir_of(last_pitch, last_yaw, ours), dir_of(pitch, yaw, want);
                const float g = degrees_between(game, to), o = degrees_between(ours, to), w = degrees_between(want, to);
                homing = o - g > HOMING_DEG && w > o - HOMING_DEG ? homing + 1 : 0;
                if (homing >= HOMING_UPDATES) why = "the game turned it toward Link (its return)";
            }
        }
    }
    if (why) {
        steering = false;
        LOG("[beetle] steering let go after %d updates: %s\n", updates, why);
        return;
    }
    ang[0] = pitch, ang[1] = yaw, ang[2] = roll;
    *(int16_t *)(uintptr_t)(a + BEETLE_PITCH2) = (int16_t)-pitch, *(int16_t *)(uintptr_t)(a + BEETLE_YAW2) = yaw;
    last_pitch = pitch, last_yaw = yaw;
    float *v = (float *)(uintptr_t)(a + BEETLE_VEL);
    const float speed = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (speed > 0) {  // (along the heading as set)
        float h[3];
        dir_of(pitch, yaw, h);
        for (int k = 0; k < 3; k++) v[k] = h[k] * speed;
    }
    if (getenv("SKYRECOMP_BEETLE_LOG") || first)
        LOG("[beetle] steered (update %d): pitch %04x yaw %04x roll %04x, speed %.1f\n", updates, (uint16_t)pitch,
            (uint16_t)yaw, (uint16_t)roll, speed);
}

static void beetle_update_hook(Cpu *c) {
    const uint64_t a = c->x[0];
    g_update(c);
    if (!a || *(const uint64_t *)(uintptr_t)a != BEETLE_VTABLE) return;
    static uint64_t launched;  // (the Beetle this flight was turned for: once a flight)
    if (!player_beetle_flying()) { launched = 0; return; }
    if (launched == a) { beetle_steer(a, false); return; }
    launched = a;
    float o[3], d[3];
    if (!launch_ray(o, d)) return;
    float *p = (float *)(uintptr_t)(a + BEETLE_POS), *v = (float *)(uintptr_t)(a + BEETLE_VEL);
    const float speed = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const int16_t yaw = angle16(atan2f(d[0], d[2]));
    const int16_t pitch = angle16(-atan2f(d[1], sqrtf(d[0] * d[0] + d[2] * d[2])));
    *(int16_t *)(uintptr_t)(a + BEETLE_PITCH) = pitch, *(int16_t *)(uintptr_t)(a + BEETLE_YAW) = yaw;
    *(int16_t *)(uintptr_t)(a + BEETLE_PITCH2) = (int16_t)-pitch, *(int16_t *)(uintptr_t)(a + BEETLE_YAW2) = yaw;
    for (int k = 0; k < 3; k++) p[k] = o[k], v[k] = d[k] * speed;
    for (unsigned i = 0; i < sizeof k_past_pos / sizeof k_past_pos[0]; i++)  // (no sweep from the game's start to here)
        memcpy((void *)(uintptr_t)(a + k_past_pos[i]), o, 12);
    static int said;
    if (said++ < 3 || getenv("SKYRECOMP_BEETLE_LOG"))
        LOG("[beetle] launched along the controller: %.3f %.3f %.3f (pitch %04x yaw %04x), speed %.1f\n", d[0], d[1], d[2],
            (uint16_t)pitch, (uint16_t)yaw, speed);
    beetle_steer(a, true);  // (from here, as the controller turns: read against the view at the launch)
}

void vr_beetle_init(void) {  // (main.c, after the game code is loaded)
    const char *e = getenv("SKYRECOMP_BEETLE_TEST_DIR");
    if (e && sscanf(e, "%f %f %f", &g_test_dir[0], &g_test_dir[1], &g_test_dir[2]) == 3) {
        const float n = sqrtf(g_test_dir[0] * g_test_dir[0] + g_test_dir[1] * g_test_dir[1] + g_test_dir[2] * g_test_dir[2]);
        if (n > 0) {
            for (int k = 0; k < 3; k++) g_test_dir[k] /= n;
            g_test_on = true;
        }
    }
    g_update = dispatch_lookup(BEETLE_UPDATE);
    if (g_update) hle_override(BEETLE_UPDATE, beetle_update_hook, "vr_beetle update");
    LOG("[beetle] update %s%s\n", g_update ? "hooked" : "not found (the Beetle keeps the game's aim)",
        g_test_on ? "; test: every launch along SKYRECOMP_BEETLE_TEST_DIR" : "");
}
