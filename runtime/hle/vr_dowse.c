// VR: dowsing points where the blade in your right hand points (1st person).
//
// Dowsing (D-pad up with the sword) is Link's action 0x68 (his actor +0x468, LINK_ACTION in vr_sword.c). While it
// lasts the game's camera looks from his eye along two 16-bit angles of his actor: +0x420 pitch (negative upward) and
// +0x422 yaw (0x10000 a turn; 0 along +z, a quarter turn along +x) - the direction it dowses along. The right stick or
// the Joy-Con's pointing turns them. Measured 2026-11-20 headset-free (SKYRECOMP_LINK_DIFF): dowsing, a turn right
// and back moved +0x422 0x7e7d -> 0x5589 -> 0x7e7d (and its copy at +0x645e), a look up and back moved +0x420
// 0 -> 0xec18 -> 0; nothing else in his actor did.
//
// In VR 1st person the view is your head and not the game's camera, and the Joy-Con's pointing (the controller's
// turn from where the game centred it) turned the dowsing somewhere unrelated to where you looked or pointed. Instead,
// around each of Link's updates while he dowses, the two angles are set to the drawn blade's direction in the world
// (vr_rig_held_world: the sword model's +x; the right controller's -z while the drawn pose isn't known) - before
// the update, so the game dowses along it, and after, so what it draws and the next frame start from it.
// SKYRECOMP_DOWSE_LOG=1 logs the angles every 30 updates. VR Options > Aim Items with the Controller off leaves dowsing
// to the game.
// Test (no headset): SKYRECOMP_DOWSE_TEST_DIR="x y z" dowses along that world direction.
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../hle.h"
#include "../vr/vr_hud.h"
#include "../vr/vr_rig.h"
#include "../vr/vr_settings.h"

#define LINK_ACTION 0x468
#define ACTION_DOWSE 0x68
#define DOWSE_PITCH 0x420
#define DOWSE_YAW 0x422
#define DOWSE_YAW_COPY 0x645e

bool vr_running(void);

static bool test_dir(float d[3]) {
    static int init;
    static bool on;
    static float t[3];
    if (!init) {
        init = 1;
        const char *e = getenv("SKYRECOMP_DOWSE_TEST_DIR");
        on = e && sscanf(e, "%f %f %f", &t[0], &t[1], &t[2]) == 3;
    }
    if (on) d[0] = t[0], d[1] = t[1], d[2] = t[2];
    return on;
}

// The blade's direction in the world (unit), or false
static bool blade_dir(float d[3]) {
    if (test_dir(d)) return true;
    if (!vr_running() || g_vr_config.view_mode != 1 || !g_vr_hud_config.aim_with_controller) return false;
    float m[12], n = 0;
    if (vr_rig_held_world(0, m)) {
        for (int i = 0; i < 3; i++) d[i] = m[i * 4], n += d[i] * d[i];
    }
    if (n < 1e-8f) {
        if (!vr_rig_controller_pose(1, m)) return false;
        n = 0;
        for (int i = 0; i < 3; i++) d[i] = -m[i * 4 + 2], n += d[i] * d[i];
    }
    n = sqrtf(n);
    if (!(n > 1e-4f)) return false;
    for (int i = 0; i < 3; i++) d[i] /= n;
    return true;
}

static int16_t angle16(float radians) { return (int16_t)(int32_t)lrintf(radians * (32768.0f / (float)M_PI)); }

// ---- the dowsing ring (the game's plain circle at its screen's centre: texpatch "th_circle") over the sword's gem
// While Link dowses, runtime/nvn/vulkan/exec.c aim_piece draws that ring in each eye at this point instead of on the HUD
// panel (through the 3D reticle's path, runtime/hle/vr_aim.c vr_aim_target): above the gem in the drawn sword's hilt,
// so it floats where you look along the blade. The gem is DOWSE_GEM_ALONG units along the blade (the sword model's +x)
// from the model's origin; the ring DOWSE_GEM_UP units above it (world up). SKYRECOMP_VR_DOWSE_RING="along up" moves it.
#define DOWSE_GEM_ALONG 6.0f
#define DOWSE_GEM_UP 6.0f
extern uint64_t renderer_present_count(void);
static _Atomic uint64_t g_dowse_frame = ~0ull;  // (the last frame Link was seen dowsing)

bool vr_dowse_active(void) {
    const uint64_t f = g_dowse_frame;
    return f != ~0ull && f + 10 >= renderer_present_count();
}

bool vr_dowse_ring(float p[3]) {
    if (!vr_dowse_active() || !vr_running() || g_vr_config.view_mode != 1) return false;
    static int init;
    static float along = DOWSE_GEM_ALONG, up = DOWSE_GEM_UP;
    if (!init) {
        init = 1;
        const char *e = getenv("SKYRECOMP_VR_DOWSE_RING");
        if (e) sscanf(e, "%f %f", &along, &up);
    }
    float m[12], n = 0;
    if (!vr_rig_held_world(0, m)) return false;
    for (int i = 0; i < 3; i++) n += m[i * 4] * m[i * 4];
    n = sqrtf(n);
    if (!(n > 1e-4f)) return false;
    for (int i = 0; i < 3; i++) p[i] = m[i * 4 + 3] + m[i * 4] / n * along + (i == 1 ? up : 0.0f);
    return true;
}

void vr_dowse_update(uint64_t pl, bool after) {
    if (*(const volatile uint32_t *)(uintptr_t)(pl + LINK_ACTION) != ACTION_DOWSE) return;
    g_dowse_frame = renderer_present_count();
    float d[3];
    if (!blade_dir(d)) return;
    const float flat = sqrtf(d[0] * d[0] + d[2] * d[2]);
    const float lim = 80.0f * (float)M_PI / 180.0f;  // (straight up or down has no yaw: kept short of it)
    const float pitch = -fmaxf(-lim, fminf(lim, atan2f(d[1], flat)));
    const int16_t p = angle16(pitch);
    int16_t *ang = (int16_t *)(uintptr_t)(pl + DOWSE_PITCH);
    const int16_t y = flat > 1e-4f ? angle16(atan2f(d[0], d[2])) : ang[1];
    static int log = -1;
    if (log < 0) log = getenv("SKYRECOMP_DOWSE_LOG") != NULL;
    static unsigned n;
    if (after && (n++ == 0 || (log && n % 30 == 0)))
        LOG("[dowse] along the blade: pitch %.1f yaw %.1f (the game had %.1f %.1f)\n", p * (180.0f / 32768.0f),
            y * (180.0f / 32768.0f), ang[0] * (180.0f / 32768.0f), ang[1] * (180.0f / 32768.0f));
    ang[0] = p, ang[1] = y;
    *(int16_t *)(uintptr_t)(pl + DOWSE_YAW_COPY) = y;
}
