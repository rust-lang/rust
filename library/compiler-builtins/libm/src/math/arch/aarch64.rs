use core::arch::aarch64::{
    float32x2_t, float64x1_t, vdup_n_f32, vdup_n_f64, vget_lane_f32, vget_lane_f64, vrndn_f32,
    vrndn_f64,
};

pub fn rint(x: f64) -> f64 {
    // SAFETY: only requires target_feature=neon, ensured by `cfg_if` in parent module.
    let x_vec: float64x1_t = unsafe { vdup_n_f64(x) };

    // SAFETY: only requires target_feature=neon, ensured by `cfg_if` in parent module.
    let result_vec: float64x1_t = unsafe { vrndn_f64(x_vec) };

    // SAFETY: only requires target_feature=neon, ensured by `cfg_if` in parent module.
    let result: f64 = unsafe { vget_lane_f64::<0>(result_vec) };

    result
}

pub fn rintf(x: f32) -> f32 {
    // There's a scalar form of this instruction (FRINTN) but core::arch doesn't expose it, so we
    // have to use the vector form and drop the other lanes afterwards.

    // SAFETY: only requires target_feature=neon, ensured by `cfg_if` in parent module.
    let x_vec: float32x2_t = unsafe { vdup_n_f32(x) };

    // SAFETY: only requires target_feature=neon, ensured by `cfg_if` in parent module.
    let result_vec: float32x2_t = unsafe { vrndn_f32(x_vec) };

    // SAFETY: only requires target_feature=neon, ensured by `cfg_if` in parent module.
    let result: f32 = unsafe { vget_lane_f32::<0>(result_vec) };

    result
}
