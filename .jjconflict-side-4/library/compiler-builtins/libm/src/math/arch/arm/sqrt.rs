#[cfg(target_feature = "vfp2sp")]
pub fn sqrtf(mut x: f32) -> f32 {
    // SAFETY: vsqrt.f32 is available with `vfp2sp`.
    unsafe {
        core::arch::asm!(
            "vsqrt.f32 {x}, {x}",
            x = inout(sreg) x,
            options(nostack, nomem, pure),
        );
    }
    x
}

#[cfg(target_feature = "vfp2")]
pub fn sqrt(mut x: f64) -> f64 {
    // SAFETY: vsqrt.f64 is available with `vfp2`.
    // Can't just use plain old `dreg` because `vfp2` only provides a base
    // level of FPU support with a non-specified amount of fp64 registers.
    // It's fine to use `dreg_low16` here, but use `dreg_low8` instead to
    // keep the requirements as minimal as possible.
    unsafe {
        core::arch::asm!(
            "vsqrt.f64 {x}, {x}",
            x = inout(dreg_low8) x,
            options(nostack, nomem, pure),
        );
    }
    x
}
