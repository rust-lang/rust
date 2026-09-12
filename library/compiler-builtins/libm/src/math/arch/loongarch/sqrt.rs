use core::arch::asm;

#[cfg(target_feature = "d")]
pub fn sqrt(mut x: f64) -> f64 {
    // SAFETY: `fsqrt.d` is available with `d` and has no side effects.
    unsafe {
        asm!(
            "fsqrt.d {x}, {x}",
            x = inout(freg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}

#[cfg(target_feature = "f")]
pub fn sqrtf(mut x: f32) -> f32 {
    // SAFETY: `fsqrt.s` is available with `f` and has no side effects.
    unsafe {
        asm!(
            "fsqrt.s {x}, {x}",
            x = inout(freg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}
