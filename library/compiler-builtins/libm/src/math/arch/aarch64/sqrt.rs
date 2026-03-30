use core::arch::asm;

pub fn sqrtf(mut x: f32) -> f32 {
    // SAFETY: `fsqrt` is available with neon and has no side effects.
    unsafe {
        asm!(
            "fsqrt {x:s}, {x:s}",
            x = inout(vreg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}

pub fn sqrt(mut x: f64) -> f64 {
    // SAFETY: `fsqrt` is available with neon and has no side effects.
    unsafe {
        asm!(
            "fsqrt {x:d}, {x:d}",
            x = inout(vreg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}

#[cfg(all(f16_enabled, target_feature = "fp16"))]
pub fn sqrtf16(mut x: f16) -> f16 {
    // SAFETY: `fsqrt` is available for `f16` with `fp16` (implies `neon`) and has no
    // side effects.
    unsafe {
        asm!(
            "fsqrt {x:h}, {x:h}",
            x = inout(vreg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}
