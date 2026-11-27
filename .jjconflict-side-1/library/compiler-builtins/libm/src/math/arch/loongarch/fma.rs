use core::arch::asm;

#[cfg(target_feature = "d")]
pub fn fma(mut x: f64, y: f64, z: f64) -> f64 {
    // SAFETY: `fmadd.d` is available with `d and has no side effects.
    unsafe {
        asm!(
            "fmadd.d {x}, {x}, {y}, {z}",
            x = inout(freg) x,
            y = in(freg) y,
            z = in(freg) z,
            options(nomem, nostack, pure)
        );
    }
    x
}

#[cfg(target_feature = "f")]
pub fn fmaf(mut x: f32, y: f32, z: f32) -> f32 {
    // SAFETY: `fmadd.s` is available with `f` and has no side effects.
    unsafe {
        asm!(
            "fmadd.s {x}, {x}, {y}, {z}",
            x = inout(freg) x,
            y = in(freg) y,
            z = in(freg) z,
            options(nomem, nostack, pure)
        );
    }
    x
}
