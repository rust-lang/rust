use core::arch::asm;

pub fn fmaf(mut x: f32, y: f32, z: f32) -> f32 {
    // SAFETY: `fmadd` is available with neon and has no side effects.
    unsafe {
        asm!(
            "fmadd {x:s}, {x:s}, {y:s}, {z:s}",
            x = inout(vreg) x,
            y = in(vreg) y,
            z = in(vreg) z,
            options(nomem, nostack, pure)
        );
    }
    x
}

pub fn fma(mut x: f64, y: f64, z: f64) -> f64 {
    // SAFETY: `fmadd` is available with neon and has no side effects.
    unsafe {
        asm!(
            "fmadd {x:d}, {x:d}, {y:d}, {z:d}",
            x = inout(vreg) x,
            y = in(vreg) y,
            z = in(vreg) z,
            options(nomem, nostack, pure)
        );
    }
    x
}
