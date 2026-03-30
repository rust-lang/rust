//! NB: `frintx` is technically the correct instruction for C's `rint`. However, in Rust (and LLVM
//! by default), `rint` is identical to `roundeven` (no fpenv interaction) so we use the
//! side-effect-free `frintn`.
//!
//! In general, C code that calls Rust's libm should assume that fpenv is ignored.

use core::arch::asm;

#[cfg(all(f16_enabled, target_feature = "fp16"))]
pub fn rintf16(mut x: f16) -> f16 {
    // SAFETY: `frintn` is available for `f16` with `fp16` (implies `neon`) and has no side effects.
    //
    // `frintn` is always round-to-nearest which does not match the C specification, but Rust does
    // not support rounding modes.
    unsafe {
        asm!(
            "frintn {x:h}, {x:h}",
            x = inout(vreg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}

pub fn rintf(mut x: f32) -> f32 {
    // SAFETY: `frintn` is available with neon and has no side effects.
    //
    // `frintn` is always round-to-nearest which does not match the C specification, but Rust does
    // not support rounding modes.
    unsafe {
        asm!(
            "frintn {x:s}, {x:s}",
            x = inout(vreg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}

pub fn rint(mut x: f64) -> f64 {
    // SAFETY: `frintn` is available with neon and has no side effects.
    //
    // `frintn` is always round-to-nearest which does not match the C specification, but Rust does
    // not support rounding modes.
    unsafe {
        asm!(
            "frintn {x:d}, {x:d}",
            x = inout(vreg) x,
            options(nomem, nostack, pure)
        );
    }
    x
}
