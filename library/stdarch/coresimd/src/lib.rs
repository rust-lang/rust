//! SIMD and vendor intrinsics support library.
//!
//! This documentation is only for one particular architecture, you can find
//! others at:
//!
//! * [i686](https://rust-lang-nursery.github.io/stdsimd/i686/stdsimd/)
//! * [`x86_64`](https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/)
//! * [arm](https://rust-lang-nursery.github.io/stdsimd/arm/stdsimd/)
//! * [aarch64](https://rust-lang-nursery.github.io/stdsimd/aarch64/stdsimd/)

#![cfg_attr(feature = "strict", deny(warnings))]
#![allow(dead_code)]
#![allow(unused_features)]
#![feature(const_fn, link_llvm_intrinsics, platform_intrinsics, repr_simd,
           simd_ffi, target_feature, cfg_target_feature, i128_type, asm,
           const_atomic_usize_new, stmt_expr_attributes)]
#![cfg_attr(test, feature(proc_macro, test, repr_align, attr_literals))]
#![cfg_attr(feature = "cargo-clippy",
            allow(inline_always, too_many_arguments, cast_sign_loss,
                  cast_lossless, cast_possible_wrap,
                  cast_possible_truncation, cast_precision_loss,
                  shadow_reuse, cyclomatic_complexity, similar_names,
                  many_single_char_names))]
#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
extern crate stdsimd_test;

#[cfg(test)]
extern crate test;

/// Platform independent SIMD vector types and operations.
pub mod simd {
    pub use v128::*;
    pub use v256::*;
    pub use v512::*;
    pub use v64::*;
}

/// Platform dependent vendor intrinsics.
pub mod vendor {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use x86::*;

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    pub use arm::*;

    #[cfg(target_arch = "aarch64")]
    pub use aarch64::*;

    // FIXME: rust does not expose the nvptx and nvptx64 targets yet
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64",
                  target_arch = "arm", target_arch = "aarch64")))]
    pub use nvptx::*;

    #[cfg(
        // x86/x86_64:
        any(target_arch = "x86", target_arch = "x86_64")
    )]
    pub use runtime::{__unstable_detect_feature, __Feature};
}

#[cfg(
    // x86/x86_64:
    any(target_arch = "x86", target_arch = "x86_64")
)]
#[macro_use]
mod runtime;

#[macro_use]
mod macros;
mod simd_llvm;
mod v128;
mod v256;
mod v512;
mod v64;

/// 32-bit wide vector tpyes
mod v32 {
    use simd_llvm::*;

    define_ty! { i16x2, i16, i16 }
    define_impl! { i16x2, i16, 2, i16x2, x0, x1 }
    define_ty! { u16x2, u16, u16 }
    define_impl! { u16x2, u16, 2, i16x2, x0, x1 }

    define_ty! { i8x4, i8, i8, i8, i8 }
    define_impl! { i8x4, i8, 4, i8x4, x0, x1, x2, x3 }
    define_ty! { u8x4, u8, u8, u8, u8 }
    define_impl! { u8x4, u8, 4, i8x4, x0, x1, x2, x3 }

    define_casts!(
        (i16x2, i64x2, as_i64x2),
        (u16x2, i64x2, as_i64x2),
        (i8x4, i32x4, as_i32x4),
        (u8x4, i32x4, as_i32x4)
    );
}

/// 16-bit wide vector tpyes
mod v16 {
    use simd_llvm::*;

    define_ty! { i8x2, i8, i8 }
    define_impl! { i8x2, i8, 2, i8x2, x0, x1 }
    define_ty! { u8x2, u8, u8 }
    define_impl! { u8x2, u8, 2, i8x2, x0, x1 }

    define_casts!((i8x2, i64x2, as_i64x2), (u8x2, i64x2, as_i64x2));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(target_arch = "aarch64")]
mod aarch64;

mod nvptx;
