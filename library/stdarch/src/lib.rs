//! SIMD support
//!
//! This crate provides the fundamentals of supporting SIMD in Rust. This crate
//! should compile on all platforms and provide `simd` and `vendor` modules at
//! the top-level. The `simd` module contains *portable vector types* which
//! should work across all platforms and be implemented in the most efficient
//! manner possible for the platform at hand. The `vendor` module contains
//! vendor intrinsics that operate over these SIMD types, typically
//! corresponding to a particular CPU instruction
//!
//! ```rust
//! extern crate stdsimd;
//! use stdsimd::simd::u32x4;
//!
//! fn main() {
//!     let a = u32x4::new(1, 2, 3, 4);
//!     let b = u32x4::splat(10);
//!     assert_eq!(a + b, u32x4::new(11, 12, 13, 14));
//! }
//! ```
//!
//! > **Note**: This crate is *nightly only* at the moment, and requires a
//! > nightly rust toolchain to compile.
//!
//! This documentation is only for one particular architecture, you can find
//! others at:
//!
//! * [i686](https://rust-lang-nursery.github.io/stdsimd/i686/stdsimd/)
//! * [`x86_64`](https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/)
//! * [arm](https://rust-lang-nursery.github.io/stdsimd/arm/stdsimd/)
//! * [aarch64](https://rust-lang-nursery.github.io/stdsimd/aarch64/stdsimd/)
//!
//! ## Portability
//!
//! The `simd` module and its types should be portable to all platforms. The
//! runtime characteristics of these types may vary per platform and per CPU
//! feature enabled, but they should always have the most optimized
//! implementation for the target at hand.
//!
//! The `vendor` module provides no portability guarantees. The `vendor` module
//! is per CPU architecture currently and provides intrinsics corresponding to
//! functions for that particular CPU architecture. Note that the functions
//! provided in this module are intended to correspond to CPU instructions and
//! have no runtime support for whether you CPU actually supports the
//! instruction.
//!
//! CPU target feature detection is done via the `cfg_feature_enabled!` macro
//! at runtime. This macro will detect at runtime whether the specified feature
//! is available or not, returning true or false depending on the current CPU.
//!
//! ```
//! #![feature(cfg_target_feature)]
//!
//! #[macro_use]
//! extern crate stdsimd;
//!
//! fn main() {
//!     if cfg_feature_enabled!("avx2") {
//!         println!("avx2 intrinsics will work");
//!     } else {
//!         println!("avx2 intrinsics will not work");
//!         // undefined behavior: may generate a `SIGILL`.
//!     }
//! }
//! ```
//!
//! After verifying that a specified feature is available, use `target_feature`
//! to enable a given feature and use the desired intrinsic.
//!
//! ```ignore
//! # #![feature(cfg_target_feature)]
//! # #![feature(target_feature)]
//! # #[macro_use]
//! # extern crate stdsimd;
//! # fn main() {
//! #     if cfg_feature_enabled!("avx2") {
//! // avx2 specific code may be used in this function
//! #[target_feature = "+avx2"]
//! fn and_256() {
//!     // avx2 feature specific intrinsics will work here!
//!     use stdsimd::vendor::{__m256i, _mm256_and_si256};
//!
//!     let a = __m256i::splat(5);
//!     let b = __m256i::splat(3);
//!
//!     let got = unsafe { _mm256_and_si256(a, b) };
//!
//!     assert_eq!(got, __m256i::splat(1));
//! }
//! #         and_256();
//! #     }
//! # }
//! ```
//!
//! # Status
//!
//! This crate is intended for eventual inclusion into the standard library,
//! but some work and experimentation is needed to get there! First and
//! foremost you can help out by kicking the tires on this crate and seeing if
//! it works for your use case! Next up you can help us fill out the [vendor
//! intrinsics][vendor] to ensure that we've got all the SIMD support
//! necessary.
//!
//! The language support and status of SIMD is also still a little up in the
//! air right now, you may be interested in a few issues along these lines:
//!
//! * [Overal tracking issue for SIMD support]
//!   (https://github.com/rust-lang/rust/issues/27731)
//! * [`cfg_target_feature` tracking issue]
//!   (https://github.com/rust-lang/rust/issues/29717)
//! * [SIMD types currently not sound]
//!   (https://github.com/rust-lang/rust/issues/44367)
//! * [`#[target_feature]` improvements]
//!   (https://github.com/rust-lang/rust/issues/44839)
//!
//! [vendor]: https://github.com/rust-lang-nursery/stdsimd/issues/40

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
                  doc_markdown, many_single_char_names))]
#![no_std]

#[cfg(any(feature = "std", test))]
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

    #[cfg(any(
        // x86/x86_64:
        any(target_arch = "x86", target_arch = "x86_64"),
        // linux + std + (arm|aarch64):
        all(target_os = "linux",
            feature = "std",
            any(target_arch = "arm", target_arch = "aarch64"))
    ))]
    pub use runtime::{__unstable_detect_feature, __Feature};
}

#[cfg(any(
    // x86/x86_64:
    any(target_arch = "x86", target_arch = "x86_64"),
    // linux + std + (arm|aarch64):
    all(target_os = "linux",
        feature = "std",
        any(target_arch = "arm", target_arch = "aarch64"))
))]
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

// FIXME: rust does not expose the nvptx and nvptx64 targets yet
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64",
              target_arch = "arm", target_arch = "aarch64")))]
mod nvptx;
