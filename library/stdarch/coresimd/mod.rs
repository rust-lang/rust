/// Platform independent SIMD vector types and operations.
///
/// This is an **unstable** module for portable SIMD operations. This module has
/// not yet gone through an RFC and is likely to change, but feedback is always
/// welcome!
#[unstable(feature = "stdsimd", issue = "0")]
pub mod simd {
    pub use coresimd::v128::*;
    pub use coresimd::v256::*;
    pub use coresimd::v512::*;
    pub use coresimd::v64::*;
}

/// Platform dependent vendor intrinsics.
///
/// This documentation is for the version of this module in the `coresimd`
/// crate, but you probably want to use the [`stdsimd` crate][stdsimd] which
/// should have more complete documentation.
///
/// [stdsimd]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/arch/index.html
///
/// Also note that while this module may appear to contains the intrinsics for
/// only one platform it actually contains intrinsics for multiple platforms
/// compiled in conditionally. For other platforms of stdsimd see:
///
/// * [x86]
/// * [x86_64]
/// * [arm]
/// * [aarch64]
///
/// [x86]: https://rust-lang-nursery.github.io/stdsimd/x86/stdsimd/arch/index.html
/// [x86_64]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/arch/index.html
/// [arm]: https://rust-lang-nursery.github.io/stdsimd/arm/stdsimd/arch/index.html
/// [aarch64]: https://rust-lang-nursery.github.io/stdsimd/aarch64/stdsimd/arch/index.html
#[unstable(feature = "stdsimd", issue = "0")]
pub mod arch {
    /// Platform-specific intrinsics for the `x86` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(target_arch = "x86")]
    pub mod x86 {
        pub use coresimd::x86::*;
    }

    /// Platform-specific intrinsics for the `x86_64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(target_arch = "x86_64")]
    pub mod x86_64 {
        pub use coresimd::x86::*;
        pub use coresimd::x86_64::*;
    }

    /// Platform-specific intrinsics for the `arm` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(target_arch = "arm")]
    pub mod arm {
        pub use coresimd::arm::*;
    }

    /// Platform-specific intrinsics for the `aarch64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(target_arch = "aarch64")]
    pub mod aarch64 {
        pub use coresimd::arm::*;
        pub use coresimd::aarch64::*;
    }
}

#[macro_use]
mod macros;
mod simd_llvm;
mod v128;
mod v256;
mod v512;
mod v64;

/// 32-bit wide vector tpyes
mod v32 {
    #[cfg(not(test))]
    use prelude::v1::*;
    use coresimd::simd_llvm::*;

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
    #[cfg(not(test))]
    use prelude::v1::*;
    use coresimd::simd_llvm::*;

    define_ty! { i8x2, i8, i8 }
    define_impl! { i8x2, i8, 2, i8x2, x0, x1 }
    define_ty! { u8x2, u8, u8 }
    define_impl! { u8x2, u8, 2, i8x2, x0, x1 }

    define_casts!((i8x2, i64x2, as_i64x2), (u8x2, i64x2, as_i64x2));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(target_arch = "aarch64")]
mod aarch64;

mod nvptx;
