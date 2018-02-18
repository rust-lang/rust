/// Platform independent SIMD vector types and operations.
pub mod simd {
    pub use coresimd::v128::*;
    pub use coresimd::v256::*;
    pub use coresimd::v512::*;
    pub use coresimd::v64::*;
}

/// Platform dependent vendor intrinsics.
pub mod vendor {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub use coresimd::x86::*;

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    pub use coresimd::arm::*;

    #[cfg(target_arch = "aarch64")]
    pub use coresimd::aarch64::*;

    // FIXME: rust does not expose the nvptx and nvptx64 targets yet
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64",
                  target_arch = "arm", target_arch = "aarch64")))]
    pub use coresimd::nvptx::*;
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

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod arm;
#[cfg(target_arch = "aarch64")]
mod aarch64;

mod nvptx;
