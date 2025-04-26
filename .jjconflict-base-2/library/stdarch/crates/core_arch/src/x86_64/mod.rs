//! `x86_64` intrinsics

#[macro_use]
mod macros;

// Any 1024-byte vector should work
type Tile = crate::core_arch::simd::Simd<u8, 1024>;

/// A tile register, used by AMX instructions.
///
/// This type is the same as the `__tile1024i` type defined by Intel, representing a 1024-byte tile register.
/// Usage of this type typically corresponds to the `amx-tile` and up target features for x86_64.
///
/// This struct contains the tile configuration information as well as the tile itself.
/// The tile configuration information consists of the row count and the size of each column in bytes,
/// with `row * colsb` never exceeding 1024.
///
/// The typical usage pattern looks like
/// ```ignore
/// let tile = MaybeUninit::uninit();
/// let tile_ptr = tile.as_mut_ptr();
///
/// (*tile_ptr).rows = rows;
/// (*tile_ptr).colsb = colsb;
/// __tile_zero(tile_ptr);
///
/// let tile = tile.assume_init();
/// ```
/// Most intrinsics using `__tile1024i` (except for the store intrinsics) have a destination parameter
/// of type `*mut __tile1024i`, and it expects the `rows` and `colsb` fields of the destination
/// to be initialized. After the function call, the whole struct can be assumed to be initialized.
/// Moreover, for dot-product intrinsics, it is UB if the shape of two operands are not compatible
/// as a matrix product or if the shape of the destination doesn't match the expected shape.
#[derive(Copy, Clone, Debug)]
#[allow(non_camel_case_types)]
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub struct __tile1024i {
    pub rows: u16,
    pub colsb: u16,
    tile: Tile,
}

mod fxsr;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::fxsr::*;

mod sse;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse::*;

mod sse2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse2::*;

mod sse41;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse41::*;

mod sse42;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::sse42::*;

mod xsave;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::xsave::*;

mod abm;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::abm::*;

mod avx;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::avx::*;

mod bmi;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bmi::*;
mod bmi2;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bmi2::*;

mod tbm;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::tbm::*;

mod avx512f;
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub use self::avx512f::*;

mod avx512bw;
#[stable(feature = "stdarch_x86_avx512", since = "1.89")]
pub use self::avx512bw::*;

mod bswap;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::bswap::*;

mod rdrand;
#[stable(feature = "simd_x86", since = "1.27.0")]
pub use self::rdrand::*;

mod cmpxchg16b;
#[stable(feature = "cmpxchg16b_intrinsic", since = "1.67.0")]
pub use self::cmpxchg16b::*;

mod adx;
#[stable(feature = "simd_x86_adx", since = "1.33.0")]
pub use self::adx::*;

mod bt;
#[stable(feature = "simd_x86_bittest", since = "1.55.0")]
pub use self::bt::*;

mod avx512fp16;
#[stable(feature = "stdarch_x86_avx512fp16", since = "1.94.0")]
pub use self::avx512fp16::*;

mod amx;
#[unstable(feature = "x86_amx_intrinsics", issue = "126622")]
pub use self::amx::*;

mod movrs;
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub use self::movrs::*;
