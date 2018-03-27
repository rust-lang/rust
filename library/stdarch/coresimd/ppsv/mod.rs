//! Portable Packed-SIMD Vectors.
//!
//! These types are:
//!
//! * portable: work correctly on all architectures,
//! * packed: have a size fixed at compile-time.
//!
//! These two terms are the opposites of:
//!
//! * architecture-specific: only available in a particular architecture,
//! * scalable: the vector's size is dynamic.
//!
//! This module is structured as follows:
//!
//! * `api`: defines the API of the portable packed vector types.
//! * `v{width}`: defines the portable vector types for a particular `width`.
//!
//! The portable packed vector types are named using the following schema:
//! `{t}{l_w}x{l_n}`:
//!
//! * `t`: type - single letter corresponding to the following Rust literal
//! types:   * `i`: signed integer
//!   * `u`: unsigned integer
//!   * `f`: floating point
//!   * `b`: boolean
//! * `l_w`: lane width in bits
//! * `l_n`: number of lanes
//!
//! For example, `f32x4` is a vector type containing four 32-bit wide
//! floating-point numbers. The total width of this type is 32 bit times 4
//! lanes, that is, 128 bits, and is thus defined in the `v128` module.

#[macro_use]
mod api;

mod v128;
mod v16;
mod v256;
mod v32;
mod v512;
mod v64;

pub use self::v128::*;
pub use self::v16::*;
pub use self::v256::*;
pub use self::v32::*;
pub use self::v512::*;
pub use self::v64::*;

/// Safe lossless bitwise conversion from `T` to `Self`.
pub trait FromBits<T>: ::marker::Sized {
    /// Safe lossless bitwise from `T` to `Self`.
    fn from_bits(T) -> Self;
}

/// Safe lossless bitwise conversion from `Self` to `T`.
pub trait IntoBits<T>: ::marker::Sized {
    /// Safe lossless bitwise transmute from `self` to `T`.
    fn into_bits(self) -> T;
}

// FromBits implies IntoBits.
impl<T, U> IntoBits<U> for T
where
    U: FromBits<T>,
{
    #[inline]
    fn into_bits(self) -> U {
        debug_assert!(::mem::size_of::<Self>() == ::mem::size_of::<U>());
        U::from_bits(self)
    }
}

// FromBits (and thus IntoBits) is reflexive.
impl<T> FromBits<T> for T {
    #[inline]
    fn from_bits(t: Self) -> Self {
        t
    }
}
