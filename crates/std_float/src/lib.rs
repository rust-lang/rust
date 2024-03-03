#![cfg_attr(feature = "as_crate", no_std)] // We are std!
#![cfg_attr(
    feature = "as_crate",
    feature(core_intrinsics),
    feature(portable_simd),
    allow(internal_features)
)]
#[cfg(not(feature = "as_crate"))]
use core::simd;
#[cfg(feature = "as_crate")]
use core_simd::simd;

use core::intrinsics::simd as intrinsics;

use simd::{LaneCount, Simd, SupportedLaneCount};

#[cfg(feature = "as_crate")]
mod experimental {
    pub trait Sealed {}
}

#[cfg(feature = "as_crate")]
use experimental as sealed;

use crate::sealed::Sealed;

/// This trait provides a possibly-temporary implementation of float functions
/// that may, in the absence of hardware support, canonicalize to calling an
/// operating system's `math.h` dynamically-loaded library (also known as a
/// shared object). As these conditionally require runtime support, they
/// should only appear in binaries built assuming OS support: `std`.
///
/// However, there is no reason SIMD types, in general, need OS support,
/// as for many architectures an embedded binary may simply configure that
/// support itself. This means these types must be visible in `core`
/// but have these functions available in `std`.
///
/// [`f32`] and [`f64`] achieve a similar trick by using "lang items", but
/// due to compiler limitations, it is harder to implement this approach for
/// abstract data types like [`Simd`]. From that need, this trait is born.
///
/// It is possible this trait will be replaced in some manner in the future,
/// when either the compiler or its supporting runtime functions are improved.
/// For now this trait is available to permit experimentation with SIMD float
/// operations that may lack hardware support, such as `mul_add`.
pub trait StdFloat: Sealed + Sized {
    /// Fused multiply-add.  Computes `(self * a) + b` with only one rounding error,
    /// yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
    /// architecture has a dedicated `fma` CPU instruction.  However, this is not always
    /// true, and will be heavily dependent on designing algorithms with specific target
    /// hardware in mind.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { intrinsics::simd_fma(self, a, b) }
    }

    /// Produces a vector where every lane has the square root value
    /// of the equivalently-indexed lane in `self`
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn sqrt(self) -> Self {
        unsafe { intrinsics::simd_fsqrt(self) }
    }

    /// Produces a vector where every lane has the sine of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn sin(self) -> Self {
        unsafe { intrinsics::simd_fsin(self) }
    }

    /// Produces a vector where every lane has the cosine of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn cos(self) -> Self {
        unsafe { intrinsics::simd_fcos(self) }
    }

    /// Produces a vector where every lane has the exponential (base e) of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn exp(self) -> Self {
        unsafe { intrinsics::simd_fexp(self) }
    }

    /// Produces a vector where every lane has the exponential (base 2) of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn exp2(self) -> Self {
        unsafe { intrinsics::simd_fexp2(self) }
    }

    /// Produces a vector where every lane has the natural logarithm of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn log(self) -> Self {
        unsafe { intrinsics::simd_flog(self) }
    }

    /// Produces a vector where every lane has the base-2 logarithm of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn log2(self) -> Self {
        unsafe { intrinsics::simd_flog2(self) }
    }

    /// Produces a vector where every lane has the base-10 logarithm of the value
    /// in the equivalently-indexed lane in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn log10(self) -> Self {
        unsafe { intrinsics::simd_flog10(self) }
    }

    /// Returns the smallest integer greater than or equal to each lane.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn ceil(self) -> Self {
        unsafe { intrinsics::simd_ceil(self) }
    }

    /// Returns the largest integer value less than or equal to each lane.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn floor(self) -> Self {
        unsafe { intrinsics::simd_floor(self) }
    }

    /// Rounds to the nearest integer value. Ties round toward zero.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn round(self) -> Self {
        unsafe { intrinsics::simd_round(self) }
    }

    /// Returns the floating point's integer value, with its fractional part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn trunc(self) -> Self {
        unsafe { intrinsics::simd_trunc(self) }
    }

    /// Returns the floating point's fractional value, with its integer part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn fract(self) -> Self;
}

impl<const N: usize> Sealed for Simd<f32, N> where LaneCount<N>: SupportedLaneCount {}
impl<const N: usize> Sealed for Simd<f64, N> where LaneCount<N>: SupportedLaneCount {}

// We can safely just use all the defaults.
impl<const N: usize> StdFloat for Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    /// Returns the floating point's fractional value, with its integer part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
}

impl<const N: usize> StdFloat for Simd<f64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    /// Returns the floating point's fractional value, with its integer part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
}

#[cfg(test)]
mod tests_simd_floats {
    use super::*;
    use simd::prelude::*;

    #[test]
    fn everything_works_f32() {
        let x = f32x4::from_array([0.1, 0.5, 0.6, -1.5]);

        let x2 = x + x;
        let _xc = x.ceil();
        let _xf = x.floor();
        let _xr = x.round();
        let _xt = x.trunc();
        let _xfma = x.mul_add(x, x);
        let _xsqrt = x.sqrt();
        let _abs_mul = x2.abs() * x2;

        let _fexp = x.exp();
        let _fexp2 = x.exp2();
        let _flog = x.log();
        let _flog2 = x.log2();
        let _flog10 = x.log10();
        let _fsin = x.sin();
        let _fcos = x.cos();
    }

    #[test]
    fn everything_works_f64() {
        let x = f64x4::from_array([0.1, 0.5, 0.6, -1.5]);

        let x2 = x + x;
        let _xc = x.ceil();
        let _xf = x.floor();
        let _xr = x.round();
        let _xt = x.trunc();
        let _xfma = x.mul_add(x, x);
        let _xsqrt = x.sqrt();
        let _abs_mul = x2.abs() * x2;

        let _fexp = x.exp();
        let _fexp2 = x.exp2();
        let _flog = x.log();
        let _flog2 = x.log2();
        let _flog10 = x.log10();
        let _fsin = x.sin();
        let _fcos = x.cos();
    }
}
