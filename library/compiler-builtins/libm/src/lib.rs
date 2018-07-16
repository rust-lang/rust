//! Port of MUSL's libm to Rust
//!
//! # Usage
//!
//! You can use this crate in two ways:
//!
//! - By directly using its free functions, e.g. `libm::powf`.
//!
//! - By importing the `F32Ext` and / or `F64Ext` extension traits to add methods like `powf` to the
//! `f32` and `f64` types. Then you'll be able to invoke math functions as methods, e.g. `x.sqrt()`.

#![deny(warnings)]
#![no_std]

mod math;

use core::{f32, f64};

pub use math::*;

/// Approximate equality with 1 ULP of tolerance
#[doc(hidden)]
#[inline]
pub fn _eqf(a: u32, b: u32) -> bool {
    (a as i32).wrapping_sub(b as i32).abs() <= 1
}

#[doc(hidden)]
#[inline]
pub fn _eq(a: u64, b: u64) -> bool {
    (a as i64).wrapping_sub(b as i64).abs() <= 1
}

/// Math support for `f32`
///
/// This trait is sealed and cannot be implemented outside of `libm`.
pub trait F32Ext: private::Sealed + Sized {
    fn floor(self) -> Self;

    fn ceil(self) -> Self;

    fn round(self) -> Self;

    fn trunc(self) -> Self;

    fn fdim(self, rhs: Self) -> Self;

    fn fract(self) -> Self;

    fn abs(self) -> Self;

    // NOTE depends on unstable intrinsics::copysignf32
    // fn signum(self) -> Self;

    fn mul_add(self, a: Self, b: Self) -> Self;

    fn div_euc(self, rhs: Self) -> Self;

    fn mod_euc(self, rhs: Self) -> Self;

    // NOTE depends on unstable intrinsics::powif32
    // fn powi(self, n: i32) -> Self;

    fn powf(self, n: Self) -> Self;

    fn sqrt(self) -> Self;

    fn exp(self) -> Self;

    fn exp2(self) -> Self;

    fn ln(self) -> Self;

    fn log(self, base: Self) -> Self;

    fn log2(self) -> Self;

    fn log10(self) -> Self;

    fn cbrt(self) -> Self;

    fn hypot(self, other: Self) -> Self;

    fn sin(self) -> Self;

    fn cos(self) -> Self;

    fn tan(self) -> Self;

    fn asin(self) -> Self;

    fn acos(self) -> Self;

    fn atan(self) -> Self;

    fn atan2(self, other: Self) -> Self;

    #[inline]
    fn sin_cos(self) -> (Self, Self)
    where
        Self: Copy,
    {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Self;

    fn ln_1p(self) -> Self;

    fn sinh(self) -> Self;

    fn cosh(self) -> Self;

    fn tanh(self) -> Self;

    fn asinh(self) -> Self;

    fn acosh(self) -> Self;

    fn atanh(self) -> Self;
}

impl F32Ext for f32 {
    #[inline]
    fn floor(self) -> Self {
        floorf(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        ceilf(self)
    }

    #[inline]
    fn round(self) -> Self {
        roundf(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        truncf(self)
    }

    #[inline]
    fn fdim(self, rhs: Self) -> Self {
        fdimf(self, rhs)
    }

    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    #[inline]
    fn abs(self) -> Self {
        fabsf(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        fmaf(self, a, b)
    }

    #[inline]
    fn div_euc(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    #[inline]
    fn mod_euc(self, rhs: f32) -> f32 {
        let r = self % rhs;
        if r < 0.0 {
            r + rhs.abs()
        } else {
            r
        }
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        powf(self, n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        sqrtf(self)
    }

    #[inline]
    fn exp(self) -> Self {
        expf(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        exp2f(self)
    }

    #[inline]
    fn ln(self) -> Self {
        logf(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        log2f(self)
    }

    #[inline]
    fn log10(self) -> Self {
        log10f(self)
    }

    #[inline]
    fn cbrt(self) -> Self {
        cbrtf(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypotf(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        sinf(self)
    }

    #[inline]
    fn cos(self) -> Self {
        cosf(self)
    }

    #[inline]
    fn tan(self) -> Self {
        tanf(self)
    }

    #[inline]
    fn asin(self) -> Self {
        asinf(self)
    }

    #[inline]
    fn acos(self) -> Self {
        acosf(self)
    }

    #[inline]
    fn atan(self) -> Self {
        atanf(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        atan2f(self, other)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        expm1f(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        log1pf(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        sinhf(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        coshf(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        tanhf(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        if self == f32::NEG_INFINITY {
            f32::NEG_INFINITY
        } else {
            (self + ((self * self) + 1.0).sqrt()).ln()
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        match self {
            x if x < 1.0 => f32::NAN,
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    #[inline]
    fn atanh(self) -> Self {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }
}

/// Math support for `f64`
///
/// This trait is sealed and cannot be implemented outside of `libm`.
pub trait F64Ext: private::Sealed + Sized {
    fn floor(self) -> Self;

    fn ceil(self) -> Self;

    fn round(self) -> Self;

    fn trunc(self) -> Self;

    fn fdim(self, rhs: Self) -> Self;

    fn fract(self) -> Self;

    fn abs(self) -> Self;

    // NOTE depends on unstable intrinsics::copysignf64
    // fn signum(self) -> Self;

    fn mul_add(self, a: Self, b: Self) -> Self;

    fn div_euc(self, rhs: Self) -> Self;

    fn mod_euc(self, rhs: Self) -> Self;

    // NOTE depends on unstable intrinsics::powif64
    // fn powi(self, n: i32) -> Self;

    #[cfg(todo)]
    fn powf(self, n: Self) -> Self;

    fn sqrt(self) -> Self;

    fn exp(self) -> Self;

    fn exp2(self) -> Self;

    fn ln(self) -> Self;

    fn log(self, base: Self) -> Self;

    fn log2(self) -> Self;

    fn log10(self) -> Self;

    fn cbrt(self) -> Self;

    fn hypot(self, other: Self) -> Self;

    fn sin(self) -> Self;

    fn cos(self) -> Self;

    fn tan(self) -> Self;

    fn asin(self) -> Self;

    fn acos(self) -> Self;

    fn atan(self) -> Self;

    #[cfg(todo)]
    fn atan2(self, other: Self) -> Self;

    #[inline]
    fn sin_cos(self) -> (Self, Self)
    where
        Self: Copy,
    {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Self;

    fn ln_1p(self) -> Self;

    fn sinh(self) -> Self;

    fn cosh(self) -> Self;

    fn tanh(self) -> Self;

    fn asinh(self) -> Self;

    fn acosh(self) -> Self;

    fn atanh(self) -> Self;
}

impl F64Ext for f64 {
    #[inline]
    fn floor(self) -> Self {
        floor(self)
    }

    #[inline]
    fn ceil(self) -> Self {
        ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        trunc(self)
    }

    #[inline]
    fn fdim(self, rhs: Self) -> Self {
        fdim(self, rhs)
    }

    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    #[inline]
    fn abs(self) -> Self {
        fabs(self)
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        fma(self, a, b)
    }

    #[inline]
    fn div_euc(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    #[inline]
    fn mod_euc(self, rhs: f64) -> f64 {
        let r = self % rhs;
        if r < 0.0 {
            r + rhs.abs()
        } else {
            r
        }
    }

    #[cfg(todo)]
    #[inline]
    fn powf(self, n: Self) -> Self {
        pow(self, n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        exp2(self)
    }

    #[inline]
    fn ln(self) -> Self {
        log(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[inline]
    fn log2(self) -> Self {
        log2(self)
    }

    #[inline]
    fn log10(self) -> Self {
        log10(self)
    }

    #[inline]
    fn cbrt(self) -> Self {
        cbrt(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypot(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        atan(self)
    }

    #[cfg(todo)]
    #[inline]
    fn atan2(self, other: Self) -> Self {
        atan2(self, other)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        expm1(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        log1p(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        sinh(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        cosh(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        tanh(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        if self == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            (self + ((self * self) + 1.0).sqrt()).ln()
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        match self {
            x if x < 1.0 => f64::NAN,
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    #[inline]
    fn atanh(self) -> Self {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
