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

#[cfg(todo)]
use core::{f32, f64};

pub use math::*;

/// Approximate equality with 1 ULP of tolerance
#[doc(hidden)]
pub fn _eqf(a: u32, b: u32) -> bool {
    (a as i32).wrapping_sub(b as i32).abs() <= 1
}

#[doc(hidden)]
pub fn _eq(a: u64, b: u64) -> bool {
    (a as i64).wrapping_sub(b as i64).abs() <= 1
}

/// Math support for `f32`
///
/// This trait is sealed and cannot be implemented outside of `libm`.
pub trait F32Ext: private::Sealed {
    #[cfg(todo)]
    fn floor(self) -> Self;

    #[cfg(todo)]
    fn ceil(self) -> Self;

    #[cfg(todo)]
    fn round(self) -> Self;

    fn trunc(self) -> Self;

    #[cfg(todo)]
    fn fract(self) -> Self;

    fn abs(self) -> Self;

    #[cfg(todo)]
    fn signum(self) -> Self;

    #[cfg(todo)]
    fn mul_add(self, a: Self, b: Self) -> Self;

    #[cfg(todo)]
    fn div_euc(self, rhs: Self) -> Self;

    #[cfg(todo)]
    fn mod_euc(self, rhs: Self) -> Self;

    // NOTE depends on unstable intrinsics::powif32
    // fn powi(self, n: i32) -> Self;

    fn powf(self, n: Self) -> Self;

    fn sqrt(self) -> Self;

    fn exp(self) -> Self;

    #[cfg(todo)]
    fn exp2(self) -> Self;

    fn ln(self) -> Self;

    fn log(self, base: Self) -> Self;

    #[cfg(todo)]
    fn log2(self) -> Self;

    #[cfg(todo)]
    fn log10(self) -> Self;

    #[cfg(todo)]
    fn cbrt(self) -> Self;

    fn hypot(self, other: Self) -> Self;

    #[cfg(todo)]
    fn sin(self) -> Self;

    #[cfg(todo)]
    fn cos(self) -> Self;

    #[cfg(todo)]
    fn tan(self) -> Self;

    #[cfg(todo)]
    fn asin(self) -> Self;

    #[cfg(todo)]
    fn acos(self) -> Self;

    #[cfg(todo)]
    fn atan(self) -> Self;

    #[cfg(todo)]
    fn atan2(self, other: Self) -> Self;

    #[cfg(todo)]
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[cfg(todo)]
    fn exp_m1(self) -> Self;

    #[cfg(todo)]
    fn ln_1p(self) -> Self;

    #[cfg(todo)]
    fn sinh(self) -> Self;

    #[cfg(todo)]
    fn cosh(self) -> Self;

    #[cfg(todo)]
    fn tanh(self) -> Self;

    #[cfg(todo)]
    fn asinh(self) -> Self;

    #[cfg(todo)]
    fn acosh(self) -> Self;

    #[cfg(todo)]
    fn atanh(self) -> Self;
}

impl F32Ext for f32 {
    #[cfg(todo)]
    #[inline]
    fn floor(self) -> Self {
        floorf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn ceil(self) -> Self {
        ceilf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn round(self) -> Self {
        roundf(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        truncf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    #[inline]
    fn abs(self) -> Self {
        fabsf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        fmaf(self, a, b)
    }

    #[cfg(todo)]
    #[inline]
    fn div_euc(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    #[cfg(todo)]
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

    #[cfg(todo)]
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

    #[cfg(todo)]
    #[inline]
    fn log2(self) -> Self {
        log2f(self)
    }

    #[cfg(todo)]
    #[inline]
    fn log10(self) -> Self {
        log10f(self)
    }

    #[cfg(todo)]
    #[inline]
    fn cbrt(self) -> Self {
        cbrtf(self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypotf(self, other)
    }

    #[cfg(todo)]
    #[inline]
    fn sin(self) -> Self {
        sinf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn cos(self) -> Self {
        cosf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn tan(self) -> Self {
        tanf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn asin(self) -> Self {
        asinf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn acos(self) -> Self {
        acosf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn atan(self) -> Self {
        atanf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn atan2(self, other: Self) -> Self {
        atan2f(self, other)
    }

    #[cfg(todo)]
    #[inline]
    fn exp_m1(self) -> Self {
        expm1f(self)
    }

    #[cfg(todo)]
    #[inline]
    fn ln_1p(self) -> Self {
        log1pf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn sinh(self) -> Self {
        sinhf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn cosh(self) -> Self {
        coshf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn tanh(self) -> Self {
        tanhf(self)
    }

    #[cfg(todo)]
    #[inline]
    fn asinh(self) -> Self {
        if self == f32::NEG_INFINITY {
            f32::NEG_INFINITY
        } else {
            (self + ((self * self) + 1.0).sqrt()).ln()
        }
    }

    #[cfg(todo)]
    #[inline]
    fn acosh(self) -> Self {
        match self {
            x if x < 1.0 => f32::NAN,
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    #[cfg(todo)]
    #[inline]
    fn atanh(self) -> Self {
        0.5 * ((2.0 * self) / (1.0 - self)).ln_1p()
    }
}

/// Math support for `f32`
///
/// This trait is sealed and cannot be implemented outside of `libm`.
pub trait F64Ext: private::Sealed {
    fn floor(self) -> Self;

    #[cfg(todo)]
    fn ceil(self) -> Self;

    fn round(self) -> Self;

    fn trunc(self) -> Self;

    #[cfg(todo)]
    fn fract(self) -> Self;

    fn abs(self) -> Self;

    #[cfg(todo)]
    fn signum(self) -> Self;

    #[cfg(todo)]
    fn mul_add(self, a: Self, b: Self) -> Self;

    #[cfg(todo)]
    fn div_euc(self, rhs: Self) -> Self;

    #[cfg(todo)]
    fn mod_euc(self, rhs: Self) -> Self;

    // NOTE depends on unstable intrinsics::powif64
    // fn powi(self, n: i32) -> Self;

    #[cfg(todo)]
    fn powf(self, n: Self) -> Self;

    #[cfg(todo)]
    fn sqrt(self) -> Self;

    #[cfg(todo)]
    fn exp(self) -> Self;

    #[cfg(todo)]
    fn exp2(self) -> Self;

    #[cfg(todo)]
    fn ln(self) -> Self;

    #[cfg(todo)]
    fn log(self, base: Self) -> Self;

    #[cfg(todo)]
    fn log2(self) -> Self;

    #[cfg(todo)]
    fn log10(self) -> Self;

    #[cfg(todo)]
    fn cbrt(self) -> Self;

    #[cfg(todo)]
    fn hypot(self, other: Self) -> Self;

    #[cfg(todo)]
    fn sin(self) -> Self;

    #[cfg(todo)]
    fn cos(self) -> Self;

    #[cfg(todo)]
    fn tan(self) -> Self;

    #[cfg(todo)]
    fn asin(self) -> Self;

    #[cfg(todo)]
    fn acos(self) -> Self;

    #[cfg(todo)]
    fn atan(self) -> Self;

    #[cfg(todo)]
    fn atan2(self, other: Self) -> Self;

    #[cfg(todo)]
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    #[cfg(todo)]
    fn exp_m1(self) -> Self;

    #[cfg(todo)]
    fn ln_1p(self) -> Self;

    #[cfg(todo)]
    fn sinh(self) -> Self;

    #[cfg(todo)]
    fn cosh(self) -> Self;

    #[cfg(todo)]
    fn tanh(self) -> Self;

    #[cfg(todo)]
    fn asinh(self) -> Self;

    #[cfg(todo)]
    fn acosh(self) -> Self;

    #[cfg(todo)]
    fn atanh(self) -> Self;
}

impl F64Ext for f64 {
    #[inline]
    fn floor(self) -> Self {
        floor(self)
    }

    #[cfg(todo)]
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

    #[cfg(todo)]
    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }

    #[inline]
    fn abs(self) -> Self {
        fabs(self)
    }

    #[cfg(todo)]
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        fma(self, a, b)
    }

    #[cfg(todo)]
    #[inline]
    fn div_euc(self, rhs: Self) -> Self {
        let q = (self / rhs).trunc();
        if self % rhs < 0.0 {
            return if rhs > 0.0 { q - 1.0 } else { q + 1.0 };
        }
        q
    }

    #[cfg(todo)]
    #[inline]
    fn mod_euc(self, rhs: f32) -> f32 {
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

    #[cfg(todo)]
    #[inline]
    fn sqrt(self) -> Self {
        sqrt(self)
    }

    #[cfg(todo)]
    #[inline]
    fn exp(self) -> Self {
        exp(self)
    }

    #[cfg(todo)]
    #[inline]
    fn exp2(self) -> Self {
        exp2(self)
    }

    #[cfg(todo)]
    #[inline]
    fn ln(self) -> Self {
        log(self)
    }

    #[cfg(todo)]
    #[inline]
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    #[cfg(todo)]
    #[inline]
    fn log2(self) -> Self {
        log2(self)
    }

    #[cfg(todo)]
    #[inline]
    fn log10(self) -> Self {
        log10(self)
    }

    #[cfg(todo)]
    #[inline]
    fn cbrt(self) -> Self {
        cbrt(self)
    }

    #[cfg(todo)]
    #[inline]
    fn hypot(self, other: Self) -> Self {
        hypot(self, other)
    }

    #[cfg(todo)]
    #[inline]
    fn sin(self) -> Self {
        sin(self)
    }

    #[cfg(todo)]
    #[inline]
    fn cos(self) -> Self {
        cos(self)
    }

    #[cfg(todo)]
    #[inline]
    fn tan(self) -> Self {
        tan(self)
    }

    #[cfg(todo)]
    #[inline]
    fn asin(self) -> Self {
        asin(self)
    }

    #[cfg(todo)]
    #[inline]
    fn acos(self) -> Self {
        acos(self)
    }

    #[cfg(todo)]
    #[inline]
    fn atan(self) -> Self {
        atan(self)
    }

    #[cfg(todo)]
    #[inline]
    fn atan2(self, other: Self) -> Self {
        atan2(self, other)
    }

    #[cfg(todo)]
    #[inline]
    fn exp_m1(self) -> Self {
        expm1(self)
    }

    #[cfg(todo)]
    #[inline]
    fn ln_1p(self) -> Self {
        log1p(self)
    }

    #[cfg(todo)]
    #[inline]
    fn sinh(self) -> Self {
        sinh(self)
    }

    #[cfg(todo)]
    #[inline]
    fn cosh(self) -> Self {
        cosh(self)
    }

    #[cfg(todo)]
    #[inline]
    fn tanh(self) -> Self {
        tanh(self)
    }

    #[cfg(todo)]
    #[inline]
    fn asinh(self) -> Self {
        if self == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else {
            (self + ((self * self) + 1.0).sqrt()).ln()
        }
    }

    #[cfg(todo)]
    #[inline]
    fn acosh(self) -> Self {
        match self {
            x if x < 1.0 => f64::NAN,
            x => (x + ((x * x) - 1.0).sqrt()).ln(),
        }
    }

    #[cfg(todo)]
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
