macro_rules! force_eval {
    ($e:expr) => {
        unsafe { ::core::ptr::read_volatile(&$e) }
    };
}

#[cfg(not(debug_assertions))]
macro_rules! i {
    ($array:expr, $index:expr) => {
        unsafe { *$array.get_unchecked($index) }
    };
    ($array:expr, $index:expr, = , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) = $rhs;
        }
    };
    ($array:expr, $index:expr, += , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) += $rhs;
        }
    };
    ($array:expr, $index:expr, -= , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) -= $rhs;
        }
    };
    ($array:expr, $index:expr, &= , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) &= $rhs;
        }
    };
    ($array:expr, $index:expr, == , $rhs:expr) => {
        unsafe { *$array.get_unchecked_mut($index) == $rhs }
    };
}

#[cfg(debug_assertions)]
macro_rules! i {
    ($array:expr, $index:expr) => {
        *$array.get($index).unwrap()
    };
    ($array:expr, $index:expr, = , $rhs:expr) => {
        *$array.get_mut($index).unwrap() = $rhs;
    };
    ($array:expr, $index:expr, -= , $rhs:expr) => {
        *$array.get_mut($index).unwrap() -= $rhs;
    };
    ($array:expr, $index:expr, += , $rhs:expr) => {
        *$array.get_mut($index).unwrap() += $rhs;
    };
    ($array:expr, $index:expr, &= , $rhs:expr) => {
        *$array.get_mut($index).unwrap() &= $rhs;
    };
    ($array:expr, $index:expr, == , $rhs:expr) => {
        *$array.get_mut($index).unwrap() == $rhs
    };
}

// Temporary macro to avoid panic codegen for division (in debug mode too). At
// the time of this writing this is only used in a few places, and once
// rust-lang/rust#72751 is fixed then this macro will no longer be necessary and
// the native `/` operator can be used and panics won't be codegen'd.
#[cfg(any(debug_assertions, not(intrinsics_enabled)))]
macro_rules! div {
    ($a:expr, $b:expr) => {
        $a / $b
    };
}

#[cfg(all(not(debug_assertions), intrinsics_enabled))]
macro_rules! div {
    ($a:expr, $b:expr) => {
        unsafe { core::intrinsics::unchecked_div($a, $b) }
    };
}

// `support` may be public for testing
#[macro_use]
#[cfg(feature = "unstable-public-internals")]
pub mod support;

#[macro_use]
#[cfg(not(feature = "unstable-public-internals"))]
mod support;

cfg_if! {
    if #[cfg(feature = "unstable-public-internals")] {
        pub mod generic;
    } else {
        mod generic;
    }
}

// Private modules
mod arch;
mod expo2;
mod fenv;
mod k_cos;
mod k_cosf;
mod k_expo2;
mod k_expo2f;
mod k_sin;
mod k_sinf;
mod k_tan;
mod k_tanf;
mod rem_pio2;
mod rem_pio2_large;
mod rem_pio2f;

// Private re-imports
use self::expo2::expo2;
use self::k_cos::k_cos;
use self::k_cosf::k_cosf;
use self::k_expo2::k_expo2;
use self::k_expo2f::k_expo2f;
use self::k_sin::k_sin;
use self::k_sinf::k_sinf;
use self::k_tan::k_tan;
use self::k_tanf::k_tanf;
use self::rem_pio2::rem_pio2;
use self::rem_pio2_large::rem_pio2_large;
use self::rem_pio2f::rem_pio2f;
#[allow(unused_imports)]
use self::support::{CastFrom, CastInto, DInt, Float, HInt, Int, IntTy, MinInt};

// Public modules
mod acos;
mod acosf;
mod acosh;
mod acoshf;
mod asin;
mod asinf;
mod asinh;
mod asinhf;
mod atan;
mod atan2;
mod atan2f;
mod atanf;
mod atanh;
mod atanhf;
mod cbrt;
mod cbrtf;
mod ceil;
mod ceilf;
mod copysign;
mod copysignf;
mod cos;
mod cosf;
mod cosh;
mod coshf;
mod erf;
mod erff;
mod exp;
mod exp10;
mod exp10f;
mod exp2;
mod exp2f;
mod expf;
mod expm1;
mod expm1f;
mod fabs;
mod fabsf;
mod fdim;
mod fdimf;
mod floor;
mod floorf;
mod fma;
mod fmaf;
mod fmax;
mod fmaxf;
mod fmin;
mod fminf;
mod fmod;
mod fmodf;
mod frexp;
mod frexpf;
mod hypot;
mod hypotf;
mod ilogb;
mod ilogbf;
mod j0;
mod j0f;
mod j1;
mod j1f;
mod jn;
mod jnf;
mod ldexp;
mod ldexpf;
mod lgamma;
mod lgamma_r;
mod lgammaf;
mod lgammaf_r;
mod log;
mod log10;
mod log10f;
mod log1p;
mod log1pf;
mod log2;
mod log2f;
mod logf;
mod modf;
mod modff;
mod nextafter;
mod nextafterf;
mod pow;
mod powf;
mod remainder;
mod remainderf;
mod remquo;
mod remquof;
mod rint;
mod rintf;
mod round;
mod roundf;
mod scalbn;
mod scalbnf;
mod sin;
mod sincos;
mod sincosf;
mod sinf;
mod sinh;
mod sinhf;
mod sqrt;
mod sqrtf;
mod tan;
mod tanf;
mod tanh;
mod tanhf;
mod tgamma;
mod tgammaf;
mod trunc;
mod truncf;

// Use separated imports instead of {}-grouped imports for easier merging.
pub use self::acos::acos;
pub use self::acosf::acosf;
pub use self::acosh::acosh;
pub use self::acoshf::acoshf;
pub use self::asin::asin;
pub use self::asinf::asinf;
pub use self::asinh::asinh;
pub use self::asinhf::asinhf;
pub use self::atan::atan;
pub use self::atan2::atan2;
pub use self::atan2f::atan2f;
pub use self::atanf::atanf;
pub use self::atanh::atanh;
pub use self::atanhf::atanhf;
pub use self::cbrt::cbrt;
pub use self::cbrtf::cbrtf;
pub use self::ceil::ceil;
pub use self::ceilf::ceilf;
pub use self::copysign::copysign;
pub use self::copysignf::copysignf;
pub use self::cos::cos;
pub use self::cosf::cosf;
pub use self::cosh::cosh;
pub use self::coshf::coshf;
pub use self::erf::{erf, erfc};
pub use self::erff::{erfcf, erff};
pub use self::exp::exp;
pub use self::exp2::exp2;
pub use self::exp2f::exp2f;
pub use self::exp10::exp10;
pub use self::exp10f::exp10f;
pub use self::expf::expf;
pub use self::expm1::expm1;
pub use self::expm1f::expm1f;
pub use self::fabs::fabs;
pub use self::fabsf::fabsf;
pub use self::fdim::fdim;
pub use self::fdimf::fdimf;
pub use self::floor::floor;
pub use self::floorf::floorf;
pub use self::fma::fma;
pub use self::fmaf::fmaf;
pub use self::fmax::fmax;
pub use self::fmaxf::fmaxf;
pub use self::fmin::fmin;
pub use self::fminf::fminf;
pub use self::fmod::fmod;
pub use self::fmodf::fmodf;
pub use self::frexp::frexp;
pub use self::frexpf::frexpf;
pub use self::hypot::hypot;
pub use self::hypotf::hypotf;
pub use self::ilogb::ilogb;
pub use self::ilogbf::ilogbf;
pub use self::j0::{j0, y0};
pub use self::j0f::{j0f, y0f};
pub use self::j1::{j1, y1};
pub use self::j1f::{j1f, y1f};
pub use self::jn::{jn, yn};
pub use self::jnf::{jnf, ynf};
pub use self::ldexp::ldexp;
pub use self::ldexpf::ldexpf;
pub use self::lgamma::lgamma;
pub use self::lgamma_r::lgamma_r;
pub use self::lgammaf::lgammaf;
pub use self::lgammaf_r::lgammaf_r;
pub use self::log::log;
pub use self::log1p::log1p;
pub use self::log1pf::log1pf;
pub use self::log2::log2;
pub use self::log2f::log2f;
pub use self::log10::log10;
pub use self::log10f::log10f;
pub use self::logf::logf;
pub use self::modf::modf;
pub use self::modff::modff;
pub use self::nextafter::nextafter;
pub use self::nextafterf::nextafterf;
pub use self::pow::pow;
pub use self::powf::powf;
pub use self::remainder::remainder;
pub use self::remainderf::remainderf;
pub use self::remquo::remquo;
pub use self::remquof::remquof;
pub use self::rint::rint;
pub use self::rintf::rintf;
pub use self::round::round;
pub use self::roundf::roundf;
pub use self::scalbn::scalbn;
pub use self::scalbnf::scalbnf;
pub use self::sin::sin;
pub use self::sincos::sincos;
pub use self::sincosf::sincosf;
pub use self::sinf::sinf;
pub use self::sinh::sinh;
pub use self::sinhf::sinhf;
pub use self::sqrt::sqrt;
pub use self::sqrtf::sqrtf;
pub use self::tan::tan;
pub use self::tanf::tanf;
pub use self::tanh::tanh;
pub use self::tanhf::tanhf;
pub use self::tgamma::tgamma;
pub use self::tgammaf::tgammaf;
pub use self::trunc::trunc;
pub use self::truncf::truncf;

cfg_if! {
    if #[cfg(f16_enabled)] {
        mod copysignf16;
        mod fabsf16;
        mod fdimf16;
        mod truncf16;

        pub use self::copysignf16::copysignf16;
        pub use self::fabsf16::fabsf16;
        pub use self::fdimf16::fdimf16;
        pub use self::truncf16::truncf16;
    }
}

cfg_if! {
    if #[cfg(f128_enabled)] {
        mod copysignf128;
        mod fabsf128;
        mod fdimf128;
        mod truncf128;

        pub use self::copysignf128::copysignf128;
        pub use self::fabsf128::fabsf128;
        pub use self::fdimf128::fdimf128;
        pub use self::truncf128::truncf128;
    }
}

#[inline]
fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn get_low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline]
fn with_set_high_word(f: f64, hi: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0x00000000_ffffffff;
    tmp |= (hi as u64) << 32;
    f64::from_bits(tmp)
}

#[inline]
fn with_set_low_word(f: f64, lo: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0xffffffff_00000000;
    tmp |= lo as u64;
    f64::from_bits(tmp)
}

#[inline]
fn combine_words(hi: u32, lo: u32) -> f64 {
    f64::from_bits(((hi as u64) << 32) | lo as u64)
}
