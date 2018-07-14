macro_rules! force_eval {
    ($e:expr) => {
        unsafe {
            ::core::ptr::read_volatile(&$e);
        }
    };
}

// Public modules
mod acos;
mod acosf;
mod asin;
mod asinf;
mod atan2f;
mod atanf;
mod cbrt;
mod cbrtf;
mod ceil;
mod ceilf;
mod cos;
mod cosf;
mod coshf;
mod exp;
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
mod fmod;
mod fmodf;
mod hypot;
mod hypotf;
mod log;
mod log10;
mod log10f;
mod log1p;
mod log1pf;
mod log2;
mod log2f;
mod logf;
mod powf;
mod round;
mod roundf;
mod scalbn;
mod scalbnf;
mod sin;
mod sinf;
mod sqrt;
mod sqrtf;
mod tanf;
mod tanhf;
mod trunc;
mod truncf;

// Use separated imports instead of {}-grouped imports for easier merging.
pub use self::acos::acos;
pub use self::acosf::acosf;
pub use self::asin::asin;
pub use self::asinf::asinf;
pub use self::atan2f::atan2f;
pub use self::atanf::atanf;
pub use self::cbrt::cbrt;
pub use self::cbrtf::cbrtf;
pub use self::ceil::ceil;
pub use self::ceilf::ceilf;
pub use self::cos::cos;
pub use self::cosf::cosf;
pub use self::coshf::coshf;
pub use self::exp::exp;
pub use self::exp2::exp2;
pub use self::exp2f::exp2f;
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
pub use self::fmod::fmod;
pub use self::fmodf::fmodf;
pub use self::hypot::hypot;
pub use self::hypotf::hypotf;
pub use self::log::log;
pub use self::log10::log10;
pub use self::log10f::log10f;
pub use self::log1p::log1p;
pub use self::log1pf::log1pf;
pub use self::log2::log2;
pub use self::log2f::log2f;
pub use self::logf::logf;
pub use self::powf::powf;
pub use self::round::round;
pub use self::roundf::roundf;
pub use self::scalbn::scalbn;
pub use self::scalbnf::scalbnf;
pub use self::sin::sin;
pub use self::sinf::sinf;
pub use self::sqrt::sqrt;
pub use self::sqrtf::sqrtf;
pub use self::tanf::tanf;
pub use self::tanhf::tanhf;
pub use self::trunc::trunc;
pub use self::truncf::truncf;

// Private modules
mod k_cos;
mod k_cosf;
mod k_expo2f;
mod k_sin;
mod k_sinf;
mod k_tanf;
mod rem_pio2;
mod rem_pio2_large;
mod rem_pio2f;

use self::k_cos::k_cos;
use self::k_cosf::k_cosf;
use self::k_expo2f::k_expo2f;
use self::k_sin::k_sin;
use self::k_sinf::k_sinf;
use self::k_tanf::k_tanf;
use self::rem_pio2::rem_pio2;
use self::rem_pio2_large::rem_pio2_large;
use self::rem_pio2f::rem_pio2f;

#[inline]
pub fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
pub fn get_low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline]
pub fn with_set_high_word(f: f64, hi: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0x00000000_ffffffff;
    tmp |= (hi as u64) << 32;
    f64::from_bits(tmp)
}

#[inline]
pub fn with_set_low_word(f: f64, lo: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0xffffffff_00000000;
    tmp |= lo as u64;
    f64::from_bits(tmp)
}
