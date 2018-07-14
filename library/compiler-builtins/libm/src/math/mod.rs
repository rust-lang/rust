macro_rules! force_eval {
    ($e:expr) => {
        unsafe {
            ::core::ptr::read_volatile(&$e);
        }
    };
}

mod acos;
mod cbrt;
mod cbrtf;
mod ceil;
mod ceilf;
mod cosf;
mod exp;
mod expf;
mod fabs;
mod fabsf;
mod fdim;
mod fdimf;
mod floor;
mod floorf;
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
mod sqrt;
mod sqrtf;
mod trunc;
mod truncf;

// Use separated imports instead of {}-grouped imports for easier merging.
pub use self::acos::acos;
pub use self::cbrt::cbrt;
pub use self::cbrtf::cbrtf;
pub use self::ceil::ceil;
pub use self::ceilf::ceilf;
pub use self::cosf::cosf;
pub use self::exp::exp;
pub use self::expf::expf;
pub use self::fabs::fabs;
pub use self::fabsf::fabsf;
pub use self::fdim::fdim;
pub use self::fdimf::fdimf;
pub use self::floor::floor;
pub use self::floorf::floorf;
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
pub use self::sqrt::sqrt;
pub use self::sqrtf::sqrtf;
pub use self::trunc::trunc;
pub use self::truncf::truncf;

mod k_cosf;
mod k_sinf;
mod rem_pio2_large;
mod rem_pio2f;

use self::{k_cosf::k_cosf, k_sinf::k_sinf, rem_pio2_large::rem_pio2_large, rem_pio2f::rem_pio2f};
