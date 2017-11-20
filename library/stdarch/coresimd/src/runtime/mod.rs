//! Run-time feature detection
mod cache;
mod bit;

#[macro_use]
mod macros;

#[macro_use]
mod x86;
pub use self::x86::__Feature;
use self::x86::detect_features;

/// Performs run-time feature detection.
#[doc(hidden)]
pub fn __unstable_detect_feature(x: __Feature) -> bool {
    cache::test(x as u32, detect_features)
}
