//! Run-time feature detection
mod cache;
mod bit;

#[macro_use]
mod macros;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[macro_use]
mod x86;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::x86::__Feature;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use self::x86::detect_features;

#[cfg(all(target_arch = "arm", target_os = "linux", feature = "std"))]
#[macro_use]
mod arm;
#[cfg(all(target_arch = "arm", target_os = "linux", feature = "std"))]
pub use self::arm::__Feature;

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
#[macro_use]
mod aarch64;
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "std"))]
pub use self::aarch64::__Feature;

#[cfg(all(feature = "std", target_os = "linux",
          any(target_arch = "arm", target_arch = "aarch64")))]
mod linux;

#[cfg(all(feature = "std", target_os = "linux",
          any(target_arch = "arm", target_arch = "aarch64")))]
pub use self::linux::detect_features;

/// Performs run-time feature detection.
#[doc(hidden)]
pub fn __unstable_detect_feature(x: __Feature) -> bool {
    cache::test(x as u32, detect_features)
}
