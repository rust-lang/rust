//! Run-time feature detection
mod cache;
mod bit;

#[macro_use]
mod macros;

#[cfg(all(target_arch = "arm", target_os = "linux"))]
#[macro_use]
mod arm;
#[cfg(all(target_arch = "arm", target_os = "linux"))]
pub use self::arm::__Feature;

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[macro_use]
mod aarch64;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub use self::aarch64::__Feature;

#[cfg(all(target_arch = "powerpc64", target_os = "linux"))]
#[macro_use]
mod powerpc64;
#[cfg(all(target_arch = "powerpc64", target_os = "linux"))]
pub use self::powerpc64::__Feature;

#[cfg(all(target_os = "linux",
          any(target_arch = "arm", target_arch = "aarch64", target_arch = "powerpc64")))]
mod linux;

#[cfg(all(target_os = "linux",
          any(target_arch = "arm", target_arch = "aarch64", target_arch = "powerpc64")))]
pub use self::linux::detect_features;

/// Performs run-time feature detection.
#[doc(hidden)]
pub fn __unstable_detect_feature(x: __Feature) -> bool {
    cache::test(x as u32, detect_features)
}
