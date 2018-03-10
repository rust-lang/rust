//! Run-time feature detection on Linux

cfg_if! {
    if #[cfg(target_arch = "aarch64")] {
        mod aarch64;
        pub use self::aarch64::check_for;
    } else if #[cfg(target_arch = "arm")] {
        mod arm;
        pub use self::arm::check_for;
    } else  if #[cfg(any(target_arch = "mips", target_arch = "mips64"))] {
        mod mips;
        pub use self::mips::check_for;
    } else if #[cfg(target_arch = "powerpc64")] {
        mod powerpc64;
        pub use self::powerpc64::check_for;
    } else {
        use arch::detect::Feature;
        /// Performs run-time feature detection.
        pub fn check_for(_x: Feature) -> bool {
            false
        }
    }
}
