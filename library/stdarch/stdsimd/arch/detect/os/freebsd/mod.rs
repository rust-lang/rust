//! Run-time feature detection on FreeBSD

cfg_if! {
    if #[cfg(target_arch = "aarch64")] {
        mod aarch64;
        pub use self::aarch64::check_for;
    } else {
        use arch::detect::Feature;
        /// Performs run-time feature detection.
        pub fn check_for(_x: Feature) -> bool {
            false
        }
    }
}
