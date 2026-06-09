//! Run-time feature detection on Linux

mod auxvec;

#[cfg(feature = "std_detect_file_io")]
mod cpuinfo;

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
    } else if #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))] {
        mod powerpc;
        pub use self::powerpc::check_for;
    } else {
        use crate::detect::Feature;
        /// Performs run-time feature detection.
        pub fn check_for(_x: Feature) -> bool {
            false
        }
    }
}
