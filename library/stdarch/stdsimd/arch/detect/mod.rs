//! Run-time feature detection

mod cache;
mod bit;

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        #[path = "x86.rs"]
        mod arch;
    } else if #[cfg(target_arch = "arm")] {
        #[path = "arm.rs"]
        mod arch;
    } else if #[cfg(target_arch = "aarch64")] {
        #[path = "aarch64.rs"]
        mod arch;
    } else if #[cfg(target_arch = "powerpc64")] {
        #[path = "powerpc64.rs"]
        mod arch;
    } else {
        mod arch {
            pub enum Feature {
                Null
            }
            pub fn detect_features() -> super::cache::Initializer {
                Default::default()
            }
        }
    }
}

mod linux;

pub use self::arch::Feature;

/// Performs run-time feature detection.
pub fn check_for(x: Feature) -> bool {
    cache::test(x as u32, arch::detect_features)
}
