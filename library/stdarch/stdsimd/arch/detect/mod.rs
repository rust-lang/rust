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

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_x86_feature_detected {
    ($t:tt) => {
        compile_error!(r#"
is_x86_feature_detected can only be used on x86 and x86_64 targets.
You can prevent it from being used in other architectures by
guarding it behind a cfg(target_arch) as follows:

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        if is_x86_feature_detected(...) { ... }
    }
"#)
    };
}

#[cfg(not(target_arch = "arm"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_arm_feature_detected {
    ($t:tt) => {
        compile_error!(r#"
is_arm_feature_detected can only be used on ARM targets.
You can prevent it from being used in other architectures by
guarding it behind a cfg(target_arch) as follows:

    #[cfg(target_arch = "arm")] {
        if is_arm_feature_detected(...) { ... }
    }
"#)
    };
}

#[cfg(not(target_arch = "aarch64"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_aarch64_feature_detected {
    ($t:tt) => {
        compile_error!(r#"
is_aarch64_feature_detected can only be used on AArch64 targets.
You can prevent it from being used in other architectures by
guarding it behind a cfg(target_arch) as follows:

    #[cfg(target_arch = "aarch64")] {
        if is_aarch64_feature_detected(...) { ... }
    }
"#)
    };
}

#[cfg(not(target_arch = "powerpc64"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_powerpc64_feature_detected {
    ($t:tt) => {
        compile_error!(r#"
is_powerpc64_feature_detected can only be used on PowerPC64 targets.
You can prevent it from being used in other architectures by
guarding it behind a cfg(target_arch) as follows:

    #[cfg(target_arch = "powerpc64")] {
        if is_powerpc64_feature_detected(...) { ... }
    }
"#)
    };
}
