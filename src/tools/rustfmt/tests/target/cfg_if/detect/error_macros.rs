//! The `is_{target_arch}_feature_detected!` macro are only available on their
//! architecture. These macros provide a better error messages when the user
//! attempts to call them in a different architecture.

/// Prevents compilation if `is_x86_feature_detected` is used somewhere
/// else than `x86` and `x86_64` targets.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_x86_feature_detected {
    ($t: tt) => {
        compile_error!(
            r#"
        is_x86_feature_detected can only be used on x86 and x86_64 targets.
        You can prevent it from being used in other architectures by
        guarding it behind a cfg(target_arch) as follows:

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if is_x86_feature_detected(...) { ... }
            }
        "#
        )
    };
}

/// Prevents compilation if `is_arm_feature_detected` is used somewhere else
/// than `ARM` targets.
#[cfg(not(target_arch = "arm"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_arm_feature_detected {
    ($t:tt) => {
        compile_error!(
            r#"
        is_arm_feature_detected can only be used on ARM targets.
        You can prevent it from being used in other architectures by
        guarding it behind a cfg(target_arch) as follows:

            #[cfg(target_arch = "arm")] {
                if is_arm_feature_detected(...) { ... }
            }
        "#
        )
    };
}

/// Prevents compilation if `is_aarch64_feature_detected` is used somewhere else
/// than `aarch64` targets.
#[cfg(not(target_arch = "aarch64"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_aarch64_feature_detected {
    ($t: tt) => {
        compile_error!(
            r#"
        is_aarch64_feature_detected can only be used on AArch64 targets.
        You can prevent it from being used in other architectures by
        guarding it behind a cfg(target_arch) as follows:

            #[cfg(target_arch = "aarch64")] {
                if is_aarch64_feature_detected(...) { ... }
            }
        "#
        )
    };
}

/// Prevents compilation if `is_powerpc_feature_detected` is used somewhere else
/// than `PowerPC` targets.
#[cfg(not(target_arch = "powerpc"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_powerpc_feature_detected {
    ($t:tt) => {
        compile_error!(
            r#"
is_powerpc_feature_detected can only be used on PowerPC targets.
You can prevent it from being used in other architectures by
guarding it behind a cfg(target_arch) as follows:

    #[cfg(target_arch = "powerpc")] {
        if is_powerpc_feature_detected(...) { ... }
    }
"#
        )
    };
}

/// Prevents compilation if `is_powerpc64_feature_detected` is used somewhere
/// else than `PowerPC64` targets.
#[cfg(not(target_arch = "powerpc64"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_powerpc64_feature_detected {
    ($t:tt) => {
        compile_error!(
            r#"
is_powerpc64_feature_detected can only be used on PowerPC64 targets.
You can prevent it from being used in other architectures by
guarding it behind a cfg(target_arch) as follows:

    #[cfg(target_arch = "powerpc64")] {
        if is_powerpc64_feature_detected(...) { ... }
    }
"#
        )
    };
}

/// Prevents compilation if `is_mips_feature_detected` is used somewhere else
/// than `MIPS` targets.
#[cfg(not(target_arch = "mips"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_mips_feature_detected {
    ($t:tt) => {
        compile_error!(
            r#"
        is_mips_feature_detected can only be used on MIPS targets.
        You can prevent it from being used in other architectures by
        guarding it behind a cfg(target_arch) as follows:

            #[cfg(target_arch = "mips")] {
                if is_mips_feature_detected(...) { ... }
            }
        "#
        )
    };
}

/// Prevents compilation if `is_mips64_feature_detected` is used somewhere else
/// than `MIPS64` targets.
#[cfg(not(target_arch = "mips64"))]
#[macro_export]
#[unstable(feature = "stdsimd", issue = "27731")]
macro_rules! is_mips64_feature_detected {
    ($t:tt) => {
        compile_error!(
            r#"
        is_mips64_feature_detected can only be used on MIPS64 targets.
        You can prevent it from being used in other architectures by
        guarding it behind a cfg(target_arch) as follows:

            #[cfg(target_arch = "mips64")] {
                if is_mips64_feature_detected(...) { ... }
            }
        "#
        )
    };
}
