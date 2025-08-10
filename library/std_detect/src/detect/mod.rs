//! This module implements run-time feature detection.
//!
//! The `is_{arch}_feature_detected!("feature-name")` macros take the name of a
//! feature as a string-literal, and return a boolean indicating whether the
//! feature is enabled at run-time or not.
//!
//! These macros do two things:
//! * map the string-literal into an integer stored as a `Feature` enum,
//! * call a `os::check_for(x: Feature)` function that returns `true` if the
//! feature is enabled.
//!
//! The `Feature` enums are also implemented in the `arch/{target_arch}.rs`
//! modules.
//!
//! The `check_for` functions are, in general, Operating System dependent. Most
//! architectures do not allow user-space programs to query the feature bits
//! due to security concerns (x86 is the big exception). These functions are
//! implemented in the `os/{target_os}.rs` modules.

#[macro_use]
mod macros;

mod arch;

// This module needs to be public because the `is_{arch}_feature_detected!`
// macros expand calls to items within it in user crates.
#[doc(hidden)]
#[unstable(feature = "stdarch_internal", issue = "none")]
pub use self::arch::__is_feature_detected;
pub(crate) use self::arch::Feature;

mod bit;
mod cache;

cfg_select! {
    miri => {
        // When running under miri all target-features that are not enabled at
        // compile-time are reported as disabled at run-time.
        //
        // For features for which `cfg(target_feature)` returns true,
        // this run-time detection logic is never called.
        #[path = "os/other.rs"]
        mod os;
    }
    any(target_arch = "x86", target_arch = "x86_64") => {
        // On x86/x86_64 no OS specific functionality is required.
        #[path = "os/x86.rs"]
        mod os;
    }
    all(any(target_os = "linux", target_os = "android"), feature = "libc") => {
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        #[path = "os/riscv.rs"]
        mod riscv;
        #[path = "os/linux/mod.rs"]
        mod os;
    }
    all(target_os = "freebsd", feature = "libc") => {
        #[cfg(target_arch = "aarch64")]
        #[path = "os/aarch64.rs"]
        mod aarch64;
        #[path = "os/freebsd/mod.rs"]
        mod os;
    }
    all(target_os = "openbsd", target_arch = "aarch64", feature = "libc") => {
        #[allow(dead_code)] // we don't use code that calls the mrs instruction.
        #[path = "os/aarch64.rs"]
        mod aarch64;
        #[path = "os/openbsd/aarch64.rs"]
        mod os;
    }
    all(target_os = "windows", any(target_arch = "aarch64", target_arch = "arm64ec")) => {
        #[path = "os/windows/aarch64.rs"]
        mod os;
    }
    all(target_vendor = "apple", target_arch = "aarch64", feature = "libc") => {
        #[path = "os/darwin/aarch64.rs"]
        mod os;
    }
    _ => {
        #[path = "os/other.rs"]
        mod os;
    }
}

/// Performs run-time feature detection.
#[inline]
#[allow(dead_code)]
fn check_for(x: Feature) -> bool {
    cache::test(x as u32)
}

/// Returns an `Iterator<Item=(&'static str, bool)>` where
/// `Item.0` is the feature name, and `Item.1` is a `bool` which
/// is `true` if the feature is supported by the host and `false` otherwise.
#[unstable(feature = "stdarch_internal", issue = "none")]
pub fn features() -> impl Iterator<Item = (&'static str, bool)> {
    cfg_select! {
        any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64",
            target_arch = "arm64ec",
            target_arch = "riscv32",
            target_arch = "riscv64",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "mips",
            target_arch = "mips64",
            target_arch = "loongarch32",
            target_arch = "loongarch64",
            target_arch = "s390x",
        ) => {
            (0_u8..Feature::_last as u8).map(|discriminant: u8| {
                #[allow(bindings_with_variant_name)] // RISC-V has Feature::f
                let f: Feature = unsafe { core::mem::transmute(discriminant) };
                let name: &'static str = f.to_str();
                let enabled: bool = check_for(f);
                (name, enabled)
            })
        }
        _ => None.into_iter(),
    }
}
