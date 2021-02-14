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

use cfg_if::cfg_if;

#[macro_use]
mod error_macros;

#[macro_use]
mod macros;

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        #[path = "arch/x86.rs"]
        #[macro_use]
        mod arch;
    } else if #[cfg(target_arch = "arm")] {
        #[path = "arch/arm.rs"]
        #[macro_use]
        mod arch;
    } else if #[cfg(target_arch = "aarch64")] {
        #[path = "arch/aarch64.rs"]
        #[macro_use]
        mod arch;
    } else if #[cfg(target_arch = "powerpc")] {
        #[path = "arch/powerpc.rs"]
        #[macro_use]
        mod arch;
    } else if #[cfg(target_arch = "powerpc64")] {
        #[path = "arch/powerpc64.rs"]
        #[macro_use]
        mod arch;
    } else if #[cfg(target_arch = "mips")] {
        #[path = "arch/mips.rs"]
        #[macro_use]
        mod arch;
    } else if #[cfg(target_arch = "mips64")] {
        #[path = "arch/mips64.rs"]
        #[macro_use]
        mod arch;
    } else {
        // Unimplemented architecture:
        #[allow(dead_code)]
        mod arch {
            #[doc(hidden)]
            pub(crate) enum Feature {
                Null
            }
            #[doc(hidden)]
            pub mod __is_feature_detected {}

            impl Feature {
                #[doc(hidden)]
                pub(crate) fn from_str(_s: &str) -> Result<Feature, ()> { Err(()) }
                #[doc(hidden)]
                pub(crate) fn to_str(self) -> &'static str { "" }
            }
        }
    }
}

// This module needs to be public because the `is_{arch}_feature_detected!`
// macros expand calls to items within it in user crates.
#[doc(hidden)]
pub use self::arch::__is_feature_detected;

pub(crate) use self::arch::Feature;

mod bit;
mod cache;

cfg_if! {
    if #[cfg(miri)] {
        // When running under miri all target-features that are not enabled at
        // compile-time are reported as disabled at run-time.
        //
        // For features for which `cfg(target_feature)` returns true,
        // this run-time detection logic is never called.
        #[path = "os/other.rs"]
        mod os;
    } else if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        // On x86/x86_64 no OS specific functionality is required.
        #[path = "os/x86.rs"]
        mod os;
    } else if #[cfg(all(target_os = "linux", feature = "libc"))] {
        #[path = "os/linux/mod.rs"]
        mod os;
    } else if #[cfg(all(target_os = "freebsd", feature = "libc"))] {
        #[cfg(target_arch = "aarch64")]
        #[path = "os/aarch64.rs"]
        mod aarch64;
        #[path = "os/freebsd/mod.rs"]
        mod os;
    } else if #[cfg(all(target_os = "windows", target_arch = "aarch64"))] {
        #[path = "os/windows/aarch64.rs"]
        mod os;
    } else {
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
#[unstable(feature = "stdsimd", issue = "27731")]
pub fn features() -> impl Iterator<Item = (&'static str, bool)> {
    cfg_if! {
        if #[cfg(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "mips",
            target_arch = "mips64",
        ))] {
            (0_u8..Feature::_last as u8).map(|discriminant: u8| {
                let f: Feature = unsafe { core::mem::transmute(discriminant) };
                let name: &'static str = f.to_str();
                let enabled: bool = check_for(f);
                (name, enabled)
            })
        } else {
            None.into_iter()
        }
    }
}
