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
mod error_macros;

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
        mod arch {
            pub enum Feature {
                Null
            }
        }
    }
}
pub use self::arch::Feature;

mod bit;
mod cache;

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        // On x86/x86_64 no OS specific functionality is required.
        #[path = "os/x86.rs"]
        mod os;
    } else if #[cfg(all(target_os = "linux", feature = "use_std"))] {
        #[path = "os/linux/mod.rs"]
        mod os;
    } else if #[cfg(target_os = "freebsd")] {
        #[cfg(target_arch = "aarch64")]
        #[path = "os/aarch64.rs"]
        mod aarch64;
        #[path = "os/freebsd/mod.rs"]
        mod os;
    } else {
        #[path = "os/other.rs"]
        mod os;
    }
}
pub use self::os::check_for;
