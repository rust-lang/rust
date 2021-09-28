//! Platform-dependent platform abstraction.
//!
//! The `std::sys` module is the abstracted interface through which
//! `std` talks to the underlying operating system. It has different
//! implementations for different operating system families, today
//! just Unix and Windows, and initial support for Redox.
//!
//! The centralization of platform-specific code in this module is
//! enforced by the "platform abstraction layer" tidy script in
//! `tools/tidy/src/pal.rs`.
//!
//! This module is closely related to the platform-independent system
//! integration code in `std::sys_common`. See that module's
//! documentation for details.
//!
//! In the future it would be desirable for the independent
//! implementations of this module to be extracted to their own crates
//! that `std` can link to, thus enabling their implementation
//! out-of-tree via crate replacement. Though due to the complex
//! inter-dependencies within `std` that will be a challenging goal to
//! achieve.

#![allow(missing_debug_implementations)]

mod common;

cfg_if::cfg_if! {
    if #[cfg(unix)] {
        mod unix;
        pub use self::unix::*;
    } else if #[cfg(windows)] {
        mod windows;
        pub use self::windows::*;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod solid;
        pub use self::solid::*;
    } else if #[cfg(target_os = "hermit")] {
        mod hermit;
        pub use self::hermit::*;
    } else if #[cfg(target_os = "wasi")] {
        mod wasi;
        pub use self::wasi::*;
    } else if #[cfg(target_arch = "wasm32")] {
        mod wasm;
        pub use self::wasm::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use self::sgx::*;
    } else {
        mod unsupported;
        pub use self::unsupported::*;
    }
}

// Import essential modules from platforms used in `std::os` when documenting.
//
// Note that on some platforms those modules don't compile
// (missing things in `libc` which is empty), so they are not included in `std::os` and can be
// omitted here as well.

#[cfg(doc)]
#[cfg(not(any(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    all(target_vendor = "fortanix", target_env = "sgx")
)))]
cfg_if::cfg_if! {
    if #[cfg(not(windows))] {
        // On non-Windows platforms (aka linux/osx/etc) pull in a "minimal"
        // amount of windows goop which ends up compiling

        #[macro_use]
        #[path = "windows/compat.rs"]
        pub mod compat;

        #[path = "windows/c.rs"]
        pub mod c;
    }
}
