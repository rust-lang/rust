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

cfg_if::cfg_if! {
    if #[cfg(target_os = "vxworks")] {
        mod vxworks;
        pub use self::vxworks::*;
    } else if #[cfg(unix)] {
        mod unix;
        pub use self::unix::*;
    } else if #[cfg(windows)] {
        mod windows;
        pub use self::windows::*;
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

// Import essential modules from both platforms when documenting. These are
// then later used in the `std::os` module when documenting, for example,
// Windows when we're compiling for Linux.

#[cfg(doc)]
cfg_if::cfg_if! {
    if #[cfg(unix)] {
        // On unix we'll document what's already available
        #[stable(feature = "rust1", since = "1.0.0")]
        pub use self::ext as unix_ext;
    } else if #[cfg(any(target_os = "hermit",
                        target_arch = "wasm32",
                        all(target_vendor = "fortanix", target_env = "sgx")))] {
        // On wasm right now the module below doesn't compile
        // (missing things in `libc` which is empty) so just omit everything
        // with an empty module
        #[unstable(issue = "none", feature = "std_internals")]
        #[allow(missing_docs)]
        pub mod unix_ext {}
    } else {
        // On other platforms like Windows document the bare bones of unix
        use crate::os::linux as platform;
        #[path = "unix/ext/mod.rs"]
        pub mod unix_ext;
    }
}

#[cfg(doc)]
cfg_if::cfg_if! {
    if #[cfg(windows)] {
        // On windows we'll just be documenting what's already available
        #[allow(missing_docs)]
        #[stable(feature = "rust1", since = "1.0.0")]
        pub use self::ext as windows_ext;
    } else if #[cfg(any(target_os = "hermit",
                        target_arch = "wasm32",
                        all(target_vendor = "fortanix", target_env = "sgx")))] {
        // On wasm right now the shim below doesn't compile, so
        // just omit it
        #[unstable(issue = "none", feature = "std_internals")]
        #[allow(missing_docs)]
        pub mod windows_ext {}
    } else {
        // On all other platforms (aka linux/osx/etc) then pull in a "minimal"
        // amount of windows goop which ends up compiling
        #[macro_use]
        #[path = "windows/compat.rs"]
        mod compat;

        #[path = "windows/c.rs"]
        mod c;

        #[path = "windows/ext/mod.rs"]
        pub mod windows_ext;
    }
}
