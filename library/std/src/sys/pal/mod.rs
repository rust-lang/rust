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

pub mod common;

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
    } else if #[cfg(target_os = "trusty")] {
        mod trusty;
        pub use self::trusty::*;
    } else if #[cfg(all(target_os = "wasi", target_env = "p2"))] {
        mod wasip2;
        pub use self::wasip2::*;
    } else if #[cfg(target_os = "wasi")] {
        mod wasi;
        pub use self::wasi::*;
    } else if #[cfg(target_family = "wasm")] {
        mod wasm;
        pub use self::wasm::*;
    } else if #[cfg(target_os = "xous")] {
        mod xous;
        pub use self::xous::*;
    } else if #[cfg(target_os = "uefi")] {
        mod uefi;
        pub use self::uefi::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use self::sgx::*;
    } else if #[cfg(target_os = "teeos")] {
        mod teeos;
        pub use self::teeos::*;
    } else if #[cfg(target_os = "zkvm")] {
        mod zkvm;
        pub use self::zkvm::*;
    } else {
        mod unsupported;
        pub use self::unsupported::*;
    }
}

cfg_if::cfg_if! {
    // Fuchsia components default to full backtrace.
    if #[cfg(target_os = "fuchsia")] {
        pub const FULL_BACKTRACE_DEFAULT: bool = true;
    } else {
        pub const FULL_BACKTRACE_DEFAULT: bool = false;
    }
}

#[cfg(not(target_os = "uefi"))]
pub type RawOsError = i32;
