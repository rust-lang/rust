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

cfg_select! {
    unix => {
        mod unix;
        pub use self::unix::*;
    }
    windows => {
        mod windows;
        pub use self::windows::*;
    }
    target_os = "solid_asp3" => {
        mod solid;
        pub use self::solid::*;
    }
    target_os = "hermit" => {
        mod hermit;
        pub use self::hermit::*;
    }
    target_os = "trusty" => {
        mod trusty;
        pub use self::trusty::*;
    }
    target_os = "vexos" => {
        mod vexos;
        pub use self::vexos::*;
    }
    all(target_os = "wasi", target_env = "p2") => {
        mod wasip2;
        pub use self::wasip2::*;
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod wasip1;
        pub use self::wasip1::*;
    }
    target_family = "wasm" => {
        mod wasm;
        pub use self::wasm::*;
    }
    target_os = "xous" => {
        mod xous;
        pub use self::xous::*;
    }
    target_os = "uefi" => {
        mod uefi;
        pub use self::uefi::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use self::sgx::*;
    }
    target_os = "teeos" => {
        mod teeos;
        pub use self::teeos::*;
    }
    target_os = "zkvm" => {
        mod zkvm;
        pub use self::zkvm::*;
    }
    _ => {
        mod unsupported;
        pub use self::unsupported::*;
    }
}

pub const FULL_BACKTRACE_DEFAULT: bool = cfg_select! {
    // Fuchsia components default to full backtrace.
    target_os = "fuchsia" => true,
    _ => false,
};

#[cfg(not(target_os = "uefi"))]
pub type RawOsError = i32;
