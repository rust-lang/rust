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
mod personality;

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
    } else if #[cfg(target_family = "wasm")] {
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

cfg_if::cfg_if! {
    // Fuchsia components default to full backtrace.
    if #[cfg(target_os = "fuchsia")] {
        pub const FULL_BACKTRACE_DEFAULT: bool = true;
    } else {
        pub const FULL_BACKTRACE_DEFAULT: bool = false;
    }
}

#[cfg(not(test))]
cfg_if::cfg_if! {
    if #[cfg(target_os = "android")] {
        pub use self::android::log2f32;
        pub use self::android::log2f64;
    } else {
        #[inline]
        pub fn log2f32(n: f32) -> f32 {
            unsafe { crate::intrinsics::log2f32(n) }
        }

        #[inline]
        pub fn log2f64(n: f64) -> f64 {
            unsafe { crate::intrinsics::log2f64(n) }
        }
    }
}

// Solaris/Illumos requires a wrapper around log, log2, and log10 functions
// because of their non-standard behavior (e.g., log(-n) returns -Inf instead
// of expected NaN).
#[cfg(not(test))]
#[cfg(any(target_os = "solaris", target_os = "illumos"))]
#[inline]
pub fn log_wrapper<F: Fn(f64) -> f64>(n: f64, log_fn: F) -> f64 {
    if n.is_finite() {
        if n > 0.0 {
            log_fn(n)
        } else if n == 0.0 {
            f64::NEG_INFINITY // log(0) = -Inf
        } else {
            f64::NAN // log(-n) = NaN
        }
    } else if n.is_nan() {
        n // log(NaN) = NaN
    } else if n > 0.0 {
        n // log(Inf) = Inf
    } else {
        f64::NAN // log(-Inf) = NaN
    }
}

#[cfg(not(test))]
#[cfg(not(any(target_os = "solaris", target_os = "illumos")))]
#[inline]
pub fn log_wrapper<F: Fn(f64) -> f64>(n: f64, log_fn: F) -> f64 {
    log_fn(n)
}
