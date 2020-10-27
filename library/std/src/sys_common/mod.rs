//! Platform-independent platform abstraction
//!
//! This is the platform-independent portion of the standard library's
//! platform abstraction layer, whereas `std::sys` is the
//! platform-specific portion.
//!
//! The relationship between `std::sys_common`, `std::sys` and the
//! rest of `std` is complex, with dependencies going in all
//! directions: `std` depending on `sys_common`, `sys_common`
//! depending on `sys`, and `sys` depending on `sys_common` and `std`.
//! Ideally `sys_common` would be split into two and the dependencies
//! between them all would form a dag, facilitating the extraction of
//! `std::sys` from the standard library.

#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

#[cfg(test)]
mod tests;

use crate::sync::Once;
use crate::sys;

macro_rules! rtabort {
    ($($t:tt)*) => (crate::sys_common::util::abort(format_args!($($t)*)))
}

macro_rules! rtassert {
    ($e:expr) => {
        if !$e {
            rtabort!(concat!("assertion failed: ", stringify!($e)));
        }
    };
}

#[allow(unused_macros)] // not used on all platforms
macro_rules! rtunwrap {
    ($ok:ident, $e:expr) => {
        match $e {
            $ok(v) => v,
            ref err => {
                let err = err.as_ref().map(drop); // map Ok/Some which might not be Debug
                rtabort!(concat!("unwrap failed: ", stringify!($e), " = {:?}"), err)
            }
        }
    };
}

pub mod alloc;
pub mod at_exit_imp;
pub mod backtrace;
pub mod bytestring;
pub mod condvar;
pub mod fs;
pub mod io;
pub mod mutex;
// `doc` is required because `sys/mod.rs` imports `unix/ext/mod.rs` on Windows
// when generating documentation.
#[cfg(any(doc, not(windows)))]
pub mod os_str_bytes;
pub mod poison;
pub mod process;
pub mod remutex;
pub mod rwlock;
pub mod thread;
pub mod thread_info;
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod thread_parker;
pub mod util;
pub mod wtf8;

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "l4re",
                 target_os = "hermit",
                 feature = "restricted-std",
                 all(target_arch = "wasm32", not(target_os = "emscripten")),
                 all(target_vendor = "fortanix", target_env = "sgx")))] {
        pub use crate::sys::net;
    } else {
        pub mod net;
    }
}

// common error constructors

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types
#[doc(hidden)]
pub trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations
#[doc(hidden)]
pub trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}

/// Enqueues a procedure to run when the main thread exits.
///
/// Currently these closures are only run once the main *Rust* thread exits.
/// Once the `at_exit` handlers begin running, more may be enqueued, but not
/// infinitely so. Eventually a handler registration will be forced to fail.
///
/// Returns `Ok` if the handler was successfully registered, meaning that the
/// closure will be run once the main thread exits. Returns `Err` to indicate
/// that the closure could not be registered, meaning that it is not scheduled
/// to be run.
pub fn at_exit<F: FnOnce() + Send + 'static>(f: F) -> Result<(), ()> {
    if at_exit_imp::push(Box::new(f)) { Ok(()) } else { Err(()) }
}

/// One-time runtime cleanup.
pub fn cleanup() {
    static CLEANUP: Once = Once::new();
    CLEANUP.call_once(|| unsafe {
        sys::args::cleanup();
        sys::stack_overflow::cleanup();
        at_exit_imp::cleanup();
    });
}

// Computes (value*numer)/denom without overflow, as long as both
// (numer*denom) and the overall result fit into i64 (which is the case
// for our time conversions).
#[allow(dead_code)] // not used on all platforms
pub fn mul_div_u64(value: u64, numer: u64, denom: u64) -> u64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numer)/denom and simplify.
    // r < denom, so (denom*numer) is the upper bound of (r*numer)
    q * numer + r * numer / denom
}
