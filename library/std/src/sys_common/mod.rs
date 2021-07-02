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
//! This is because `sys_common` not only contains platform-independent code,
//! but also code that is shared between the different platforms in `sys`.
//! Ideally all that shared code should be moved to `sys::common`,
//! and the dependencies between `std`, `sys_common` and `sys` all would form a dag.
//! Progress on this is tracked in #84187.

#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

#[cfg(test)]
mod tests;

pub mod backtrace;
pub mod condvar;
pub mod fs;
pub mod io;
pub mod memchr;
pub mod mutex;
// `doc` is required because `sys/mod.rs` imports `unix/ext/mod.rs` on Windows
// when generating documentation.
#[cfg(any(doc, not(windows)))]
pub mod os_str_bytes;
pub mod process;
pub mod remutex;
#[macro_use]
pub mod rt;
pub mod rwlock;
pub mod thread;
pub mod thread_info;
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod thread_parker;
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
