#![allow(dead_code)]
#![allow(missing_docs, nonstandard_style)]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod abi;

#[path = "../itron"]
pub mod itron {
    pub mod abi;
    pub mod error;
    pub mod spin;
    pub mod task;
    pub mod thread;
    pub mod thread_parking;
    pub mod time;
    use super::unsupported;
}

#[path = "../unsupported/args.rs"]
pub mod args;
pub mod env;
// `error` is `pub(crate)` so that it can be accessed by `itron/error.rs` as
// `crate::sys::error`
pub(crate) mod error;
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
pub use self::itron::{thread, thread_parking};
pub mod time;

// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::UNSUPPORTED_PLATFORM
}

#[inline]
pub fn is_interrupted(code: i32) -> bool {
    crate::sys::net::is_interrupted(code)
}

pub fn decode_error_kind(code: i32) -> crate::io::ErrorKind {
    error::decode_error_kind(code)
}

#[inline]
pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}
