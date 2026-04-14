//! ThingOS platform abstraction layer (PAL).
//!
//! Exports the public surface expected by `library/std/src/sys/pal/mod.rs`:
//! `os`, `time`, and `futex` modules, plus `init`, `cleanup`,
//! `abort_internal`, `unsupported`, and `unsupported_err`.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(missing_docs, nonstandard_style)]

pub mod common;
pub mod futex;
pub mod os;
pub mod time;

use crate::io;

/// Initialise the ThingOS runtime.
///
/// # Safety
/// Must be called exactly once, before any other std functionality.
pub unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    // Store argc/argv for later retrieval by `std::env::args()`.
    unsafe {
        crate::sys::args::init(argc, argv);
    }
}

/// Clean up after the ThingOS runtime.
///
/// # Safety
/// Must be called exactly once, after all std functionality.
pub unsafe fn cleanup() {}

/// Abort the process immediately.
pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}

/// Return an `Err` for operations that are not yet supported on ThingOS.
pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

/// The `io::Error` value returned for unsupported operations.
pub fn unsupported_err() -> io::Error {
    io::const_error!(io::ErrorKind::Unsupported, "operation not supported on ThingOS yet")
}
