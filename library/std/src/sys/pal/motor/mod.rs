#![allow(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod pipe;
pub mod time;

pub use moto_rt::futex;

use crate::io as std_io;
use crate::sys::RawOsError;

pub(crate) fn map_motor_error(err: moto_rt::ErrorCode) -> crate::io::Error {
    crate::io::Error::from_raw_os_error(err.into())
}

#[cfg(not(test))]
#[unsafe(no_mangle)]
pub extern "C" fn motor_start() -> ! {
    // Initialize the runtime.
    moto_rt::start();

    // Call main.
    unsafe extern "C" {
        fn main(_: isize, _: *const *const u8, _: u8) -> i32;
    }
    let result = unsafe { main(0, core::ptr::null(), 0) };

    // Terminate the process.
    moto_rt::process::exit(result)
}

// SAFETY: must be called only once during runtime initialization.
// NOTE: Motor OS uses moto_rt::start() to initialize runtime (see above).
pub unsafe fn init(_argc: isize, _argv: *const *const u8, _sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> std_io::Error {
    std_io::Error::UNSUPPORTED_PLATFORM
}

pub fn is_interrupted(_code: RawOsError) -> bool {
    false // Motor OS doesn't have signals.
}

pub fn decode_error_kind(code: RawOsError) -> crate::io::ErrorKind {
    use moto_rt::error::*;
    use std_io::ErrorKind;

    if code < 0 || code > u16::MAX.into() {
        return std_io::ErrorKind::Uncategorized;
    }

    match code as moto_rt::ErrorCode /* u16 */ {
        E_ALREADY_IN_USE => ErrorKind::AlreadyExists,
        E_INVALID_FILENAME => ErrorKind::InvalidFilename,
        E_NOT_FOUND => ErrorKind::NotFound,
        E_TIMED_OUT => ErrorKind::TimedOut,
        E_NOT_IMPLEMENTED => ErrorKind::Unsupported,
        E_FILE_TOO_LARGE => ErrorKind::FileTooLarge,
        E_UNEXPECTED_EOF => ErrorKind::UnexpectedEof,
        E_INVALID_ARGUMENT => ErrorKind::InvalidInput,
        E_NOT_READY => ErrorKind::WouldBlock,
        E_NOT_CONNECTED => ErrorKind::NotConnected,
        _ => crate::io::ErrorKind::Uncategorized,
    }
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}
