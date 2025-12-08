#![allow(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod time;

pub use moto_rt::futex;

use crate::io;
use crate::sys::RawOsError;

pub(crate) fn map_motor_error(err: moto_rt::ErrorCode) -> io::Error {
    io::Error::from_raw_os_error(err.into())
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

pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::UNSUPPORTED_PLATFORM
}

pub fn is_interrupted(_code: RawOsError) -> bool {
    false // Motor OS doesn't have signals.
}

pub fn decode_error_kind(code: RawOsError) -> io::ErrorKind {
    use moto_rt::error::*;

    if code < 0 || code > u16::MAX.into() {
        return io::ErrorKind::Uncategorized;
    }

    match code as moto_rt::ErrorCode /* u16 */ {
        E_UNSPECIFIED => io::ErrorKind::Uncategorized,
        E_UNKNOWN => io::ErrorKind::Uncategorized,
        E_NOT_READY => io::ErrorKind::WouldBlock,
        E_NOT_IMPLEMENTED => io::ErrorKind::Unsupported,
        E_VERSION_TOO_HIGH => io::ErrorKind::Unsupported,
        E_VERSION_TOO_LOW => io::ErrorKind::Unsupported,
        E_INVALID_ARGUMENT => io::ErrorKind::InvalidInput,
        E_OUT_OF_MEMORY => io::ErrorKind::OutOfMemory,
        E_NOT_ALLOWED => io::ErrorKind::PermissionDenied,
        E_NOT_FOUND => io::ErrorKind::NotFound,
        E_INTERNAL_ERROR => io::ErrorKind::Other,
        E_TIMED_OUT => io::ErrorKind::TimedOut,
        E_ALREADY_IN_USE => io::ErrorKind::AlreadyExists,
        E_UNEXPECTED_EOF => io::ErrorKind::UnexpectedEof,
        E_INVALID_FILENAME => io::ErrorKind::InvalidFilename,
        E_NOT_A_DIRECTORY => io::ErrorKind::NotADirectory,
        E_BAD_HANDLE => io::ErrorKind::InvalidInput,
        E_FILE_TOO_LARGE => io::ErrorKind::FileTooLarge,
        E_NOT_CONNECTED => io::ErrorKind::NotConnected,
        E_STORAGE_FULL => io::ErrorKind::StorageFull,
        E_INVALID_DATA => io::ErrorKind::InvalidData,
        _ => io::ErrorKind::Uncategorized,
    }
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}
