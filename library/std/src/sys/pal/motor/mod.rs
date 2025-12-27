#![allow(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod time;

pub use moto_rt::futex;

use crate::io as std_io;
use crate::sys::io::RawOsError;

pub(crate) fn map_motor_error(err: moto_rt::Error) -> crate::io::Error {
    let error_code: moto_rt::ErrorCode = err.into();
    crate::io::Error::from_raw_os_error(error_code.into())
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
    use std_io::ErrorKind;

    if code < 0 || code > u16::MAX.into() {
        return std_io::ErrorKind::Uncategorized;
    }

    let error = moto_rt::Error::from(code as moto_rt::ErrorCode);

    match error {
        moto_rt::Error::Unspecified => ErrorKind::Uncategorized,
        moto_rt::Error::Unknown => ErrorKind::Uncategorized,
        moto_rt::Error::NotReady => ErrorKind::WouldBlock,
        moto_rt::Error::NotImplemented => ErrorKind::Unsupported,
        moto_rt::Error::VersionTooHigh => ErrorKind::Unsupported,
        moto_rt::Error::VersionTooLow => ErrorKind::Unsupported,
        moto_rt::Error::InvalidArgument => ErrorKind::InvalidInput,
        moto_rt::Error::OutOfMemory => ErrorKind::OutOfMemory,
        moto_rt::Error::NotAllowed => ErrorKind::PermissionDenied,
        moto_rt::Error::NotFound => ErrorKind::NotFound,
        moto_rt::Error::InternalError => ErrorKind::Other,
        moto_rt::Error::TimedOut => ErrorKind::TimedOut,
        moto_rt::Error::AlreadyInUse => ErrorKind::AlreadyExists,
        moto_rt::Error::UnexpectedEof => ErrorKind::UnexpectedEof,
        moto_rt::Error::InvalidFilename => ErrorKind::InvalidFilename,
        moto_rt::Error::NotADirectory => ErrorKind::NotADirectory,
        moto_rt::Error::BadHandle => ErrorKind::InvalidInput,
        moto_rt::Error::FileTooLarge => ErrorKind::FileTooLarge,
        moto_rt::Error::NotConnected => ErrorKind::NotConnected,
        moto_rt::Error::StorageFull => ErrorKind::StorageFull,
        moto_rt::Error::InvalidData => ErrorKind::InvalidData,
        _ => crate::io::ErrorKind::Uncategorized,
    }
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}
