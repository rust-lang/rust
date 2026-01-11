#![allow(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod time;

pub use moto_rt::futex;

use crate::io;

pub(crate) fn map_motor_error(err: moto_rt::Error) -> io::Error {
    let error_code: moto_rt::ErrorCode = err.into();
    io::Error::from_raw_os_error(error_code.into())
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

pub fn is_interrupted(_code: io::RawOsError) -> bool {
    false // Motor OS doesn't have signals.
}

pub fn decode_error_kind(code: io::RawOsError) -> io::ErrorKind {
    use moto_rt::error::*;

    if code < 0 || code > u16::MAX.into() {
        return io::ErrorKind::Uncategorized;
    }

    let error = moto_rt::Error::from(code as moto_rt::ErrorCode);

    match error {
        moto_rt::Error::Unspecified => io::ErrorKind::Uncategorized,
        moto_rt::Error::Unknown => io::ErrorKind::Uncategorized,
        moto_rt::Error::NotReady => io::ErrorKind::WouldBlock,
        moto_rt::Error::NotImplemented => io::ErrorKind::Unsupported,
        moto_rt::Error::VersionTooHigh => io::ErrorKind::Unsupported,
        moto_rt::Error::VersionTooLow => io::ErrorKind::Unsupported,
        moto_rt::Error::InvalidArgument => io::ErrorKind::InvalidInput,
        moto_rt::Error::OutOfMemory => io::ErrorKind::OutOfMemory,
        moto_rt::Error::NotAllowed => io::ErrorKind::PermissionDenied,
        moto_rt::Error::NotFound => io::ErrorKind::NotFound,
        moto_rt::Error::InternalError => io::ErrorKind::Other,
        moto_rt::Error::TimedOut => io::ErrorKind::TimedOut,
        moto_rt::Error::AlreadyInUse => io::ErrorKind::AlreadyExists,
        moto_rt::Error::UnexpectedEof => io::ErrorKind::UnexpectedEof,
        moto_rt::Error::InvalidFilename => io::ErrorKind::InvalidFilename,
        moto_rt::Error::NotADirectory => io::ErrorKind::NotADirectory,
        moto_rt::Error::BadHandle => io::ErrorKind::InvalidInput,
        moto_rt::Error::FileTooLarge => io::ErrorKind::FileTooLarge,
        moto_rt::Error::NotConnected => io::ErrorKind::NotConnected,
        moto_rt::Error::StorageFull => io::ErrorKind::StorageFull,
        moto_rt::Error::InvalidData => io::ErrorKind::InvalidData,
        _ => io::ErrorKind::Uncategorized,
    }
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}
