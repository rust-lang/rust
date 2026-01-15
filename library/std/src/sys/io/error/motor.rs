use crate::io;
use crate::sys::io::RawOsError;

pub fn errno() -> RawOsError {
    // Not used in Motor OS because it is ambiguous: Motor OS
    // is micro-kernel-based, and I/O happens via a shared-memory
    // ring buffer, so an I/O operation that on a unix is a syscall
    // may involve no sycalls on Motor OS at all, or a syscall
    // that e.g. waits for a notification from the I/O driver
    // (sys-io); and the wait syscall may succeed, but the
    // driver may report an I/O error; or a bunch of results
    // for several I/O operations, some successful and some
    // not.
    //
    // Also I/O operations in a Motor OS process are handled by a
    // separate runtime background/I/O thread, so it is really hard
    // to define what "last system error in the current thread"
    // actually means.
    let error_code: moto_rt::ErrorCode = moto_rt::Error::Unknown.into();
    error_code.into()
}

pub fn is_interrupted(_code: io::RawOsError) -> bool {
    false // Motor OS doesn't have signals.
}

pub fn decode_error_kind(code: io::RawOsError) -> io::ErrorKind {
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

pub fn error_string(errno: RawOsError) -> String {
    let error: moto_rt::Error = match errno {
        x if x < 0 => moto_rt::Error::Unknown,
        x if x > u16::MAX.into() => moto_rt::Error::Unknown,
        x => (x as moto_rt::ErrorCode).into(), /* u16 */
    };
    format!("{}", error)
}
