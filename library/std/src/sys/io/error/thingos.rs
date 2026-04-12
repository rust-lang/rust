//! ThingOS I/O error mapping.
//!
//! ThingOS uses POSIX-compatible errno values, so mapping is straightforward.

use crate::io;
use crate::sys::io::RawOsError;

/// Returns the value of `errno` for the current thread.
///
/// On ThingOS, errno is stored in a kernel-managed thread-local variable
/// accessible via a dedicated syscall.  For now we return 0 as a safe
/// placeholder; individual syscall wrappers return `-errno` directly.
pub fn errno() -> RawOsError {
    0
}

/// Returns `true` when the error code represents an interrupted operation.
#[inline]
pub fn is_interrupted(code: io::RawOsError) -> bool {
    code == 4 // EINTR
}

/// Map a raw ThingOS errno value to an `io::ErrorKind`.
pub fn decode_error_kind(code: io::RawOsError) -> io::ErrorKind {
    use io::ErrorKind::*;
    match code {
        1 => PermissionDenied,   // EPERM
        2 => NotFound,           // ENOENT
        4 => Interrupted,        // EINTR
        5 => InvalidData,        // EIO (general I/O)
        11 => WouldBlock,        // EAGAIN / EWOULDBLOCK
        12 => OutOfMemory,       // ENOMEM
        13 => PermissionDenied,  // EACCES
        17 => AlreadyExists,     // EEXIST
        20 => NotADirectory,     // ENOTDIR
        21 => IsADirectory,      // EISDIR
        22 => InvalidInput,      // EINVAL
        23 => QuotaExceeded,     // ENFILE (used as quota)
        24 => QuotaExceeded,     // EMFILE
        28 => StorageFull,       // ENOSPC
        32 => BrokenPipe,        // EPIPE
        36 => InvalidFilename,   // ENAMETOOLONG
        61 => ConnectionRefused, // ECONNREFUSED
        62 => AddrInUse,         // EADDRINUSE
        95 => Unsupported,       // EOPNOTSUPP
        97 => Unsupported,       // EAFNOSUPPORT (IPv6)
        98 => AddrInUse,         // EADDRINUSE
        104 => ConnectionReset,  // ECONNRESET
        107 => NotConnected,     // ENOTCONN
        108 => NotConnected,     // ESHUTDOWN
        110 => TimedOut,         // ETIMEDOUT
        111 => ConnectionRefused, // ECONNREFUSED
        113 => HostUnreachable,  // EHOSTUNREACH
        // 116 is often ESTALE; treat as other
        _ => Uncategorized,
    }
}

/// Format a ThingOS errno value as a human-readable string.
pub fn error_string(errno: RawOsError) -> String {
    // For a production implementation this would call into a kernel
    // error-string facility; for now we format the number.
    format!("ThingOS error code {errno}")
}
