use crate::io::ErrorKind;

pub fn errno() -> i32 {
    0
}

pub fn is_interrupted(code: i32) -> bool {
    code == 4
}

pub fn decode_error_kind(code: i32) -> ErrorKind {
    match code {
        1 => ErrorKind::PermissionDenied,    // EPERM
        2 => ErrorKind::NotFound,            // ENOENT
        4 => ErrorKind::Interrupted,         // EINTR
        5 => ErrorKind::Uncategorized,       // EIO
        9 => ErrorKind::InvalidInput,        // EBADF
        11 => ErrorKind::WouldBlock,         // EAGAIN / EWOULDBLOCK
        12 => ErrorKind::OutOfMemory,        // ENOMEM
        13 => ErrorKind::PermissionDenied,   // EACCES
        16 => ErrorKind::ResourceBusy,       // EBUSY
        17 => ErrorKind::AlreadyExists,      // EEXIST
        20 => ErrorKind::NotADirectory,      // ENOTDIR
        21 => ErrorKind::IsADirectory,       // EISDIR
        22 => ErrorKind::InvalidInput,       // EINVAL
        28 => ErrorKind::StorageFull,        // ENOSPC
        32 => ErrorKind::BrokenPipe,         // EPIPE
        38 => ErrorKind::Unsupported,        // ENOSYS
        39 => ErrorKind::DirectoryNotEmpty,  // ENOTEMPTY
        75 => ErrorKind::InvalidData,        // EOVERFLOW
        95 | 97 => ErrorKind::Unsupported,   // EOPNOTSUPP / EAFNOSUPPORT
        110 => ErrorKind::TimedOut,          // ETIMEDOUT
        111 => ErrorKind::ConnectionRefused, // ECONNREFUSED
        _ => ErrorKind::Uncategorized,
    }
}

pub fn error_string(errno: i32) -> String {
    match errno {
        1 => "operation not permitted",
        2 => "no such file or directory",
        4 => "interrupted system call",
        5 => "i/o error",
        9 => "bad file descriptor",
        11 => "resource temporarily unavailable",
        12 => "cannot allocate memory",
        13 => "permission denied",
        16 => "device or resource busy",
        17 => "file exists",
        20 => "not a directory",
        21 => "is a directory",
        22 => "invalid argument",
        28 => "no space left on device",
        32 => "broken pipe",
        38 => "function not implemented",
        39 => "directory not empty",
        75 => "value too large for defined data type",
        95 => "operation not supported",
        97 => "address family not supported",
        110 => "timed out",
        111 => "connection refused",
        _ => "uncategorized error",
    }
    .to_string()
}
