use crate::{fmt, io};

pub fn errno() -> i32 {
    unsafe { hermit_abi::get_errno() }
}

#[inline]
pub fn is_interrupted(errno: i32) -> bool {
    errno == hermit_abi::errno::EINTR
}

pub fn decode_error_kind(errno: i32) -> io::ErrorKind {
    use hermit_abi::errno::*;
    use io::ErrorKind::*;
    match errno {
        E2BIG => ArgumentListTooLong,
        EADDRINUSE => AddrInUse,
        EADDRNOTAVAIL => AddrNotAvailable,
        EBUSY => ResourceBusy,
        ECONNABORTED => ConnectionAborted,
        ECONNREFUSED => ConnectionRefused,
        ECONNRESET => ConnectionReset,
        EDEADLK => Deadlock,
        EDQUOT => QuotaExceeded,
        EEXIST => AlreadyExists,
        EFBIG => FileTooLarge,
        EHOSTUNREACH => HostUnreachable,
        EINTR => Interrupted,
        EINVAL => InvalidInput,
        EISDIR => IsADirectory,
        ELOOP => FilesystemLoop,
        ENOENT => NotFound,
        ENOMEM => OutOfMemory,
        ENOSPC => StorageFull,
        EMLINK => TooManyLinks,
        ENAMETOOLONG => InvalidFilename,
        ENETDOWN => NetworkDown,
        ENETUNREACH => NetworkUnreachable,
        ENOTCONN => NotConnected,
        ENOTDIR => NotADirectory,
        ENOTEMPTY => DirectoryNotEmpty,
        EPIPE => BrokenPipe,
        EROFS => ReadOnlyFilesystem,
        ESPIPE => NotSeekable,
        ESTALE => StaleNetworkFileHandle,
        ETIMEDOUT => TimedOut,
        ETXTBSY => ExecutableFileBusy,
        EXDEV => CrossesDevices,
        EINPROGRESS => InProgress,
        EMFILE | ENFILE => TooManyOpenFiles,
        EIO => InputOutputError,
        EACCES | EPERM => PermissionDenied,
        ENOSYS | EOPNOTSUPP => Unsupported,
        EAGAIN => WouldBlock,
        _ => Uncategorized,
    }
}

pub fn format_error(errno: i32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let description = hermit_abi::error_string(errno);
    f.write_str(description)
}
