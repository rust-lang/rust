#![forbid(unsafe_op_in_unsafe_fn)]

use crate::io as std_io;

#[inline]
pub fn is_interrupted(errno: i32) -> bool {
    errno == wasi::ERRNO_INTR.raw().into()
}

pub fn decode_error_kind(errno: i32) -> std_io::ErrorKind {
    use std_io::ErrorKind;

    let Ok(errno) = u16::try_from(errno) else {
        return ErrorKind::Uncategorized;
    };

    macro_rules! match_errno {
        ($($($errno:ident)|+ => $errkind:ident),*, _ => $wildcard:ident $(,)?) => {
            match errno {
                $(e if $(e == ::wasi::$errno.raw())||+ => ErrorKind::$errkind),*,
                _ => ErrorKind::$wildcard,
            }
        };
    }

    match_errno! {
        ERRNO_2BIG           => ArgumentListTooLong,
        ERRNO_ACCES          => PermissionDenied,
        ERRNO_ADDRINUSE      => AddrInUse,
        ERRNO_ADDRNOTAVAIL   => AddrNotAvailable,
        ERRNO_AFNOSUPPORT    => Unsupported,
        ERRNO_AGAIN          => WouldBlock,
        //    ALREADY        => "connection already in progress",
        //    BADF           => "bad file descriptor",
        //    BADMSG         => "bad message",
        ERRNO_BUSY           => ResourceBusy,
        //    CANCELED       => "operation canceled",
        //    CHILD          => "no child processes",
        ERRNO_CONNABORTED    => ConnectionAborted,
        ERRNO_CONNREFUSED    => ConnectionRefused,
        ERRNO_CONNRESET      => ConnectionReset,
        ERRNO_DEADLK         => Deadlock,
        //    DESTADDRREQ    => "destination address required",
        ERRNO_DOM            => InvalidInput,
        //    DQUOT          => /* reserved */,
        ERRNO_EXIST          => AlreadyExists,
        //    FAULT          => "bad address",
        ERRNO_FBIG           => FileTooLarge,
        ERRNO_HOSTUNREACH    => HostUnreachable,
        //    IDRM           => "identifier removed",
        //    ILSEQ          => "illegal byte sequence",
        //    INPROGRESS     => "operation in progress",
        ERRNO_INTR           => Interrupted,
        ERRNO_INVAL          => InvalidInput,
        ERRNO_IO             => Uncategorized,
        //    ISCONN         => "socket is connected",
        ERRNO_ISDIR          => IsADirectory,
        ERRNO_LOOP           => FilesystemLoop,
        //    MFILE          => "file descriptor value too large",
        ERRNO_MLINK          => TooManyLinks,
        //    MSGSIZE        => "message too large",
        //    MULTIHOP       => /* reserved */,
        ERRNO_NAMETOOLONG    => InvalidFilename,
        ERRNO_NETDOWN        => NetworkDown,
        //    NETRESET       => "connection aborted by network",
        ERRNO_NETUNREACH     => NetworkUnreachable,
        //    NFILE          => "too many files open in system",
        //    NOBUFS         => "no buffer space available",
        ERRNO_NODEV          => NotFound,
        ERRNO_NOENT          => NotFound,
        //    NOEXEC         => "executable file format error",
        //    NOLCK          => "no locks available",
        //    NOLINK         => /* reserved */,
        ERRNO_NOMEM          => OutOfMemory,
        //    NOMSG          => "no message of the desired type",
        //    NOPROTOOPT     => "protocol not available",
        ERRNO_NOSPC          => StorageFull,
        ERRNO_NOSYS          => Unsupported,
        ERRNO_NOTCONN        => NotConnected,
        ERRNO_NOTDIR         => NotADirectory,
        ERRNO_NOTEMPTY       => DirectoryNotEmpty,
        //    NOTRECOVERABLE => "state not recoverable",
        //    NOTSOCK        => "not a socket",
        ERRNO_NOTSUP         => Unsupported,
        //    NOTTY          => "inappropriate I/O control operation",
        ERRNO_NXIO           => NotFound,
        //    OVERFLOW       => "value too large to be stored in data type",
        //    OWNERDEAD      => "previous owner died",
        ERRNO_PERM           => PermissionDenied,
        ERRNO_PIPE           => BrokenPipe,
        //    PROTO          => "protocol error",
        ERRNO_PROTONOSUPPORT => Unsupported,
        //    PROTOTYPE      => "protocol wrong type for socket",
        //    RANGE          => "result too large",
        ERRNO_ROFS           => ReadOnlyFilesystem,
        ERRNO_SPIPE          => NotSeekable,
        ERRNO_SRCH           => NotFound,
        //    STALE          => /* reserved */,
        ERRNO_TIMEDOUT       => TimedOut,
        ERRNO_TXTBSY         => ResourceBusy,
        ERRNO_XDEV           => CrossesDevices,
        ERRNO_NOTCAPABLE     => PermissionDenied,
        _                    => Uncategorized,
    }
}

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

#[inline]
pub(crate) fn err2io(err: wasi::Errno) -> std_io::Error {
    std_io::Error::from_raw_os_error(err.raw().into())
}
