//! Syscall error conventions and errno helpers.
//!
//! # Examples
//! ```
//! use abi::errors::{Errno, errno};
//!
//! let rc: isize = Errno::ENOENT.as_isize();
//! assert_eq!(rc, -2);
//! assert_eq!(errno(rc), Err(Errno::ENOENT));
//! assert_eq!(errno(0), Ok(0));
//! ```

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Errno {
    Success = 0,
    EPERM = 1,
    ENOENT = 2,
    ESRCH = 3,
    EINTR = 4,
    EIO = 5,
    ENXIO = 6,
    E2BIG = 7,
    ENOEXEC = 8,
    EBADF = 9,
    ECHILD = 10,
    EAGAIN = 11,
    ENOMEM = 12,
    EACCES = 13,
    EFAULT = 14,
    ENOTBLK = 15,
    EBUSY = 16,
    EEXIST = 17,
    EXDEV = 18,
    ENODEV = 19,
    ENOTDIR = 20,
    EISDIR = 21,
    EINVAL = 22,
    ENFILE = 23,
    EMFILE = 24,
    ENOTTY = 25,
    ETXTBSY = 26,
    EFBIG = 27,
    ENOSPC = 28,
    ESPIPE = 29,
    EROFS = 30,
    EMLINK = 31,
    EPIPE = 32,
    EDOM = 33,
    ERANGE = 34,
    ENAMETOOLONG = 36,
    ENOSYS = 38,
    /// Too many levels of symbolic links.
    ELOOP = 40,
    EOVERFLOW = 75,
    ENOBUFS = 105,
    EMSGSIZE = 90,
    ETIMEDOUT = 110,
    ECONNREFUSED = 111,
    // Add more as needed, following Linux numbers usually helps debugging

    // Socket-related errors (Linux numbers)
    /// Operation not supported on transport endpoint.
    EOPNOTSUPP = 95,
    /// Address family not supported by protocol.
    EAFNOSUPPORT = 97,
    /// Address already in use.
    EADDRINUSE = 98,
    /// Protocol wrong type for socket.
    EPROTOTYPE = 91,
    /// Socket operation on non-socket.
    ENOTSOCK = 88,
    /// Transport endpoint is not connected.
    ENOTCONN = 107,

    // Custom/Extension
}

pub type SysResult<T> = core::result::Result<T, Errno>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Errno(Errno),
    BufferTooSmall { required: usize, available: usize },
    InvalidDataLength { expected: usize, actual: usize },
}

impl From<Errno> for Error {
    fn from(e: Errno) -> Self {
        Error::Errno(e)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

impl Errno {
    #[allow(non_upper_case_globals)]
    pub const NotSupported: Errno = Errno::ENOSYS;
    pub fn as_isize(self) -> isize {
        -(self as isize)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Errno::Success => "success",
            Errno::EPERM => "operation not permitted",
            Errno::ENOENT => "no such file or directory",
            Errno::ESRCH => "no such process",
            Errno::EINTR => "interrupted system call",
            Errno::EIO => "i/o error",
            Errno::ENXIO => "no such device or address",
            Errno::E2BIG => "argument list too long",
            Errno::ENOEXEC => "exec format error",
            Errno::EBADF => "bad file descriptor",
            Errno::ECHILD => "no child processes",
            Errno::EAGAIN => "resource temporarily unavailable",
            Errno::ENOMEM => "cannot allocate memory",
            Errno::EACCES => "permission denied",
            Errno::EFAULT => "bad address",
            Errno::ENOTBLK => "block device required",
            Errno::EBUSY => "device or resource busy",
            Errno::EEXIST => "file exists",
            Errno::EXDEV => "invalid cross-device link",
            Errno::ENODEV => "no such device",
            Errno::ENOTDIR => "not a directory",
            Errno::EISDIR => "is a directory",
            Errno::EINVAL => "invalid argument",
            Errno::ENFILE => "too many open files in system",
            Errno::EMFILE => "too many open files",
            Errno::ENOTTY => "inappropriate ioctl for device",
            Errno::ETXTBSY => "text file busy",
            Errno::EFBIG => "file too large",
            Errno::ENOSPC => "no space left on device",
            Errno::ESPIPE => "illegal seek",
            Errno::EROFS => "read-only file system",
            Errno::EMLINK => "too many links",
            Errno::EPIPE => "broken pipe",
            Errno::EDOM => "numerical argument out of domain",
            Errno::ERANGE => "numerical result out of range",
            Errno::ENAMETOOLONG => "file name too long",
            Errno::ENOSYS => "function not implemented",
            Errno::ELOOP => "too many levels of symbolic links",
            Errno::EOVERFLOW => "value too large for defined data type",
            Errno::EMSGSIZE => "message too long",
            Errno::ENOBUFS => "no buffer space available",
            Errno::ETIMEDOUT => "connection timed out",
            Errno::ECONNREFUSED => "connection refused",
            Errno::ENOTSOCK => "socket operation on non-socket",
            Errno::EPROTOTYPE => "protocol wrong type for socket",
            Errno::EOPNOTSUPP => "operation not supported",
            Errno::EAFNOSUPPORT => "address family not supported by protocol",
            Errno::EADDRINUSE => "address already in use",
            Errno::ENOTCONN => "transport endpoint is not connected",
        }
    }
}

impl core::fmt::Display for Errno {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub fn errno(ret: isize) -> core::result::Result<usize, Errno> {
    if ret < 0 && ret >= -4096 {
        // This is a rough mapping back, optimizing for common case
        // In a real impl we'd match every value.
        // For now let's just assume it's valid if negative.
        // But to be safe in Rust enum, we might want to transmute if we trust the source,
        // or just return a generic error.
        // Let's do a basic match for the ones we care about.
        let code = -ret;
        match code {
            1 => Err(Errno::EPERM),
            2 => Err(Errno::ENOENT),
            3 => Err(Errno::ESRCH),
            4 => Err(Errno::EINTR),
            5 => Err(Errno::EIO),
            6 => Err(Errno::ENXIO),
            7 => Err(Errno::E2BIG),
            8 => Err(Errno::ENOEXEC),
            9 => Err(Errno::EBADF),
            10 => Err(Errno::ECHILD),
            11 => Err(Errno::EAGAIN),
            12 => Err(Errno::ENOMEM),
            13 => Err(Errno::EACCES),
            14 => Err(Errno::EFAULT),
            15 => Err(Errno::ENOTBLK),
            16 => Err(Errno::EBUSY),
            17 => Err(Errno::EEXIST),
            18 => Err(Errno::EXDEV),
            19 => Err(Errno::ENODEV),
            20 => Err(Errno::ENOTDIR),
            21 => Err(Errno::EISDIR),
            22 => Err(Errno::EINVAL),
            23 => Err(Errno::ENFILE),
            24 => Err(Errno::EMFILE),
            25 => Err(Errno::ENOTTY),
            26 => Err(Errno::ETXTBSY),
            27 => Err(Errno::EFBIG),
            28 => Err(Errno::ENOSPC),
            29 => Err(Errno::ESPIPE),
            30 => Err(Errno::EROFS),
            31 => Err(Errno::EMLINK),
            32 => Err(Errno::EPIPE),
            33 => Err(Errno::EDOM),
            34 => Err(Errno::ERANGE),
            36 => Err(Errno::ENAMETOOLONG),
            38 => Err(Errno::ENOSYS),
            40 => Err(Errno::ELOOP),
            75 => Err(Errno::EOVERFLOW),
            90 => Err(Errno::EMSGSIZE),
            105 => Err(Errno::ENOBUFS),
            110 => Err(Errno::ETIMEDOUT),
            111 => Err(Errno::ECONNREFUSED),
            88 => Err(Errno::ENOTSOCK),
            91 => Err(Errno::EPROTOTYPE),
            95 => Err(Errno::EOPNOTSUPP),
            97 => Err(Errno::EAFNOSUPPORT),
            98 => Err(Errno::EADDRINUSE),
            107 => Err(Errno::ENOTCONN),
            _ => Err(Errno::EINVAL), // Fallback for unknown codes
        }
    } else {
        Ok(ret as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_as_isize_returns_negative_value() {
        assert_eq!(Errno::EPERM.as_isize(), -1);
        assert_eq!(Errno::ENOENT.as_isize(), -2);
        assert_eq!(Errno::EINVAL.as_isize(), -22);
        assert_eq!(Errno::ENOSYS.as_isize(), -38);
    }

    #[test]
    fn errno_success_as_isize_is_zero() {
        assert_eq!(Errno::Success.as_isize(), 0);
    }

    #[test]
    fn errno_recognizes_eperm() {
        assert_eq!(errno(-1), Err(Errno::EPERM));
    }

    #[test]
    fn errno_recognizes_enoent() {
        assert_eq!(errno(-2), Err(Errno::ENOENT));
    }

    #[test]
    fn errno_recognizes_eio() {
        assert_eq!(errno(-5), Err(Errno::EIO));
    }

    #[test]
    fn errno_recognizes_eagain() {
        assert_eq!(errno(-11), Err(Errno::EAGAIN));
    }

    #[test]
    fn errno_recognizes_enomem() {
        assert_eq!(errno(-12), Err(Errno::ENOMEM));
    }

    #[test]
    fn errno_recognizes_efault() {
        assert_eq!(errno(-14), Err(Errno::EFAULT));
    }

    #[test]
    fn errno_recognizes_einval() {
        assert_eq!(errno(-22), Err(Errno::EINVAL));
    }

    #[test]
    fn errno_recognizes_enosys() {
        assert_eq!(errno(-38), Err(Errno::ENOSYS));
    }

    #[test]
    fn errno_unknown_code_fallback_to_einval() {
        assert_eq!(errno(-999), Err(Errno::EINVAL));
    }

    #[test]
    fn errno_success_returns_ok() {
        assert_eq!(errno(0), Ok(0usize));
    }

    #[test]
    fn errno_positive_returns_ok() {
        assert_eq!(errno(42), Ok(42usize));
        assert_eq!(errno(1000), Ok(1000usize));
    }

    #[test]
    fn errno_large_negative_is_fallback() {
        // Values < -4096 are not treated as errors
        assert_eq!(errno(-5000), Ok((-5000isize) as usize));
    }
}
