use crate::ffi::c_int;
#[cfg(not(target_os = "teeos"))]
use crate::ffi::{CStr, c_char};
use crate::io;

unsafe extern "C" {
    #[cfg(not(any(
        target_os = "dragonfly",
        target_os = "vxworks",
        target_os = "rtems",
        target_os = "wasi"
    )))]
    #[cfg_attr(
        any(
            target_os = "linux",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "l4re",
            target_os = "hurd",
            target_os = "teeos",
        ),
        link_name = "__errno_location"
    )]
    #[cfg_attr(
        any(
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "cygwin",
            target_os = "android",
            target_os = "redox",
            target_os = "nuttx",
            target_env = "newlib"
        ),
        link_name = "__errno"
    )]
    #[cfg_attr(any(target_os = "solaris", target_os = "illumos"), link_name = "___errno")]
    #[cfg_attr(target_os = "nto", link_name = "__get_errno_ptr")]
    #[cfg_attr(target_os = "qnx", link_name = "__get_errno_ptr")]
    #[cfg_attr(any(target_os = "freebsd", target_vendor = "apple"), link_name = "__error")]
    #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
    #[cfg_attr(target_os = "aix", link_name = "_Errno")]
    // SAFETY: this will always return the same pointer on a given thread.
    #[unsafe(ffi_const)]
    pub safe fn errno_location() -> *mut c_int;
}

/// Returns the platform-specific value of errno
#[cfg(not(any(
    target_os = "dragonfly",
    target_os = "vxworks",
    target_os = "rtems",
    target_os = "wasi"
)))]
#[inline]
pub fn errno() -> i32 {
    unsafe { (*errno_location()) as i32 }
}

/// Sets the platform-specific value of errno
// needed for readdir and syscall!
#[cfg(not(any(
    target_os = "dragonfly",
    target_os = "vxworks",
    target_os = "rtems",
    target_os = "wasi",
)))]
#[allow(dead_code)] // but not all target cfgs actually end up using it
#[inline]
pub fn set_errno(e: i32) {
    unsafe { *errno_location() = e as c_int }
}

#[cfg(target_os = "vxworks")]
#[inline]
pub fn errno() -> i32 {
    unsafe { libc::errnoGet() }
}

#[cfg(target_os = "rtems")]
#[inline]
pub fn errno() -> i32 {
    unsafe extern "C" {
        #[thread_local]
        static _tls_errno: c_int;
    }

    unsafe { _tls_errno as i32 }
}

#[cfg(target_os = "dragonfly")]
#[inline]
pub fn errno() -> i32 {
    unsafe extern "C" {
        #[thread_local]
        static errno: c_int;
    }

    unsafe { errno as i32 }
}

#[cfg(target_os = "dragonfly")]
#[allow(dead_code)]
#[inline]
pub fn set_errno(e: i32) {
    unsafe extern "C" {
        #[thread_local]
        static mut errno: c_int;
    }

    unsafe {
        errno = e;
    }
}

#[cfg(target_os = "wasi")]
unsafe extern "C" {
    #[thread_local]
    #[link_name = "errno"]
    static mut libc_errno: libc::c_int;
}

#[cfg(target_os = "wasi")]
pub fn errno() -> i32 {
    unsafe { libc_errno as i32 }
}

#[cfg(target_os = "wasi")]
pub fn set_errno(val: i32) {
    unsafe {
        libc_errno = val;
    }
}

#[inline]
pub fn is_interrupted(errno: i32) -> bool {
    errno == libc::EINTR
}

pub fn decode_error_kind(errno: i32) -> io::ErrorKind {
    use io::ErrorKind::*;
    match errno as libc::c_int {
        libc::E2BIG => ArgumentListTooLong,
        libc::EADDRINUSE => AddrInUse,
        libc::EADDRNOTAVAIL => AddrNotAvailable,
        libc::EBUSY => ResourceBusy,
        libc::ECONNABORTED => ConnectionAborted,
        libc::ECONNREFUSED => ConnectionRefused,
        libc::ECONNRESET => ConnectionReset,
        libc::EDEADLK => Deadlock,
        libc::EDQUOT => QuotaExceeded,
        libc::EEXIST => AlreadyExists,
        libc::EFBIG => FileTooLarge,
        libc::EHOSTUNREACH => HostUnreachable,
        libc::EINTR => Interrupted,
        libc::EINVAL => InvalidInput,
        libc::EISDIR => IsADirectory,
        libc::ELOOP => FilesystemLoop,
        libc::ENOENT => NotFound,
        libc::ENOMEM => OutOfMemory,
        libc::ENOSPC => StorageFull,
        libc::ENOSYS => Unsupported,
        libc::EMLINK => TooManyLinks,
        libc::ENAMETOOLONG => InvalidFilename,
        libc::ENETDOWN => NetworkDown,
        libc::ENETUNREACH => NetworkUnreachable,
        libc::ENOTCONN => NotConnected,
        libc::ENOTDIR => NotADirectory,
        #[cfg(not(target_os = "aix"))]
        libc::ENOTEMPTY => DirectoryNotEmpty,
        libc::EPIPE => BrokenPipe,
        libc::EROFS => ReadOnlyFilesystem,
        libc::ESPIPE => NotSeekable,
        libc::ESTALE => StaleNetworkFileHandle,
        libc::ETIMEDOUT => TimedOut,
        libc::ETXTBSY => ExecutableFileBusy,
        libc::EXDEV => CrossesDevices,
        libc::EINPROGRESS => InProgress,
        libc::EMFILE | libc::ENFILE => TooManyOpenFiles,
        libc::EOPNOTSUPP => Unsupported,
        libc::EIO => InputOutputError,

        libc::EACCES | libc::EPERM => PermissionDenied,

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK => WouldBlock,

        _ => Uncategorized,
    }
}

/// Gets a detailed string description for the given error number.
#[cfg(any(target_family = "unix", target_os = "wasi"))]
pub fn error_string(errno: i32) -> String {
    const TMPBUF_SZ: usize = if cfg!(target_os = "wasi") { 1024 } else { 128 };

    unsafe extern "C" {
        #[cfg_attr(
            all(
                any(
                    target_os = "linux",
                    target_os = "hurd",
                    target_env = "newlib",
                    target_os = "cygwin"
                ),
                not(target_env = "ohos")
            ),
            link_name = "__xpg_strerror_r"
        )]
        fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: libc::size_t) -> c_int;
    }

    let mut buf = [0 as c_char; TMPBUF_SZ];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as c_int, p, buf.len()) < 0 {
            panic!("strerror_r failure");
        }

        let p = p as *const _;
        // We can't always expect a UTF-8 environment. When we don't get that luxury,
        // it's better to give a low-quality error message than none at all.
        String::from_utf8_lossy(CStr::from_ptr(p).to_bytes()).into()
    }
}

#[cfg(target_os = "teeos")]
pub fn error_string(_errno: i32) -> String {
    "error string unimplemented".to_string()
}
