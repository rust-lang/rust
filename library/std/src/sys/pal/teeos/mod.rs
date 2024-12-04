//! System bindings for the Teeos platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for Teeos.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unused_variables)]
#![allow(dead_code)]

#[path = "../unsupported/args.rs"]
pub mod args;
#[path = "../unsupported/env.rs"]
pub mod env;
//pub mod fd;
#[path = "../unsupported/fs.rs"]
pub mod fs;
#[path = "../unsupported/io.rs"]
pub mod io;
pub mod net;
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
pub mod thread;
#[allow(non_upper_case_globals)]
#[path = "../unix/time.rs"]
pub mod time;

#[path = "../unix/sync"]
pub mod sync {
    mod condvar;
    mod mutex;
    pub use condvar::Condvar;
    pub use mutex::Mutex;
}

use crate::io::ErrorKind;

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

// Trusted Applications are loaded as dynamic libraries on Teeos,
// so this should never be called.
pub fn init(argc: isize, argv: *const *const u8, sigpipe: u8) {}

// SAFETY: must be called only once during runtime cleanup.
// this is not guaranteed to run, for example when the program aborts.
pub unsafe fn cleanup() {
    unimplemented!()
    // We do NOT have stack overflow handler, because TEE OS will kill TA when it happens.
    // So cleanup is commented
    // stack_overflow::cleanup();
}

#[inline]
pub(crate) fn is_interrupted(errno: i32) -> bool {
    errno == libc::EINTR
}

// Note: code below is 1:1 copied from unix/mod.rs
pub fn decode_error_kind(errno: i32) -> ErrorKind {
    use ErrorKind::*;
    match errno as libc::c_int {
        libc::E2BIG => ArgumentListTooLong,
        libc::EADDRINUSE => AddrInUse,
        libc::EADDRNOTAVAIL => AddrNotAvailable,
        libc::EBUSY => ResourceBusy,
        libc::ECONNABORTED => ConnectionAborted,
        libc::ECONNREFUSED => ConnectionRefused,
        libc::ECONNRESET => ConnectionReset,
        libc::EDEADLK => Deadlock,
        libc::EDQUOT => FilesystemQuotaExceeded,
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
        libc::ENOTEMPTY => DirectoryNotEmpty,
        libc::EPIPE => BrokenPipe,
        libc::EROFS => ReadOnlyFilesystem,
        libc::ESPIPE => NotSeekable,
        libc::ESTALE => StaleNetworkFileHandle,
        libc::ETIMEDOUT => TimedOut,
        libc::ETXTBSY => ExecutableFileBusy,
        libc::EXDEV => CrossesDevices,

        libc::EACCES | libc::EPERM => PermissionDenied,

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK => WouldBlock,

        _ => Uncategorized,
    }
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

pub fn cvt<T: IsMinusOne>(t: T) -> crate::io::Result<T> {
    if t.is_minus_one() { Err(crate::io::Error::last_os_error()) } else { Ok(t) }
}

pub fn cvt_r<T, F>(mut f: F) -> crate::io::Result<T>
where
    T: IsMinusOne,
    F: FnMut() -> T,
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

pub fn cvt_nz(error: libc::c_int) -> crate::io::Result<()> {
    if error == 0 { Ok(()) } else { Err(crate::io::Error::from_raw_os_error(error)) }
}

use crate::io as std_io;
pub fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> std_io::Error {
    std_io::Error::new(std_io::ErrorKind::Unsupported, "operation not supported on this platform")
}
