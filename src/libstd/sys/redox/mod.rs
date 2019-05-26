#![allow(dead_code, missing_docs, nonstandard_style)]

use crate::io::ErrorKind;

pub use libc::strlen;
pub use self::rand::hashmap_random_keys;

#[path = "../unix/alloc.rs"]
pub mod alloc;
pub mod args;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fast_thread_local;
pub mod fd;
pub mod fs;
pub mod io;
pub mod memchr;
pub mod mutex;
pub mod net;
pub mod os;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod stack_overflow;
pub mod stdio;
pub mod syscall;
pub mod thread;
pub mod thread_local;
pub mod time;

pub use crate::sys_common::os_str_bytes as os_str;

#[cfg(not(test))]
pub fn init() {}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno {
        syscall::ECONNREFUSED => ErrorKind::ConnectionRefused,
        syscall::ECONNRESET => ErrorKind::ConnectionReset,
        syscall::EPERM | syscall::EACCES => ErrorKind::PermissionDenied,
        syscall::EPIPE => ErrorKind::BrokenPipe,
        syscall::ENOTCONN => ErrorKind::NotConnected,
        syscall::ECONNABORTED => ErrorKind::ConnectionAborted,
        syscall::EADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        syscall::EADDRINUSE => ErrorKind::AddrInUse,
        syscall::ENOENT => ErrorKind::NotFound,
        syscall::EINTR => ErrorKind::Interrupted,
        syscall::EINVAL => ErrorKind::InvalidInput,
        syscall::ETIMEDOUT => ErrorKind::TimedOut,
        syscall::EEXIST => ErrorKind::AlreadyExists,

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == syscall::EAGAIN || x == syscall::EWOULDBLOCK =>
            ErrorKind::WouldBlock,

        _ => ErrorKind::Other,
    }
}

pub fn cvt(result: Result<usize, syscall::Error>) -> crate::io::Result<usize> {
    result.map_err(|err| crate::io::Error::from_raw_os_error(err.errno))
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

pub fn cvt_libc<T: IsMinusOne>(t: T) -> crate::io::Result<T> {
    if t.is_minus_one() {
        Err(crate::io::Error::last_os_error())
    } else {
        Ok(t)
    }
}

/// On Redox, use an illegal instruction to abort
pub unsafe fn abort_internal() -> ! {
    core::intrinsics::abort();
}
