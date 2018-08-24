#![allow(dead_code, missing_docs, bad_style)]

use io::{self, ErrorKind};

pub use libc::strlen;
pub use self::rand::hashmap_random_keys;

pub mod args;
#[cfg(feature = "backtrace")]
pub mod backtrace;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fast_thread_local;
pub mod fd;
pub mod fs;
pub mod memchr;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
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

pub fn cvt(result: Result<usize, syscall::Error>) -> io::Result<usize> {
    result.map_err(|err| io::Error::from_raw_os_error(err.errno))
}

/// On Redox, use an illegal instruction to abort
pub unsafe fn abort_internal() -> ! {
    ::core::intrinsics::abort();
}
