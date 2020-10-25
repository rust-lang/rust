#![allow(dead_code)]
#![allow(missing_docs, nonstandard_style)]

use crate::io::ErrorKind;

pub use self::rand::hashmap_random_keys;
pub use crate::os::vxworks as platform;
pub use libc::strlen;

#[macro_use]
#[path = "../unix/weak.rs"]
pub mod weak;

#[path = "../unix/alloc.rs"]
pub mod alloc;
#[path = "../unix/args.rs"]
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
#[path = "../unix/condvar.rs"]
pub mod condvar;
pub mod env;
#[path = "../unix/ext/mod.rs"]
pub mod ext;
#[path = "../unix/fd.rs"]
pub mod fd;
#[path = "../unix/fs.rs"]
pub mod fs;
#[path = "../unix/io.rs"]
pub mod io;
#[path = "../unix/memchr.rs"]
pub mod memchr;
#[path = "../unix/mutex.rs"]
pub mod mutex;
#[path = "../unix/net.rs"]
pub mod net;
#[path = "../unix/os.rs"]
pub mod os;
#[path = "../unix/path.rs"]
pub mod path;
#[path = "../unix/pipe.rs"]
pub mod pipe;
pub mod process;
pub mod rand;
#[path = "../unix/rwlock.rs"]
pub mod rwlock;
#[path = "../unix/stack_overflow.rs"]
pub mod stack_overflow;
#[path = "../unix/stdio.rs"]
pub mod stdio;
#[path = "../unix/thread.rs"]
pub mod thread;
pub mod thread_local_dtor;
#[path = "../unix/thread_local_key.rs"]
pub mod thread_local_key;
#[path = "../unix/time.rs"]
pub mod time;

pub use crate::sys_common::os_str_bytes as os_str;

#[cfg(not(test))]
pub fn init() {
    // ignore SIGPIPE
    unsafe {
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != libc::SIG_ERR);
    }
}

pub use libc::signal;

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno as libc::c_int {
        libc::ECONNREFUSED => ErrorKind::ConnectionRefused,
        libc::ECONNRESET => ErrorKind::ConnectionReset,
        libc::EPERM | libc::EACCES => ErrorKind::PermissionDenied,
        libc::EPIPE => ErrorKind::BrokenPipe,
        libc::ENOTCONN => ErrorKind::NotConnected,
        libc::ECONNABORTED => ErrorKind::ConnectionAborted,
        libc::EADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        libc::EADDRINUSE => ErrorKind::AddrInUse,
        libc::ENOENT => ErrorKind::NotFound,
        libc::EINTR => ErrorKind::Interrupted,
        libc::EINVAL => ErrorKind::InvalidInput,
        libc::ETIMEDOUT => ErrorKind::TimedOut,
        libc::EEXIST => ErrorKind::AlreadyExists,

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK => ErrorKind::WouldBlock,

        _ => ErrorKind::Other,
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

// On Unix-like platforms, libc::abort will unregister signal handlers
// including the SIGABRT handler, preventing the abort from being blocked, and
// fclose streams, with the side effect of flushing them so libc buffered
// output will be printed.  Additionally the shell will generally print a more
// understandable error message like "Abort trap" rather than "Illegal
// instruction" that intrinsics::abort would cause, as intrinsics::abort is
// implemented as an illegal instruction.
pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}
