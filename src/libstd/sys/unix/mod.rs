#![allow(missing_docs, nonstandard_style)]

use crate::io::ErrorKind;

#[cfg(any(rustdoc, target_os = "linux"))] pub use crate::os::linux as platform;

#[cfg(all(not(rustdoc), target_os = "android"))]   pub use crate::os::android as platform;
#[cfg(all(not(rustdoc), target_os = "dragonfly"))] pub use crate::os::dragonfly as platform;
#[cfg(all(not(rustdoc), target_os = "freebsd"))]   pub use crate::os::freebsd as platform;
#[cfg(all(not(rustdoc), target_os = "haiku"))]     pub use crate::os::haiku as platform;
#[cfg(all(not(rustdoc), target_os = "ios"))]       pub use crate::os::ios as platform;
#[cfg(all(not(rustdoc), target_os = "macos"))]     pub use crate::os::macos as platform;
#[cfg(all(not(rustdoc), target_os = "netbsd"))]    pub use crate::os::netbsd as platform;
#[cfg(all(not(rustdoc), target_os = "openbsd"))]   pub use crate::os::openbsd as platform;
#[cfg(all(not(rustdoc), target_os = "solaris"))]   pub use crate::os::solaris as platform;
#[cfg(all(not(rustdoc), target_os = "emscripten"))] pub use crate::os::emscripten as platform;
#[cfg(all(not(rustdoc), target_os = "fuchsia"))]   pub use crate::os::fuchsia as platform;
#[cfg(all(not(rustdoc), target_os = "l4re"))]      pub use crate::os::linux as platform;
#[cfg(all(not(rustdoc), target_os = "hermit"))]    pub use crate::os::hermit as platform;

pub use self::rand::hashmap_random_keys;
pub use libc::strlen;

#[macro_use]
pub mod weak;

pub mod alloc;
pub mod args;
pub mod android;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fast_thread_local;
pub mod fd;
pub mod fs;
pub mod memchr;
pub mod io;
pub mod mutex;
#[cfg(not(target_os = "l4re"))]
pub mod net;
#[cfg(target_os = "l4re")]
mod l4re;
#[cfg(target_os = "l4re")]
pub use self::l4re::net;
pub mod os;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

pub use crate::sys_common::os_str_bytes as os_str;

#[cfg(not(test))]
pub fn init() {
    // By default, some platforms will send a *signal* when an EPIPE error
    // would otherwise be delivered. This runtime doesn't install a SIGPIPE
    // handler, causing it to kill the program, which isn't exactly what we
    // want!
    //
    // Hence, we set SIGPIPE to ignore when the program starts up in order
    // to prevent this problem.
    unsafe {
        reset_sigpipe();
    }

    #[cfg(not(any(target_os = "emscripten", target_os = "fuchsia")))]
    unsafe fn reset_sigpipe() {
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != libc::SIG_ERR);
    }
    #[cfg(any(target_os = "emscripten", target_os = "fuchsia"))]
    unsafe fn reset_sigpipe() {}
}

#[cfg(target_os = "android")]
pub use crate::sys::android::signal;
#[cfg(not(target_os = "android"))]
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
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
            ErrorKind::WouldBlock,

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
    if t.is_minus_one() {
        Err(crate::io::Error::last_os_error())
    } else {
        Ok(t)
    }
}

pub fn cvt_r<T, F>(mut f: F) -> crate::io::Result<T>
    where T: IsMinusOne,
          F: FnMut() -> T
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
// fclose streams, with the side effect of flushing them so libc bufferred
// output will be printed.  Additionally the shell will generally print a more
// understandable error message like "Abort trap" rather than "Illegal
// instruction" that intrinsics::abort would cause, as intrinsics::abort is
// implemented as an illegal instruction.
pub unsafe fn abort_internal() -> ! {
    libc::abort()
}
