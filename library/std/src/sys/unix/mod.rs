#![allow(missing_docs, nonstandard_style)]

use crate::io::ErrorKind;

#[cfg(any(doc, target_os = "linux"))]
pub use crate::os::linux as platform;

#[cfg(all(not(doc), target_os = "android"))]
pub use crate::os::android as platform;
#[cfg(all(not(doc), target_os = "dragonfly"))]
pub use crate::os::dragonfly as platform;
#[cfg(all(not(doc), target_os = "emscripten"))]
pub use crate::os::emscripten as platform;
#[cfg(all(not(doc), target_os = "freebsd"))]
pub use crate::os::freebsd as platform;
#[cfg(all(not(doc), target_os = "fuchsia"))]
pub use crate::os::fuchsia as platform;
#[cfg(all(not(doc), target_os = "haiku"))]
pub use crate::os::haiku as platform;
#[cfg(all(not(doc), target_os = "illumos"))]
pub use crate::os::illumos as platform;
#[cfg(all(not(doc), target_os = "ios"))]
pub use crate::os::ios as platform;
#[cfg(all(not(doc), target_os = "l4re"))]
pub use crate::os::linux as platform;
#[cfg(all(not(doc), target_os = "macos"))]
pub use crate::os::macos as platform;
#[cfg(all(not(doc), target_os = "netbsd"))]
pub use crate::os::netbsd as platform;
#[cfg(all(not(doc), target_os = "openbsd"))]
pub use crate::os::openbsd as platform;
#[cfg(all(not(doc), target_os = "redox"))]
pub use crate::os::redox as platform;
#[cfg(all(not(doc), target_os = "solaris"))]
pub use crate::os::solaris as platform;

pub use self::rand::hashmap_random_keys;
pub use libc::strlen;

#[macro_use]
pub mod weak;

pub mod alloc;
pub mod android;
pub mod args;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fd;
pub mod fs;
pub mod futex;
pub mod io;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod kernel_copy;
#[cfg(target_os = "l4re")]
mod l4re;
pub mod memchr;
pub mod mutex;
#[cfg(not(target_os = "l4re"))]
pub mod net;
#[cfg(target_os = "l4re")]
pub use self::l4re::net;
pub mod os;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod time;

pub use crate::sys_common::os_str_bytes as os_str;

#[cfg(not(test))]
pub fn init() {
    // The standard streams might be closed on application startup. To prevent
    // std::io::{stdin, stdout,stderr} objects from using other unrelated file
    // resources opened later, we reopen standards streams when they are closed.
    unsafe {
        sanitize_standard_fds();
    }

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

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // The standard fds are always available in Miri.
            unsafe fn sanitize_standard_fds() {}
        } else if #[cfg(not(any(
            target_os = "emscripten",
            target_os = "fuchsia",
            // The poll on Darwin doesn't set POLLNVAL for closed fds.
            target_os = "macos",
            target_os = "ios",
            target_os = "redox",
        )))] {
            // In the case when all file descriptors are open, the poll has been
            // observed to perform better than fcntl (on GNU/Linux).
            unsafe fn sanitize_standard_fds() {
                use crate::sys::os::errno;
                let pfds: &mut [_] = &mut [
                    libc::pollfd { fd: 0, events: 0, revents: 0 },
                    libc::pollfd { fd: 1, events: 0, revents: 0 },
                    libc::pollfd { fd: 2, events: 0, revents: 0 },
                ];
                while libc::poll(pfds.as_mut_ptr(), 3, 0) == -1 {
                    if errno() == libc::EINTR {
                        continue;
                    }
                    libc::abort();
                }
                for pfd in pfds {
                    if pfd.revents & libc::POLLNVAL == 0 {
                        continue;
                    }
                    if libc::open("/dev/null\0".as_ptr().cast(), libc::O_RDWR, 0) == -1 {
                        // If the stream is closed but we failed to reopen it, abort the
                        // process. Otherwise we wouldn't preserve the safety of
                        // operations on the corresponding Rust object Stdin, Stdout, or
                        // Stderr.
                        libc::abort();
                    }
                }
            }
        } else if #[cfg(any(target_os = "macos", target_os = "ios", target_os = "redox"))] {
            unsafe fn sanitize_standard_fds() {
                use crate::sys::os::errno;
                for fd in 0..3 {
                    if libc::fcntl(fd, libc::F_GETFD) == -1 && errno() == libc::EBADF {
                        if libc::open("/dev/null\0".as_ptr().cast(), libc::O_RDWR, 0) == -1 {
                            libc::abort();
                        }
                    }
                }
            }
        } else {
            unsafe fn sanitize_standard_fds() {}
        }
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

pub fn cvt_nz(error: libc::c_int) -> crate::io::Result<()> {
    if error == 0 { Ok(()) } else { Err(crate::io::Error::from_raw_os_error(error)) }
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
