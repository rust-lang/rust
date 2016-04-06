// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs, bad_style)]

use io::{self, ErrorKind};
use libc;
use num::One;
use ops::Neg;

#[cfg(target_os = "android")]   pub use os::android as platform;
#[cfg(target_os = "bitrig")]    pub use os::bitrig as platform;
#[cfg(target_os = "dragonfly")] pub use os::dragonfly as platform;
#[cfg(target_os = "freebsd")]   pub use os::freebsd as platform;
#[cfg(target_os = "ios")]       pub use os::ios as platform;
#[cfg(target_os = "linux")]     pub use os::linux as platform;
#[cfg(target_os = "macos")]     pub use os::macos as platform;
#[cfg(target_os = "nacl")]      pub use os::nacl as platform;
#[cfg(target_os = "netbsd")]    pub use os::netbsd as platform;
#[cfg(target_os = "openbsd")]   pub use os::openbsd as platform;
#[cfg(target_os = "solaris")]   pub use os::solaris as platform;
#[cfg(target_os = "emscripten")] pub use os::emscripten as platform;

#[macro_use]
pub mod weak;

pub mod backtrace;
pub mod condvar;
pub mod ext;
pub mod fd;
pub mod fs;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

#[cfg(not(test))]
pub fn init() {
    use alloc::oom;

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

    oom::set_oom_handler(oom_handler);

    // A nicer handler for out-of-memory situations than the default one. This
    // one prints a message to stderr before aborting. It is critical that this
    // code does not allocate any memory since we are in an OOM situation. Any
    // errors are ignored while printing since there's nothing we can do about
    // them and we are about to exit anyways.
    fn oom_handler() -> ! {
        use intrinsics;
        let msg = "fatal runtime error: out of memory\n";
        unsafe {
            libc::write(libc::STDERR_FILENO,
                        msg.as_ptr() as *const libc::c_void,
                        msg.len() as libc::size_t);
            intrinsics::abort();
        }
    }

    #[cfg(not(target_os = "nacl"))]
    unsafe fn reset_sigpipe() {
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != !0);
    }
    #[cfg(target_os = "nacl")]
    unsafe fn reset_sigpipe() {}
}

// Currently the minimum supported Android version of the standard library is
// API level 18 (android-18). Back in those days [1] the `signal` function was
// just an inline wrapper around `bsd_signal`, but starting in API level
// android-20 the `signal` symbols was introduced [2]. Finally, in android-21
// the API `bsd_signal` was removed [3].
//
// Basically this means that if we want to be binary compatible with multiple
// Android releases (oldest being 18 and newest being 21) then we need to check
// for both symbols and not actually link against either.
//
// Note that if we're not on android we just link against the `android` symbol
// itself.
//
// [1]: https://chromium.googlesource.com/android_tools/+/20ee6d20/ndk/platforms
//                                       /android-18/arch-arm/usr/include/signal.h
// [2]: https://chromium.googlesource.com/android_tools/+/fbd420/ndk_experimental
//                                       /platforms/android-20/arch-arm
//                                       /usr/include/signal.h
// [3]: https://chromium.googlesource.com/android_tools/+/20ee6d/ndk/platforms
//                                       /android-21/arch-arm/usr/include/signal.h
#[cfg(target_os = "android")]
unsafe fn signal(signum: libc::c_int,
                 handler: libc::sighandler_t) -> libc::sighandler_t {
    weak!(fn signal(libc::c_int, libc::sighandler_t) -> libc::sighandler_t);
    weak!(fn bsd_signal(libc::c_int, libc::sighandler_t) -> libc::sighandler_t);

    let f = signal.get().or_else(|| bsd_signal.get());
    let f = f.expect("neither `signal` nor `bsd_signal` symbols found");
    f(signum, handler)
}

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

pub fn cvt<T: One + PartialEq + Neg<Output=T>>(t: T) -> io::Result<T> {
    let one: T = T::one();
    if t == -one {
        Err(io::Error::last_os_error())
    } else {
        Ok(t)
    }
}

pub fn cvt_r<T, F>(mut f: F) -> io::Result<T>
    where T: One + PartialEq + Neg<Output=T>, F: FnMut() -> T
{
    loop {
        match cvt(f()) {
            Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}
