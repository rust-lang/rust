// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]
#![allow(non_camel_case_types)]

use io::{self, ErrorKind};
use libc;
use num::One;
use ops::Neg;
use alloc::oom;

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
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

// A nicer handler for out-of-memory situations than the default one. This one
// prints a message to stderr before aborting. It is critical that this code
// does not allocate any memory since we are in an OOM situation. Any errors are
// ignored while printing since there's nothing we can do about them and we are
// about to exit anyways.
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

#[cfg(not(any(target_os = "nacl", test)))]
pub fn init() {
    use libc::signal;
    // By default, some platforms will send a *signal* when an EPIPE error
    // would otherwise be delivered. This runtime doesn't install a SIGPIPE
    // handler, causing it to kill the program, which isn't exactly what we
    // want!
    //
    // Hence, we set SIGPIPE to ignore when the program starts up in order
    // to prevent this problem.
    unsafe {
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != !0);
    }

    oom::set_oom_handler(oom_handler);
}

#[cfg(all(target_os = "nacl", not(test)))]
pub fn init() {
    oom::set_oom_handler(oom_handler);
}

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
