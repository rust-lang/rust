// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code, missing_docs, bad_style)]

use io::{self, ErrorKind};

pub mod args;
#[cfg(any(not(cargobuild), feature = "backtrace"))]
pub mod backtrace;
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
pub fn init() {
    use alloc::oom;

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
            let _ = syscall::write(2, msg.as_bytes());
            intrinsics::abort();
        }
    }
}

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
