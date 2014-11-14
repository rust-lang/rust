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
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_unsafe)]
#![allow(unused_mut)]

extern crate libc;

use num;
use num::{Int, SignedInt};
use prelude::*;
use io::{mod, IoResult, IoError};
use sys_common::mkerr_libc;

macro_rules! helper_init( (static $name:ident: Helper<$m:ty>) => (
    static $name: Helper<$m> = Helper {
        lock: ::rustrt::mutex::NATIVE_MUTEX_INIT,
        chan: ::cell::UnsafeCell { value: 0 as *mut Sender<$m> },
        signal: ::cell::UnsafeCell { value: 0 },
        initialized: ::cell::UnsafeCell { value: false },
    };
) )

pub mod c;
pub mod fs;
pub mod helper_signal;
pub mod os;
pub mod pipe;
pub mod process;
pub mod tcp;
pub mod timer;
pub mod thread_local;
pub mod tty;
pub mod udp;

pub mod addrinfo {
    pub use sys_common::net::get_host_addresses;
}

// FIXME: move these to c module
pub type sock_t = self::fs::fd_t;
pub type wrlen = libc::size_t;
pub type msglen_t = libc::size_t;
pub unsafe fn close_sock(sock: sock_t) { let _ = libc::close(sock); }

pub fn last_error() -> IoError {
    decode_error_detailed(os::errno() as i32)
}

pub fn last_net_error() -> IoError {
    last_error()
}

extern "system" {
    fn gai_strerror(errcode: libc::c_int) -> *const libc::c_char;
}

pub fn last_gai_error(s: libc::c_int) -> IoError {
    use c_str::CString;

    let mut err = decode_error(s);
    err.detail = Some(unsafe {
        CString::new(gai_strerror(s), false).as_str().unwrap().to_string()
    });
    err
}

/// Convert an `errno` value into a high-level error variant and description.
pub fn decode_error(errno: i32) -> IoError {
    // FIXME: this should probably be a bit more descriptive...
    let (kind, desc) = match errno {
        libc::EOF => (io::EndOfFile, "end of file"),
        libc::ECONNREFUSED => (io::ConnectionRefused, "connection refused"),
        libc::ECONNRESET => (io::ConnectionReset, "connection reset"),
        libc::EPERM | libc::EACCES =>
            (io::PermissionDenied, "permission denied"),
        libc::EPIPE => (io::BrokenPipe, "broken pipe"),
        libc::ENOTCONN => (io::NotConnected, "not connected"),
        libc::ECONNABORTED => (io::ConnectionAborted, "connection aborted"),
        libc::EADDRNOTAVAIL => (io::ConnectionRefused, "address not available"),
        libc::EADDRINUSE => (io::ConnectionRefused, "address in use"),
        libc::ENOENT => (io::FileNotFound, "no such file or directory"),
        libc::EISDIR => (io::InvalidInput, "illegal operation on a directory"),
        libc::ENOSYS => (io::IoUnavailable, "function not implemented"),
        libc::EINVAL => (io::InvalidInput, "invalid argument"),
        libc::ENOTTY =>
            (io::MismatchedFileTypeForOperation,
             "file descriptor is not a TTY"),
        libc::ETIMEDOUT => (io::TimedOut, "operation timed out"),
        libc::ECANCELED => (io::TimedOut, "operation aborted"),

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
            (io::ResourceUnavailable, "resource temporarily unavailable"),

        _ => (io::OtherIoError, "unknown error")
    };
    IoError { kind: kind, desc: desc, detail: None }
}

pub fn decode_error_detailed(errno: i32) -> IoError {
    let mut err = decode_error(errno);
    err.detail = Some(os::error_string(errno));
    err
}

#[inline]
pub fn retry<T: SignedInt> (f: || -> T) -> T {
    let one: T = Int::one();
    loop {
        let n = f();
        if n == -one && os::errno() == libc::EINTR as int { }
        else { return n }
    }
}

pub fn ms_to_timeval(ms: u64) -> libc::timeval {
    libc::timeval {
        tv_sec: (ms / 1000) as libc::time_t,
        tv_usec: ((ms % 1000) * 1000) as libc::suseconds_t,
    }
}

pub fn wouldblock() -> bool {
    let err = os::errno();
    err == libc::EWOULDBLOCK as int || err == libc::EAGAIN as int
}

pub fn set_nonblocking(fd: sock_t, nb: bool) -> IoResult<()> {
    let set = nb as libc::c_int;
    mkerr_libc(retry(|| unsafe { c::ioctl(fd, c::FIONBIO, &set) }))
}

// nothing needed on unix platforms
pub fn init_net() {}
