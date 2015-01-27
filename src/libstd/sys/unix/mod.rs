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

use prelude::v1::*;

use ffi;
use old_io::{self, IoResult, IoError};
use libc;
use num::{Int, SignedInt};
use num;
use str;
use sys_common::mkerr_libc;

macro_rules! helper_init { (static $name:ident: Helper<$m:ty>) => (
    static $name: Helper<$m> = Helper {
        lock: ::sync::MUTEX_INIT,
        cond: ::sync::CONDVAR_INIT,
        chan: ::cell::UnsafeCell { value: 0 as *mut Sender<$m> },
        signal: ::cell::UnsafeCell { value: 0 },
        initialized: ::cell::UnsafeCell { value: false },
        shutdown: ::cell::UnsafeCell { value: false },
    };
) }

pub mod backtrace;
pub mod c;
pub mod ext;
pub mod condvar;
pub mod fs;
pub mod helper_signal;
pub mod mutex;
pub mod os;
pub mod os_str;
pub mod pipe;
pub mod process;
pub mod rwlock;
pub mod stack_overflow;
pub mod sync;
pub mod tcp;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod timer;
pub mod tty;
pub mod udp;

pub mod addrinfo {
    pub use sys_common::net::get_host_addresses;
    pub use sys_common::net::get_address_name;
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

    let mut err = decode_error(s);
    err.detail = Some(unsafe {
        str::from_utf8(ffi::c_str_to_bytes(&gai_strerror(s))).unwrap().to_string()
    });
    err
}

/// Convert an `errno` value into a high-level error variant and description.
pub fn decode_error(errno: i32) -> IoError {
    // FIXME: this should probably be a bit more descriptive...
    let (kind, desc) = match errno {
        libc::EOF => (old_io::EndOfFile, "end of file"),
        libc::ECONNREFUSED => (old_io::ConnectionRefused, "connection refused"),
        libc::ECONNRESET => (old_io::ConnectionReset, "connection reset"),
        libc::EPERM | libc::EACCES =>
            (old_io::PermissionDenied, "permission denied"),
        libc::EPIPE => (old_io::BrokenPipe, "broken pipe"),
        libc::ENOTCONN => (old_io::NotConnected, "not connected"),
        libc::ECONNABORTED => (old_io::ConnectionAborted, "connection aborted"),
        libc::EADDRNOTAVAIL => (old_io::ConnectionRefused, "address not available"),
        libc::EADDRINUSE => (old_io::ConnectionRefused, "address in use"),
        libc::ENOENT => (old_io::FileNotFound, "no such file or directory"),
        libc::EISDIR => (old_io::InvalidInput, "illegal operation on a directory"),
        libc::ENOSYS => (old_io::IoUnavailable, "function not implemented"),
        libc::EINVAL => (old_io::InvalidInput, "invalid argument"),
        libc::ENOTTY =>
            (old_io::MismatchedFileTypeForOperation,
             "file descriptor is not a TTY"),
        libc::ETIMEDOUT => (old_io::TimedOut, "operation timed out"),
        libc::ECANCELED => (old_io::TimedOut, "operation aborted"),
        libc::consts::os::posix88::EEXIST =>
            (old_io::PathAlreadyExists, "path already exists"),

        // These two constants can have the same value on some systems,
        // but different values on others, so we can't use a match
        // clause
        x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
            (old_io::ResourceUnavailable, "resource temporarily unavailable"),

        _ => (old_io::OtherIoError, "unknown error")
    };
    IoError { kind: kind, desc: desc, detail: None }
}

pub fn decode_error_detailed(errno: i32) -> IoError {
    let mut err = decode_error(errno);
    err.detail = Some(os::error_string(errno));
    err
}

#[inline]
pub fn retry<T, F> (mut f: F) -> T where
    T: SignedInt,
    F: FnMut() -> T,
{
    let one: T = Int::one();
    loop {
        let n = f();
        if n == -one && os::errno() == libc::EINTR as i32 { }
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
    err == libc::EWOULDBLOCK as i32 || err == libc::EAGAIN as i32
}

pub fn set_nonblocking(fd: sock_t, nb: bool) -> IoResult<()> {
    let set = nb as libc::c_int;
    mkerr_libc(retry(|| unsafe { c::ioctl(fd, c::FIONBIO, &set) }))
}

// nothing needed on unix platforms
pub fn init_net() {}
