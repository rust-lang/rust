// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! C definitions used by libnative that don't belong in liblibc

pub use self::select::fd_set;

use libc;

#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
pub static FIONBIO: libc::c_ulong = 0x8004667e;
#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
pub static FIONBIO: libc::c_ulong = 0x5421;
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
pub static FIOCLEX: libc::c_ulong = 0x20006601;
#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
pub static FIOCLEX: libc::c_ulong = 0x5451;

extern {
    pub fn gettimeofday(timeval: *mut libc::timeval,
                        tzp: *libc::c_void) -> libc::c_int;
    pub fn select(nfds: libc::c_int,
                  readfds: *fd_set,
                  writefds: *fd_set,
                  errorfds: *fd_set,
                  timeout: *libc::timeval) -> libc::c_int;
    pub fn getsockopt(sockfd: libc::c_int,
                      level: libc::c_int,
                      optname: libc::c_int,
                      optval: *mut libc::c_void,
                      optlen: *mut libc::socklen_t) -> libc::c_int;
    pub fn ioctl(fd: libc::c_int, req: libc::c_ulong, ...) -> libc::c_int;

}

#[cfg(target_os = "macos")]
mod select {
    pub static FD_SETSIZE: uint = 1024;

    pub struct fd_set {
        fds_bits: [i32, ..(FD_SETSIZE / 32)]
    }

    pub fn fd_set(set: &mut fd_set, fd: i32) {
        set.fds_bits[(fd / 32) as uint] |= 1 << (fd % 32);
    }
}

#[cfg(target_os = "android")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "linux")]
mod select {
    use std::uint;

    pub static FD_SETSIZE: uint = 1024;

    pub struct fd_set {
        fds_bits: [uint, ..(FD_SETSIZE / uint::BITS)]
    }

    pub fn fd_set(set: &mut fd_set, fd: i32) {
        let fd = fd as uint;
        set.fds_bits[fd / uint::BITS] |= 1 << (fd % uint::BITS);
    }
}
