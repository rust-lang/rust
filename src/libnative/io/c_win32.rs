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

#![allow(type_overflow)]

use libc;

pub static WSADESCRIPTION_LEN: uint = 256;
pub static WSASYS_STATUS_LEN: uint = 128;
pub static FIONBIO: libc::c_long = 0x8004667e;
static FD_SETSIZE: uint = 64;

pub struct WSADATA {
    pub wVersion: libc::WORD,
    pub wHighVersion: libc::WORD,
    pub szDescription: [u8, ..WSADESCRIPTION_LEN + 1],
    pub szSystemStatus: [u8, ..WSASYS_STATUS_LEN + 1],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *u8,
}

pub type LPWSADATA = *mut WSADATA;

pub struct fd_set {
    fd_count: libc::c_uint,
    fd_array: [libc::SOCKET, ..FD_SETSIZE],
}

pub fn fd_set(set: &mut fd_set, s: libc::SOCKET) {
    set.fd_array[set.fd_count as uint] = s;
    set.fd_count += 1;
}

#[link(name = "ws2_32")]
extern "system" {
    pub fn WSAStartup(wVersionRequested: libc::WORD,
                      lpWSAData: LPWSADATA) -> libc::c_int;
    pub fn WSAGetLastError() -> libc::c_int;

    pub fn ioctlsocket(s: libc::SOCKET, cmd: libc::c_long,
                       argp: *mut libc::c_ulong) -> libc::c_int;
    pub fn select(nfds: libc::c_int,
                  readfds: *fd_set,
                  writefds: *fd_set,
                  exceptfds: *fd_set,
                  timeout: *libc::timeval) -> libc::c_int;
    pub fn getsockopt(sockfd: libc::SOCKET,
                      level: libc::c_int,
                      optname: libc::c_int,
                      optval: *mut libc::c_char,
                      optlen: *mut libc::c_int) -> libc::c_int;
}
