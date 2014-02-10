// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Dummy functions for raw sockets

    Ordinarily raw sockets require root access to the machine. By
    re-implementing the library functions used and replacing them
    with dummy functions we are able to test the functionality without
    root access.
*/
#[crate_type="staticlib"];

extern mod rustuv;

use std::libc::{c_int, c_void, size_t, ssize_t};
use std::libc::{sockaddr, sockaddr_storage, socklen_t};


static mut data: *c_void = 0 as *c_void;
static mut datalen: ssize_t = 0;

static mut fromaddr: sockaddr_storage = sockaddr_storage { ss_family: 0, __ss_align: 0, __ss_pad2: [0, ..112] };
static mut fromaddrlen: socklen_t = 0;

// Only one socket, always succeeds
#[no_mangle]
pub extern "C" fn socket(_domain: c_int, _ty: c_int, _protocol: c_int) -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn close(_fd: c_int) -> c_int {
    1
}
#[no_mangle]
pub extern "C" fn setsockopt(_socket: c_int, _level: c_int, _name: c_int,
                             _value: *c_void, _option_len: socklen_t) -> c_int {
    1
}

// FIXME This doesn't match the C definition, so may stop working if
//       the usage in the raw socket backend code changes
#[no_mangle]
pub extern "C" fn fcntl(_fd: c_int, _cmd: c_int, _opt: c_int) -> c_int {
    1
}

// Receive data from previous sendto()
#[no_mangle]
pub extern "C" fn recvfrom(_socket: c_int, buf: *mut c_void, len: size_t,
                           _flags: c_int, addr: *mut sockaddr,
                           addrlen: *mut socklen_t) -> ssize_t {
    unsafe {
        // This could/should be replaced with memcpy
        for i in range(0, datalen) {
            if i >= len as ssize_t {
                break;
            }
            *(((buf as size_t) + i as size_t) as *mut char) = *(((data as size_t) + i as size_t) as *char);
        }
        let stoAddr = addr as *mut sockaddr_storage;
        *stoAddr = fromaddr;
        *addrlen = fromaddrlen;

        datalen
    }
}

// Send data
#[no_mangle]
pub extern "C" fn sendto(_socket: c_int, buf: *c_void, len: size_t,
                         _flags: c_int, addr: *sockaddr,
                         addrlen: socklen_t) -> ssize_t {
    unsafe {
        datalen = len as ssize_t;
        data = buf;
        fromaddr = *(addr as *sockaddr_storage);
        fromaddrlen = addrlen;

        datalen
    }
}
