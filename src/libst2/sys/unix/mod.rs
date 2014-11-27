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
pub unsafe fn close_sock(sock: sock_t) { unimplemented!() }

pub fn last_error() -> IoError { unimplemented!() }

pub fn last_net_error() -> IoError { unimplemented!() }

extern "system" {
    fn gai_strerror(errcode: libc::c_int) -> *const libc::c_char;
}

pub fn last_gai_error(s: libc::c_int) -> IoError { unimplemented!() }

/// Convert an `errno` value into a high-level error variant and description.
pub fn decode_error(errno: i32) -> IoError { unimplemented!() }

pub fn decode_error_detailed(errno: i32) -> IoError { unimplemented!() }

#[inline]
pub fn retry<T: SignedInt> (f: || -> T) -> T { unimplemented!() }

pub fn ms_to_timeval(ms: u64) -> libc::timeval { unimplemented!() }

pub fn wouldblock() -> bool { unimplemented!() }

pub fn set_nonblocking(fd: sock_t, nb: bool) -> IoResult<()> { unimplemented!() }

// nothing needed on unix platforms
pub fn init_net() { unimplemented!() }
