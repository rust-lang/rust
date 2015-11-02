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
#![allow(non_snake_case)]

pub mod deps;

#[macro_use] pub mod compat;
pub mod handle;
pub mod time;
pub mod stack_overflow;
pub mod stdio;
pub mod c;
pub mod error;
pub mod thread;
pub mod backtrace;
pub mod path;
pub mod rand;
pub mod dynamic_lib;
pub mod net;
pub mod rt;
pub mod fs;
pub mod process;
pub mod env;
pub mod thread_local;

mod pipe;
mod mutex;
mod rwlock;
mod condvar;

pub mod sync {
    pub use super::mutex::{Mutex, ReentrantMutex};
    pub use super::rwlock::RwLock;
    pub use super::condvar::Condvar;
}

pub use sys::common::unwind;
pub use sys::common::os_str::wtf8 as os_str;

/*use prelude::v1::*;

use ffi::{OsStr, OsString};
use io::{self, ErrorKind};
use libc;
use num::Zero;
use os::windows::ffi::{OsStrExt, OsStringExt};
use path::PathBuf;
use time::Duration;

pub use sys::common::os_str::wtf8 as os_str;
pub use sys::common::unwind;

#[macro_use] pub mod compat;

pub mod backtrace;
pub mod c;
pub mod condvar;
pub mod fs;
pub mod handle;
pub mod mutex;
pub mod net;
pub mod os;
pub mod pipe;
pub mod process;
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

pub fn init() {}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno as libc::c_int {
        libc::ERROR_ACCESS_DENIED => ErrorKind::PermissionDenied,
        libc::ERROR_ALREADY_EXISTS => ErrorKind::AlreadyExists,
        libc::ERROR_BROKEN_PIPE => ErrorKind::BrokenPipe,
        libc::ERROR_FILE_NOT_FOUND => ErrorKind::NotFound,
        c::ERROR_PATH_NOT_FOUND => ErrorKind::NotFound,
        libc::ERROR_NO_DATA => ErrorKind::BrokenPipe,
        libc::ERROR_OPERATION_ABORTED => ErrorKind::TimedOut,

        libc::WSAEACCES => ErrorKind::PermissionDenied,
        libc::WSAEADDRINUSE => ErrorKind::AddrInUse,
        libc::WSAEADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        libc::WSAECONNABORTED => ErrorKind::ConnectionAborted,
        libc::WSAECONNREFUSED => ErrorKind::ConnectionRefused,
        libc::WSAECONNRESET => ErrorKind::ConnectionReset,
        libc::WSAEINVAL => ErrorKind::InvalidInput,
        libc::WSAENOTCONN => ErrorKind::NotConnected,
        libc::WSAEWOULDBLOCK => ErrorKind::WouldBlock,
        libc::WSAETIMEDOUT => ErrorKind::TimedOut,

        _ => ErrorKind::Other,
    }
}

fn os2path(s: &[u16]) -> PathBuf {
    PathBuf::from(OsString::from_wide(s))
}

pub fn truncate_utf16_at_nul<'a>(v: &'a [u16]) -> &'a [u16] {
    match v.iter().position(|c| *c == 0) {
        // don't include the 0
        Some(i) => &v[..i],
        None => v
    }
}

fn dur2timeout(dur: Duration) -> libc::DWORD {
    // Note that a duration is a (u64, u32) (seconds, nanoseconds) pair, and the
    // timeouts in windows APIs are typically u32 milliseconds. To translate, we
    // have two pieces to take care of:
    //
    // * Nanosecond precision is rounded up
    // * Greater than u32::MAX milliseconds (50 days) is rounded up to INFINITE
    //   (never time out).
    dur.as_secs().checked_mul(1000).and_then(|ms| {
        ms.checked_add((dur.subsec_nanos() as u64) / 1_000_000)
    }).and_then(|ms| {
        ms.checked_add(if dur.subsec_nanos() % 1_000_000 > 0 {1} else {0})
    }).map(|ms| {
        if ms > <libc::DWORD>::max_value() as u64 {
            libc::INFINITE
        } else {
            ms as libc::DWORD
        }
    }).unwrap_or(libc::INFINITE)
}*/
