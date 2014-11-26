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
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_unsafe)]
#![allow(unused_mut)]

extern crate libc;

use num;
use mem;
use prelude::*;
use io::{mod, IoResult, IoError};
use sync::{Once, ONCE_INIT};

macro_rules! helper_init( (static $name:ident: Helper<$m:ty>) => (
    static $name: Helper<$m> = Helper {
        lock: ::rustrt::mutex::NATIVE_MUTEX_INIT,
        chan: ::cell::UnsafeCell { value: 0 as *mut Sender<$m> },
        signal: ::cell::UnsafeCell { value: 0 },
        initialized: ::cell::UnsafeCell { value: false },
    };
) )

pub mod c;
pub mod ext;
pub mod fs;
pub mod helper_signal;
pub mod os;
pub mod pipe;
pub mod process;
pub mod tcp;
pub mod thread_local;
pub mod timer;
pub mod tty;
pub mod udp;

pub mod addrinfo {
    pub use sys_common::net::get_host_addresses;
}

// FIXME: move these to c module
pub type sock_t = libc::SOCKET;
pub type wrlen = libc::c_int;
pub type msglen_t = libc::c_int;
pub unsafe fn close_sock(sock: sock_t) { let _ = libc::closesocket(sock); }

// windows has zero values as errors
fn mkerr_winbool(ret: libc::c_int) -> IoResult<()> {
    if ret == 0 {
        Err(last_error())
    } else {
        Ok(())
    }
}

pub fn last_error() -> IoError {
    let errno = os::errno() as i32;
    let mut err = decode_error(errno);
    err.detail = Some(os::error_string(errno));
    err
}

pub fn last_net_error() -> IoError {
    let errno = unsafe { c::WSAGetLastError() as i32 };
    let mut err = decode_error(errno);
    err.detail = Some(os::error_string(errno));
    err
}

pub fn last_gai_error(_errno: i32) -> IoError {
    last_net_error()
}

/// Convert an `errno` value into a high-level error variant and description.
pub fn decode_error(errno: i32) -> IoError {
    let (kind, desc) = match errno {
        libc::EOF => (io::EndOfFile, "end of file"),
        libc::ERROR_NO_DATA => (io::BrokenPipe, "the pipe is being closed"),
        libc::ERROR_FILE_NOT_FOUND => (io::FileNotFound, "file not found"),
        libc::ERROR_INVALID_NAME => (io::InvalidInput, "invalid file name"),
        libc::WSAECONNREFUSED => (io::ConnectionRefused, "connection refused"),
        libc::WSAECONNRESET => (io::ConnectionReset, "connection reset"),
        libc::ERROR_ACCESS_DENIED | libc::WSAEACCES =>
            (io::PermissionDenied, "permission denied"),
        libc::WSAEWOULDBLOCK => {
            (io::ResourceUnavailable, "resource temporarily unavailable")
        }
        libc::WSAENOTCONN => (io::NotConnected, "not connected"),
        libc::WSAECONNABORTED => (io::ConnectionAborted, "connection aborted"),
        libc::WSAEADDRNOTAVAIL => (io::ConnectionRefused, "address not available"),
        libc::WSAEADDRINUSE => (io::ConnectionRefused, "address in use"),
        libc::ERROR_BROKEN_PIPE => (io::EndOfFile, "the pipe has ended"),
        libc::ERROR_OPERATION_ABORTED =>
            (io::TimedOut, "operation timed out"),
        libc::WSAEINVAL => (io::InvalidInput, "invalid argument"),
        libc::ERROR_CALL_NOT_IMPLEMENTED =>
            (io::IoUnavailable, "function not implemented"),
        libc::ERROR_INVALID_HANDLE =>
            (io::MismatchedFileTypeForOperation,
             "invalid handle provided to function"),
        libc::ERROR_NOTHING_TO_TERMINATE =>
            (io::InvalidInput, "no process to kill"),

        // libuv maps this error code to EISDIR. we do too. if it is found
        // to be incorrect, we can add in some more machinery to only
        // return this message when ERROR_INVALID_FUNCTION after certain
        // Windows calls.
        libc::ERROR_INVALID_FUNCTION => (io::InvalidInput,
                                         "illegal operation on a directory"),

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
pub fn retry<I> (f: || -> I) -> I { f() } // PR rust-lang/rust/#17020

pub fn ms_to_timeval(ms: u64) -> libc::timeval {
    libc::timeval {
        tv_sec: (ms / 1000) as libc::c_long,
        tv_usec: ((ms % 1000) * 1000) as libc::c_long,
    }
}

pub fn wouldblock() -> bool {
    let err = os::errno();
    err == libc::WSAEWOULDBLOCK as uint
}

pub fn set_nonblocking(fd: sock_t, nb: bool) -> IoResult<()> {
    let mut set = nb as libc::c_ulong;
    if unsafe { c::ioctlsocket(fd, c::FIONBIO, &mut set) != 0 } {
        Err(last_error())
    } else {
        Ok(())
    }
}

pub fn init_net() {
    unsafe {
        static START: Once = ONCE_INIT;

        START.doit(|| {
            let mut data: c::WSADATA = mem::zeroed();
            let ret = c::WSAStartup(0x202, // version 2.2
                                    &mut data);
            assert_eq!(ret, 0);
        });
    }
}

pub fn unimpl() -> IoError {
    IoError {
        kind: io::IoUnavailable,
        desc: "operation is not implemented",
        detail: None,
    }
}

pub fn to_utf16(s: Option<&str>) -> IoResult<Vec<u16>> {
    match s {
        Some(s) => Ok({
            let mut s = s.utf16_units().collect::<Vec<u16>>();
            s.push(0);
            s
        }),
        None => Err(IoError {
            kind: io::InvalidInput,
            desc: "valid unicode input required",
            detail: None
        })
    }
}
