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

use prelude::v1::*;

use ffi::OsStr;
use io::{self, ErrorKind};
use libc;
use mem;
use old_io::{self, IoResult, IoError};
use num::Int;
use os::windows::OsStrExt;
use sync::{Once, ONCE_INIT};

macro_rules! helper_init { (static $name:ident: Helper<$m:ty>) => (
    static $name: Helper<$m> = Helper {
        lock: ::sync::MUTEX_INIT,
        cond: ::sync::CONDVAR_INIT,
        chan: ::cell::UnsafeCell { value: 0 as *mut ::sync::mpsc::Sender<$m> },
        signal: ::cell::UnsafeCell { value: 0 },
        initialized: ::cell::UnsafeCell { value: false },
        shutdown: ::cell::UnsafeCell { value: false },
    };
) }

pub mod backtrace;
pub mod c;
pub mod condvar;
pub mod ext;
pub mod fs;
pub mod fs2;
pub mod handle;
pub mod helper_signal;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
pub mod pipe;
pub mod pipe2;
pub mod process;
pub mod process2;
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
        libc::EOF => (old_io::EndOfFile, "end of file"),
        libc::ERROR_NO_DATA => (old_io::BrokenPipe, "the pipe is being closed"),
        libc::ERROR_FILE_NOT_FOUND => (old_io::FileNotFound, "file not found"),
        libc::ERROR_INVALID_NAME => (old_io::InvalidInput, "invalid file name"),
        libc::WSAECONNREFUSED => (old_io::ConnectionRefused, "connection refused"),
        libc::WSAECONNRESET => (old_io::ConnectionReset, "connection reset"),
        libc::ERROR_ACCESS_DENIED | libc::WSAEACCES =>
            (old_io::PermissionDenied, "permission denied"),
        libc::WSAEWOULDBLOCK => {
            (old_io::ResourceUnavailable, "resource temporarily unavailable")
        }
        libc::WSAENOTCONN => (old_io::NotConnected, "not connected"),
        libc::WSAECONNABORTED => (old_io::ConnectionAborted, "connection aborted"),
        libc::WSAEADDRNOTAVAIL => (old_io::ConnectionRefused, "address not available"),
        libc::WSAEADDRINUSE => (old_io::ConnectionRefused, "address in use"),
        libc::ERROR_BROKEN_PIPE => (old_io::EndOfFile, "the pipe has ended"),
        libc::ERROR_OPERATION_ABORTED =>
            (old_io::TimedOut, "operation timed out"),
        libc::WSAEINVAL => (old_io::InvalidInput, "invalid argument"),
        libc::ERROR_CALL_NOT_IMPLEMENTED =>
            (old_io::IoUnavailable, "function not implemented"),
        libc::ERROR_INVALID_HANDLE =>
            (old_io::MismatchedFileTypeForOperation,
             "invalid handle provided to function"),
        libc::ERROR_NOTHING_TO_TERMINATE =>
            (old_io::InvalidInput, "no process to kill"),
        libc::ERROR_ALREADY_EXISTS =>
            (old_io::PathAlreadyExists, "path already exists"),

        // libuv maps this error code to EISDIR. we do too. if it is found
        // to be incorrect, we can add in some more machinery to only
        // return this message when ERROR_INVALID_FUNCTION after certain
        // Windows calls.
        libc::ERROR_INVALID_FUNCTION => (old_io::InvalidInput,
                                         "illegal operation on a directory"),

        _ => (old_io::OtherIoError, "unknown error")
    };
    IoError { kind: kind, desc: desc, detail: None }
}

pub fn decode_error_detailed(errno: i32) -> IoError {
    let mut err = decode_error(errno);
    err.detail = Some(os::error_string(errno));
    err
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno as libc::c_int {
        libc::ERROR_ACCESS_DENIED => ErrorKind::PermissionDenied,
        libc::ERROR_ALREADY_EXISTS => ErrorKind::PathAlreadyExists,
        libc::ERROR_BROKEN_PIPE => ErrorKind::BrokenPipe,
        libc::ERROR_FILE_NOT_FOUND => ErrorKind::FileNotFound,
        libc::ERROR_INVALID_FUNCTION => ErrorKind::InvalidInput,
        libc::ERROR_INVALID_HANDLE => ErrorKind::MismatchedFileTypeForOperation,
        libc::ERROR_INVALID_NAME => ErrorKind::InvalidInput,
        libc::ERROR_NOTHING_TO_TERMINATE => ErrorKind::InvalidInput,
        libc::ERROR_NO_DATA => ErrorKind::BrokenPipe,
        libc::ERROR_OPERATION_ABORTED => ErrorKind::TimedOut,

        libc::WSAEACCES => ErrorKind::PermissionDenied,
        libc::WSAEADDRINUSE => ErrorKind::ConnectionRefused,
        libc::WSAEADDRNOTAVAIL => ErrorKind::ConnectionRefused,
        libc::WSAECONNABORTED => ErrorKind::ConnectionAborted,
        libc::WSAECONNREFUSED => ErrorKind::ConnectionRefused,
        libc::WSAECONNRESET => ErrorKind::ConnectionReset,
        libc::WSAEINVAL => ErrorKind::InvalidInput,
        libc::WSAENOTCONN => ErrorKind::NotConnected,
        libc::WSAEWOULDBLOCK => ErrorKind::ResourceUnavailable,

        _ => ErrorKind::Other,
    }
}


#[inline]
pub fn retry<I, F>(f: F) -> I where F: FnOnce() -> I { f() } // PR rust-lang/rust/#17020

pub fn ms_to_timeval(ms: u64) -> libc::timeval {
    libc::timeval {
        tv_sec: (ms / 1000) as libc::c_long,
        tv_usec: ((ms % 1000) * 1000) as libc::c_long,
    }
}

pub fn wouldblock() -> bool {
    let err = os::errno();
    err == libc::WSAEWOULDBLOCK as i32
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

        START.call_once(|| {
            let mut data: c::WSADATA = mem::zeroed();
            let ret = c::WSAStartup(0x202, // version 2.2
                                    &mut data);
            assert_eq!(ret, 0);
        });
    }
}

pub fn unimpl() -> IoError {
    IoError {
        kind: old_io::IoUnavailable,
        desc: "operation is not implemented",
        detail: None,
    }
}

fn to_utf16(s: Option<&str>) -> IoResult<Vec<u16>> {
    match s {
        Some(s) => Ok(to_utf16_os(OsStr::from_str(s))),
        None => Err(IoError {
            kind: old_io::InvalidInput,
            desc: "valid unicode input required",
            detail: None,
        }),
    }
}

fn to_utf16_os(s: &OsStr) -> Vec<u16> {
    let mut v: Vec<_> = s.encode_wide().collect();
    v.push(0);
    v
}

// Many Windows APIs follow a pattern of where we hand the a buffer and then
// they will report back to us how large the buffer should be or how many bytes
// currently reside in the buffer. This function is an abstraction over these
// functions by making them easier to call.
//
// The first callback, `f1`, is yielded a (pointer, len) pair which can be
// passed to a syscall. The `ptr` is valid for `len` items (u16 in this case).
// The closure is expected to return what the syscall returns which will be
// interpreted by this function to determine if the syscall needs to be invoked
// again (with more buffer space).
//
// Once the syscall has completed (errors bail out early) the second closure is
// yielded the data which has been read from the syscall. The return value
// from this closure is then the return value of the function.
fn fill_utf16_buf_base<F1, F2, T>(mut f1: F1, f2: F2) -> Result<T, ()>
    where F1: FnMut(*mut u16, libc::DWORD) -> libc::DWORD,
          F2: FnOnce(&[u16]) -> T
{
    // Start off with a stack buf but then spill over to the heap if we end up
    // needing more space.
    let mut stack_buf = [0u16; 512];
    let mut heap_buf = Vec::new();
    unsafe {
        let mut n = stack_buf.len();
        loop {
            let buf = if n <= stack_buf.len() {
                &mut stack_buf[..]
            } else {
                let extra = n - heap_buf.len();
                heap_buf.reserve(extra);
                heap_buf.set_len(n);
                &mut heap_buf[..]
            };

            // This function is typically called on windows API functions which
            // will return the correct length of the string, but these functions
            // also return the `0` on error. In some cases, however, the
            // returned "correct length" may actually be 0!
            //
            // To handle this case we call `SetLastError` to reset it to 0 and
            // then check it again if we get the "0 error value". If the "last
            // error" is still 0 then we interpret it as a 0 length buffer and
            // not an actual error.
            c::SetLastError(0);
            let k = match f1(buf.as_mut_ptr(), n as libc::DWORD) {
                0 if libc::GetLastError() == 0 => 0,
                0 => return Err(()),
                n => n,
            } as usize;
            if k == n && libc::GetLastError() ==
                            libc::ERROR_INSUFFICIENT_BUFFER as libc::DWORD {
                n *= 2;
            } else if k >= n {
                n = k;
            } else {
                return Ok(f2(&buf[..k]))
            }
        }
    }
}

fn fill_utf16_buf<F1, F2, T>(f1: F1, f2: F2) -> IoResult<T>
    where F1: FnMut(*mut u16, libc::DWORD) -> libc::DWORD,
          F2: FnOnce(&[u16]) -> T
{
    fill_utf16_buf_base(f1, f2).map_err(|()| IoError::last_error())
}

fn fill_utf16_buf_new<F1, F2, T>(f1: F1, f2: F2) -> io::Result<T>
    where F1: FnMut(*mut u16, libc::DWORD) -> libc::DWORD,
          F2: FnOnce(&[u16]) -> T
{
    fill_utf16_buf_base(f1, f2).map_err(|()| io::Error::last_os_error())
}

fn os2path(s: &[u16]) -> Path {
    // FIXME: this should not be a panicking conversion (aka path reform)
    Path::new(String::from_utf16(s).unwrap())
}

pub fn truncate_utf16_at_nul<'a>(v: &'a [u16]) -> &'a [u16] {
    match v.iter().position(|c| *c == 0) {
        // don't include the 0
        Some(i) => &v[..i],
        None => v
    }
}

fn cvt<I: Int>(i: I) -> io::Result<I> {
    if i == Int::zero() {
        Err(io::Error::last_os_error())
    } else {
        Ok(i)
    }
}

fn ms_to_filetime(ms: u64) -> libc::FILETIME {
    // A FILETIME is a count of 100 nanosecond intervals, so we multiply by
    // 10000 b/c there are 10000 intervals in 1 ms
    let ms = ms * 10000;
    libc::FILETIME {
        dwLowDateTime: ms as u32,
        dwHighDateTime: (ms >> 32) as u32,
    }
}
