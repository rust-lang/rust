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

use ffi::{OsStr, OsString};
use io::{self, ErrorKind};
use num::Zero;
use os::windows::ffi::{OsStrExt, OsStringExt};
use path::PathBuf;
use time::Duration;
use alloc::oom;

#[macro_use] pub mod compat;

pub mod backtrace;
pub mod c;
pub mod condvar;
pub mod ext;
pub mod fs;
pub mod handle;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
pub mod pipe;
pub mod process;
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

// See comment in sys/unix/mod.rs
fn oom_handler() -> ! {
    use intrinsics;
    use ptr;
    let msg = "fatal runtime error: out of memory\n";
    unsafe {
        // WriteFile silently fails if it is passed an invalid handle, so there
        // is no need to check the result of GetStdHandle.
        c::WriteFile(c::GetStdHandle(c::STD_ERROR_HANDLE),
                     msg.as_ptr() as c::LPVOID,
                     msg.len() as DWORD,
                     ptr::null_mut(),
                     ptr::null_mut());
        intrinsics::abort();
    }
}

pub fn init() {
    oom::set_oom_handler(oom_handler);
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno as c::DWORD {
        c::ERROR_ACCESS_DENIED => return ErrorKind::PermissionDenied,
        c::ERROR_ALREADY_EXISTS => return ErrorKind::AlreadyExists,
        c::ERROR_BROKEN_PIPE => return ErrorKind::BrokenPipe,
        c::ERROR_FILE_NOT_FOUND => return ErrorKind::NotFound,
        c::ERROR_PATH_NOT_FOUND => return ErrorKind::NotFound,
        c::ERROR_NO_DATA => return ErrorKind::BrokenPipe,
        c::ERROR_OPERATION_ABORTED => return ErrorKind::TimedOut,
        _ => {}
    }

    match errno {
        c::WSAEACCES => ErrorKind::PermissionDenied,
        c::WSAEADDRINUSE => ErrorKind::AddrInUse,
        c::WSAEADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
        c::WSAECONNABORTED => ErrorKind::ConnectionAborted,
        c::WSAECONNREFUSED => ErrorKind::ConnectionRefused,
        c::WSAECONNRESET => ErrorKind::ConnectionReset,
        c::WSAEINVAL => ErrorKind::InvalidInput,
        c::WSAENOTCONN => ErrorKind::NotConnected,
        c::WSAEWOULDBLOCK => ErrorKind::WouldBlock,
        c::WSAETIMEDOUT => ErrorKind::TimedOut,

        _ => ErrorKind::Other,
    }
}

pub fn to_u16s<S: AsRef<OsStr>>(s: S) -> io::Result<Vec<u16>> {
    fn inner(s: &OsStr) -> io::Result<Vec<u16>> {
        let mut maybe_result: Vec<u16> = s.encode_wide().collect();
        if maybe_result.iter().any(|&u| u == 0) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "strings passed to WinAPI cannot contain NULs"));
        }
        maybe_result.push(0);
        Ok(maybe_result)
    }
    inner(s.as_ref())
}

// Many Windows APIs follow a pattern of where we hand a buffer and then they
// will report back to us how large the buffer should be or how many bytes
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
fn fill_utf16_buf<F1, F2, T>(mut f1: F1, f2: F2) -> io::Result<T>
    where F1: FnMut(*mut u16, c::DWORD) -> c::DWORD,
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
            let k = match f1(buf.as_mut_ptr(), n as c::DWORD) {
                0 if c::GetLastError() == 0 => 0,
                0 => return Err(io::Error::last_os_error()),
                n => n,
            } as usize;
            if k == n && c::GetLastError() == c::ERROR_INSUFFICIENT_BUFFER {
                n *= 2;
            } else if k >= n {
                n = k;
            } else {
                return Ok(f2(&buf[..k]))
            }
        }
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

fn cvt<I: PartialEq + Zero>(i: I) -> io::Result<I> {
    if i == I::zero() {
        Err(io::Error::last_os_error())
    } else {
        Ok(i)
    }
}

fn dur2timeout(dur: Duration) -> c::DWORD {
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
        if ms > <c::DWORD>::max_value() as u64 {
            c::INFINITE
        } else {
            ms as c::DWORD
        }
    }).unwrap_or(c::INFINITE)
}
