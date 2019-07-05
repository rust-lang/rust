#![allow(missing_docs, nonstandard_style)]

use crate::ptr;
use crate::ffi::{OsStr, OsString};
use crate::io::ErrorKind;
use crate::os::windows::ffi::{OsStrExt, OsStringExt};
use crate::path::PathBuf;
use crate::time::Duration;

pub use libc::strlen;
pub use self::rand::hashmap_random_keys;

#[macro_use] pub mod compat;

pub mod alloc;
pub mod args;
pub mod c;
pub mod cmath;
pub mod condvar;
pub mod env;
pub mod ext;
pub mod fast_thread_local;
pub mod fs;
pub mod handle;
pub mod io;
pub mod memchr;
pub mod mutex;
pub mod net;
pub mod os;
pub mod os_str;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
pub mod rwlock;
pub mod stack_overflow;
pub mod thread;
pub mod thread_local;
pub mod time;
pub mod stdio;

#[cfg(not(test))]
pub fn init() {
}

pub fn decode_error_kind(errno: i32) -> ErrorKind {
    match errno as c::DWORD {
        c::ERROR_ACCESS_DENIED => return ErrorKind::PermissionDenied,
        c::ERROR_ALREADY_EXISTS => return ErrorKind::AlreadyExists,
        c::ERROR_FILE_EXISTS => return ErrorKind::AlreadyExists,
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

pub fn to_u16s<S: AsRef<OsStr>>(s: S) -> crate::io::Result<Vec<u16>> {
    fn inner(s: &OsStr) -> crate::io::Result<Vec<u16>> {
        let mut maybe_result: Vec<u16> = s.encode_wide().collect();
        if maybe_result.iter().any(|&u| u == 0) {
            return Err(crate::io::Error::new(ErrorKind::InvalidInput,
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
fn fill_utf16_buf<F1, F2, T>(mut f1: F1, f2: F2) -> crate::io::Result<T>
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
                0 => return Err(crate::io::Error::last_os_error()),
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

#[allow(dead_code)] // Only used in backtrace::gnu::get_executable_filename()
fn wide_char_to_multi_byte(code_page: u32,
                           flags: u32,
                           s: &[u16],
                           no_default_char: bool)
                           -> crate::io::Result<Vec<i8>> {
    unsafe {
        let mut size = c::WideCharToMultiByte(code_page,
                                              flags,
                                              s.as_ptr(),
                                              s.len() as i32,
                                              ptr::null_mut(),
                                              0,
                                              ptr::null(),
                                              ptr::null_mut());
        if size == 0 {
            return Err(crate::io::Error::last_os_error());
        }

        let mut buf = Vec::with_capacity(size as usize);
        buf.set_len(size as usize);

        let mut used_default_char = c::FALSE;
        size = c::WideCharToMultiByte(code_page,
                                      flags,
                                      s.as_ptr(),
                                      s.len() as i32,
                                      buf.as_mut_ptr(),
                                      buf.len() as i32,
                                      ptr::null(),
                                      if no_default_char { &mut used_default_char }
                                      else { ptr::null_mut() });
        if size == 0 {
            return Err(crate::io::Error::last_os_error());
        }
        if no_default_char && used_default_char == c::TRUE {
            return Err(crate::io::Error::new(crate::io::ErrorKind::InvalidData,
                                      "string cannot be converted to requested code page"));
        }

        buf.set_len(size as usize);

        Ok(buf)
    }
}

pub fn truncate_utf16_at_nul(v: &[u16]) -> &[u16] {
    match v.iter().position(|c| *c == 0) {
        // don't include the 0
        Some(i) => &v[..i],
        None => v
    }
}

pub trait IsZero {
    fn is_zero(&self) -> bool;
}

macro_rules! impl_is_zero {
    ($($t:ident)*) => ($(impl IsZero for $t {
        fn is_zero(&self) -> bool {
            *self == 0
        }
    })*)
}

impl_is_zero! { i8 i16 i32 i64 isize u8 u16 u32 u64 usize }

pub fn cvt<I: IsZero>(i: I) -> crate::io::Result<I> {
    if i.is_zero() {
        Err(crate::io::Error::last_os_error())
    } else {
        Ok(i)
    }
}

pub fn dur2timeout(dur: Duration) -> c::DWORD {
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

// On Windows, use the processor-specific __fastfail mechanism.  In Windows 8
// and later, this will terminate the process immediately without running any
// in-process exception handlers.  In earlier versions of Windows, this
// sequence of instructions will be treated as an access violation,
// terminating the process but without necessarily bypassing all exception
// handlers.
//
// https://msdn.microsoft.com/en-us/library/dn774154.aspx
#[allow(unreachable_code)]
pub unsafe fn abort_internal() -> ! {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        asm!("int $$0x29" :: "{ecx}"(7) ::: volatile); // 7 is FAST_FAIL_FATAL_APP_EXIT
        crate::intrinsics::unreachable();
    }
    crate::intrinsics::abort();
}
