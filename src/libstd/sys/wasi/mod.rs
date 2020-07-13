//! System bindings for the wasm/web platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for wasm. Note that this wasm is *not* the emscripten
//! wasm, so we have no runtime here.
//!
//! This is all super highly experimental and not actually intended for
//! wide/production use yet, it's still all in the experimental category. This
//! will likely change over time.
//!
//! Currently all functions here are basically stubs that immediately return
//! errors. The hope is that with a portability lint we can turn actually just
//! remove all this and just omit parts of the standard library if we're
//! compiling for wasm. That way it's a compile time error for something that's
//! guaranteed to be a runtime error!

use crate::io as std_io;
use crate::mem;
use crate::os::raw::c_char;

pub mod alloc;
pub mod args;
#[path = "../wasm/cmath.rs"]
pub mod cmath;
#[path = "../wasm/condvar.rs"]
pub mod condvar;
pub mod env;
pub mod fd;
pub mod fs;
pub mod io;
#[path = "../wasm/memchr.rs"]
pub mod memchr;
#[path = "../wasm/mutex.rs"]
pub mod mutex;
pub mod net;
pub mod os;
pub use crate::sys_common::os_str_bytes as os_str;
pub mod ext;
#[path = "../wasm/fast_thread_local.rs"]
pub mod fast_thread_local;
pub mod path;
pub mod pipe;
pub mod process;
#[path = "../wasm/rwlock.rs"]
pub mod rwlock;
#[path = "../wasm/stack_overflow.rs"]
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
#[path = "../wasm/thread_local.rs"]
pub mod thread_local;
pub mod time;

#[cfg(not(test))]
pub fn init() {}

pub fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> std_io::Error {
    std_io::Error::new(std_io::ErrorKind::Other, "operation not supported on wasm yet")
}

pub fn decode_error_kind(errno: i32) -> std_io::ErrorKind {
    use std_io::ErrorKind::*;
    if errno > u16::MAX as i32 || errno < 0 {
        return Other;
    }
    match errno as u16 {
        wasi::ERRNO_CONNREFUSED => ConnectionRefused,
        wasi::ERRNO_CONNRESET => ConnectionReset,
        wasi::ERRNO_PERM | wasi::ERRNO_ACCES => PermissionDenied,
        wasi::ERRNO_PIPE => BrokenPipe,
        wasi::ERRNO_NOTCONN => NotConnected,
        wasi::ERRNO_CONNABORTED => ConnectionAborted,
        wasi::ERRNO_ADDRNOTAVAIL => AddrNotAvailable,
        wasi::ERRNO_ADDRINUSE => AddrInUse,
        wasi::ERRNO_NOENT => NotFound,
        wasi::ERRNO_INTR => Interrupted,
        wasi::ERRNO_INVAL => InvalidInput,
        wasi::ERRNO_TIMEDOUT => TimedOut,
        wasi::ERRNO_EXIST => AlreadyExists,
        wasi::ERRNO_AGAIN => WouldBlock,
        _ => Other,
    }
}

// This enum is used as the storage for a bunch of types which can't actually
// exist.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Void {}

pub unsafe fn strlen(mut s: *const c_char) -> usize {
    let mut n = 0;
    while *s != 0 {
        n += 1;
        s = s.offset(1);
    }
    return n;
}

pub fn abort_internal() -> ! {
    unsafe { libc::abort() }
}

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut ret = (0u64, 0u64);
    unsafe {
        let base = &mut ret as *mut (u64, u64) as *mut u8;
        let len = mem::size_of_val(&ret);
        wasi::random_get(base, len).expect("random_get failure");
    }
    return ret;
}

fn err2io(err: wasi::Error) -> std_io::Error {
    std_io::Error::from_raw_os_error(err.raw_error().into())
}
