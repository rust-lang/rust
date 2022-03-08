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

#[path = "../unix/alloc.rs"]
pub mod alloc;
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
#[path = "../unsupported/condvar.rs"]
pub mod condvar;
pub mod env;
pub mod fd;
pub mod fs;
pub mod io;
#[path = "../unsupported/mutex.rs"]
pub mod mutex;
pub mod net;
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
#[path = "../unsupported/rwlock.rs"]
pub mod rwlock;
pub mod stdio;
pub mod thread;
#[path = "../unsupported/thread_local_dtor.rs"]
pub mod thread_local_dtor;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
pub mod time;

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;
pub use common::*;

pub fn decode_error_kind(errno: i32) -> std_io::ErrorKind {
    use std_io::ErrorKind::*;
    if errno > u16::MAX as i32 || errno < 0 {
        return Uncategorized;
    }

    match errno {
        e if e == wasi::ERRNO_CONNREFUSED.raw().into() => ConnectionRefused,
        e if e == wasi::ERRNO_CONNRESET.raw().into() => ConnectionReset,
        e if e == wasi::ERRNO_PERM.raw().into() || e == wasi::ERRNO_ACCES.raw().into() => {
            PermissionDenied
        }
        e if e == wasi::ERRNO_PIPE.raw().into() => BrokenPipe,
        e if e == wasi::ERRNO_NOTCONN.raw().into() => NotConnected,
        e if e == wasi::ERRNO_CONNABORTED.raw().into() => ConnectionAborted,
        e if e == wasi::ERRNO_ADDRNOTAVAIL.raw().into() => AddrNotAvailable,
        e if e == wasi::ERRNO_ADDRINUSE.raw().into() => AddrInUse,
        e if e == wasi::ERRNO_NOENT.raw().into() => NotFound,
        e if e == wasi::ERRNO_INTR.raw().into() => Interrupted,
        e if e == wasi::ERRNO_INVAL.raw().into() => InvalidInput,
        e if e == wasi::ERRNO_TIMEDOUT.raw().into() => TimedOut,
        e if e == wasi::ERRNO_EXIST.raw().into() => AlreadyExists,
        e if e == wasi::ERRNO_AGAIN.raw().into() => WouldBlock,
        e if e == wasi::ERRNO_NOSYS.raw().into() => Unsupported,
        e if e == wasi::ERRNO_NOMEM.raw().into() => OutOfMemory,
        _ => Uncategorized,
    }
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

fn err2io(err: wasi::Errno) -> std_io::Error {
    std_io::Error::from_raw_os_error(err.raw().into())
}
