//! System bindings for the wasm/web platform on WASIX

use crate::io as std_io;
use crate::mem;

#[path = "../unix/alloc.rs"]
pub mod alloc;
#[path = "../wasi/args.rs"]
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
#[path = "../wasi/env.rs"]
pub mod env;
pub mod fd;
#[path = "../wasi/fs.rs"]
pub mod fs;
#[path = "../wasi/io.rs"]
pub mod io;
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
#[path = "../wasi/stdio.rs"]
pub mod stdio;
#[path = "../unsupported/thread_local_dtor.rs"]
pub mod thread_local_dtor;
#[path = "../unsupported/thread_local_key.rs"]
pub mod thread_local_key;
#[path = "../wasi/time.rs"]
pub mod time;

#[path = "../unix/locks"]
pub mod locks {
    #![allow(unsafe_op_in_unsafe_fn)]
    mod futex;
    mod futex_rwlock;
    pub use futex::{Mutex, MovableMutex, Condvar, MovableCondvar};
    pub use futex_rwlock::{RwLock, MovableRwLock};
}
#[path = "atomics/futex.rs"]
pub mod futex;
#[path = "atomics/thread.rs"]
pub mod thread;

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
        wasi::random_get(base, len as usize).expect("random_get failure");
    }
    return ret;
}

pub use super::common::cvt::{
    IsMinusOne,
    cvt,
    cvt_r
};

pub(crate) fn err2io(err: wasi::Errno) -> std_io::Error {
    std_io::Error::from_raw_os_error(err.raw().into())
}
