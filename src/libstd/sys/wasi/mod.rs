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

use crate::io;
use crate::mem;
use crate::os::raw::c_char;
use wasi::wasi_unstable as wasi;

pub mod alloc;
pub mod args;
#[path = "../wasm/cmath.rs"]
pub mod cmath;
#[path = "../wasm/condvar.rs"]
pub mod condvar;
pub mod env;
pub mod fd;
pub mod fs;
#[path = "../wasm/memchr.rs"]
pub mod memchr;
#[path = "../wasm/mutex.rs"]
pub mod mutex;
pub mod net;
pub mod io;
pub mod os;
pub use crate::sys_common::os_str_bytes as os_str;
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
#[path = "../wasm/fast_thread_local.rs"]
pub mod fast_thread_local;
pub mod time;
pub mod ext;

#[cfg(not(test))]
pub fn init() {
}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "operation not supported on wasm yet")
}

pub fn decode_error_kind(_code: i32) -> io::ErrorKind {
    io::ErrorKind::Other
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
    return n
}

pub unsafe fn abort_internal() -> ! {
    libc::abort()
}

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut ret = (0u64, 0u64);
    unsafe {
        let base = &mut ret as *mut (u64, u64) as *mut core::ffi::c_void;
        let len = mem::size_of_val(&ret);
        let ret = wasi::raw::__wasi_random_get(base, len);
        if ret != 0 {
            panic!("__wasi_random_get failure")
        }
    }
    return ret
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

pub fn cvt<T: IsMinusOne>(t: T) -> crate::io::Result<T> {
    if t.is_minus_one() {
        Err(io::Error::last_os_error())
    } else {
        Ok(t)
    }
}

impl From<wasi::Error> for io::Error {
    fn from(err: wasi::Error) -> Self {
        Self::from_raw_os_error(err as i32)
    }
}
