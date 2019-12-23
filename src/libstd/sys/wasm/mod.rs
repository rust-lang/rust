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

use crate::os::raw::c_char;

pub mod alloc;
pub mod args;
pub mod cmath;
pub mod env;
pub mod fast_thread_local;
pub mod fs;
pub mod io;
pub mod memchr;
pub mod net;
pub mod os;
pub mod path;
pub mod pipe;
pub mod process;
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
pub mod thread_local;
pub mod time;

pub use crate::sys_common::os_str_bytes as os_str;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "atomics")] {
        #[path = "condvar_atomics.rs"]
        pub mod condvar;
        #[path = "mutex_atomics.rs"]
        pub mod mutex;
        #[path = "rwlock_atomics.rs"]
        pub mod rwlock;
    } else {
        pub mod condvar;
        pub mod mutex;
        pub mod rwlock;
    }
}

#[cfg(not(test))]
pub fn init() {}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::new(crate::io::ErrorKind::Other, "operation not supported on wasm yet")
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Other
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

pub unsafe fn abort_internal() -> ! {
    crate::arch::wasm32::unreachable()
}

// We don't have randomness yet, but I totally used a random number generator to
// generate these numbers.
//
// More seriously though this is just for DOS protection in hash maps. It's ok
// if we don't do that on wasm just yet.
pub fn hashmap_random_keys() -> (u64, u64) {
    (1, 2)
}
