// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

use io;
use os::raw::c_char;

#[path = "../wasm/alloc.rs"]
pub mod alloc;
pub mod args;
#[cfg(feature = "backtrace")]
#[path = "../wasm/backtrace.rs"]
pub mod backtrace;
#[path = "../wasm/cmath.rs"]
pub mod cmath;
pub mod env;
pub mod ext;
pub mod fd;
pub mod fs;
#[path = "../wasm/memchr.rs"]
pub mod memchr;
pub mod net;
pub mod os;
pub mod os_str;
pub mod path;
pub mod pipe;
pub mod process;
pub mod rand;
#[path = "../wasm/stack_overflow.rs"]
pub mod stack_overflow;
pub mod stdio;
pub mod thread;
pub mod time;

cfg_if! {
    if #[cfg(target_feature = "atomics")] {
        #[path = "../wasm/condvar_atomics.rs"]
        pub mod condvar;
        #[path = "../wasm/mutex_atomics.rs"]
        pub mod mutex;
        #[path = "../wasm/rwlock_atomics.rs"]
        pub mod rwlock;
        #[path = "../wasm/thread_local_atomics.rs"]
        pub mod thread_local;
    } else {
        #[path = "../wasm/condvar.rs"]
        pub mod condvar;
        #[path = "../wasm/mutex.rs"]
        pub mod mutex;
        #[path = "../wasm/rwlock.rs"]
        pub mod rwlock;
        #[path = "../wasm/thread_local.rs"]
        pub mod thread_local;
    }
}

#[path = "../cloudabi/abi/mod.rs"]
mod cloudabi;
mod err;

#[cfg(not(test))]
pub fn init() {
}

pub fn unsupported<T>() -> io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> io::Error {
    io::Error::new(io::ErrorKind::Other,
                   "operation not supported on wasm yet")
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
    cloudabi::proc_exit(1)
}

pub use self::rand::hashmap_random_keys;
