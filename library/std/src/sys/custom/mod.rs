//! System bindings for custom platforms

#![unstable(issue = "none", feature = "std_internals")]

use crate::custom_os_impl;
use crate::io as std_io;

#[path = "../unix/cmath.rs"]
pub mod cmath;
#[path = "../unix/os_str.rs"]
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;

// unsupported provides a working implementation
#[path = "../unsupported/io.rs"]
pub mod io;

// unsupported provides an empty argument iterator
#[path = "../unsupported/args.rs"]
pub mod args;

// TODO: unsupported provides a thread-unsafe implementation
#[path = "../unsupported/once.rs"]
pub mod once;

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;
pub use common::{cleanup, init, memchr};

pub mod alloc;
pub mod env;
pub mod fs;
pub mod locks;
pub mod net;
pub mod os;
pub mod pipe;
pub mod process;
pub mod stdio;
pub mod thread;
pub mod thread_parking;
pub mod time;

// really bad implementation
pub mod thread_local_key;

pub fn decode_error_kind(errno: i32) -> std_io::ErrorKind {
    custom_os_impl!(os, decode_error_kind, errno)
}

#[inline]
pub(crate) fn is_interrupted(errno: i32) -> bool {
    custom_os_impl!(os, is_interrupted, errno)
}

pub fn abort_internal() -> ! {
    fn infinite_loop() -> ! {
        loop {}
    }

    let rwlock = &crate::os::custom::os::IMPL;
    let reader = match rwlock.read().ok() {
        Some(thing) => thing,
        None => infinite_loop(),
    };
    let some_impl = match reader.as_ref() {
        Some(thing) => thing,
        None => infinite_loop(),
    };

    let exit_status = process::ExitCode::FAILURE;
    some_impl.exit(exit_status.as_i32())
}

pub fn hashmap_random_keys() -> (u64, u64) {
    custom_os_impl!(os, hashmap_random_keys)
}
