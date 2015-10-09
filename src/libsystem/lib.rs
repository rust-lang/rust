// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(no_std)]
#![feature(nonzero)]
#![feature(collections)]
#![feature(alloc)]
#![cfg_attr(any(unix, windows), feature(libc))]
#![feature(rand)]
#![feature(unicode)]
#![feature(char_internals)]
#![feature(str_internals)]
#![feature(decode_utf16)]
#![cfg_attr(any(unix, windows), feature(zero_one))]
#![feature(linkage)]
#![feature(slice_bytes)]
#![feature(char_from_unchecked)]
#![feature(vec_push_all)]
#![feature(const_fn)]
#![feature(lang_items)]
#![feature(unwind_attributes)]
#![cfg_attr(any(unix, windows), feature(fnbox))]
#![feature(rustc_private)]
#![feature(slice_patterns)]
#![feature(staged_api)]
#![cfg_attr(any(unix, windows), feature(core_intrinsics))]
#![feature(associated_consts)]

#![no_std]
#![staged_api]
#![unstable(feature = "system", issue = "0")]

#[macro_use]
extern crate collections as collections;

#[macro_use]
extern crate rustc_bitflags;
mod std { pub use core::{option, ops}; }

extern crate rand as core_rand;
extern crate rustc_unicode as unicode;
extern crate alloc;
#[cfg(any(unix, windows))]
extern crate libc;

#[macro_use]
mod common;

#[cfg(unix)]
pub mod unix;

pub mod thread_local;
pub mod unwind;
pub mod error;
pub mod time;
pub mod sync;
pub mod thread;
pub mod backtrace;
pub mod env;
pub mod stdio;
pub mod path;
pub mod stack_overflow;
pub mod rand;
pub mod fs;
pub mod dynamic_lib;
pub mod net;
pub mod os_str;
pub mod process;
pub mod rt;
pub mod c;

pub mod inner;
pub mod io;
mod wtf8;
mod c_str;
mod deps;

pub mod os;
#[cfg(not(any(unix, windows)))]
pub mod bind;

#[cfg(not(any(unix, windows)))]
#[path = ""] pub mod imp {
    pub use bind::{
        thread_local,
        error,
        backtrace,
        unwind,
        env,
        sync,
        stdio,
        net,
        path,
        time,
        thread,
        stack_overflow,
        rand,
        dynamic_lib,
        fs,
        process,
        rt,
        c
    };

    pub use os_str::u8 as os_str;
}

#[cfg(unix)]
#[path = ""] pub mod imp {
    pub use unix::ext;

    pub use common::{
        thread_local,
        c,
        unwind,
        net,
    };

    pub use unix::{
        error,
        backtrace,
        env,
        stdio,
        path,
        time,
        thread,
        stack_overflow,
        rand,
        dynamic_lib,
        fs,
        process,
        rt
    };

    pub mod sync {
        pub use unix::sync::Sync;
        pub use unix::mutex::{Mutex, ReentrantMutex};
        pub use unix::condvar::Condvar;
        pub use unix::rwlock::RwLock;
        pub use common::once::Once;
    }

    pub use os_str::u8 as os_str;
}

/*#[cfg(unix)]
#[path = ""] pub mod imp {
    #[path = "unix/ext/mod.rs"] pub mod ext;

    pub use super::unix;
    pub use super::unix as sys;

    pub mod common {
        pub mod thread_local;
        pub mod libunwind;
        pub mod unwind;
        pub mod args;
        pub mod gnu;
        pub mod stdio;
        pub mod net;
    }

    pub mod thread_local {
        pub use super::unix::thread_local as imp;

        pub use super::common::thread_local::{Key, OsKey, StaticOsKey};
    }

    pub mod error {
        pub use super::unix::os::{errno, error_string};
        pub use super::unix::decode_error_kind;
    }

    pub mod backtrace {
        #[cfg(not(feature = "disable-backtrace"))]
        pub use super::unix::backtrace::write;

        #[cfg(feature = "disable-backtrace")]
        pub use super::none::backtrace::write;
    }

    pub mod unwind {
        pub use super::common::unwind::{begin_unwind_fmt, begin_unwind, panicking, try};
    }

    pub mod env {
        pub use super::unix::os::{Env, split_paths, join_paths, SplitPaths, JoinPathsError};
    }

    pub mod sync {
        pub use super::unix::mutex::{Mutex, ReentrantMutex};
        pub use super::unix::rwlock::RwLock;
        pub use super::unix::condvar::Condvar;
    }

    pub mod stdio {
        pub use super::unix::stdio::{Stdin, Stdout, Stderr};
        pub use super::common::stdio::handle_ebadf;
    }

    pub mod net {
        pub use super::common::net::Net;
    }

    pub use self::unix::{os_str, path, time, thread, stack_overflow, rand, dynamic_lib, fs, process};

    pub mod none;
}

#[cfg(windows)]
#[path = ""] pub mod imp {
    #[path = "windows/mod.rs"] pub mod imp;

    pub mod thread_local {
        pub use sys::common::thread_local::{Key, OsKey, StaticOsKey};
    }
}

#[cfg(not(any(unix, windows)))]
#[path = ""] pub mod imp {
    pub mod unix {
        pub mod os_str;
        pub mod path;
    }

    pub mod none;
    pub use self::none::{thread_local, unwind, error, time, sync, thread, backtrace, env, stdio, stack_overflow, rand, fs, dynamic_lib, process, net};

    pub mod os_str {
        pub use super::unix::os_str::{Buf, Slice};
    }

    pub mod path {
        pub use super::unix::path::{is_sep_byte, is_verbatim_sep, MAIN_SEP, MAIN_SEP_STR, parse_prefix};
    }
}*/
