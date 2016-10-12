// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs, bad_style)]

use io::ErrorKind;

pub use os::none as platform;

#[path = "../unix/args.rs"]           pub mod args;
#[path = "../unix/android.rs"]        pub mod android;
#[cfg(any(not(cargobuild), feature = "backtrace"))]
#[path = "../unix/backtrace/mod.rs"]  pub mod backtrace;
#[path = "../unix/ext/mod.rs"]        pub mod ext;
#[path = "../unix/memchr.rs"]         pub mod memchr;
#[path = "../unix/os.rs"]             pub mod os;
#[path = "../unix/os_str.rs"]         pub mod os_str;
#[path = "../unix/path.rs"]           pub mod path;
#[path = "../unix/pipe.rs"]           pub mod pipe;
#[path = "../unix/process.rs"]        pub mod process;
#[path = "../unix/rand.rs"]           pub mod rand;
#[path = "../unix/stack_overflow.rs"] pub mod stack_overflow;
#[path = "../unix/time.rs"]           pub mod time;
                                      pub mod env;
                                      pub mod fd;
                                      pub mod fs;
                                      pub mod net;
                                      pub mod stdio;
                                      pub mod condvar;
                                      pub mod mutex;
                                      pub mod rwlock;
#[path = "../unix/thread.rs"]         pub mod thread;
                                      pub mod thread_local;

#[cfg(not(test))]
pub fn init() {}

pub fn decode_error_kind(_errno: i32) -> ErrorKind {
    ErrorKind::Other
}
