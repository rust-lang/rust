// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Deprecated in favor of `thread`.

#![deprecated = "use std::thread instead"]

use any::Any;
use boxed::Box;
use thread;
use kinds::Send;
use result::Result;

/// Deprecate: use `std::thread::Cfg` instead.
#[deprecated = "use std::thread::Cfg instead"]
pub type TaskBuilder = thread::Cfg;

/// Deprecated: use `std::thread::Thread::spawn` instead.
#[deprecated = "use std::thread::Thread::spawn instead"]
pub fn spawn(f: proc(): Send) {
    thread::Thread::spawn(f);
}

/// Deprecated: use `std::thread::Thread::with_join instead`.
#[deprecated = "use std::thread::Thread::with_join instead"]
pub fn try<T: Send>(f: proc(): Send -> T) -> Result<T, Box<Any + Send>> {
    thread::Thread::with_join(f).join()
}

/// Deprecated: use `std::thread::Thread::yield_now instead`.
#[deprecated = "use std::thread::Thread::yield_now instead"]
pub fn deschedule() {
    thread::Thread::yield_now()
}
