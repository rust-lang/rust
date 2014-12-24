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
use ops::FnOnce;

/// Deprecate: use `std::thread::Builder` instead.
#[deprecated = "use std::thread::Builder instead"]
pub type TaskBuilder = thread::Builder;

/// Deprecated: use `std::thread::Thread::spawn` and `detach` instead.
#[deprecated = "use std::thread::Thread::spawn and detach instead"]
pub fn spawn<F>(f: F) where F: FnOnce(), F: Send {
    thread::Thread::spawn(f).detach();
}

/// Deprecated: use `std::thread::Thread::spawn` and `join` instead.
#[deprecated = "use std::thread::Thread::spawn and join instead"]
pub fn try<T, F>(f: F) -> Result<T, Box<Any + Send>> where
    T: Send, F: FnOnce() -> T, F: Send
{
    thread::Thread::spawn(f).join()
}

/// Deprecated: use `std::thread::Thread::yield_now instead`.
#[deprecated = "use std::thread::Thread::yield_now instead"]
pub fn deschedule() {
    thread::Thread::yield_now()
}
