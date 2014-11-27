// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![experimental]

use alloc::boxed::Box;
use any::{Any, AnyRefExt};
use cell::RefCell;
use fmt;
use io::{Writer, IoResult};
use kinds::Send;
use option::{Some, None, Option};
use result::Ok;
use rt::backtrace;
use rustrt::{Stderr, Stdio};
use rustrt::local::Local;
use rustrt::task::Task;
use str::Str;
use string::String;

// Defined in this module instead of io::stdio so that the unwinding
thread_local!(pub static LOCAL_STDERR: RefCell<Option<Box<Writer + Send>>> = {
    RefCell::new(None)
})

impl Writer for Stdio {
    fn write(&mut self, bytes: &[u8]) -> IoResult<()> { unimplemented!() }
}

pub fn on_fail(obj: &Any + Send, file: &'static str, line: uint) { unimplemented!() }
