// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(deprecated)]

use dynamic_lib::DynamicLibrary;
use io::prelude::*;
use io;
use sys::c;
use libc::c_void;

use sys_common::gnu::libbacktrace;

pub fn print(w: &mut Write, i: isize, addr: u64, _: &DynamicLibrary, _: c::HANDLE)
        -> io::Result<()> {
    let addr = addr as usize as *mut c_void;
    libbacktrace::print(w, i, addr, addr)
}
