// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use os;

use sys::fs::FileDesc;

pub type signal = libc::c_int;

pub fn new() -> (signal, signal) {
    let os::Pipe { reader, writer } = unsafe { os::pipe().unwrap() };
    (reader, writer)
}

pub fn signal(fd: libc::c_int) {
    FileDesc::new(fd, false).write(&[0]).ok().unwrap();
}

pub fn close(fd: libc::c_int) {
    let _fd = FileDesc::new(fd, true);
}
