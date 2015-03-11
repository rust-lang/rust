// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use io;
use libc;
use sys::fd::FileDesc;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

impl Stdin {
    pub fn new() -> Stdin { Stdin(()) }

    pub fn read(&self, data: &mut [u8]) -> io::Result<usize> {
        let fd = FileDesc::new(libc::STDIN_FILENO);
        let ret = fd.read(data);
        fd.into_raw();
        return ret;
    }
}

impl Stdout {
    pub fn new() -> Stdout { Stdout(()) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(libc::STDOUT_FILENO);
        let ret = fd.write(data);
        fd.into_raw();
        return ret;
    }
}

impl Stderr {
    pub fn new() -> Stderr { Stderr(()) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(libc::STDERR_FILENO);
        let ret = fd.write(data);
        fd.into_raw();
        return ret;
    }
}

// FIXME: right now this raw stderr handle is used in a few places because
//        std::io::stderr_raw isn't exposed, but once that's exposed this impl
//        should go away
impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Stderr::write(self, data)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
