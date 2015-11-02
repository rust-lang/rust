// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::inner::*;
use sys::error::Result;
use io::prelude::*;
use io;
use libc;
use sys::unix::fd::FileDesc;

pub use sys::common::stdio::dumb_print;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

pub fn stdin() -> Result<Stdin> { Ok(Stdin(())) }
pub fn stdout() -> Result<Stdout> { Ok(Stdout(())) }
pub fn stderr() -> Result<Stderr> { Ok(Stderr(())) }

impl Read for Stdin {
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        let fd = FileDesc::from_inner(libc::STDIN_FILENO);
        let ret = fd.read(data);
        fd.into_inner();
        ret.map_err(From::from)
    }
}

impl Write for Stdout {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::from_inner(libc::STDOUT_FILENO);
        let ret = fd.write(data);
        fd.into_inner();
        ret.map_err(From::from)
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

impl Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::from_inner(libc::STDERR_FILENO);
        let ret = fd.write(data);
        fd.into_inner();
        ret.map_err(From::from)
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

pub fn is_ebadf(e: &io::Error) -> bool {
    e.raw_os_error() == Some(libc::EBADF)
}
