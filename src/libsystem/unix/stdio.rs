// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error::prelude::*;
use inner::prelude::*;
use io;
use libc;
use unix::fd::FileDesc;
use stdio as sys;

pub struct Stdio(());
pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

impl sys::Stdio for Stdio {
    type Stdin = Stdin;
    type Stdout = Stdout;
    type Stderr = Stderr;

    fn stdin() -> Result<Stdin> { Ok(Stdin(())) }
    fn stdout() -> Result<Stdout> { Ok(Stdout(())) }
    fn stderr() -> Result<Stderr> { Ok(Stderr(())) }
    fn ebadf() -> i32 { libc::EBADF }
}

impl io::Read for Stdin {
    fn read(&self, data: &mut [u8]) -> Result<usize> {
        let fd = FileDesc::from_inner(libc::STDIN_FILENO);
        let ret = io::Read::read(&fd, data);
        fd.into_inner();
        ret
    }
}

impl io::Write for Stdout {
    fn write(&self, data: &[u8]) -> Result<usize> {
        let fd = FileDesc::from_inner(libc::STDOUT_FILENO);
        let ret = io::Write::write(&fd, data);
        fd.into_inner();
        ret
    }
}

impl io::Write for Stderr {
    fn write(&self, data: &[u8]) -> Result<usize> {
        let fd = FileDesc::from_inner(libc::STDERR_FILENO);
        let ret = io::Write::write(&fd, data);
        fd.into_inner();
        ret
    }
}
