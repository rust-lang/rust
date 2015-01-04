// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use sys::fs::FileDesc;
use libc::{self, c_int};
use io::{self, IoResult, IoError};
use sys_common;

pub struct TTY {
    pub fd: FileDesc,
}

impl TTY {
    pub fn new(fd: c_int) -> IoResult<TTY> {
        if unsafe { libc::isatty(fd) } != 0 {
            Ok(TTY { fd: FileDesc::new(fd, true) })
        } else {
            Err(IoError {
                kind: io::MismatchedFileTypeForOperation,
                desc: "file descriptor is not a TTY",
                detail: None,
            })
        }
    }

    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.fd.read(buf)
    }
    pub fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.fd.write(buf)
    }
    pub fn set_raw(&mut self, _raw: bool) -> IoResult<()> {
        Err(sys_common::unimpl())
    }
    pub fn get_winsize(&mut self) -> IoResult<(int, int)> {
        Err(sys_common::unimpl())
    }
    pub fn isatty(&self) -> bool { false }
}
