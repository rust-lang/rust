// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use super::*;
use super::super::*;
use super::super::misc::PathLike;

pub struct UnixStream;

impl UnixStream {
    pub fn connect<P: PathLike>(_path: &P) -> Result<UnixStream, IoError> {
        fail!()
    }
}

impl Reader for UnixStream {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for UnixStream {
    fn write(&mut self, _v: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl Close for UnixStream {
    fn close(&mut self) { fail!() }
}

pub struct UnixListener;

impl UnixListener {
    pub fn new<P: PathLike>(_path: &P) -> UnixListener {
        fail!()
    }
}

impl Listener<UnixStream> for UnixListener {
    fn accept(&mut self) -> Option<UnixStream> { fail!() }
}

