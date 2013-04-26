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
use super::{Reader, Writer, Close};

pub fn stdin() -> StdReader { fail!() }

pub fn stdout() -> StdWriter { fail!() }

pub fn stderr() -> StdReader { fail!() }

pub fn print(_s: &str) { fail!() }

pub fn println(_s: &str) { fail!() }

pub enum StdStream {
    StdIn,
    StdOut,
    StdErr
}

pub struct StdReader;

impl StdReader {
    pub fn new(_stream: StdStream) -> StdReader { fail!() }
}

impl Reader for StdReader {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl Close for StdReader {
    fn close(&mut self) { fail!() }
}

pub struct StdWriter;

impl StdWriter {
    pub fn new(_stream: StdStream) -> StdWriter { fail!() }
}

impl Writer for StdWriter {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl Close for StdWriter {
    fn close(&mut self) { fail!() }
}
