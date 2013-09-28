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
use super::{Reader, Writer};

pub fn stdin() -> StdReader { fail2!() }

pub fn stdout() -> StdWriter { fail2!() }

pub fn stderr() -> StdReader { fail2!() }

pub fn print(_s: &str) { fail2!() }

pub fn println(_s: &str) { fail2!() }

pub enum StdStream {
    StdIn,
    StdOut,
    StdErr
}

pub struct StdReader;

impl StdReader {
    pub fn new(_stream: StdStream) -> StdReader { fail2!() }
}

impl Reader for StdReader {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail2!() }

    fn eof(&mut self) -> bool { fail2!() }
}

pub struct StdWriter;

impl StdWriter {
    pub fn new(_stream: StdStream) -> StdWriter { fail2!() }
}

impl Writer for StdWriter {
    fn write(&mut self, _buf: &[u8]) { fail2!() }

    fn flush(&mut self) { fail2!() }
}
