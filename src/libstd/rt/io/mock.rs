// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::{Option, None};
use rt::io::{Reader, Writer};

pub struct MockReader {
    read: ~fn(buf: &mut [u8]) -> Option<uint>,
    priv eof: ~fn() -> bool
}

impl MockReader {
    pub fn new() -> MockReader {
        MockReader {
            read: |_| None,
            eof: || false
        }
    }
}

impl Reader for MockReader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { (self.read)(buf) }
    fn eof(&mut self) -> bool { (self.eof)() }
}

pub struct MockWriter {
    priv write: ~fn(buf: &[u8]),
    priv flush: ~fn()
}

impl MockWriter {
    pub fn new() -> MockWriter {
        MockWriter {
            write: |_| (),
            flush: || ()
        }
    }
}

impl Writer for MockWriter {
    fn write(&mut self, buf: &[u8]) { (self.write)(buf) }
    fn flush(&mut self) { (self.flush)() }
}
