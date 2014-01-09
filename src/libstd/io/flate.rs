// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some various other I/O types

// FIXME(#3660): should move to libextra

use prelude::*;
use super::*;

/// A Writer decorator that compresses using the 'deflate' scheme
pub struct DeflateWriter<W> {
    priv inner_writer: W
}

impl<W: Writer> DeflateWriter<W> {
    pub fn new(inner_writer: W) -> DeflateWriter<W> {
        DeflateWriter {
            inner_writer: inner_writer
        }
    }
}

impl<W: Writer> Writer for DeflateWriter<W> {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

/// A Reader decorator that decompresses using the 'deflate' scheme
pub struct InflateReader<R> {
    priv inner_reader: R
}

impl<R: Reader> InflateReader<R> {
    pub fn new(inner_reader: R) -> InflateReader<R> {
        InflateReader {
            inner_reader: inner_reader
        }
    }
}

impl<R: Reader> Reader for InflateReader<R> {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}
