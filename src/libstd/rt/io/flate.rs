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

// NOTE: These ultimately belong somewhere else

use prelude::*;
use super::*;

/// A Writer decorator that compresses using the 'deflate' scheme
pub struct DeflateWriter<W> {
    inner_writer: W
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

impl<W: Writer> Decorator<W> for DeflateWriter<W> {
    fn inner(self) -> W {
        match self {
            DeflateWriter { inner_writer: w } => w
        }
    }

    fn inner_ref<'a>(&'a self) -> &'a W {
        match *self {
            DeflateWriter { inner_writer: ref w } => w
        }
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut W {
        match *self {
            DeflateWriter { inner_writer: ref mut w } => w
        }
    }
}

/// A Reader decorator that decompresses using the 'deflate' scheme
pub struct InflateReader<R> {
    inner_reader: R
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

impl<R: Reader> Decorator<R> for InflateReader<R> {
    fn inner(self) -> R {
        match self {
            InflateReader { inner_reader: r } => r
        }
    }

    fn inner_ref<'a>(&'a self) -> &'a R {
        match *self {
            InflateReader { inner_reader: ref r } => r
        }
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut R {
        match *self {
            InflateReader { inner_reader: ref mut r } => r
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;
    use super::super::mem::*;
    use super::super::Decorator;

    use str;

    #[test]
    #[ignore]
    fn smoke_test() {
        let mem_writer = MemWriter::new();
        let mut deflate_writer = DeflateWriter::new(mem_writer);
        let in_msg = "test";
        let in_bytes = in_msg.to_bytes();
        deflate_writer.write(in_bytes);
        deflate_writer.flush();
        let buf = deflate_writer.inner().inner();
        let mem_reader = MemReader::new(buf);
        let mut inflate_reader = InflateReader::new(mem_reader);
        let mut out_bytes = [0, .. 100];
        let bytes_read = inflate_reader.read(out_bytes).get();
        assert_eq!(bytes_read, in_bytes.len());
        let out_msg = str::from_bytes(out_bytes);
        assert!(in_msg == out_msg);
    }
}
