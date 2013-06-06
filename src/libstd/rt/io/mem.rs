// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Readers and Writers for in-memory buffers
//!
//! # XXX
//!
//! * Should probably have something like this for strings.
//! * Should they implement Closable? Would take extra state.

use cmp::min;
use prelude::*;
use super::*;
use vec;

/// Writes to an owned, growable byte vector
pub struct MemWriter {
    buf: ~[u8]
}

impl MemWriter {
    pub fn new() -> MemWriter { MemWriter { buf: ~[] } }
}

impl Writer for MemWriter {
    fn write(&mut self, buf: &[u8]) {
        self.buf.push_all(buf)
    }

    fn flush(&mut self) { /* no-op */ }
}

impl Seek for MemWriter {
    fn tell(&self) -> u64 { self.buf.len() as u64 }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

impl Decorator<~[u8]> for MemWriter {

    fn inner(self) -> ~[u8] {
        match self {
            MemWriter { buf: buf } => buf
        }
    }

    fn inner_ref<'a>(&'a self) -> &'a ~[u8] {
        match *self {
            MemWriter { buf: ref buf } => buf
        }
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut ~[u8] {
        match *self {
            MemWriter { buf: ref mut buf } => buf
        }
    }
}

/// Reads from an owned byte vector
pub struct MemReader {
    buf: ~[u8],
    pos: uint
}

impl MemReader {
    pub fn new(buf: ~[u8]) -> MemReader {
        MemReader {
            buf: buf,
            pos: 0
        }
    }
}

impl Reader for MemReader {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        { if self.eof() { return None; } }

        let write_len = min(buf.len(), self.buf.len() - self.pos);
        {
            let input = self.buf.slice(self.pos, self.pos + write_len);
            let output = vec::mut_slice(buf, 0, write_len);
            assert_eq!(input.len(), output.len());
            vec::bytes::copy_memory(output, input, write_len);
        }
        self.pos += write_len;
        assert!(self.pos <= self.buf.len());

        return Some(write_len);
    }

    fn eof(&mut self) -> bool { self.pos == self.buf.len() }
}

impl Seek for MemReader {
    fn tell(&self) -> u64 { self.pos as u64 }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

impl Decorator<~[u8]> for MemReader {

    fn inner(self) -> ~[u8] {
        match self {
            MemReader { buf: buf, _ } => buf
        }
    }

    fn inner_ref<'a>(&'a self) -> &'a ~[u8] {
        match *self {
            MemReader { buf: ref buf, _ } => buf
        }
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut ~[u8] {
        match *self {
            MemReader { buf: ref mut buf, _ } => buf
        }
    }
}


/// Writes to a fixed-size byte slice
struct BufWriter<'self> {
    buf: &'self mut [u8],
    pos: uint
}

impl<'self> BufWriter<'self> {
    pub fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a> {
        BufWriter {
            buf: buf,
            pos: 0
        }
    }
}

impl<'self> Writer for BufWriter<'self> {
    fn write(&mut self, _buf: &[u8]) { fail!() }

    fn flush(&mut self) { fail!() }
}

impl<'self> Seek for BufWriter<'self> {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}


/// Reads from a fixed-size byte slice
struct BufReader<'self> {
    buf: &'self [u8],
    pos: uint
}

impl<'self> BufReader<'self> {
    pub fn new<'a>(buf: &'a [u8]) -> BufReader<'a> {
        BufReader {
            buf: buf,
            pos: 0
        }
    }
}

impl<'self> Reader for BufReader<'self> {
    fn read(&mut self, _buf: &mut [u8]) -> Option<uint> { fail!() }

    fn eof(&mut self) -> bool { fail!() }
}

impl<'self> Seek for BufReader<'self> {
    fn tell(&self) -> u64 { fail!() }

    fn seek(&mut self, _pos: i64, _style: SeekStyle) { fail!() }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;

    #[test]
    fn test_mem_writer() {
        let mut writer = MemWriter::new();
        assert_eq!(writer.tell(), 0);
        writer.write([0]);
        assert_eq!(writer.tell(), 1);
        writer.write([1, 2, 3]);
        writer.write([4, 5, 6, 7]);
        assert_eq!(writer.tell(), 8);
        assert_eq!(writer.inner(), ~[0, 1, 2, 3, 4, 5 , 6, 7]);
    }

    #[test]
    fn test_mem_reader() {
        let mut reader = MemReader::new(~[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut buf = [];
        assert_eq!(reader.read(buf), Some(0));
        assert_eq!(reader.tell(), 0);
        let mut buf = [0];
        assert_eq!(reader.read(buf), Some(1));
        assert_eq!(reader.tell(), 1);
        assert_eq!(buf, [0]);
        let mut buf = [0, ..4];
        assert_eq!(reader.read(buf), Some(4));
        assert_eq!(reader.tell(), 5);
        assert_eq!(buf, [1, 2, 3, 4]);
        assert_eq!(reader.read(buf), Some(3));
        assert_eq!(buf.slice(0, 3), [5, 6, 7]);
        assert!(reader.eof());
        assert_eq!(reader.read(buf), None);
        assert!(reader.eof());
    }
}
