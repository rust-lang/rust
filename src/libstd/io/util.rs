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
use cmp;
use io;
use vec::bytes::MutableByteVector;

/// Wraps a `Reader`, limiting the number of bytes that can be read from it.
pub struct LimitReader<'a, R> {
    priv limit: uint,
    priv inner: &'a mut R
}

impl<'a, R: Reader> LimitReader<'a, R> {
    /// Creates a new `LimitReader`
    pub fn new<'a>(r: &'a mut R, limit: uint) -> LimitReader<'a, R> {
        LimitReader { limit: limit, inner: r }
    }
}

impl<'a, R: Reader> Reader for LimitReader<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
        if self.limit == 0 {
            return Err(io::standard_error(io::EndOfFile));
        }

        let len = cmp::min(self.limit, buf.len());
        self.inner.read(buf.mut_slice_to(len)).map(|len| {
            self.limit -= len;
            len
        })
    }
}

/// A `Writer` which ignores bytes written to it, like /dev/null.
pub struct NullWriter;

impl Writer for NullWriter {
    #[inline]
    fn write(&mut self, _buf: &[u8]) -> io::IoResult<()> { Ok(()) }
}

/// A `Reader` which returns an infinite stream of 0 bytes, like /dev/zero.
pub struct ZeroReader;

impl Reader for ZeroReader {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
        buf.set_memory(0);
        Ok(buf.len())
    }
}

/// A `Reader` which is always at EOF, like /dev/null.
pub struct NullReader;

impl Reader for NullReader {
    #[inline]
    fn read(&mut self, _buf: &mut [u8]) -> io::IoResult<uint> {
        Err(io::standard_error(io::EndOfFile))
    }
}

/// A `Writer` which multiplexes writes to a set of `Writers`.
pub struct MultiWriter {
    priv writers: ~[~Writer]
}

impl MultiWriter {
    /// Creates a new `MultiWriter`
    pub fn new(writers: ~[~Writer]) -> MultiWriter {
        MultiWriter { writers: writers }
    }
}

impl Writer for MultiWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::IoResult<()> {
        let mut ret = Ok(());
        for writer in self.writers.mut_iter() {
            ret = ret.and(writer.write(buf));
        }
        return ret;
    }

    #[inline]
    fn flush(&mut self) -> io::IoResult<()> {
        let mut ret = Ok(());
        for writer in self.writers.mut_iter() {
            ret = ret.and(writer.flush());
        }
        return ret;
    }
}

/// A `Reader` which chains input from multiple `Readers`, reading each to
/// completion before moving onto the next.
pub struct ChainedReader<I, R> {
    priv readers: I,
    priv cur_reader: Option<R>,
}

impl<R: Reader, I: Iterator<R>> ChainedReader<I, R> {
    /// Creates a new `ChainedReader`
    pub fn new(mut readers: I) -> ChainedReader<I, R> {
        let r = readers.next();
        ChainedReader { readers: readers, cur_reader: r }
    }
}

impl<R: Reader, I: Iterator<R>> Reader for ChainedReader<I, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
        loop {
            let err = match self.cur_reader {
                Some(ref mut r) => {
                    match r.read(buf) {
                        Ok(len) => return Ok(len),
                        Err(ref e) if e.kind == io::EndOfFile => None,
                        Err(e) => Some(e),
                    }
                }
                None => break
            };
            self.cur_reader = self.readers.next();
            match err {
                Some(e) => return Err(e),
                None => {}
            }
        }
        Err(io::standard_error(io::EndOfFile))
    }
}

/// A `Reader` which forwards input from another `Reader`, passing it along to
/// a `Writer` as well. Similar to the `tee(1)` command.
pub struct TeeReader<R, W> {
    priv reader: R,
    priv writer: W
}

impl<R: Reader, W: Writer> TeeReader<R, W> {
    /// Creates a new `TeeReader`
    pub fn new(r: R, w: W) -> TeeReader<R, W> {
        TeeReader { reader: r, writer: w }
    }

    /// Consumes the `TeeReader`, returning the underlying `Reader` and
    /// `Writer`.
    pub fn unwrap(self) -> (R, W) {
        let TeeReader { reader, writer } = self;
        (reader, writer)
    }
}

impl<R: Reader, W: Writer> Reader for TeeReader<R, W> {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> {
        self.reader.read(buf).and_then(|len| {
            self.writer.write(buf.slice_to(len)).map(|()| len)
        })
    }
}

/// Copies all data from a `Reader` to a `Writer`.
pub fn copy<R: Reader, W: Writer>(r: &mut R, w: &mut W) -> io::IoResult<()> {
    let mut buf = [0, ..super::DEFAULT_BUF_SIZE];
    loop {
        let len = match r.read(buf) {
            Ok(len) => len,
            Err(ref e) if e.kind == io::EndOfFile => return Ok(()),
            Err(e) => return Err(e),
        };
        if_ok!(w.write(buf.slice_to(len)));
    }
}

#[cfg(test)]
mod test {
    use io;
    use io::{MemReader, MemWriter};
    use super::*;
    use prelude::*;

    #[test]
    fn test_bounded_reader_unlimited() {
        let mut r = MemReader::new(~[0, 1, 2]);
        {
            let mut r = LimitReader::new(&mut r, 4);
            assert_eq!(~[0, 1, 2], r.read_to_end().unwrap());
        }
    }

    #[test]
    fn test_bound_reader_limited() {
        let mut r = MemReader::new(~[0, 1, 2]);
        {
            let mut r = LimitReader::new(&mut r, 2);
            assert_eq!(~[0, 1], r.read_to_end().unwrap());
        }
        assert_eq!(~[2], r.read_to_end().unwrap());
    }

    #[test]
    fn test_null_writer() {
        let mut s = NullWriter;
        let buf = ~[0, 0, 0];
        s.write(buf).unwrap();
        s.flush().unwrap();
    }

    #[test]
    fn test_zero_reader() {
        let mut s = ZeroReader;
        let mut buf = ~[1, 2, 3];
        assert_eq!(s.read(buf), Ok(3));
        assert_eq!(~[0, 0, 0], buf);
    }

    #[test]
    fn test_null_reader() {
        let mut r = NullReader;
        let mut buf = ~[0];
        assert!(r.read(buf).is_err());
    }

    #[test]
    fn test_multi_writer() {
        static mut writes: uint = 0;
        static mut flushes: uint = 0;

        struct TestWriter;
        impl Writer for TestWriter {
            fn write(&mut self, _buf: &[u8]) -> io::IoResult<()> {
                unsafe { writes += 1 }
                Ok(())
            }

            fn flush(&mut self) -> io::IoResult<()> {
                unsafe { flushes += 1 }
                Ok(())
            }
        }

        let mut multi = MultiWriter::new(~[~TestWriter as ~Writer,
                                           ~TestWriter as ~Writer]);
        multi.write([1, 2, 3]).unwrap();
        assert_eq!(2, unsafe { writes });
        assert_eq!(0, unsafe { flushes });
        multi.flush().unwrap();
        assert_eq!(2, unsafe { writes });
        assert_eq!(2, unsafe { flushes });
    }

    #[test]
    fn test_chained_reader() {
        let rs = ~[MemReader::new(~[0, 1]), MemReader::new(~[]),
                   MemReader::new(~[2, 3])];
        let mut r = ChainedReader::new(rs.move_iter());
        assert_eq!(~[0, 1, 2, 3], r.read_to_end().unwrap());
    }

    #[test]
    fn test_tee_reader() {
        let mut r = TeeReader::new(MemReader::new(~[0, 1, 2]),
                                   MemWriter::new());
        assert_eq!(~[0, 1, 2], r.read_to_end().unwrap());
        let (_, w) = r.unwrap();
        assert_eq!(~[0, 1, 2], w.unwrap());
    }

    #[test]
    fn test_copy() {
        let mut r = MemReader::new(~[0, 1, 2, 3, 4]);
        let mut w = MemWriter::new();
        copy(&mut r, &mut w).unwrap();
        assert_eq!(~[0, 1, 2, 3, 4], w.unwrap());
    }
}
