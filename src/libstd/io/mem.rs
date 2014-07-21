// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

//! Readers and Writers for in-memory buffers

use cmp::min;
use collections::Collection;
use option::None;
use result::{Err, Ok};
use io;
use io::{Reader, Writer, Seek, Buffer, IoError, SeekStyle, IoResult};
use slice;
use slice::{Vector, ImmutableVector, MutableVector};
use vec::Vec;

fn combine(seek: SeekStyle, cur: uint, end: uint, offset: i64) -> IoResult<u64> {
    // compute offset as signed and clamp to prevent overflow
    let pos = match seek {
        io::SeekSet => 0,
        io::SeekEnd => end,
        io::SeekCur => cur,
    } as i64;

    if offset + pos < 0 {
        Err(IoError {
            kind: io::InvalidInput,
            desc: "invalid seek to a negative offset",
            detail: None
        })
    } else {
        Ok((offset + pos) as u64)
    }
}

/// Writes to an owned, growable byte vector
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::MemWriter;
///
/// let mut w = MemWriter::new();
/// w.write([0, 1, 2]);
///
/// assert_eq!(w.unwrap(), vec!(0, 1, 2));
/// ```
pub struct MemWriter {
    buf: Vec<u8>,
    pos: uint,
}

impl MemWriter {
    /// Create a new `MemWriter`.
    #[inline]
    pub fn new() -> MemWriter {
        MemWriter::with_capacity(128)
    }
    /// Create a new `MemWriter`, allocating at least `n` bytes for
    /// the internal buffer.
    #[inline]
    pub fn with_capacity(n: uint) -> MemWriter {
        MemWriter { buf: Vec::with_capacity(n), pos: 0 }
    }

    /// Acquires an immutable reference to the underlying buffer of this
    /// `MemWriter`.
    ///
    /// No method is exposed for acquiring a mutable reference to the buffer
    /// because it could corrupt the state of this `MemWriter`.
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a [u8] { self.buf.as_slice() }

    /// Unwraps this `MemWriter`, returning the underlying buffer
    #[inline]
    pub fn unwrap(self) -> Vec<u8> { self.buf }
}

impl Writer for MemWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        // Make sure the internal buffer is as least as big as where we
        // currently are
        let difference = self.pos as i64 - self.buf.len() as i64;
        if difference > 0 {
            self.buf.grow(difference as uint, &0);
        }

        // Figure out what bytes will be used to overwrite what's currently
        // there (left), and what will be appended on the end (right)
        let cap = self.buf.len() - self.pos;
        let (left, right) = if cap <= buf.len() {
            (buf.slice_to(cap), buf.slice_from(cap))
        } else {
            (buf, &[])
        };

        // Do the necessary writes
        if left.len() > 0 {
            slice::bytes::copy_memory(self.buf.mut_slice_from(self.pos), left);
        }
        if right.len() > 0 {
            self.buf.push_all(right);
        }

        // Bump us forward
        self.pos += buf.len();
        Ok(())
    }
}

impl Seek for MemWriter {
    #[inline]
    fn tell(&self) -> IoResult<u64> { Ok(self.pos as u64) }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        let new = try!(combine(style, self.pos, self.buf.len(), pos));
        self.pos = new as uint;
        Ok(())
    }
}

/// Reads from an owned byte vector
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::MemReader;
///
/// let mut r = MemReader::new(vec!(0, 1, 2));
///
/// assert_eq!(r.read_to_end().unwrap(), vec!(0, 1, 2));
/// ```
pub struct MemReader {
    buf: Vec<u8>,
    pos: uint
}

impl MemReader {
    /// Creates a new `MemReader` which will read the buffer given. The buffer
    /// can be re-acquired through `unwrap`
    #[inline]
    pub fn new(buf: Vec<u8>) -> MemReader {
        MemReader {
            buf: buf,
            pos: 0
        }
    }

    /// Tests whether this reader has read all bytes in its buffer.
    ///
    /// If `true`, then this will no longer return bytes from `read`.
    #[inline]
    pub fn eof(&self) -> bool { self.pos >= self.buf.len() }

    /// Acquires an immutable reference to the underlying buffer of this
    /// `MemReader`.
    ///
    /// No method is exposed for acquiring a mutable reference to the buffer
    /// because it could corrupt the state of this `MemReader`.
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a [u8] { self.buf.as_slice() }

    /// Unwraps this `MemReader`, returning the underlying buffer
    #[inline]
    pub fn unwrap(self) -> Vec<u8> { self.buf }
}

impl Reader for MemReader {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        if self.eof() { return Err(io::standard_error(io::EndOfFile)) }

        let write_len = min(buf.len(), self.buf.len() - self.pos);
        {
            let input = self.buf.slice(self.pos, self.pos + write_len);
            let output = buf.mut_slice(0, write_len);
            assert_eq!(input.len(), output.len());
            slice::bytes::copy_memory(output, input);
        }
        self.pos += write_len;
        assert!(self.pos <= self.buf.len());

        return Ok(write_len);
    }
}

impl Seek for MemReader {
    #[inline]
    fn tell(&self) -> IoResult<u64> { Ok(self.pos as u64) }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        let new = try!(combine(style, self.pos, self.buf.len(), pos));
        self.pos = new as uint;
        Ok(())
    }
}

impl Buffer for MemReader {
    #[inline]
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> {
        if self.pos < self.buf.len() {
            Ok(self.buf.slice_from(self.pos))
        } else {
            Err(io::standard_error(io::EndOfFile))
        }
    }

    #[inline]
    fn consume(&mut self, amt: uint) { self.pos += amt; }
}

/// Writes to a fixed-size byte slice
///
/// If a write will not fit in the buffer, it returns an error and does not
/// write any data.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::BufWriter;
///
/// let mut buf = [0, ..4];
/// {
///     let mut w = BufWriter::new(buf);
///     w.write([0, 1, 2]);
/// }
/// assert!(buf == [0, 1, 2, 0]);
/// ```
pub struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: uint
}

impl<'a> BufWriter<'a> {
    /// Creates a new `BufWriter` which will wrap the specified buffer. The
    /// writer initially starts at position 0.
    #[inline]
    pub fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a> {
        BufWriter {
            buf: buf,
            pos: 0
        }
    }
}

impl<'a> Writer for BufWriter<'a> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        // return an error if the entire write does not fit in the buffer
        let cap = if self.pos >= self.buf.len() { 0 } else { self.buf.len() - self.pos };
        if buf.len() > cap {
            return Err(IoError {
                kind: io::OtherIoError,
                desc: "Trying to write past end of buffer",
                detail: None
            })
        }

        slice::bytes::copy_memory(self.buf.mut_slice_from(self.pos), buf);
        self.pos += buf.len();
        Ok(())
    }
}

impl<'a> Seek for BufWriter<'a> {
    #[inline]
    fn tell(&self) -> IoResult<u64> { Ok(self.pos as u64) }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        let new = try!(combine(style, self.pos, self.buf.len(), pos));
        self.pos = new as uint;
        Ok(())
    }
}

/// Reads from a fixed-size byte slice
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::BufReader;
///
/// let mut buf = [0, 1, 2, 3];
/// let mut r = BufReader::new(buf);
///
/// assert_eq!(r.read_to_end().unwrap(), vec!(0, 1, 2, 3));
/// ```
pub struct BufReader<'a> {
    buf: &'a [u8],
    pos: uint
}

impl<'a> BufReader<'a> {
    /// Creates a new buffered reader which will read the specified buffer
    #[inline]
    pub fn new<'a>(buf: &'a [u8]) -> BufReader<'a> {
        BufReader {
            buf: buf,
            pos: 0
        }
    }

    /// Tests whether this reader has read all bytes in its buffer.
    ///
    /// If `true`, then this will no longer return bytes from `read`.
    #[inline]
    pub fn eof(&self) -> bool { self.pos >= self.buf.len() }
}

impl<'a> Reader for BufReader<'a> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        if self.eof() { return Err(io::standard_error(io::EndOfFile)) }

        let write_len = min(buf.len(), self.buf.len() - self.pos);
        {
            let input = self.buf.slice(self.pos, self.pos + write_len);
            let output = buf.mut_slice(0, write_len);
            assert_eq!(input.len(), output.len());
            slice::bytes::copy_memory(output, input);
        }
        self.pos += write_len;
        assert!(self.pos <= self.buf.len());

        return Ok(write_len);
     }
}

impl<'a> Seek for BufReader<'a> {
    #[inline]
    fn tell(&self) -> IoResult<u64> { Ok(self.pos as u64) }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> {
        let new = try!(combine(style, self.pos, self.buf.len(), pos));
        self.pos = new as uint;
        Ok(())
    }
}

impl<'a> Buffer for BufReader<'a> {
    #[inline]
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> {
        if self.pos < self.buf.len() {
            Ok(self.buf.slice_from(self.pos))
        } else {
            Err(io::standard_error(io::EndOfFile))
        }
    }

    #[inline]
    fn consume(&mut self, amt: uint) { self.pos += amt; }
}

#[cfg(test)]
mod test {
    extern crate test;
    use prelude::*;
    use super::*;
    use io::*;
    use io;
    use self::test::Bencher;
    use str::StrSlice;

    #[test]
    fn test_mem_writer() {
        let mut writer = MemWriter::new();
        assert_eq!(writer.tell(), Ok(0));
        writer.write([0]).unwrap();
        assert_eq!(writer.tell(), Ok(1));
        writer.write([1, 2, 3]).unwrap();
        writer.write([4, 5, 6, 7]).unwrap();
        assert_eq!(writer.tell(), Ok(8));
        assert_eq!(writer.get_ref(), &[0, 1, 2, 3, 4, 5, 6, 7]);

        writer.seek(0, SeekSet).unwrap();
        assert_eq!(writer.tell(), Ok(0));
        writer.write([3, 4]).unwrap();
        assert_eq!(writer.get_ref(), &[3, 4, 2, 3, 4, 5, 6, 7]);

        writer.seek(1, SeekCur).unwrap();
        writer.write([0, 1]).unwrap();
        assert_eq!(writer.get_ref(), &[3, 4, 2, 0, 1, 5, 6, 7]);

        writer.seek(-1, SeekEnd).unwrap();
        writer.write([1, 2]).unwrap();
        assert_eq!(writer.get_ref(), &[3, 4, 2, 0, 1, 5, 6, 1, 2]);

        writer.seek(1, SeekEnd).unwrap();
        writer.write([1]).unwrap();
        assert_eq!(writer.get_ref(), &[3, 4, 2, 0, 1, 5, 6, 1, 2, 0, 1]);
    }

    #[test]
    fn test_buf_writer() {
        let mut buf = [0 as u8, ..8];
        {
            let mut writer = BufWriter::new(buf);
            assert_eq!(writer.tell(), Ok(0));
            writer.write([0]).unwrap();
            assert_eq!(writer.tell(), Ok(1));
            writer.write([1, 2, 3]).unwrap();
            writer.write([4, 5, 6, 7]).unwrap();
            assert_eq!(writer.tell(), Ok(8));
            writer.write([]).unwrap();
            assert_eq!(writer.tell(), Ok(8));
        }
        assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_buf_writer_seek() {
        let mut buf = [0 as u8, ..8];
        {
            let mut writer = BufWriter::new(buf);
            assert_eq!(writer.tell(), Ok(0));
            writer.write([1]).unwrap();
            assert_eq!(writer.tell(), Ok(1));

            writer.seek(2, SeekSet).unwrap();
            assert_eq!(writer.tell(), Ok(2));
            writer.write([2]).unwrap();
            assert_eq!(writer.tell(), Ok(3));

            writer.seek(-2, SeekCur).unwrap();
            assert_eq!(writer.tell(), Ok(1));
            writer.write([3]).unwrap();
            assert_eq!(writer.tell(), Ok(2));

            writer.seek(-1, SeekEnd).unwrap();
            assert_eq!(writer.tell(), Ok(7));
            writer.write([4]).unwrap();
            assert_eq!(writer.tell(), Ok(8));

        }
        assert_eq!(buf.as_slice(), &[1, 3, 2, 0, 0, 0, 0, 4]);
    }

    #[test]
    fn test_buf_writer_error() {
        let mut buf = [0 as u8, ..2];
        let mut writer = BufWriter::new(buf);
        writer.write([0]).unwrap();

        match writer.write([0, 0]) {
            Ok(..) => fail!(),
            Err(e) => assert_eq!(e.kind, io::OtherIoError),
        }
    }

    #[test]
    fn test_mem_reader() {
        let mut reader = MemReader::new(vec!(0, 1, 2, 3, 4, 5, 6, 7));
        let mut buf = [];
        assert_eq!(reader.read(buf), Ok(0));
        assert_eq!(reader.tell(), Ok(0));
        let mut buf = [0];
        assert_eq!(reader.read(buf), Ok(1));
        assert_eq!(reader.tell(), Ok(1));
        assert_eq!(buf.as_slice(), &[0]);
        let mut buf = [0, ..4];
        assert_eq!(reader.read(buf), Ok(4));
        assert_eq!(reader.tell(), Ok(5));
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(reader.read(buf), Ok(3));
        assert_eq!(buf.slice(0, 3), &[5, 6, 7]);
        assert!(reader.read(buf).is_err());
        let mut reader = MemReader::new(vec!(0, 1, 2, 3, 4, 5, 6, 7));
        assert_eq!(reader.read_until(3).unwrap(), vec!(0, 1, 2, 3));
        assert_eq!(reader.read_until(3).unwrap(), vec!(4, 5, 6, 7));
        assert!(reader.read(buf).is_err());
    }

    #[test]
    fn test_buf_reader() {
        let in_buf = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut reader = BufReader::new(in_buf.as_slice());
        let mut buf = [];
        assert_eq!(reader.read(buf), Ok(0));
        assert_eq!(reader.tell(), Ok(0));
        let mut buf = [0];
        assert_eq!(reader.read(buf), Ok(1));
        assert_eq!(reader.tell(), Ok(1));
        assert_eq!(buf.as_slice(), &[0]);
        let mut buf = [0, ..4];
        assert_eq!(reader.read(buf), Ok(4));
        assert_eq!(reader.tell(), Ok(5));
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(reader.read(buf), Ok(3));
        assert_eq!(buf.slice(0, 3), &[5, 6, 7]);
        assert!(reader.read(buf).is_err());
        let mut reader = BufReader::new(in_buf.as_slice());
        assert_eq!(reader.read_until(3).unwrap(), vec!(0, 1, 2, 3));
        assert_eq!(reader.read_until(3).unwrap(), vec!(4, 5, 6, 7));
        assert!(reader.read(buf).is_err());
    }

    #[test]
    fn test_read_char() {
        let b = b"Vi\xE1\xBB\x87t";
        let mut r = BufReader::new(b);
        assert_eq!(r.read_char(), Ok('V'));
        assert_eq!(r.read_char(), Ok('i'));
        assert_eq!(r.read_char(), Ok('ệ'));
        assert_eq!(r.read_char(), Ok('t'));
        assert!(r.read_char().is_err());
    }

    #[test]
    fn test_read_bad_char() {
        let b = b"\x80";
        let mut r = BufReader::new(b);
        assert!(r.read_char().is_err());
    }

    #[test]
    fn test_write_strings() {
        let mut writer = MemWriter::new();
        writer.write_str("testing").unwrap();
        writer.write_line("testing").unwrap();
        writer.write_str("testing").unwrap();
        let mut r = BufReader::new(writer.get_ref());
        assert_eq!(r.read_to_string().unwrap(), "testingtesting\ntesting".to_string());
    }

    #[test]
    fn test_write_char() {
        let mut writer = MemWriter::new();
        writer.write_char('a').unwrap();
        writer.write_char('\n').unwrap();
        writer.write_char('ệ').unwrap();
        let mut r = BufReader::new(writer.get_ref());
        assert_eq!(r.read_to_string().unwrap(), "a\nệ".to_string());
    }

    #[test]
    fn test_read_whole_string_bad() {
        let buf = [0xff];
        let mut r = BufReader::new(buf);
        match r.read_to_string() {
            Ok(..) => fail!(),
            Err(..) => {}
        }
    }

    #[test]
    fn seek_past_end() {
        let buf = [0xff];
        let mut r = BufReader::new(buf);
        r.seek(10, SeekSet).unwrap();
        assert!(r.read(&mut []).is_err());

        let mut r = MemReader::new(vec!(10));
        r.seek(10, SeekSet).unwrap();
        assert!(r.read(&mut []).is_err());

        let mut r = MemWriter::new();
        r.seek(10, SeekSet).unwrap();
        assert!(r.write([3]).is_ok());

        let mut buf = [0];
        let mut r = BufWriter::new(buf);
        r.seek(10, SeekSet).unwrap();
        assert!(r.write([3]).is_err());
    }

    #[test]
    fn seek_before_0() {
        let buf = [0xff];
        let mut r = BufReader::new(buf);
        assert!(r.seek(-1, SeekSet).is_err());

        let mut r = MemReader::new(vec!(10));
        assert!(r.seek(-1, SeekSet).is_err());

        let mut r = MemWriter::new();
        assert!(r.seek(-1, SeekSet).is_err());

        let mut buf = [0];
        let mut r = BufWriter::new(buf);
        assert!(r.seek(-1, SeekSet).is_err());
    }

    #[test]
    fn io_read_at_least() {
        let mut r = MemReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let mut buf = [0, ..3];
        assert!(r.read_at_least(buf.len(), buf).is_ok());
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        assert!(r.read_at_least(0, buf.mut_slice_to(0)).is_ok());
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        assert!(r.read_at_least(buf.len(), buf).is_ok());
        assert_eq!(buf.as_slice(), &[4, 5, 6]);
        assert!(r.read_at_least(buf.len(), buf).is_err());
        assert_eq!(buf.as_slice(), &[7, 8, 6]);
    }

    fn do_bench_mem_writer(b: &mut Bencher, times: uint, len: uint) {
        let src: Vec<u8> = Vec::from_elem(len, 5);

        b.iter(|| {
            let mut wr = MemWriter::new();
            for _ in range(0, times) {
                wr.write(src.as_slice()).unwrap();
            }

            let v = wr.unwrap();
            assert_eq!(v.len(), times * len);
            assert!(v.iter().all(|x| *x == 5));
        });
    }

    #[bench]
    fn bench_mem_writer_001_0000(b: &mut Bencher) {
        do_bench_mem_writer(b, 1, 0)
    }

    #[bench]
    fn bench_mem_writer_001_0010(b: &mut Bencher) {
        do_bench_mem_writer(b, 1, 10)
    }

    #[bench]
    fn bench_mem_writer_001_0100(b: &mut Bencher) {
        do_bench_mem_writer(b, 1, 100)
    }

    #[bench]
    fn bench_mem_writer_001_1000(b: &mut Bencher) {
        do_bench_mem_writer(b, 1, 1000)
    }

    #[bench]
    fn bench_mem_writer_100_0000(b: &mut Bencher) {
        do_bench_mem_writer(b, 100, 0)
    }

    #[bench]
    fn bench_mem_writer_100_0010(b: &mut Bencher) {
        do_bench_mem_writer(b, 100, 10)
    }

    #[bench]
    fn bench_mem_writer_100_0100(b: &mut Bencher) {
        do_bench_mem_writer(b, 100, 100)
    }

    #[bench]
    fn bench_mem_writer_100_1000(b: &mut Bencher) {
        do_bench_mem_writer(b, 100, 1000)
    }

    #[bench]
    fn bench_mem_reader(b: &mut Bencher) {
        b.iter(|| {
            let buf = Vec::from_slice([5 as u8, ..100]);
            {
                let mut rdr = MemReader::new(buf);
                for _i in range(0u, 10) {
                    let mut buf = [0 as u8, .. 10];
                    rdr.read(buf).unwrap();
                    assert_eq!(buf.as_slice(), [5, .. 10].as_slice());
                }
            }
        });
    }

    #[bench]
    fn bench_buf_writer(b: &mut Bencher) {
        b.iter(|| {
            let mut buf = [0 as u8, ..100];
            {
                let mut wr = BufWriter::new(buf);
                for _i in range(0u, 10) {
                    wr.write([5, .. 10]).unwrap();
                }
            }
            assert_eq!(buf.as_slice(), [5, .. 100].as_slice());
        });
    }

    #[bench]
    fn bench_buf_reader(b: &mut Bencher) {
        b.iter(|| {
            let buf = [5 as u8, ..100];
            {
                let mut rdr = BufReader::new(buf);
                for _i in range(0u, 10) {
                    let mut buf = [0 as u8, .. 10];
                    rdr.read(buf).unwrap();
                    assert_eq!(buf.as_slice(), [5, .. 10].as_slice());
                }
            }
        });
    }
}
