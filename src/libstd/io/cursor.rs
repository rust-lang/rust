// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_copy_implementations)]

use prelude::v1::*;
use io::prelude::*;

use cmp;
use io::{self, SeekFrom, Error, ErrorKind};
use iter::repeat;
use num::Int;
use slice;

/// A `Cursor` is a type which wraps another I/O object to provide a `Seek`
/// implementation.
///
/// Cursors are currently typically used with memory buffer objects in order to
/// allow `Seek` plus `Read` and `Write` implementations. For example, common
/// cursor types include:
///
/// * `Cursor<Vec<u8>>`
/// * `Cursor<&[u8]>`
///
/// Implementations of the I/O traits for `Cursor<T>` are not currently generic
/// over `T` itself. Instead, specific implementations are provided for various
/// in-memory buffer types like `Vec<u8>` and `&[u8]`.
pub struct Cursor<T> {
    inner: T,
    pos: u64,
}

impl<T> Cursor<T> {
    /// Create a new cursor wrapping the provided underlying I/O object.
    pub fn new(inner: T) -> Cursor<T> {
        Cursor { pos: 0, inner: inner }
    }

    /// Consume this cursor, returning the underlying value.
    pub fn into_inner(self) -> T { self.inner }

    /// Get a reference to the underlying value in this cursor.
    pub fn get_ref(&self) -> &T { &self.inner }

    /// Get a mutable reference to the underlying value in this cursor.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the
    /// underlying value as it may corrupt this cursor's position.
    pub fn get_mut(&mut self) -> &mut T { &mut self.inner }

    /// Returns the current value of this cursor
    pub fn position(&self) -> u64 { self.pos }

    /// Sets the value of this cursor
    pub fn set_position(&mut self, pos: u64) { self.pos = pos; }
}

macro_rules! seek {
    () => {
        fn seek(&mut self, style: SeekFrom) -> io::Result<u64> {
            let pos = match style {
                SeekFrom::Start(n) => { self.pos = n; return Ok(n) }
                SeekFrom::End(n) => self.inner.len() as i64 + n,
                SeekFrom::Current(n) => self.pos as i64 + n,
            };

            if pos < 0 {
                Err(Error::new(ErrorKind::InvalidInput,
                               "invalid seek to a negative position",
                               None))
            } else {
                self.pos = pos as u64;
                Ok(self.pos)
            }
        }
    }
}

impl<'a> io::Seek for Cursor<&'a [u8]> { seek!(); }
impl<'a> io::Seek for Cursor<&'a mut [u8]> { seek!(); }
impl io::Seek for Cursor<Vec<u8>> { seek!(); }

macro_rules! read {
    () => {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = try!(Read::read(&mut try!(self.fill_buf()), buf));
            self.pos += n as u64;
            Ok(n)
        }
    }
}

impl<'a> Read for Cursor<&'a [u8]> { read!(); }
impl<'a> Read for Cursor<&'a mut [u8]> { read!(); }
impl Read for Cursor<Vec<u8>> { read!(); }

macro_rules! buffer {
    () => {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            let amt = cmp::min(self.pos, self.inner.len() as u64);
            Ok(&self.inner[(amt as usize)..])
        }
        fn consume(&mut self, amt: usize) { self.pos += amt as u64; }
    }
}

impl<'a> BufRead for Cursor<&'a [u8]> { buffer!(); }
impl<'a> BufRead for Cursor<&'a mut [u8]> { buffer!(); }
impl<'a> BufRead for Cursor<Vec<u8>> { buffer!(); }

impl<'a> Write for Cursor<&'a mut [u8]> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let pos = cmp::min(self.pos, self.inner.len() as u64);
        let amt = try!((&mut self.inner[(pos as usize)..]).write(data));
        self.pos += amt as u64;
        Ok(amt)
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

impl Write for Cursor<Vec<u8>> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Make sure the internal buffer is as least as big as where we
        // currently are
        let pos = self.position();
        let amt = pos.saturating_sub(self.inner.len() as u64);
        self.inner.extend(repeat(0).take(amt as usize));

        // Figure out what bytes will be used to overwrite what's currently
        // there (left), and what will be appended on the end (right)
        let space = self.inner.len() - pos as usize;
        let (left, right) = buf.split_at(cmp::min(space, buf.len()));
        slice::bytes::copy_memory(&mut self.inner[(pos as usize)..], left);
        self.inner.push_all(right);

        // Bump us forward
        self.set_position(pos + buf.len() as u64);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}


#[cfg(test)]
mod tests {
    use core::prelude::*;

    use io::prelude::*;
    use io::{Cursor, SeekFrom};
    use vec::Vec;

    #[test]
    fn test_vec_writer() {
        let mut writer = Vec::new();
        assert_eq!(writer.write(&[0]), Ok(1));
        assert_eq!(writer.write(&[1, 2, 3]), Ok(3));
        assert_eq!(writer.write(&[4, 5, 6, 7]), Ok(4));
        let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(writer, b);
    }

    #[test]
    fn test_mem_writer() {
        let mut writer = Cursor::new(Vec::new());
        assert_eq!(writer.write(&[0]), Ok(1));
        assert_eq!(writer.write(&[1, 2, 3]), Ok(3));
        assert_eq!(writer.write(&[4, 5, 6, 7]), Ok(4));
        let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(&writer.get_ref()[], b);
    }

    #[test]
    fn test_buf_writer() {
        let mut buf = [0 as u8; 9];
        {
            let mut writer = Cursor::new(&mut buf[..]);
            assert_eq!(writer.position(), 0);
            assert_eq!(writer.write(&[0]), Ok(1));
            assert_eq!(writer.position(), 1);
            assert_eq!(writer.write(&[1, 2, 3]), Ok(3));
            assert_eq!(writer.write(&[4, 5, 6, 7]), Ok(4));
            assert_eq!(writer.position(), 8);
            assert_eq!(writer.write(&[]), Ok(0));
            assert_eq!(writer.position(), 8);

            assert_eq!(writer.write(&[8, 9]), Ok(1));
            assert_eq!(writer.write(&[10]), Ok(0));
        }
        let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(buf, b);
    }

    #[test]
    fn test_buf_writer_seek() {
        let mut buf = [0 as u8; 8];
        {
            let mut writer = Cursor::new(&mut buf[..]);
            assert_eq!(writer.position(), 0);
            assert_eq!(writer.write(&[1]), Ok(1));
            assert_eq!(writer.position(), 1);

            assert_eq!(writer.seek(SeekFrom::Start(2)), Ok(2));
            assert_eq!(writer.position(), 2);
            assert_eq!(writer.write(&[2]), Ok(1));
            assert_eq!(writer.position(), 3);

            assert_eq!(writer.seek(SeekFrom::Current(-2)), Ok(1));
            assert_eq!(writer.position(), 1);
            assert_eq!(writer.write(&[3]), Ok(1));
            assert_eq!(writer.position(), 2);

            assert_eq!(writer.seek(SeekFrom::End(-1)), Ok(7));
            assert_eq!(writer.position(), 7);
            assert_eq!(writer.write(&[4]), Ok(1));
            assert_eq!(writer.position(), 8);

        }
        let b: &[_] = &[1, 3, 2, 0, 0, 0, 0, 4];
        assert_eq!(buf, b);
    }

    #[test]
    fn test_buf_writer_error() {
        let mut buf = [0 as u8; 2];
        let mut writer = Cursor::new(&mut buf[..]);
        assert_eq!(writer.write(&[0]), Ok(1));
        assert_eq!(writer.write(&[0, 0]), Ok(1));
        assert_eq!(writer.write(&[0, 0]), Ok(0));
    }

    #[test]
    fn test_mem_reader() {
        let mut reader = Cursor::new(vec!(0u8, 1, 2, 3, 4, 5, 6, 7));
        let mut buf = [];
        assert_eq!(reader.read(&mut buf), Ok(0));
        assert_eq!(reader.position(), 0);
        let mut buf = [0];
        assert_eq!(reader.read(&mut buf), Ok(1));
        assert_eq!(reader.position(), 1);
        let b: &[_] = &[0];
        assert_eq!(buf, b);
        let mut buf = [0; 4];
        assert_eq!(reader.read(&mut buf), Ok(4));
        assert_eq!(reader.position(), 5);
        let b: &[_] = &[1, 2, 3, 4];
        assert_eq!(buf, b);
        assert_eq!(reader.read(&mut buf), Ok(3));
        let b: &[_] = &[5, 6, 7];
        assert_eq!(&buf[..3], b);
        assert_eq!(reader.read(&mut buf), Ok(0));
    }

    #[test]
    fn read_to_end() {
        let mut reader = Cursor::new(vec!(0u8, 1, 2, 3, 4, 5, 6, 7));
        let mut v = Vec::new();
        reader.read_to_end(&mut v).ok().unwrap();
        assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_slice_reader() {
        let in_buf = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut reader = &mut in_buf.as_slice();
        let mut buf = [];
        assert_eq!(reader.read(&mut buf), Ok(0));
        let mut buf = [0];
        assert_eq!(reader.read(&mut buf), Ok(1));
        assert_eq!(reader.len(), 7);
        let b: &[_] = &[0];
        assert_eq!(buf.as_slice(), b);
        let mut buf = [0; 4];
        assert_eq!(reader.read(&mut buf), Ok(4));
        assert_eq!(reader.len(), 3);
        let b: &[_] = &[1, 2, 3, 4];
        assert_eq!(buf.as_slice(), b);
        assert_eq!(reader.read(&mut buf), Ok(3));
        let b: &[_] = &[5, 6, 7];
        assert_eq!(&buf[..3], b);
        assert_eq!(reader.read(&mut buf), Ok(0));
    }

    #[test]
    fn test_buf_reader() {
        let in_buf = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut reader = Cursor::new(in_buf.as_slice());
        let mut buf = [];
        assert_eq!(reader.read(&mut buf), Ok(0));
        assert_eq!(reader.position(), 0);
        let mut buf = [0];
        assert_eq!(reader.read(&mut buf), Ok(1));
        assert_eq!(reader.position(), 1);
        let b: &[_] = &[0];
        assert_eq!(buf, b);
        let mut buf = [0; 4];
        assert_eq!(reader.read(&mut buf), Ok(4));
        assert_eq!(reader.position(), 5);
        let b: &[_] = &[1, 2, 3, 4];
        assert_eq!(buf, b);
        assert_eq!(reader.read(&mut buf), Ok(3));
        let b: &[_] = &[5, 6, 7];
        assert_eq!(&buf[..3], b);
        assert_eq!(reader.read(&mut buf), Ok(0));
    }

    #[test]
    fn test_read_char() {
        let b = b"Vi\xE1\xBB\x87t";
        let mut c = Cursor::new(b).chars();
        assert_eq!(c.next(), Some(Ok('V')));
        assert_eq!(c.next(), Some(Ok('i')));
        assert_eq!(c.next(), Some(Ok('ệ')));
        assert_eq!(c.next(), Some(Ok('t')));
        assert_eq!(c.next(), None);
    }

    #[test]
    fn test_read_bad_char() {
        let b = b"\x80";
        let mut c = Cursor::new(b).chars();
        assert!(c.next().unwrap().is_err());
    }

    #[test]
    fn seek_past_end() {
        let buf = [0xff];
        let mut r = Cursor::new(&buf[..]);
        assert_eq!(r.seek(SeekFrom::Start(10)), Ok(10));
        assert_eq!(r.read(&mut [0]), Ok(0));

        let mut r = Cursor::new(vec!(10u8));
        assert_eq!(r.seek(SeekFrom::Start(10)), Ok(10));
        assert_eq!(r.read(&mut [0]), Ok(0));

        let mut buf = [0];
        let mut r = Cursor::new(&mut buf[..]);
        assert_eq!(r.seek(SeekFrom::Start(10)), Ok(10));
        assert_eq!(r.write(&[3]), Ok(0));
    }

    #[test]
    fn seek_before_0() {
        let buf = [0xff_u8];
        let mut r = Cursor::new(&buf[..]);
        assert!(r.seek(SeekFrom::End(-2)).is_err());

        let mut r = Cursor::new(vec!(10u8));
        assert!(r.seek(SeekFrom::End(-2)).is_err());

        let mut buf = [0];
        let mut r = Cursor::new(&mut buf[..]);
        assert!(r.seek(SeekFrom::End(-2)).is_err());
    }

    #[test]
    fn test_seekable_mem_writer() {
        let mut writer = Cursor::new(Vec::<u8>::new());
        assert_eq!(writer.position(), 0);
        assert_eq!(writer.write(&[0]), Ok(1));
        assert_eq!(writer.position(), 1);
        assert_eq!(writer.write(&[1, 2, 3]), Ok(3));
        assert_eq!(writer.write(&[4, 5, 6, 7]), Ok(4));
        assert_eq!(writer.position(), 8);
        let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(&writer.get_ref()[], b);

        assert_eq!(writer.seek(SeekFrom::Start(0)), Ok(0));
        assert_eq!(writer.position(), 0);
        assert_eq!(writer.write(&[3, 4]), Ok(2));
        let b: &[_] = &[3, 4, 2, 3, 4, 5, 6, 7];
        assert_eq!(&writer.get_ref()[], b);

        assert_eq!(writer.seek(SeekFrom::Current(1)), Ok(3));
        assert_eq!(writer.write(&[0, 1]), Ok(2));
        let b: &[_] = &[3, 4, 2, 0, 1, 5, 6, 7];
        assert_eq!(&writer.get_ref()[], b);

        assert_eq!(writer.seek(SeekFrom::End(-1)), Ok(7));
        assert_eq!(writer.write(&[1, 2]), Ok(2));
        let b: &[_] = &[3, 4, 2, 0, 1, 5, 6, 1, 2];
        assert_eq!(&writer.get_ref()[], b);

        assert_eq!(writer.seek(SeekFrom::End(1)), Ok(10));
        assert_eq!(writer.write(&[1]), Ok(1));
        let b: &[_] = &[3, 4, 2, 0, 1, 5, 6, 1, 2, 0, 1];
        assert_eq!(&writer.get_ref()[], b);
    }

    #[test]
    fn vec_seek_past_end() {
        let mut r = Cursor::new(Vec::new());
        assert_eq!(r.seek(SeekFrom::Start(10)), Ok(10));
        assert_eq!(r.write(&[3]), Ok(1));
    }

    #[test]
    fn vec_seek_before_0() {
        let mut r = Cursor::new(Vec::new());
        assert!(r.seek(SeekFrom::End(-2)).is_err());
    }
}
