// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! Utility implementations of Reader and Writer */

use prelude::*;
use cmp;
use io;
use slice::bytes::MutableByteVector;

/// Wraps a `Reader`, limiting the number of bytes that can be read from it.
pub struct LimitReader<R> {
    limit: uint,
    inner: R
}

impl<R: Reader> LimitReader<R> {
    /// Creates a new `LimitReader`
    pub fn new(r: R, limit: uint) -> LimitReader<R> { unimplemented!() }

    /// Consumes the `LimitReader`, returning the underlying `Reader`.
    pub fn unwrap(self) -> R { unimplemented!() }

    /// Returns the number of bytes that can be read before the `LimitReader`
    /// will return EOF.
    ///
    /// # Note
    ///
    /// The reader may reach EOF after reading fewer bytes than indicated by
    /// this method if the underlying reader reaches EOF.
    pub fn limit(&self) -> uint { unimplemented!() }
}

impl<R: Reader> Reader for LimitReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> { unimplemented!() }
}

impl<R: Buffer> Buffer for LimitReader<R> {
    fn fill_buf<'a>(&'a mut self) -> io::IoResult<&'a [u8]> { unimplemented!() }

    fn consume(&mut self, amt: uint) { unimplemented!() }

}

/// A `Writer` which ignores bytes written to it, like /dev/null.
pub struct NullWriter;

impl Writer for NullWriter {
    #[inline]
    fn write(&mut self, _buf: &[u8]) -> io::IoResult<()> { unimplemented!() }
}

/// A `Reader` which returns an infinite stream of 0 bytes, like /dev/zero.
pub struct ZeroReader;

impl Reader for ZeroReader {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> { unimplemented!() }
}

impl Buffer for ZeroReader {
    fn fill_buf<'a>(&'a mut self) -> io::IoResult<&'a [u8]> { unimplemented!() }

    fn consume(&mut self, _amt: uint) { unimplemented!() }
}

/// A `Reader` which is always at EOF, like /dev/null.
pub struct NullReader;

impl Reader for NullReader {
    #[inline]
    fn read(&mut self, _buf: &mut [u8]) -> io::IoResult<uint> { unimplemented!() }
}

impl Buffer for NullReader {
    fn fill_buf<'a>(&'a mut self) -> io::IoResult<&'a [u8]> { unimplemented!() }
    fn consume(&mut self, _amt: uint) { unimplemented!() }
}

/// A `Writer` which multiplexes writes to a set of `Writer`s.
///
/// The `Writer`s are delegated to in order. If any `Writer` returns an error,
/// that error is returned immediately and remaining `Writer`s are not called.
pub struct MultiWriter {
    writers: Vec<Box<Writer+'static>>
}

impl MultiWriter {
    /// Creates a new `MultiWriter`
    pub fn new(writers: Vec<Box<Writer+'static>>) -> MultiWriter { unimplemented!() }
}

impl Writer for MultiWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::IoResult<()> { unimplemented!() }

    #[inline]
    fn flush(&mut self) -> io::IoResult<()> { unimplemented!() }
}

/// A `Reader` which chains input from multiple `Reader`s, reading each to
/// completion before moving onto the next.
pub struct ChainedReader<I, R> {
    readers: I,
    cur_reader: Option<R>,
}

impl<R: Reader, I: Iterator<R>> ChainedReader<I, R> {
    /// Creates a new `ChainedReader`
    pub fn new(mut readers: I) -> ChainedReader<I, R> { unimplemented!() }
}

impl<R: Reader, I: Iterator<R>> Reader for ChainedReader<I, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> { unimplemented!() }
}

/// A `Reader` which forwards input from another `Reader`, passing it along to
/// a `Writer` as well. Similar to the `tee(1)` command.
pub struct TeeReader<R, W> {
    reader: R,
    writer: W,
}

impl<R: Reader, W: Writer> TeeReader<R, W> {
    /// Creates a new `TeeReader`
    pub fn new(r: R, w: W) -> TeeReader<R, W> { unimplemented!() }

    /// Consumes the `TeeReader`, returning the underlying `Reader` and
    /// `Writer`.
    pub fn unwrap(self) -> (R, W) { unimplemented!() }
}

impl<R: Reader, W: Writer> Reader for TeeReader<R, W> {
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> { unimplemented!() }
}

/// Copies all data from a `Reader` to a `Writer`.
pub fn copy<R: Reader, W: Writer>(r: &mut R, w: &mut W) -> io::IoResult<()> { unimplemented!() }

/// An adaptor converting an `Iterator<u8>` to a `Reader`.
pub struct IterReader<T> {
    iter: T,
}

impl<T: Iterator<u8>> IterReader<T> {
    /// Creates a new `IterReader` which will read from the specified
    /// `Iterator`.
    pub fn new(iter: T) -> IterReader<T> { unimplemented!() }
}

impl<T: Iterator<u8>> Reader for IterReader<T> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::IoResult<uint> { unimplemented!() }
}
