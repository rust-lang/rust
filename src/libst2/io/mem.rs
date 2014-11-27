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

#![allow(deprecated)]

use cmp::min;
use option::None;
use result::{Err, Ok};
use io;
use io::{Reader, Writer, Seek, Buffer, IoError, SeekStyle, IoResult};
use slice::{mod, AsSlice, SlicePrelude};
use vec::Vec;

const BUF_CAPACITY: uint = 128;

fn combine(seek: SeekStyle, cur: uint, end: uint, offset: i64) -> IoResult<u64> { unimplemented!() }

impl Writer for Vec<u8> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
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
/// w.write(&[0, 1, 2]);
///
/// assert_eq!(w.unwrap(), vec!(0, 1, 2));
/// ```
#[deprecated = "use the Vec<u8> Writer implementation directly"]
#[deriving(Clone)]
pub struct MemWriter {
    buf: Vec<u8>,
}

impl MemWriter {
    /// Create a new `MemWriter`.
    #[inline]
    pub fn new() -> MemWriter { unimplemented!() }
    /// Create a new `MemWriter`, allocating at least `n` bytes for
    /// the internal buffer.
    #[inline]
    pub fn with_capacity(n: uint) -> MemWriter { unimplemented!() }
    /// Create a new `MemWriter` that will append to an existing `Vec`.
    #[inline]
    pub fn from_vec(buf: Vec<u8>) -> MemWriter { unimplemented!() }

    /// Acquires an immutable reference to the underlying buffer of this
    /// `MemWriter`.
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a [u8] { unimplemented!() }

    /// Unwraps this `MemWriter`, returning the underlying buffer
    #[inline]
    pub fn unwrap(self) -> Vec<u8> { unimplemented!() }
}

impl Writer for MemWriter {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
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
    pub fn new(buf: Vec<u8>) -> MemReader { unimplemented!() }

    /// Tests whether this reader has read all bytes in its buffer.
    ///
    /// If `true`, then this will no longer return bytes from `read`.
    #[inline]
    pub fn eof(&self) -> bool { unimplemented!() }

    /// Acquires an immutable reference to the underlying buffer of this
    /// `MemReader`.
    ///
    /// No method is exposed for acquiring a mutable reference to the buffer
    /// because it could corrupt the state of this `MemReader`.
    #[inline]
    pub fn get_ref<'a>(&'a self) -> &'a [u8] { unimplemented!() }

    /// Unwraps this `MemReader`, returning the underlying buffer
    #[inline]
    pub fn unwrap(self) -> Vec<u8> { unimplemented!() }
}

impl Reader for MemReader {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl Seek for MemReader {
    #[inline]
    fn tell(&self) -> IoResult<u64> { unimplemented!() }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> { unimplemented!() }
}

impl Buffer for MemReader {
    #[inline]
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { unimplemented!() }

    #[inline]
    fn consume(&mut self, amt: uint) { unimplemented!() }
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
///     let mut w = BufWriter::new(&mut buf);
///     w.write(&[0, 1, 2]);
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
    pub fn new<'a>(buf: &'a mut [u8]) -> BufWriter<'a> { unimplemented!() }
}

impl<'a> Writer for BufWriter<'a> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
}

impl<'a> Seek for BufWriter<'a> {
    #[inline]
    fn tell(&self) -> IoResult<u64> { unimplemented!() }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> { unimplemented!() }
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
/// let mut r = BufReader::new(&mut buf);
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
    pub fn new<'a>(buf: &'a [u8]) -> BufReader<'a> { unimplemented!() }

    /// Tests whether this reader has read all bytes in its buffer.
    ///
    /// If `true`, then this will no longer return bytes from `read`.
    #[inline]
    pub fn eof(&self) -> bool { unimplemented!() }
}

impl<'a> Reader for BufReader<'a> {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl<'a> Seek for BufReader<'a> {
    #[inline]
    fn tell(&self) -> IoResult<u64> { unimplemented!() }

    #[inline]
    fn seek(&mut self, pos: i64, style: SeekStyle) -> IoResult<()> { unimplemented!() }
}

impl<'a> Buffer for BufReader<'a> {
    #[inline]
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { unimplemented!() }

    #[inline]
    fn consume(&mut self, amt: uint) { unimplemented!() }
}
