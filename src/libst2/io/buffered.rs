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
// ignore-lexer-test FIXME #15883

//! Buffering wrappers for I/O traits

use cmp;
use io::{Reader, Writer, Stream, Buffer, DEFAULT_BUF_SIZE, IoResult};
use iter::ExactSize;
use ops::Drop;
use option::{Some, None, Option};
use result::{Ok, Err};
use slice::{SlicePrelude};
use slice;
use vec::Vec;

/// Wraps a Reader and buffers input from it
///
/// It can be excessively inefficient to work directly with a `Reader`. For
/// example, every call to `read` on `TcpStream` results in a system call. A
/// `BufferedReader` performs large, infrequent reads on the underlying
/// `Reader` and maintains an in-memory buffer of the results.
///
/// # Example
///
/// ```rust
/// use std::io::{BufferedReader, File};
///
/// let file = File::open(&Path::new("message.txt"));
/// let mut reader = BufferedReader::new(file);
///
/// let mut buf = [0, ..100];
/// match reader.read(&mut buf) {
///     Ok(nread) => println!("Read {} bytes", nread),
///     Err(e) => println!("error reading: {}", e)
/// }
/// ```
pub struct BufferedReader<R> {
    inner: R,
    buf: Vec<u8>,
    pos: uint,
    cap: uint,
}

impl<R: Reader> BufferedReader<R> {
    /// Creates a new `BufferedReader` with the specified buffer capacity
    pub fn with_capacity(cap: uint, inner: R) -> BufferedReader<R> { unimplemented!() }

    /// Creates a new `BufferedReader` with a default buffer capacity
    pub fn new(inner: R) -> BufferedReader<R> { unimplemented!() }

    /// Gets a reference to the underlying reader.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a R { unimplemented!() }

    /// Unwraps this `BufferedReader`, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost.
    pub fn unwrap(self) -> R { unimplemented!() }
}

impl<R: Reader> Buffer for BufferedReader<R> {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { unimplemented!() }

    fn consume(&mut self, amt: uint) { unimplemented!() }
}

impl<R: Reader> Reader for BufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

/// Wraps a Writer and buffers output to it
///
/// It can be excessively inefficient to work directly with a `Writer`. For
/// example, every call to `write` on `TcpStream` results in a system call. A
/// `BufferedWriter` keeps an in memory buffer of data and writes it to the
/// underlying `Writer` in large, infrequent batches.
///
/// This writer will be flushed when it is dropped.
///
/// # Example
///
/// ```rust
/// use std::io::{BufferedWriter, File};
///
/// let file = File::create(&Path::new("message.txt")).unwrap();
/// let mut writer = BufferedWriter::new(file);
///
/// writer.write_str("hello, world").unwrap();
/// writer.flush().unwrap();
/// ```
pub struct BufferedWriter<W> {
    inner: Option<W>,
    buf: Vec<u8>,
    pos: uint
}

impl<W: Writer> BufferedWriter<W> {
    /// Creates a new `BufferedWriter` with the specified buffer capacity
    pub fn with_capacity(cap: uint, inner: W) -> BufferedWriter<W> { unimplemented!() }

    /// Creates a new `BufferedWriter` with a default buffer capacity
    pub fn new(inner: W) -> BufferedWriter<W> { unimplemented!() }

    fn flush_buf(&mut self) -> IoResult<()> { unimplemented!() }

    /// Gets a reference to the underlying writer.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W { unimplemented!() }

    /// Unwraps this `BufferedWriter`, returning the underlying writer.
    ///
    /// The buffer is flushed before returning the writer.
    pub fn unwrap(mut self) -> W { unimplemented!() }
}

impl<W: Writer> Writer for BufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}

#[unsafe_destructor]
impl<W: Writer> Drop for BufferedWriter<W> {
    fn drop(&mut self) { unimplemented!() }
}

/// Wraps a Writer and buffers output to it, flushing whenever a newline (`0x0a`,
/// `'\n'`) is detected.
///
/// This writer will be flushed when it is dropped.
pub struct LineBufferedWriter<W> {
    inner: BufferedWriter<W>,
}

impl<W: Writer> LineBufferedWriter<W> {
    /// Creates a new `LineBufferedWriter`
    pub fn new(inner: W) -> LineBufferedWriter<W> { unimplemented!() }

    /// Gets a reference to the underlying writer.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W { unimplemented!() }

    /// Unwraps this `LineBufferedWriter`, returning the underlying writer.
    ///
    /// The internal buffer is flushed before returning the writer.
    pub fn unwrap(self) -> W { unimplemented!() }
}

impl<W: Writer> Writer for LineBufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }

    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}

struct InternalBufferedWriter<W>(BufferedWriter<W>);

impl<W> InternalBufferedWriter<W> {
    fn get_mut<'a>(&'a mut self) -> &'a mut BufferedWriter<W> { unimplemented!() }
}

impl<W: Reader> Reader for InternalBufferedWriter<W> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

/// Wraps a Stream and buffers input and output to and from it.
///
/// It can be excessively inefficient to work directly with a `Stream`. For
/// example, every call to `read` or `write` on `TcpStream` results in a system
/// call. A `BufferedStream` keeps in memory buffers of data, making large,
/// infrequent calls to `read` and `write` on the underlying `Stream`.
///
/// The output half will be flushed when this stream is dropped.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_must_use)]
/// use std::io::{BufferedStream, File};
///
/// let file = File::open(&Path::new("message.txt"));
/// let mut stream = BufferedStream::new(file);
///
/// stream.write("hello, world".as_bytes());
/// stream.flush();
///
/// let mut buf = [0, ..100];
/// match stream.read(&mut buf) {
///     Ok(nread) => println!("Read {} bytes", nread),
///     Err(e) => println!("error reading: {}", e)
/// }
/// ```
pub struct BufferedStream<S> {
    inner: BufferedReader<InternalBufferedWriter<S>>
}

impl<S: Stream> BufferedStream<S> {
    /// Creates a new buffered stream with explicitly listed capacities for the
    /// reader/writer buffer.
    pub fn with_capacities(reader_cap: uint, writer_cap: uint, inner: S)
                           -> BufferedStream<S> { unimplemented!() }

    /// Creates a new buffered stream with the default reader/writer buffer
    /// capacities.
    pub fn new(inner: S) -> BufferedStream<S> { unimplemented!() }

    /// Gets a reference to the underlying stream.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a S { unimplemented!() }

    /// Unwraps this `BufferedStream`, returning the underlying stream.
    ///
    /// The internal buffer is flushed before returning the stream. Any leftover
    /// data in the read buffer is lost.
    pub fn unwrap(self) -> S { unimplemented!() }
}

impl<S: Stream> Buffer for BufferedStream<S> {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { unimplemented!() }
    fn consume(&mut self, amt: uint) { unimplemented!() }
}

impl<S: Stream> Reader for BufferedStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> { unimplemented!() }
}

impl<S: Stream> Writer for BufferedStream<S> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> { unimplemented!() }
    fn flush(&mut self) -> IoResult<()> { unimplemented!() }
}
