// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Buffering wrappers for I/O traits
//!
//! It can be excessively inefficient to work directly with a `Reader` or
//! `Writer`. Every call to `read` or `write` on `TcpStream` results in a
//! system call, for example. This module provides structures that wrap
//! `Readers`, `Writers`, and `Streams` and buffer input and output to them.
//!
//! # Examples
//!
//! ```
//! let tcp_stream = TcpStream::connect(addr);
//! let reader = BufferedReader::new(tcp_stream);
//!
//! let mut buf: ~[u8] = vec::from_elem(100, 0u8);
//! match reader.read(buf.as_slice()) {
//!     Some(nread) => println!("Read {} bytes", nread),
//!     None => println!("At the end of the stream!")
//! }
//! ```
//!
//! ```
//! let tcp_stream = TcpStream::connect(addr);
//! let writer = BufferedWriter::new(tcp_stream);
//!
//! writer.write("hello, world".as_bytes());
//! writer.flush();
//! ```
//!
//! ```
//! let tcp_stream = TcpStream::connect(addr);
//! let stream = BufferedStream::new(tcp_stream);
//!
//! stream.write("hello, world".as_bytes());
//! stream.flush();
//!
//! let mut buf = vec::from_elem(100, 0u8);
//! match stream.read(buf.as_slice()) {
//!     Some(nread) => println!("Read {} bytes", nread),
//!     None => println!("At the end of the stream!")
//! }
//! ```
//!

use prelude::*;

use num;
use vec;
use super::Stream;

// libuv recommends 64k buffers to maximize throughput
// https://groups.google.com/forum/#!topic/libuv/oQO1HJAIDdA
static DEFAULT_CAPACITY: uint = 64 * 1024;

/// Wraps a Reader and buffers input from it
pub struct BufferedReader<R> {
    priv inner: R,
    priv buf: ~[u8],
    priv pos: uint,
    priv cap: uint,
    priv eof: bool,
}

impl<R: Reader> BufferedReader<R> {
    /// Creates a new `BufferedReader` with with the specified buffer capacity
    pub fn with_capacity(cap: uint, inner: R) -> BufferedReader<R> {
        // It's *much* faster to create an uninitialized buffer than it is to
        // fill everything in with 0. This buffer is entirely an implementation
        // detail and is never exposed, so we're safe to not initialize
        // everything up-front. This allows creation of BufferedReader instances
        // to be very cheap (large mallocs are not nearly as expensive as large
        // callocs).
        let mut buf = vec::with_capacity(cap);
        unsafe { buf.set_len(cap); }
        BufferedReader {
            inner: inner,
            buf: buf,
            pos: 0,
            cap: 0,
            eof: false,
        }
    }

    /// Creates a new `BufferedReader` with a default buffer capacity
    pub fn new(inner: R) -> BufferedReader<R> {
        BufferedReader::with_capacity(DEFAULT_CAPACITY, inner)
    }

    /// Gets a reference to the underlying reader.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a R { &self.inner }

    /// Unwraps this buffer, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost.
    pub fn unwrap(self) -> R { self.inner }
}

impl<R: Reader> Buffer for BufferedReader<R> {
    fn fill<'a>(&'a mut self) -> &'a [u8] {
        if self.pos == self.cap {
            match self.inner.read(self.buf) {
                Some(cap) => {
                    self.pos = 0;
                    self.cap = cap;
                }
                None => { self.eof = true; }
            }
        }
        return self.buf.slice(self.pos, self.cap);
    }

    fn consume(&mut self, amt: uint) {
        self.pos += amt;
        assert!(self.pos <= self.cap);
    }
}

impl<R: Reader> Reader for BufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        let nread = {
            let available = self.fill();
            let nread = num::min(available.len(), buf.len());
            vec::bytes::copy_memory(buf, available.slice_to(nread));
            nread
        };
        self.pos += nread;
        if nread == 0 && buf.len() != 0 && self.eof {
            return None;
        }
        Some(nread)
    }
}

/// Wraps a Writer and buffers output to it
///
/// Note that `BufferedWriter` will NOT flush its buffer when dropped.
pub struct BufferedWriter<W> {
    priv inner: W,
    priv buf: ~[u8],
    priv pos: uint
}

impl<W: Writer> BufferedWriter<W> {
    /// Creates a new `BufferedWriter` with with the specified buffer capacity
    pub fn with_capacity(cap: uint, inner: W) -> BufferedWriter<W> {
        // See comments in BufferedReader for why this uses unsafe code.
        let mut buf = vec::with_capacity(cap);
        unsafe { buf.set_len(cap); }
        BufferedWriter {
            inner: inner,
            buf: buf,
            pos: 0
        }
    }

    /// Creates a new `BufferedWriter` with a default buffer capacity
    pub fn new(inner: W) -> BufferedWriter<W> {
        BufferedWriter::with_capacity(DEFAULT_CAPACITY, inner)
    }

    fn flush_buf(&mut self) {
        if self.pos != 0 {
            self.inner.write(self.buf.slice_to(self.pos));
            self.pos = 0;
        }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W { &self.inner }

    /// Unwraps this buffer, returning the underlying writer.
    ///
    /// The buffer is flushed before returning the writer.
    pub fn unwrap(mut self) -> W {
        self.flush_buf();
        self.inner
    }
}

impl<W: Writer> Writer for BufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) {
        if self.pos + buf.len() > self.buf.len() {
            self.flush_buf();
        }

        if buf.len() > self.buf.len() {
            self.inner.write(buf);
        } else {
            let dst = self.buf.mut_slice_from(self.pos);
            vec::bytes::copy_memory(dst, buf);
            self.pos += buf.len();
        }
    }

    fn flush(&mut self) {
        self.flush_buf();
        self.inner.flush();
    }
}

/// Wraps a Writer and buffers output to it, flushing whenever a newline (`0x0a`,
/// `'\n'`) is detected.
///
/// Note that this structure does NOT flush the output when dropped.
pub struct LineBufferedWriter<W> {
    priv inner: BufferedWriter<W>,
}

impl<W: Writer> LineBufferedWriter<W> {
    /// Creates a new `LineBufferedWriter`
    pub fn new(inner: W) -> LineBufferedWriter<W> {
        // Lines typically aren't that long, don't use a giant buffer
        LineBufferedWriter {
            inner: BufferedWriter::with_capacity(1024, inner)
        }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W { self.inner.get_ref() }

    /// Unwraps this buffer, returning the underlying writer.
    ///
    /// The internal buffer is flushed before returning the writer.
    pub fn unwrap(mut self) -> W { self.inner.unwrap() }
}

impl<W: Writer> Writer for LineBufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) {
        match buf.iter().rposition(|&b| b == '\n' as u8) {
            Some(i) => {
                self.inner.write(buf.slice_to(i + 1));
                self.inner.flush();
                self.inner.write(buf.slice_from(i + 1));
            }
            None => self.inner.write(buf),
        }
    }

    fn flush(&mut self) { self.inner.flush() }
}

struct InternalBufferedWriter<W>(BufferedWriter<W>);

impl<W> InternalBufferedWriter<W> {
    fn get_mut_ref<'a>(&'a mut self) -> &'a mut BufferedWriter<W> {
        let InternalBufferedWriter(ref mut w) = *self;
        return w;
    }
}

impl<W: Reader> Reader for InternalBufferedWriter<W> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.get_mut_ref().inner.read(buf) }
}

/// Wraps a Stream and buffers input and output to and from it
///
/// Note that `BufferedStream` will NOT flush its output buffer when dropped.
pub struct BufferedStream<S> {
    priv inner: BufferedReader<InternalBufferedWriter<S>>
}

impl<S: Stream> BufferedStream<S> {
    /// Creates a new buffered stream with explicitly listed capacities for the
    /// reader/writer buffer.
    pub fn with_capacities(reader_cap: uint, writer_cap: uint, inner: S)
                           -> BufferedStream<S> {
        let writer = BufferedWriter::with_capacity(writer_cap, inner);
        let internal_writer = InternalBufferedWriter(writer);
        let reader = BufferedReader::with_capacity(reader_cap,
                                                   internal_writer);
        BufferedStream { inner: reader }
    }

    /// Creates a new buffered stream with the default reader/writer buffer
    /// capacities.
    pub fn new(inner: S) -> BufferedStream<S> {
        BufferedStream::with_capacities(DEFAULT_CAPACITY, DEFAULT_CAPACITY,
                                        inner)
    }

    /// Gets a reference to the underlying stream.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a S {
        let InternalBufferedWriter(ref w) = self.inner.inner;
        w.get_ref()
    }

    /// Unwraps this buffer, returning the underlying stream.
    ///
    /// The internal buffer is flushed before returning the stream. Any leftover
    /// data in the read buffer is lost.
    pub fn unwrap(self) -> S {
        let InternalBufferedWriter(w) = self.inner.inner;
        w.unwrap()
    }
}

impl<S: Stream> Buffer for BufferedStream<S> {
    fn fill<'a>(&'a mut self) -> &'a [u8] { self.inner.fill() }
    fn consume(&mut self, amt: uint) { self.inner.consume(amt) }
}

impl<S: Stream> Reader for BufferedStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> { self.inner.read(buf) }
}

impl<S: Stream> Writer for BufferedStream<S> {
    fn write(&mut self, buf: &[u8]) { self.inner.inner.get_mut_ref().write(buf) }
    fn flush(&mut self) { self.inner.inner.get_mut_ref().flush() }
}

#[cfg(test)]
mod test {
    use io;
    use prelude::*;
    use super::*;
    use super::super::mem::{MemReader, MemWriter, BufReader};
    use Harness = extra::test::BenchHarness;

    /// A type, free to create, primarily intended for benchmarking creation of wrappers that, just
    /// for construction, don't need a Reader/Writer that does anything useful. Is equivalent to
    /// `/dev/null` in semantics.
    #[deriving(Clone,Eq,Ord)]
    pub struct NullStream;

    impl Reader for NullStream {
        fn read(&mut self, _: &mut [u8]) -> Option<uint> {
            None
        }
    }

    impl Writer for NullStream {
        fn write(&mut self, _: &[u8]) { }
    }

    /// A dummy reader intended at testing short-reads propagation.
    pub struct ShortReader {
        priv lengths: ~[uint],
    }

    impl Reader for ShortReader {
        fn read(&mut self, _: &mut [u8]) -> Option<uint> {
            self.lengths.shift_opt()
        }
    }

    #[test]
    fn test_buffered_reader() {
        let inner = MemReader::new(~[0, 1, 2, 3, 4]);
        let mut reader = BufferedReader::with_capacity(2, inner);

        let mut buf = [0, 0, 0];
        let nread = reader.read(buf);
        assert_eq!(Some(2), nread);
        assert_eq!([0, 1, 0], buf);

        let mut buf = [0];
        let nread = reader.read(buf);
        assert_eq!(Some(1), nread);
        assert_eq!([2], buf);

        let mut buf = [0, 0, 0];
        let nread = reader.read(buf);
        assert_eq!(Some(1), nread);
        assert_eq!([3, 0, 0], buf);

        let nread = reader.read(buf);
        assert_eq!(Some(1), nread);
        assert_eq!([4, 0, 0], buf);

        assert_eq!(None, reader.read(buf));
    }

    #[test]
    fn test_buffered_writer() {
        let inner = MemWriter::new();
        let mut writer = BufferedWriter::with_capacity(2, inner);

        writer.write([0, 1]);
        assert_eq!([], writer.get_ref().get_ref());

        writer.write([2]);
        assert_eq!([0, 1], writer.get_ref().get_ref());

        writer.write([3]);
        assert_eq!([0, 1], writer.get_ref().get_ref());

        writer.flush();
        assert_eq!([0, 1, 2, 3], writer.get_ref().get_ref());

        writer.write([4]);
        writer.write([5]);
        assert_eq!([0, 1, 2, 3], writer.get_ref().get_ref());

        writer.write([6]);
        assert_eq!([0, 1, 2, 3, 4, 5],
                   writer.get_ref().get_ref());

        writer.write([7, 8]);
        assert_eq!([0, 1, 2, 3, 4, 5, 6],
                   writer.get_ref().get_ref());

        writer.write([9, 10, 11]);
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   writer.get_ref().get_ref());

        writer.flush();
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   writer.get_ref().get_ref());
    }

    #[test]
    fn test_buffered_writer_inner_flushes() {
        let mut w = BufferedWriter::with_capacity(3, MemWriter::new());
        w.write([0, 1]);
        assert_eq!([], w.get_ref().get_ref());
        let w = w.unwrap();
        assert_eq!([0, 1], w.get_ref());
    }

    // This is just here to make sure that we don't infinite loop in the
    // newtype struct autoderef weirdness
    #[test]
    fn test_buffered_stream() {
        struct S;

        impl io::Writer for S {
            fn write(&mut self, _: &[u8]) {}
        }

        impl io::Reader for S {
            fn read(&mut self, _: &mut [u8]) -> Option<uint> { None }
        }

        let mut stream = BufferedStream::new(S);
        let mut buf = [];
        stream.read(buf);
        stream.write(buf);
        stream.flush();
    }

    #[test]
    fn test_read_until() {
        let inner = MemReader::new(~[0, 1, 2, 1, 0]);
        let mut reader = BufferedReader::with_capacity(2, inner);
        assert_eq!(reader.read_until(0), Some(~[0]));
        assert_eq!(reader.read_until(2), Some(~[1, 2]));
        assert_eq!(reader.read_until(1), Some(~[1]));
        assert_eq!(reader.read_until(8), Some(~[0]));
        assert_eq!(reader.read_until(9), None);
    }

    #[test]
    fn test_line_buffer() {
        let mut writer = LineBufferedWriter::new(MemWriter::new());
        writer.write([0]);
        assert_eq!(writer.get_ref().get_ref(), []);
        writer.write([1]);
        assert_eq!(writer.get_ref().get_ref(), []);
        writer.flush();
        assert_eq!(writer.get_ref().get_ref(), [0, 1]);
        writer.write([0, '\n' as u8, 1, '\n' as u8, 2]);
        assert_eq!(writer.get_ref().get_ref(),
            [0, 1, 0, '\n' as u8, 1, '\n' as u8]);
        writer.flush();
        assert_eq!(writer.get_ref().get_ref(),
            [0, 1, 0, '\n' as u8, 1, '\n' as u8, 2]);
        writer.write([3, '\n' as u8]);
        assert_eq!(writer.get_ref().get_ref(),
            [0, 1, 0, '\n' as u8, 1, '\n' as u8, 2, 3, '\n' as u8]);
    }

    #[test]
    fn test_read_line() {
        let in_buf = MemReader::new(bytes!("a\nb\nc").to_owned());
        let mut reader = BufferedReader::with_capacity(2, in_buf);
        assert_eq!(reader.read_line(), Some(~"a\n"));
        assert_eq!(reader.read_line(), Some(~"b\n"));
        assert_eq!(reader.read_line(), Some(~"c"));
        assert_eq!(reader.read_line(), None);
    }

    #[test]
    fn test_lines() {
        let in_buf = MemReader::new(bytes!("a\nb\nc").to_owned());
        let mut reader = BufferedReader::with_capacity(2, in_buf);
        let mut it = reader.lines();
        assert_eq!(it.next(), Some(~"a\n"));
        assert_eq!(it.next(), Some(~"b\n"));
        assert_eq!(it.next(), Some(~"c"));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_short_reads() {
        let inner = ShortReader{lengths: ~[0, 1, 2, 0, 1, 0]};
        let mut reader = BufferedReader::new(inner);
        let mut buf = [0, 0];
        assert_eq!(reader.read(buf), Some(0));
        assert_eq!(reader.read(buf), Some(1));
        assert_eq!(reader.read(buf), Some(2));
        assert_eq!(reader.read(buf), Some(0));
        assert_eq!(reader.read(buf), Some(1));
        assert_eq!(reader.read(buf), Some(0));
        assert_eq!(reader.read(buf), None);
    }

    #[test]
    fn read_char_buffered() {
        let buf = [195u8, 159u8];
        let mut reader = BufferedReader::with_capacity(1, BufReader::new(buf));
        assert_eq!(reader.read_char(), Some('ß'));
    }

    #[bench]
    fn bench_buffered_reader(bh: &mut Harness) {
        bh.iter(|| {
            BufferedReader::new(NullStream);
        });
    }

    #[bench]
    fn bench_buffered_writer(bh: &mut Harness) {
        bh.iter(|| {
            BufferedWriter::new(NullStream);
        });
    }

    #[bench]
    fn bench_buffered_stream(bh: &mut Harness) {
        bh.iter(|| {
            BufferedStream::new(NullStream);
        });
    }
}
