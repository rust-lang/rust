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
use str;
use super::{Reader, Writer, Stream, Decorator};

// libuv recommends 64k buffers to maximize throughput
// https://groups.google.com/forum/#!topic/libuv/oQO1HJAIDdA
static DEFAULT_CAPACITY: uint = 64 * 1024;

/// Wraps a Reader and buffers input from it
pub struct BufferedReader<R> {
    priv inner: R,
    priv buf: ~[u8],
    priv pos: uint,
    priv cap: uint
}

impl<R: Reader> BufferedReader<R> {
    /// Creates a new `BufferedReader` with with the specified buffer capacity
    pub fn with_capacity(cap: uint, inner: R) -> BufferedReader<R> {
        BufferedReader {
            inner: inner,
            buf: vec::from_elem(cap, 0u8),
            pos: 0,
            cap: 0
        }
    }

    /// Creates a new `BufferedReader` with a default buffer capacity
    pub fn new(inner: R) -> BufferedReader<R> {
        BufferedReader::with_capacity(DEFAULT_CAPACITY, inner)
    }

    /// Reads the next line of input, interpreted as a sequence of utf-8
    /// encoded unicode codepoints. If a newline is encountered, then the
    /// newline is contained in the returned string.
    pub fn read_line(&mut self) -> Option<~str> {
        self.read_until('\n' as u8).map(str::from_utf8_owned)
    }

    /// Reads a sequence of bytes leading up to a specified delimeter. Once the
    /// specified byte is encountered, reading ceases and the bytes up to and
    /// including the delimiter are returned.
    pub fn read_until(&mut self, byte: u8) -> Option<~[u8]> {
        let mut res = ~[];
        let mut used;
        loop {
            {
                let available = self.fill_buffer();
                match available.iter().position(|&b| b == byte) {
                    Some(i) => {
                        res.push_all(available.slice_to(i + 1));
                        used = i + 1;
                        break
                    }
                    None => {
                        res.push_all(available);
                        used = available.len();
                    }
                }
            }
            if used == 0 {
                break
            }
            self.pos += used;
        }
        self.pos += used;
        return if res.len() == 0 {None} else {Some(res)};
    }

    fn fill_buffer<'a>(&'a mut self) -> &'a [u8] {
        if self.pos == self.cap {
            match self.inner.read(self.buf) {
                Some(cap) => {
                    self.pos = 0;
                    self.cap = cap;
                }
                None => {}
            }
        }
        return self.buf.slice(self.pos, self.cap);
    }
}

impl<R: Reader> Reader for BufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        let nread = {
            let available = self.fill_buffer();
            if available.len() == 0 {
                return None;
            }
            let nread = num::min(available.len(), buf.len());
            vec::bytes::copy_memory(buf, available, nread);
            nread
        };
        self.pos += nread;
        Some(nread)
    }

    fn eof(&mut self) -> bool {
        self.pos == self.cap && self.inner.eof()
    }
}

impl<R: Reader> Decorator<R> for BufferedReader<R> {
    fn inner(self) -> R {
        self.inner
    }

    fn inner_ref<'a>(&'a self) -> &'a R {
        &self.inner
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut R {
        &mut self.inner
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
        BufferedWriter {
            inner: inner,
            buf: vec::from_elem(cap, 0u8),
            pos: 0
        }
    }

    /// Creates a new `BufferedWriter` with a default buffer capacity
    pub fn new(inner: W) -> BufferedWriter<W> {
        BufferedWriter::with_capacity(DEFAULT_CAPACITY, inner)
    }
}

impl<W: Writer> Writer for BufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) {
        if self.pos + buf.len() > self.buf.len() {
            self.flush();
        }

        if buf.len() > self.buf.len() {
            self.inner.write(buf);
        } else {
            let dst = self.buf.mut_slice_from(self.pos);
            vec::bytes::copy_memory(dst, buf, buf.len());
            self.pos += buf.len();
        }
    }

    fn flush(&mut self) {
        if self.pos != 0 {
            self.inner.write(self.buf.slice_to(self.pos));
            self.pos = 0;
        }
        self.inner.flush();
    }
}

impl<W: Writer> Decorator<W> for BufferedWriter<W> {
    fn inner(self) -> W {
        self.inner
    }

    fn inner_ref<'a>(&'a self) -> &'a W {
        &self.inner
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut W {
        &mut self.inner
    }
}

struct InternalBufferedWriter<W>(BufferedWriter<W>);

impl<W: Reader> Reader for InternalBufferedWriter<W> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        self.inner.read(buf)
    }

    fn eof(&mut self) -> bool {
        self.inner.eof()
    }
}

/// Wraps a Stream and buffers input and output to and from it
///
/// Note that `BufferedStream` will NOT flush its output buffer when dropped.
pub struct BufferedStream<S> {
    priv inner: BufferedReader<InternalBufferedWriter<S>>
}

impl<S: Stream> BufferedStream<S> {
    pub fn with_capacities(reader_cap: uint, writer_cap: uint, inner: S)
                           -> BufferedStream<S> {
        let writer = BufferedWriter::with_capacity(writer_cap, inner);
        let internal_writer = InternalBufferedWriter(writer);
        let reader = BufferedReader::with_capacity(reader_cap,
                                                   internal_writer);
        BufferedStream { inner: reader }
    }

    pub fn new(inner: S) -> BufferedStream<S> {
        BufferedStream::with_capacities(DEFAULT_CAPACITY, DEFAULT_CAPACITY,
                                        inner)
    }
}

impl<S: Stream> Reader for BufferedStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        self.inner.read(buf)
    }

    fn eof(&mut self) -> bool {
        self.inner.eof()
    }
}

impl<S: Stream> Writer for BufferedStream<S> {
    fn write(&mut self, buf: &[u8]) {
        self.inner.inner.write(buf)
    }

    fn flush(&mut self) {
        self.inner.inner.flush()
    }
}

impl<S: Stream> Decorator<S> for BufferedStream<S> {
    fn inner(self) -> S {
        self.inner.inner.inner()
    }

    fn inner_ref<'a>(&'a self) -> &'a S {
        self.inner.inner.inner_ref()
    }

    fn inner_mut_ref<'a>(&'a mut self) -> &'a mut S {
        self.inner.inner.inner_mut_ref()
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;
    use super::super::mem::{MemReader, MemWriter};

    #[test]
    fn test_buffered_reader() {
        let inner = MemReader::new(~[0, 1, 2, 3, 4]);
        let mut reader = BufferedReader::with_capacity(2, inner);

        let mut buf = [0, 0, 0];
        let nread = reader.read(buf);
        assert_eq!(Some(2), nread);
        assert_eq!([0, 1, 0], buf);
        assert!(!reader.eof());

        let mut buf = [0];
        let nread = reader.read(buf);
        assert_eq!(Some(1), nread);
        assert_eq!([2], buf);
        assert!(!reader.eof());

        let mut buf = [0, 0, 0];
        let nread = reader.read(buf);
        assert_eq!(Some(1), nread);
        assert_eq!([3, 0, 0], buf);
        assert!(!reader.eof());

        let nread = reader.read(buf);
        assert_eq!(Some(1), nread);
        assert_eq!([4, 0, 0], buf);
        assert!(reader.eof());

        assert_eq!(None, reader.read(buf));
    }

    #[test]
    fn test_buffered_writer() {
        let inner = MemWriter::new();
        let mut writer = BufferedWriter::with_capacity(2, inner);

        writer.write([0, 1]);
        assert_eq!([], writer.inner_ref().inner_ref().as_slice());

        writer.write([2]);
        assert_eq!([0, 1], writer.inner_ref().inner_ref().as_slice());

        writer.write([3]);
        assert_eq!([0, 1], writer.inner_ref().inner_ref().as_slice());

        writer.flush();
        assert_eq!([0, 1, 2, 3], writer.inner_ref().inner_ref().as_slice());

        writer.write([4]);
        writer.write([5]);
        assert_eq!([0, 1, 2, 3], writer.inner_ref().inner_ref().as_slice());

        writer.write([6]);
        assert_eq!([0, 1, 2, 3, 4, 5],
                   writer.inner_ref().inner_ref().as_slice());

        writer.write([7, 8]);
        assert_eq!([0, 1, 2, 3, 4, 5, 6],
                   writer.inner_ref().inner_ref().as_slice());

        writer.write([9, 10, 11]);
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   writer.inner_ref().inner_ref().as_slice());

        writer.flush();
        assert_eq!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   writer.inner_ref().inner_ref().as_slice());
    }

    // This is just here to make sure that we don't infinite loop in the
    // newtype struct autoderef weirdness
    #[test]
    fn test_buffered_stream() {
        use rt;
        struct S;

        impl rt::io::Writer for S {
            fn write(&mut self, _: &[u8]) {}
            fn flush(&mut self) {}
        }

        impl rt::io::Reader for S {
            fn read(&mut self, _: &mut [u8]) -> Option<uint> { None }
            fn eof(&mut self) -> bool { true }
        }

        let mut stream = BufferedStream::new(S);
        let mut buf = [];
        stream.read(buf);
        stream.eof();
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
}
