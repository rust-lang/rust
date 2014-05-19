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

use cmp;
use collections::Collection;
use io::{Reader, Writer, Stream, Buffer, DEFAULT_BUF_SIZE, IoResult};
use iter::ExactSize;
use ops::Drop;
use option::{Some, None, Option};
use result::{Ok, Err};
use slice::{ImmutableVector, MutableVector};
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
/// match reader.read(buf) {
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
    pub fn with_capacity(cap: uint, inner: R) -> BufferedReader<R> {
        // It's *much* faster to create an uninitialized buffer than it is to
        // fill everything in with 0. This buffer is entirely an implementation
        // detail and is never exposed, so we're safe to not initialize
        // everything up-front. This allows creation of BufferedReader instances
        // to be very cheap (large mallocs are not nearly as expensive as large
        // callocs).
        let mut buf = Vec::with_capacity(cap);
        unsafe { buf.set_len(cap); }
        BufferedReader {
            inner: inner,
            buf: buf,
            pos: 0,
            cap: 0,
        }
    }

    /// Creates a new `BufferedReader` with a default buffer capacity
    pub fn new(inner: R) -> BufferedReader<R> {
        BufferedReader::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    /// Gets a reference to the underlying reader.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a R { &self.inner }

    /// Unwraps this `BufferedReader`, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost.
    pub fn unwrap(self) -> R { self.inner }
}

impl<R: Reader> Buffer for BufferedReader<R> {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> {
        if self.pos == self.cap {
            self.cap = try!(self.inner.read(self.buf.as_mut_slice()));
            self.pos = 0;
        }
        Ok(self.buf.slice(self.pos, self.cap))
    }

    fn consume(&mut self, amt: uint) {
        self.pos += amt;
        assert!(self.pos <= self.cap);
    }
}

impl<R: Reader> Reader for BufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        let nread = {
            let available = try!(self.fill_buf());
            let nread = cmp::min(available.len(), buf.len());
            slice::bytes::copy_memory(buf, available.slice_to(nread));
            nread
        };
        self.pos += nread;
        Ok(nread)
    }
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
/// # #![allow(unused_must_use)]
/// use std::io::{BufferedWriter, File};
///
/// let file = File::open(&Path::new("message.txt"));
/// let mut writer = BufferedWriter::new(file);
///
/// writer.write_str("hello, world");
/// writer.flush();
/// ```
pub struct BufferedWriter<W> {
    inner: Option<W>,
    buf: Vec<u8>,
    pos: uint
}

impl<W: Writer> BufferedWriter<W> {
    /// Creates a new `BufferedWriter` with the specified buffer capacity
    pub fn with_capacity(cap: uint, inner: W) -> BufferedWriter<W> {
        // See comments in BufferedReader for why this uses unsafe code.
        let mut buf = Vec::with_capacity(cap);
        unsafe { buf.set_len(cap); }
        BufferedWriter {
            inner: Some(inner),
            buf: buf,
            pos: 0
        }
    }

    /// Creates a new `BufferedWriter` with a default buffer capacity
    pub fn new(inner: W) -> BufferedWriter<W> {
        BufferedWriter::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    fn flush_buf(&mut self) -> IoResult<()> {
        if self.pos != 0 {
            let ret = self.inner.get_mut_ref().write(self.buf.slice_to(self.pos));
            self.pos = 0;
            ret
        } else {
            Ok(())
        }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W { self.inner.get_ref() }

    /// Unwraps this `BufferedWriter`, returning the underlying writer.
    ///
    /// The buffer is flushed before returning the writer.
    pub fn unwrap(mut self) -> W {
        // FIXME(#12628): is failing the right thing to do if flushing fails?
        self.flush_buf().unwrap();
        self.inner.take_unwrap()
    }
}

impl<W: Writer> Writer for BufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        if self.pos + buf.len() > self.buf.len() {
            try!(self.flush_buf());
        }

        if buf.len() > self.buf.len() {
            self.inner.get_mut_ref().write(buf)
        } else {
            let dst = self.buf.mut_slice_from(self.pos);
            slice::bytes::copy_memory(dst, buf);
            self.pos += buf.len();
            Ok(())
        }
    }

    fn flush(&mut self) -> IoResult<()> {
        self.flush_buf().and_then(|()| self.inner.get_mut_ref().flush())
    }
}

#[unsafe_destructor]
impl<W: Writer> Drop for BufferedWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            // dtors should not fail, so we ignore a failed flush
            let _ = self.flush_buf();
        }
    }
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

    /// Unwraps this `LineBufferedWriter`, returning the underlying writer.
    ///
    /// The internal buffer is flushed before returning the writer.
    pub fn unwrap(self) -> W { self.inner.unwrap() }
}

impl<W: Writer> Writer for LineBufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        match buf.iter().rposition(|&b| b == '\n' as u8) {
            Some(i) => {
                try!(self.inner.write(buf.slice_to(i + 1)));
                try!(self.inner.flush());
                try!(self.inner.write(buf.slice_from(i + 1)));
                Ok(())
            }
            None => self.inner.write(buf),
        }
    }

    fn flush(&mut self) -> IoResult<()> { self.inner.flush() }
}

struct InternalBufferedWriter<W>(BufferedWriter<W>);

impl<W> InternalBufferedWriter<W> {
    fn get_mut_ref<'a>(&'a mut self) -> &'a mut BufferedWriter<W> {
        let InternalBufferedWriter(ref mut w) = *self;
        return w;
    }
}

impl<W: Reader> Reader for InternalBufferedWriter<W> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.get_mut_ref().inner.get_mut_ref().read(buf)
    }
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
/// match stream.read(buf) {
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
        BufferedStream::with_capacities(DEFAULT_BUF_SIZE, DEFAULT_BUF_SIZE,
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

    /// Unwraps this `BufferedStream`, returning the underlying stream.
    ///
    /// The internal buffer is flushed before returning the stream. Any leftover
    /// data in the read buffer is lost.
    pub fn unwrap(self) -> S {
        let InternalBufferedWriter(w) = self.inner.inner;
        w.unwrap()
    }
}

impl<S: Stream> Buffer for BufferedStream<S> {
    fn fill_buf<'a>(&'a mut self) -> IoResult<&'a [u8]> { self.inner.fill_buf() }
    fn consume(&mut self, amt: uint) { self.inner.consume(amt) }
}

impl<S: Stream> Reader for BufferedStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        self.inner.read(buf)
    }
}

impl<S: Stream> Writer for BufferedStream<S> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.inner.inner.get_mut_ref().write(buf)
    }
    fn flush(&mut self) -> IoResult<()> {
        self.inner.inner.get_mut_ref().flush()
    }
}

#[cfg(test)]
mod test {
    extern crate test;
    use io;
    use prelude::*;
    use super::*;
    use super::super::{IoResult, EndOfFile};
    use super::super::mem::{MemReader, MemWriter, BufReader};
    use self::test::Bencher;
    use str::StrSlice;

    /// A type, free to create, primarily intended for benchmarking creation of
    /// wrappers that, just for construction, don't need a Reader/Writer that
    /// does anything useful. Is equivalent to `/dev/null` in semantics.
    #[deriving(Clone,PartialEq,PartialOrd)]
    pub struct NullStream;

    impl Reader for NullStream {
        fn read(&mut self, _: &mut [u8]) -> io::IoResult<uint> {
            Err(io::standard_error(io::EndOfFile))
        }
    }

    impl Writer for NullStream {
        fn write(&mut self, _: &[u8]) -> io::IoResult<()> { Ok(()) }
    }

    /// A dummy reader intended at testing short-reads propagation.
    pub struct ShortReader {
        lengths: Vec<uint>,
    }

    impl Reader for ShortReader {
        fn read(&mut self, _: &mut [u8]) -> io::IoResult<uint> {
            match self.lengths.shift() {
                Some(i) => Ok(i),
                None => Err(io::standard_error(io::EndOfFile))
            }
        }
    }

    #[test]
    fn test_buffered_reader() {
        let inner = MemReader::new(vec!(0, 1, 2, 3, 4));
        let mut reader = BufferedReader::with_capacity(2, inner);

        let mut buf = [0, 0, 0];
        let nread = reader.read(buf);
        assert_eq!(Ok(2), nread);
        assert_eq!(buf.as_slice(), &[0, 1, 0]);

        let mut buf = [0];
        let nread = reader.read(buf);
        assert_eq!(Ok(1), nread);
        assert_eq!(buf.as_slice(), &[2]);

        let mut buf = [0, 0, 0];
        let nread = reader.read(buf);
        assert_eq!(Ok(1), nread);
        assert_eq!(buf.as_slice(), &[3, 0, 0]);

        let nread = reader.read(buf);
        assert_eq!(Ok(1), nread);
        assert_eq!(buf.as_slice(), &[4, 0, 0]);

        assert!(reader.read(buf).is_err());
    }

    #[test]
    fn test_buffered_writer() {
        let inner = MemWriter::new();
        let mut writer = BufferedWriter::with_capacity(2, inner);

        writer.write([0, 1]).unwrap();
        assert_eq!(writer.get_ref().get_ref(), &[]);

        writer.write([2]).unwrap();
        assert_eq!(writer.get_ref().get_ref(), &[0, 1]);

        writer.write([3]).unwrap();
        assert_eq!(writer.get_ref().get_ref(), &[0, 1]);

        writer.flush().unwrap();
        assert_eq!(&[0, 1, 2, 3], writer.get_ref().get_ref());

        writer.write([4]).unwrap();
        writer.write([5]).unwrap();
        assert_eq!(&[0, 1, 2, 3], writer.get_ref().get_ref());

        writer.write([6]).unwrap();
        assert_eq!(&[0, 1, 2, 3, 4, 5],
                   writer.get_ref().get_ref());

        writer.write([7, 8]).unwrap();
        assert_eq!(&[0, 1, 2, 3, 4, 5, 6],
                   writer.get_ref().get_ref());

        writer.write([9, 10, 11]).unwrap();
        assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   writer.get_ref().get_ref());

        writer.flush().unwrap();
        assert_eq!(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   writer.get_ref().get_ref());
    }

    #[test]
    fn test_buffered_writer_inner_flushes() {
        let mut w = BufferedWriter::with_capacity(3, MemWriter::new());
        w.write([0, 1]).unwrap();
        assert_eq!(&[], w.get_ref().get_ref());
        let w = w.unwrap();
        assert_eq!(&[0, 1], w.get_ref());
    }

    // This is just here to make sure that we don't infinite loop in the
    // newtype struct autoderef weirdness
    #[test]
    fn test_buffered_stream() {
        struct S;

        impl io::Writer for S {
            fn write(&mut self, _: &[u8]) -> io::IoResult<()> { Ok(()) }
        }

        impl io::Reader for S {
            fn read(&mut self, _: &mut [u8]) -> io::IoResult<uint> {
                Err(io::standard_error(io::EndOfFile))
            }
        }

        let mut stream = BufferedStream::new(S);
        let mut buf = [];
        assert!(stream.read(buf).is_err());
        stream.write(buf).unwrap();
        stream.flush().unwrap();
    }

    #[test]
    fn test_read_until() {
        let inner = MemReader::new(vec!(0, 1, 2, 1, 0));
        let mut reader = BufferedReader::with_capacity(2, inner);
        assert_eq!(reader.read_until(0), Ok(vec!(0)));
        assert_eq!(reader.read_until(2), Ok(vec!(1, 2)));
        assert_eq!(reader.read_until(1), Ok(vec!(1)));
        assert_eq!(reader.read_until(8), Ok(vec!(0)));
        assert!(reader.read_until(9).is_err());
    }

    #[test]
    fn test_line_buffer() {
        let mut writer = LineBufferedWriter::new(MemWriter::new());
        writer.write([0]).unwrap();
        assert_eq!(writer.get_ref().get_ref(), &[]);
        writer.write([1]).unwrap();
        assert_eq!(writer.get_ref().get_ref(), &[]);
        writer.flush().unwrap();
        assert_eq!(writer.get_ref().get_ref(), &[0, 1]);
        writer.write([0, '\n' as u8, 1, '\n' as u8, 2]).unwrap();
        assert_eq!(writer.get_ref().get_ref(),
                   &[0, 1, 0, '\n' as u8, 1, '\n' as u8]);
        writer.flush().unwrap();
        assert_eq!(writer.get_ref().get_ref(),
                   &[0, 1, 0, '\n' as u8, 1, '\n' as u8, 2]);
        writer.write([3, '\n' as u8]).unwrap();
        assert_eq!(writer.get_ref().get_ref(),
            &[0, 1, 0, '\n' as u8, 1, '\n' as u8, 2, 3, '\n' as u8]);
    }

    #[test]
    fn test_read_line() {
        let in_buf = MemReader::new(Vec::from_slice(bytes!("a\nb\nc")));
        let mut reader = BufferedReader::with_capacity(2, in_buf);
        assert_eq!(reader.read_line(), Ok("a\n".to_string()));
        assert_eq!(reader.read_line(), Ok("b\n".to_string()));
        assert_eq!(reader.read_line(), Ok("c".to_string()));
        assert!(reader.read_line().is_err());
    }

    #[test]
    fn test_lines() {
        let in_buf = MemReader::new(Vec::from_slice(bytes!("a\nb\nc")));
        let mut reader = BufferedReader::with_capacity(2, in_buf);
        let mut it = reader.lines();
        assert_eq!(it.next(), Some(Ok("a\n".to_string())));
        assert_eq!(it.next(), Some(Ok("b\n".to_string())));
        assert_eq!(it.next(), Some(Ok("c".to_string())));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_short_reads() {
        let inner = ShortReader{lengths: vec![0, 1, 2, 0, 1, 0]};
        let mut reader = BufferedReader::new(inner);
        let mut buf = [0, 0];
        assert_eq!(reader.read(buf), Ok(0));
        assert_eq!(reader.read(buf), Ok(1));
        assert_eq!(reader.read(buf), Ok(2));
        assert_eq!(reader.read(buf), Ok(0));
        assert_eq!(reader.read(buf), Ok(1));
        assert_eq!(reader.read(buf), Ok(0));
        assert!(reader.read(buf).is_err());
    }

    #[test]
    fn read_char_buffered() {
        let buf = [195u8, 159u8];
        let mut reader = BufferedReader::with_capacity(1, BufReader::new(buf));
        assert_eq!(reader.read_char(), Ok('ß'));
    }

    #[test]
    fn test_chars() {
        let buf = [195u8, 159u8, 'a' as u8];
        let mut reader = BufferedReader::with_capacity(1, BufReader::new(buf));
        let mut it = reader.chars();
        assert_eq!(it.next(), Some(Ok('ß')));
        assert_eq!(it.next(), Some(Ok('a')));
        assert_eq!(it.next(), None);
    }

    #[test]
    #[should_fail]
    fn dont_fail_in_drop_on_failed_flush() {
        struct FailFlushWriter;

        impl Writer for FailFlushWriter {
            fn write(&mut self, _buf: &[u8]) -> IoResult<()> { Ok(()) }
            fn flush(&mut self) -> IoResult<()> { Err(io::standard_error(EndOfFile)) }
        }

        let writer = FailFlushWriter;
        let _writer = BufferedWriter::new(writer);

        // Trigger failure. If writer fails *again* due to the flush
        // error then the process will abort.
        fail!();
    }

    #[bench]
    fn bench_buffered_reader(b: &mut Bencher) {
        b.iter(|| {
            BufferedReader::new(NullStream)
        });
    }

    #[bench]
    fn bench_buffered_writer(b: &mut Bencher) {
        b.iter(|| {
            BufferedWriter::new(NullStream)
        });
    }

    #[bench]
    fn bench_buffered_stream(b: &mut Bencher) {
        b.iter(|| {
            BufferedStream::new(NullStream);
        });
    }
}
