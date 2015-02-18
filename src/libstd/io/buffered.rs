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

use prelude::v1::*;
use io::prelude::*;

use cmp;
use error::Error as StdError;
use error::FromError;
use fmt;
use io::{self, Cursor, DEFAULT_BUF_SIZE, Error, ErrorKind};
use ptr;

/// Wraps a `Read` and buffers input from it
///
/// It can be excessively inefficient to work directly with a `Read` instance.
/// For example, every call to `read` on `TcpStream` results in a system call.
/// A `BufReader` performs large, infrequent reads on the underlying `Read`
/// and maintains an in-memory buffer of the results.
pub struct BufReader<R> {
    inner: R,
    buf: Cursor<Vec<u8>>,
}

impl<R: Read> BufReader<R> {
    /// Creates a new `BufReader` with a default buffer capacity
    pub fn new(inner: R) -> BufReader<R> {
        BufReader::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    /// Creates a new `BufReader` with the specified buffer capacity
    pub fn with_capacity(cap: usize, inner: R) -> BufReader<R> {
        BufReader {
            inner: inner,
            buf: Cursor::new(Vec::with_capacity(cap)),
        }
    }

    /// Gets a reference to the underlying reader.
    pub fn get_ref<'a>(&self) -> &R { &self.inner }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// # Warning
    ///
    /// It is inadvisable to directly read from the underlying reader.
    pub fn get_mut(&mut self) -> &mut R { &mut self.inner }

    /// Unwraps this `BufReader`, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost.
    pub fn into_inner(self) -> R { self.inner }
}

impl<R: Read> Read for BufReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.buf.get_ref().len() == self.buf.position() as usize &&
            buf.len() >= self.buf.get_ref().capacity() {
            return self.inner.read(buf);
        }
        try!(self.fill_buf());
        self.buf.read(buf)
    }
}

impl<R: Read> BufRead for BufReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the underlying reader.
        if self.buf.position() as usize == self.buf.get_ref().len() {
            self.buf.set_position(0);
            let v = self.buf.get_mut();
            v.truncate(0);
            let inner = &mut self.inner;
            try!(super::with_end_to_cap(v, |b| inner.read(b)));
        }
        self.buf.fill_buf()
    }

    fn consume(&mut self, amt: uint) {
        self.buf.consume(amt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R> fmt::Debug for BufReader<R> where R: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "BufReader {{ reader: {:?}, buffer: {}/{} }}",
               self.inner, self.buf.position(), self.buf.get_ref().len())
    }
}

/// Wraps a Writer and buffers output to it
///
/// It can be excessively inefficient to work directly with a `Write`. For
/// example, every call to `write` on `TcpStream` results in a system call. A
/// `BufWriter` keeps an in memory buffer of data and writes it to the
/// underlying `Write` in large, infrequent batches.
///
/// This writer will be flushed when it is dropped.
pub struct BufWriter<W> {
    inner: Option<W>,
    buf: Vec<u8>,
}

/// An error returned by `into_inner` which indicates whether a flush error
/// happened or not.
#[derive(Debug)]
pub struct IntoInnerError<W>(W, Error);

impl<W: Write> BufWriter<W> {
    /// Creates a new `BufWriter` with a default buffer capacity
    pub fn new(inner: W) -> BufWriter<W> {
        BufWriter::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    /// Creates a new `BufWriter` with the specified buffer capacity
    pub fn with_capacity(cap: usize, inner: W) -> BufWriter<W> {
        BufWriter {
            inner: Some(inner),
            buf: Vec::with_capacity(cap),
        }
    }

    fn flush_buf(&mut self) -> io::Result<()> {
        let mut written = 0;
        let len = self.buf.len();
        let mut ret = Ok(());
        while written < len {
            match self.inner.as_mut().unwrap().write(&self.buf[written..]) {
                Ok(0) => {
                    ret = Err(Error::new(ErrorKind::WriteZero,
                                         "failed to flush", None));
                    break;
                }
                Ok(n) => written += n,
                Err(e) => { ret = Err(e); break }

            }
        }
        if written > 0 {
            // NB: would be better expressed as .remove(0..n) if it existed
            unsafe {
                ptr::copy_memory(self.buf.as_mut_ptr(),
                                 self.buf.as_ptr().offset(written as isize),
                                 len - written);
            }
        }
        self.buf.truncate(len - written);
        ret
    }

    /// Gets a reference to the underlying writer.
    pub fn get_ref(&self) -> &W { self.inner.as_ref().unwrap() }

    /// Gets a mutable reference to the underlying write.
    ///
    /// # Warning
    ///
    /// It is inadvisable to directly read from the underlying writer.
    pub fn get_mut(&mut self) -> &mut W { self.inner.as_mut().unwrap() }

    /// Unwraps this `BufWriter`, returning the underlying writer.
    ///
    /// The buffer is flushed before returning the writer.
    pub fn into_inner(mut self) -> Result<W, IntoInnerError<BufWriter<W>>> {
        match self.flush_buf() {
            Err(e) => Err(IntoInnerError(self, e)),
            Ok(()) => Ok(self.inner.take().unwrap())
        }
    }
}

impl<W: Write> Write for BufWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.buf.len() + buf.len() > self.buf.capacity() {
            try!(self.flush_buf());
        }
        if buf.len() >= self.buf.capacity() {
            self.inner.as_mut().unwrap().write(buf)
        } else {
            let amt = cmp::min(buf.len(), self.buf.capacity());
            Write::write(&mut self.buf, &buf[..amt])
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flush_buf().and_then(|()| self.get_mut().flush())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W> fmt::Debug for BufWriter<W> where W: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "BufWriter {{ writer: {:?}, buffer: {}/{} }}",
               self.inner.as_ref().unwrap(), self.buf.len(), self.buf.capacity())
    }
}

#[unsafe_destructor]
impl<W: Write> Drop for BufWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            // dtors should not panic, so we ignore a failed flush
            let _r = self.flush_buf();
        }
    }
}

impl<W> IntoInnerError<W> {
    /// Returns the error which caused the call to `into_inner` to fail.
    ///
    /// This error was returned when attempting to flush the internal buffer.
    pub fn error(&self) -> &Error { &self.1 }

    /// Returns the underlying `BufWriter` instance which generated the error.
    ///
    /// The returned object can be used to retry a flush or re-inspect the
    /// buffer.
    pub fn into_inner(self) -> W { self.0 }
}

impl<W> FromError<IntoInnerError<W>> for Error {
    fn from_error(iie: IntoInnerError<W>) -> Error { iie.1 }
}

impl<W> StdError for IntoInnerError<W> {
    fn description(&self) -> &str { self.error().description() }
}

impl<W> fmt::Display for IntoInnerError<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.error().fmt(f)
    }
}

/// Wraps a Writer and buffers output to it, flushing whenever a newline
/// (`0x0a`, `'\n'`) is detected.
///
/// This writer will be flushed when it is dropped.
pub struct LineWriter<W> {
    inner: BufWriter<W>,
}

impl<W: Write> LineWriter<W> {
    /// Creates a new `LineWriter`
    pub fn new(inner: W) -> LineWriter<W> {
        // Lines typically aren't that long, don't use a giant buffer
        LineWriter { inner: BufWriter::with_capacity(1024, inner) }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// This type does not expose the ability to get a mutable reference to the
    /// underlying reader because that could possibly corrupt the buffer.
    pub fn get_ref<'a>(&'a self) -> &'a W { self.inner.get_ref() }

    /// Unwraps this `LineWriter`, returning the underlying writer.
    ///
    /// The internal buffer is flushed before returning the writer.
    pub fn into_inner(self) -> Result<W, IntoInnerError<LineWriter<W>>> {
        self.inner.into_inner().map_err(|IntoInnerError(buf, e)| {
            IntoInnerError(LineWriter { inner: buf }, e)
        })
    }
}

impl<W: Write> Write for LineWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match buf.rposition_elem(&b'\n') {
            Some(i) => {
                let n = try!(self.inner.write(&buf[..i + 1]));
                if n != i + 1 { return Ok(n) }
                try!(self.inner.flush());
                self.inner.write(&buf[i + 1..]).map(|i| n + i)
            }
            None => self.inner.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> { self.inner.flush() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W> fmt::Debug for LineWriter<W> where W: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "LineWriter {{ writer: {:?}, buffer: {}/{} }}",
               self.inner.inner, self.inner.buf.len(),
               self.inner.buf.capacity())
    }
}

struct InternalBufWriter<W>(BufWriter<W>);

impl<W> InternalBufWriter<W> {
    fn get_mut(&mut self) -> &mut BufWriter<W> {
        let InternalBufWriter(ref mut w) = *self;
        return w;
    }
}

impl<W: Read> Read for InternalBufWriter<W> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.get_mut().inner.as_mut().unwrap().read(buf)
    }
}

/// Wraps a Stream and buffers input and output to and from it.
///
/// It can be excessively inefficient to work directly with a `Stream`. For
/// example, every call to `read` or `write` on `TcpStream` results in a system
/// call. A `BufStream` keeps in memory buffers of data, making large,
/// infrequent calls to `read` and `write` on the underlying `Stream`.
///
/// The output half will be flushed when this stream is dropped.
pub struct BufStream<S> {
    inner: BufReader<InternalBufWriter<S>>
}

impl<S: Read + Write> BufStream<S> {
    /// Creates a new buffered stream with explicitly listed capacities for the
    /// reader/writer buffer.
    pub fn with_capacities(reader_cap: usize, writer_cap: usize, inner: S)
                           -> BufStream<S> {
        let writer = BufWriter::with_capacity(writer_cap, inner);
        let internal_writer = InternalBufWriter(writer);
        let reader = BufReader::with_capacity(reader_cap, internal_writer);
        BufStream { inner: reader }
    }

    /// Creates a new buffered stream with the default reader/writer buffer
    /// capacities.
    pub fn new(inner: S) -> BufStream<S> {
        BufStream::with_capacities(DEFAULT_BUF_SIZE, DEFAULT_BUF_SIZE, inner)
    }

    /// Gets a reference to the underlying stream.
    pub fn get_ref(&self) -> &S {
        let InternalBufWriter(ref w) = self.inner.inner;
        w.get_ref()
    }

    /// Gets a mutable reference to the underlying stream.
    ///
    /// # Warning
    ///
    /// It is inadvisable to read directly from or write directly to the
    /// underlying stream.
    pub fn get_mut(&mut self) -> &mut S {
        let InternalBufWriter(ref mut w) = self.inner.inner;
        w.get_mut()
    }

    /// Unwraps this `BufStream`, returning the underlying stream.
    ///
    /// The internal buffer is flushed before returning the stream. Any leftover
    /// data in the read buffer is lost.
    pub fn into_inner(self) -> Result<S, IntoInnerError<BufStream<S>>> {
        let BufReader { inner: InternalBufWriter(w), buf } = self.inner;
        w.into_inner().map_err(|IntoInnerError(w, e)| {
            IntoInnerError(BufStream {
                inner: BufReader { inner: InternalBufWriter(w), buf: buf },
            }, e)
        })
    }
}

impl<S: Read + Write> BufRead for BufStream<S> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> { self.inner.fill_buf() }
    fn consume(&mut self, amt: uint) { self.inner.consume(amt) }
}

impl<S: Read + Write> Read for BufStream<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<S: Read + Write> Write for BufStream<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.inner.get_mut().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.inner.get_mut().flush()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<S> fmt::Debug for BufStream<S> where S: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let reader = &self.inner;
        let writer = &self.inner.inner.0;
        write!(fmt, "BufStream {{ stream: {:?}, write_buffer: {}/{}, read_buffer: {}/{} }}",
               writer.inner,
               writer.buf.len(), writer.buf.capacity(),
               reader.buf.position(), reader.buf.get_ref().len())
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;
    use io::prelude::*;
    use io::{self, BufReader, BufWriter, BufStream, Cursor, LineWriter};
    use test;

    /// A dummy reader intended at testing short-reads propagation.
    pub struct ShortReader {
        lengths: Vec<usize>,
    }

    impl Read for ShortReader {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            if self.lengths.is_empty() {
                Ok(0)
            } else {
                Ok(self.lengths.remove(0))
            }
        }
    }

    #[test]
    fn test_buffered_reader() {
        let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
        let mut reader = BufReader::with_capacity(2, inner);

        let mut buf = [0, 0, 0];
        let nread = reader.read(&mut buf);
        assert_eq!(Ok(3), nread);
        let b: &[_] = &[5, 6, 7];
        assert_eq!(buf, b);

        let mut buf = [0, 0];
        let nread = reader.read(&mut buf);
        assert_eq!(Ok(2), nread);
        let b: &[_] = &[0, 1];
        assert_eq!(buf, b);

        let mut buf = [0];
        let nread = reader.read(&mut buf);
        assert_eq!(Ok(1), nread);
        let b: &[_] = &[2];
        assert_eq!(buf, b);

        let mut buf = [0, 0, 0];
        let nread = reader.read(&mut buf);
        assert_eq!(Ok(1), nread);
        let b: &[_] = &[3, 0, 0];
        assert_eq!(buf, b);

        let nread = reader.read(&mut buf);
        assert_eq!(Ok(1), nread);
        let b: &[_] = &[4, 0, 0];
        assert_eq!(buf, b);

        assert_eq!(reader.read(&mut buf), Ok(0));
    }

    #[test]
    fn test_buffered_writer() {
        let inner = Vec::new();
        let mut writer = BufWriter::with_capacity(2, inner);

        writer.write(&[0, 1]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1]);

        writer.write(&[2]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1]);

        writer.write(&[3]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1]);

        writer.flush().unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 2, 3]);

        writer.write(&[4]).unwrap();
        writer.write(&[5]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 2, 3]);

        writer.write(&[6]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5]);

        writer.write(&[7, 8]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5, 6, 7, 8]);

        writer.write(&[9, 10, 11]).unwrap();
        let a: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

        writer.flush().unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn test_buffered_writer_inner_flushes() {
        let mut w = BufWriter::with_capacity(3, Vec::new());
        w.write(&[0, 1]).unwrap();
        assert_eq!(*w.get_ref(), []);
        let w = w.into_inner().unwrap();
        assert_eq!(w, [0, 1]);
    }

    // This is just here to make sure that we don't infinite loop in the
    // newtype struct autoderef weirdness
    #[test]
    fn test_buffered_stream() {
        struct S;

        impl Write for S {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> { Ok(b.len()) }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }

        impl Read for S {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { Ok(0) }
        }

        let mut stream = BufStream::new(S);
        assert_eq!(stream.read(&mut [0; 10]), Ok(0));
        stream.write(&[0; 10]).unwrap();
        stream.flush().unwrap();
    }

    #[test]
    fn test_read_until() {
        let inner: &[u8] = &[0, 1, 2, 1, 0];
        let mut reader = BufReader::with_capacity(2, inner);
        let mut v = Vec::new();
        reader.read_until(0, &mut v).unwrap();
        assert_eq!(v, [0]);
        v.truncate(0);
        reader.read_until(2, &mut v).unwrap();
        assert_eq!(v, [1, 2]);
        v.truncate(0);
        reader.read_until(1, &mut v).unwrap();
        assert_eq!(v, [1]);
        v.truncate(0);
        reader.read_until(8, &mut v).unwrap();
        assert_eq!(v, [0]);
        v.truncate(0);
        reader.read_until(9, &mut v).unwrap();
        assert_eq!(v, []);
    }

    #[test]
    fn test_line_buffer() {
        let mut writer = LineWriter::new(Vec::new());
        writer.write(&[0]).unwrap();
        assert_eq!(*writer.get_ref(), []);
        writer.write(&[1]).unwrap();
        assert_eq!(*writer.get_ref(), []);
        writer.flush().unwrap();
        assert_eq!(*writer.get_ref(), [0, 1]);
        writer.write(&[0, b'\n', 1, b'\n', 2]).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 0, b'\n', 1, b'\n']);
        writer.flush().unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 0, b'\n', 1, b'\n', 2]);
        writer.write(&[3, b'\n']).unwrap();
        assert_eq!(*writer.get_ref(), [0, 1, 0, b'\n', 1, b'\n', 2, 3, b'\n']);
    }

    #[test]
    fn test_read_line() {
        let in_buf = b"a\nb\nc";
        let mut reader = BufReader::with_capacity(2, in_buf);
        let mut s = String::new();
        reader.read_line(&mut s).unwrap();
        assert_eq!(s, "a\n");
        s.truncate(0);
        reader.read_line(&mut s).unwrap();
        assert_eq!(s, "b\n");
        s.truncate(0);
        reader.read_line(&mut s).unwrap();
        assert_eq!(s, "c");
        s.truncate(0);
        reader.read_line(&mut s).unwrap();
        assert_eq!(s, "");
    }

    #[test]
    fn test_lines() {
        let in_buf = b"a\nb\nc";
        let mut reader = BufReader::with_capacity(2, in_buf);
        let mut it = reader.lines();
        assert_eq!(it.next(), Some(Ok("a".to_string())));
        assert_eq!(it.next(), Some(Ok("b".to_string())));
        assert_eq!(it.next(), Some(Ok("c".to_string())));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_short_reads() {
        let inner = ShortReader{lengths: vec![0, 1, 2, 0, 1, 0]};
        let mut reader = BufReader::new(inner);
        let mut buf = [0, 0];
        assert_eq!(reader.read(&mut buf), Ok(0));
        assert_eq!(reader.read(&mut buf), Ok(1));
        assert_eq!(reader.read(&mut buf), Ok(2));
        assert_eq!(reader.read(&mut buf), Ok(0));
        assert_eq!(reader.read(&mut buf), Ok(1));
        assert_eq!(reader.read(&mut buf), Ok(0));
        assert_eq!(reader.read(&mut buf), Ok(0));
    }

    #[test]
    fn read_char_buffered() {
        let buf = [195u8, 159u8];
        let mut reader = BufReader::with_capacity(1, &buf[..]);
        assert_eq!(reader.chars().next(), Some(Ok('ß')));
    }

    #[test]
    fn test_chars() {
        let buf = [195u8, 159u8, b'a'];
        let mut reader = BufReader::with_capacity(1, &buf[..]);
        let mut it = reader.chars();
        assert_eq!(it.next(), Some(Ok('ß')));
        assert_eq!(it.next(), Some(Ok('a')));
        assert_eq!(it.next(), None);
    }

    #[test]
    #[should_fail]
    fn dont_panic_in_drop_on_panicked_flush() {
        struct FailFlushWriter;

        impl Write for FailFlushWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> { Ok(buf.len()) }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::last_os_error())
            }
        }

        let writer = FailFlushWriter;
        let _writer = BufWriter::new(writer);

        // If writer panics *again* due to the flush error then the process will
        // abort.
        panic!();
    }

    #[bench]
    fn bench_buffered_reader(b: &mut test::Bencher) {
        b.iter(|| {
            BufReader::new(io::empty())
        });
    }

    #[bench]
    fn bench_buffered_writer(b: &mut test::Bencher) {
        b.iter(|| {
            BufWriter::new(io::sink())
        });
    }

    #[bench]
    fn bench_buffered_stream(b: &mut test::Bencher) {
        let mut buf = Cursor::new(Vec::new());
        b.iter(|| {
            BufStream::new(&mut buf);
        });
    }
}
