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

use io::prelude::*;

use cmp;
use error;
use fmt;
use io::{self, DEFAULT_BUF_SIZE, Error, ErrorKind, SeekFrom};
use memchr;

/// The `BufReader` struct adds buffering to any reader.
///
/// It can be excessively inefficient to work directly with a [`Read`] instance.
/// For example, every call to [`read`] on [`TcpStream`] results in a system call.
/// A `BufReader` performs large, infrequent reads on the underlying [`Read`]
/// and maintains an in-memory buffer of the results.
///
/// [`Read`]: ../../std/io/trait.Read.html
/// [`read`]: ../../std/net/struct.TcpStream.html#method.read
/// [`TcpStream`]: ../../std/net/struct.TcpStream.html
///
/// # Examples
///
/// ```
/// use std::io::prelude::*;
/// use std::io::BufReader;
/// use std::fs::File;
///
/// # fn foo() -> std::io::Result<()> {
/// let mut f = File::open("log.txt")?;
/// let mut reader = BufReader::new(f);
///
/// let mut line = String::new();
/// let len = reader.read_line(&mut line)?;
/// println!("First line is {} bytes long", len);
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BufReader<R> {
    inner: R,
    buf: Box<[u8]>,
    pos: usize,
    cap: usize,
}

impl<R: Read> BufReader<R> {
    /// Creates a new `BufReader` with a default buffer capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = File::open("log.txt")?;
    /// let mut reader = BufReader::new(f);
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(inner: R) -> BufReader<R> {
        BufReader::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    /// Creates a new `BufReader` with the specified buffer capacity.
    ///
    /// # Examples
    ///
    /// Creating a buffer with ten bytes of capacity:
    ///
    /// ```
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f = File::open("log.txt")?;
    /// let mut reader = BufReader::with_capacity(10, f);
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(cap: usize, inner: R) -> BufReader<R> {
        BufReader {
            inner: inner,
            buf: vec![0; cap].into_boxed_slice(),
            pos: 0,
            cap: 0,
        }
    }

    /// Gets a reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f1 = File::open("log.txt")?;
    /// let mut reader = BufReader::new(f1);
    ///
    /// let f2 = reader.get_ref();
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_ref(&self) -> &R { &self.inner }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f1 = File::open("log.txt")?;
    /// let mut reader = BufReader::new(f1);
    ///
    /// let f2 = reader.get_mut();
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut R { &mut self.inner }

    /// Unwraps this `BufReader`, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let mut f1 = File::open("log.txt")?;
    /// let mut reader = BufReader::new(f1);
    ///
    /// let f2 = reader.into_inner();
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(self) -> R { self.inner }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: Read> Read for BufReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.pos == self.cap && buf.len() >= self.buf.len() {
            return self.inner.read(buf);
        }
        let nread = {
            let mut rem = self.fill_buf()?;
            rem.read(buf)?
        };
        self.consume(nread);
        Ok(nread)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: Read> BufRead for BufReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        // If we've reached the end of our internal buffer then we need to fetch
        // some more data from the underlying reader.
        // Branch using `>=` instead of the more correct `==`
        // to tell the compiler that the pos..cap slice is always valid.
        if self.pos >= self.cap {
            debug_assert!(self.pos == self.cap);
            self.cap = self.inner.read(&mut self.buf)?;
            self.pos = 0;
        }
        Ok(&self.buf[self.pos..self.cap])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.cap);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R> fmt::Debug for BufReader<R> where R: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("BufReader")
            .field("reader", &self.inner)
            .field("buffer", &format_args!("{}/{}", self.cap - self.pos, self.buf.len()))
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: Seek> Seek for BufReader<R> {
    /// Seek to an offset, in bytes, in the underlying reader.
    ///
    /// The position used for seeking with `SeekFrom::Current(_)` is the
    /// position the underlying reader would be at if the `BufReader` had no
    /// internal buffer.
    ///
    /// Seeking always discards the internal buffer, even if the seek position
    /// would otherwise fall within it. This guarantees that calling
    /// `.into_inner()` immediately after a seek yields the underlying reader
    /// at the same position.
    ///
    /// See `std::io::Seek` for more details.
    ///
    /// Note: In the edge case where you're seeking with `SeekFrom::Current(n)`
    /// where `n` minus the internal buffer length underflows an `i64`, two
    /// seeks will be performed instead of one. If the second seek returns
    /// `Err`, the underlying reader will be left at the same position it would
    /// have if you seeked to `SeekFrom::Current(0)`.
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let result: u64;
        if let SeekFrom::Current(n) = pos {
            let remainder = (self.cap - self.pos) as i64;
            // it should be safe to assume that remainder fits within an i64 as the alternative
            // means we managed to allocate 8 exbibytes and that's absurd.
            // But it's not out of the realm of possibility for some weird underlying reader to
            // support seeking by i64::min_value() so we need to handle underflow when subtracting
            // remainder.
            if let Some(offset) = n.checked_sub(remainder) {
                result = self.inner.seek(SeekFrom::Current(offset))?;
            } else {
                // seek backwards by our remainder, and then by the offset
                self.inner.seek(SeekFrom::Current(-remainder))?;
                self.pos = self.cap; // empty the buffer
                result = self.inner.seek(SeekFrom::Current(n))?;
            }
        } else {
            // Seeking with Start/End doesn't care about our buffer length.
            result = self.inner.seek(pos)?;
        }
        self.pos = self.cap; // empty the buffer
        Ok(result)
    }
}

/// Wraps a writer and buffers its output.
///
/// It can be excessively inefficient to work directly with something that
/// implements [`Write`]. For example, every call to [`write`] on [`TcpStream`]
/// results in a system call. A `BufWriter` keeps an in-memory buffer of data
/// and writes it to an underlying writer in large, infrequent batches.
///
/// The buffer will be written out when the writer is dropped.
///
/// # Examples
///
/// Let's write the numbers one through ten to a [`TcpStream`]:
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::net::TcpStream;
///
/// let mut stream = TcpStream::connect("127.0.0.1:34254").unwrap();
///
/// for i in 1..10 {
///     stream.write(&[i]).unwrap();
/// }
/// ```
///
/// Because we're not buffering, we write each one in turn, incurring the
/// overhead of a system call per byte written. We can fix this with a
/// `BufWriter`:
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::io::BufWriter;
/// use std::net::TcpStream;
///
/// let mut stream = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
///
/// for i in 1..10 {
///     stream.write(&[i]).unwrap();
/// }
/// ```
///
/// By wrapping the stream with a `BufWriter`, these ten writes are all grouped
/// together by the buffer, and will all be written out in one system call when
/// the `stream` is dropped.
///
/// [`Write`]: ../../std/io/trait.Write.html
/// [`write`]: ../../std/net/struct.TcpStream.html#method.write
/// [`TcpStream`]: ../../std/net/struct.TcpStream.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BufWriter<W: Write> {
    inner: Option<W>,
    buf: Vec<u8>,
    // #30888: If the inner writer panics in a call to write, we don't want to
    // write the buffered data a second time in BufWriter's destructor. This
    // flag tells the Drop impl if it should skip the flush.
    panicked: bool,
}

/// An error returned by `into_inner` which combines an error that
/// happened while writing out the buffer, and the buffered writer object
/// which may be used to recover from the condition.
///
/// # Examples
///
/// ```no_run
/// use std::io::BufWriter;
/// use std::net::TcpStream;
///
/// let mut stream = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
///
/// // do stuff with the stream
///
/// // we want to get our `TcpStream` back, so let's try:
///
/// let stream = match stream.into_inner() {
///     Ok(s) => s,
///     Err(e) => {
///         // Here, e is an IntoInnerError
///         panic!("An error occurred");
///     }
/// };
/// ```
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoInnerError<W>(W, Error);

impl<W: Write> BufWriter<W> {
    /// Creates a new `BufWriter` with a default buffer capacity.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let mut buffer = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(inner: W) -> BufWriter<W> {
        BufWriter::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    /// Creates a new `BufWriter` with the specified buffer capacity.
    ///
    /// # Examples
    ///
    /// Creating a buffer with a buffer of a hundred bytes.
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:34254").unwrap();
    /// let mut buffer = BufWriter::with_capacity(100, stream);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(cap: usize, inner: W) -> BufWriter<W> {
        BufWriter {
            inner: Some(inner),
            buf: Vec::with_capacity(cap),
            panicked: false,
        }
    }

    fn flush_buf(&mut self) -> io::Result<()> {
        let mut written = 0;
        let len = self.buf.len();
        let mut ret = Ok(());
        while written < len {
            self.panicked = true;
            let r = self.inner.as_mut().unwrap().write(&self.buf[written..]);
            self.panicked = false;

            match r {
                Ok(0) => {
                    ret = Err(Error::new(ErrorKind::WriteZero,
                                         "failed to write the buffered data"));
                    break;
                }
                Ok(n) => written += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => { ret = Err(e); break }

            }
        }
        if written > 0 {
            self.buf.drain(..written);
        }
        ret
    }

    /// Gets a reference to the underlying writer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let mut buffer = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // we can use reference just like buffer
    /// let reference = buffer.get_ref();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_ref(&self) -> &W { self.inner.as_ref().unwrap() }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// It is inadvisable to directly write to the underlying writer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let mut buffer = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // we can use reference just like buffer
    /// let reference = buffer.get_mut();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut W { self.inner.as_mut().unwrap() }

    /// Unwraps this `BufWriter`, returning the underlying writer.
    ///
    /// The buffer is written out before returning the writer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let mut buffer = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // unwrap the TcpStream and flush the buffer
    /// let stream = buffer.into_inner().unwrap();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(mut self) -> Result<W, IntoInnerError<BufWriter<W>>> {
        match self.flush_buf() {
            Err(e) => Err(IntoInnerError(self, e)),
            Ok(()) => Ok(self.inner.take().unwrap())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> Write for BufWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.buf.len() + buf.len() > self.buf.capacity() {
            self.flush_buf()?;
        }
        if buf.len() >= self.buf.capacity() {
            self.panicked = true;
            let r = self.inner.as_mut().unwrap().write(buf);
            self.panicked = false;
            r
        } else {
            Write::write(&mut self.buf, buf)
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flush_buf().and_then(|()| self.get_mut().flush())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> fmt::Debug for BufWriter<W> where W: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("BufWriter")
            .field("writer", &self.inner.as_ref().unwrap())
            .field("buffer", &format_args!("{}/{}", self.buf.len(), self.buf.capacity()))
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write + Seek> Seek for BufWriter<W> {
    /// Seek to the offset, in bytes, in the underlying writer.
    ///
    /// Seeking always writes out the internal buffer before seeking.
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.flush_buf().and_then(|_| self.get_mut().seek(pos))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> Drop for BufWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some() && !self.panicked {
            // dtors should not panic, so we ignore a failed flush
            let _r = self.flush_buf();
        }
    }
}

impl<W> IntoInnerError<W> {
    /// Returns the error which caused the call to `into_inner()` to fail.
    ///
    /// This error was returned when attempting to write the internal buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let mut stream = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // do stuff with the stream
    ///
    /// // we want to get our `TcpStream` back, so let's try:
    ///
    /// let stream = match stream.into_inner() {
    ///     Ok(s) => s,
    ///     Err(e) => {
    ///         // Here, e is an IntoInnerError, let's log the inner error.
    ///         //
    ///         // We'll just 'log' to stdout for this example.
    ///         println!("{}", e.error());
    ///
    ///         panic!("An unexpected error occurred.");
    ///     }
    /// };
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn error(&self) -> &Error { &self.1 }

    /// Returns the buffered writer instance which generated the error.
    ///
    /// The returned object can be used for error recovery, such as
    /// re-inspecting the buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let mut stream = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // do stuff with the stream
    ///
    /// // we want to get our `TcpStream` back, so let's try:
    ///
    /// let stream = match stream.into_inner() {
    ///     Ok(s) => s,
    ///     Err(e) => {
    ///         // Here, e is an IntoInnerError, let's re-examine the buffer:
    ///         let buffer = e.into_inner();
    ///
    ///         // do stuff to try to recover
    ///
    ///         // afterwards, let's just return the stream
    ///         buffer.into_inner().unwrap()
    ///     }
    /// };
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(self) -> W { self.0 }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W> From<IntoInnerError<W>> for Error {
    fn from(iie: IntoInnerError<W>) -> Error { iie.1 }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Send + fmt::Debug> error::Error for IntoInnerError<W> {
    fn description(&self) -> &str {
        error::Error::description(self.error())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W> fmt::Display for IntoInnerError<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.error().fmt(f)
    }
}

/// Wraps a writer and buffers output to it, flushing whenever a newline
/// (`0x0a`, `'\n'`) is detected.
///
/// The [`BufWriter`][bufwriter] struct wraps a writer and buffers its output.
/// But it only does this batched write when it goes out of scope, or when the
/// internal buffer is full. Sometimes, you'd prefer to write each line as it's
/// completed, rather than the entire buffer at once. Enter `LineWriter`. It
/// does exactly that.
///
/// [bufwriter]: struct.BufWriter.html
///
/// If there's still a partial line in the buffer when the `LineWriter` is
/// dropped, it will flush those contents.
///
/// # Examples
///
/// We can use `LineWriter` to write one line at a time, significantly
/// reducing the number of actual writes to the file.
///
/// ```
/// use std::fs::File;
/// use std::io::prelude::*;
/// use std::io::LineWriter;
///
/// # fn foo() -> std::io::Result<()> {
/// let road_not_taken = b"I shall be telling this with a sigh
/// Somewhere ages and ages hence:
/// Two roads diverged in a wood, and I -
/// I took the one less traveled by,
/// And that has made all the difference.";
///
/// let file = File::create("poem.txt")?;
/// let mut file = LineWriter::new(file);
///
/// for &byte in road_not_taken.iter() {
///    file.write(&[byte]).unwrap();
/// }
///
/// // let's check we did the right thing.
/// let mut file = File::open("poem.txt")?;
/// let mut contents = String::new();
///
/// file.read_to_string(&mut contents)?;
///
/// assert_eq!(contents.as_bytes(), &road_not_taken[..]);
/// # Ok(())
/// # }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LineWriter<W: Write> {
    inner: BufWriter<W>,
    need_flush: bool,
}

impl<W: Write> LineWriter<W> {
    /// Creates a new `LineWriter`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let file = File::create("poem.txt")?;
    /// let file = LineWriter::new(file);
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(inner: W) -> LineWriter<W> {
        // Lines typically aren't that long, don't use a giant buffer
        LineWriter::with_capacity(1024, inner)
    }

    /// Creates a new `LineWriter` with a specified capacity for the internal
    /// buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let file = File::create("poem.txt")?;
    /// let file = LineWriter::with_capacity(100, file);
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(cap: usize, inner: W) -> LineWriter<W> {
        LineWriter {
            inner: BufWriter::with_capacity(cap, inner),
            need_flush: false,
        }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let file = File::create("poem.txt")?;
    /// let file = LineWriter::new(file);
    ///
    /// let reference = file.get_ref();
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_ref(&self) -> &W { self.inner.get_ref() }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// Caution must be taken when calling methods on the mutable reference
    /// returned as extra writes could corrupt the output stream.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let file = File::create("poem.txt")?;
    /// let mut file = LineWriter::new(file);
    ///
    /// // we can use reference just like file
    /// let reference = file.get_mut();
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut W { self.inner.get_mut() }

    /// Unwraps this `LineWriter`, returning the underlying writer.
    ///
    /// The internal buffer is written out before returning the writer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// # fn foo() -> std::io::Result<()> {
    /// let file = File::create("poem.txt")?;
    ///
    /// let writer: LineWriter<File> = LineWriter::new(file);
    ///
    /// let file: File = writer.into_inner()?;
    /// # Ok(())
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(self) -> Result<W, IntoInnerError<LineWriter<W>>> {
        self.inner.into_inner().map_err(|IntoInnerError(buf, e)| {
            IntoInnerError(LineWriter {
                inner: buf,
                need_flush: false,
            }, e)
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> Write for LineWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.need_flush {
            self.flush()?;
        }

        // Find the last newline character in the buffer provided. If found then
        // we're going to write all the data up to that point and then flush,
        // otherewise we just write the whole block to the underlying writer.
        let i = match memchr::memrchr(b'\n', buf) {
            Some(i) => i,
            None => return self.inner.write(buf),
        };


        // Ok, we're going to write a partial amount of the data given first
        // followed by flushing the newline. After we've successfully written
        // some data then we *must* report that we wrote that data, so future
        // errors are ignored. We set our internal `need_flush` flag, though, in
        // case flushing fails and we need to try it first next time.
        let n = self.inner.write(&buf[..i + 1])?;
        self.need_flush = true;
        if self.flush().is_err() || n != i + 1 {
            return Ok(n)
        }

        // At this point we successfully wrote `i + 1` bytes and flushed it out,
        // meaning that the entire line is now flushed out on the screen. While
        // we can attempt to finish writing the rest of the data provided.
        // Remember though that we ignore errors here as we've successfully
        // written data, so we need to report that.
        match self.inner.write(&buf[i + 1..]) {
            Ok(i) => Ok(n + i),
            Err(_) => Ok(n),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()?;
        self.need_flush = false;
        Ok(())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> fmt::Debug for LineWriter<W> where W: fmt::Debug {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("LineWriter")
            .field("writer", &self.inner.inner)
            .field("buffer",
                   &format_args!("{}/{}", self.inner.buf.len(), self.inner.buf.capacity()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use io::prelude::*;
    use io::{self, BufReader, BufWriter, LineWriter, SeekFrom};
    use sync::atomic::{AtomicUsize, Ordering};
    use thread;
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
        assert_eq!(nread.unwrap(), 3);
        let b: &[_] = &[5, 6, 7];
        assert_eq!(buf, b);

        let mut buf = [0, 0];
        let nread = reader.read(&mut buf);
        assert_eq!(nread.unwrap(), 2);
        let b: &[_] = &[0, 1];
        assert_eq!(buf, b);

        let mut buf = [0];
        let nread = reader.read(&mut buf);
        assert_eq!(nread.unwrap(), 1);
        let b: &[_] = &[2];
        assert_eq!(buf, b);

        let mut buf = [0, 0, 0];
        let nread = reader.read(&mut buf);
        assert_eq!(nread.unwrap(), 1);
        let b: &[_] = &[3, 0, 0];
        assert_eq!(buf, b);

        let nread = reader.read(&mut buf);
        assert_eq!(nread.unwrap(), 1);
        let b: &[_] = &[4, 0, 0];
        assert_eq!(buf, b);

        assert_eq!(reader.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn test_buffered_reader_seek() {
        let inner: &[u8] = &[5, 6, 7, 0, 1, 2, 3, 4];
        let mut reader = BufReader::with_capacity(2, io::Cursor::new(inner));

        assert_eq!(reader.seek(SeekFrom::Start(3)).ok(), Some(3));
        assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
        assert_eq!(reader.seek(SeekFrom::Current(0)).ok(), Some(3));
        assert_eq!(reader.fill_buf().ok(), Some(&[0, 1][..]));
        assert_eq!(reader.seek(SeekFrom::Current(1)).ok(), Some(4));
        assert_eq!(reader.fill_buf().ok(), Some(&[1, 2][..]));
        reader.consume(1);
        assert_eq!(reader.seek(SeekFrom::Current(-2)).ok(), Some(3));
    }

    #[test]
    fn test_buffered_reader_seek_underflow() {
        // gimmick reader that yields its position modulo 256 for each byte
        struct PositionReader {
            pos: u64
        }
        impl Read for PositionReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                let len = buf.len();
                for x in buf {
                    *x = self.pos as u8;
                    self.pos = self.pos.wrapping_add(1);
                }
                Ok(len)
            }
        }
        impl Seek for PositionReader {
            fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
                match pos {
                    SeekFrom::Start(n) => {
                        self.pos = n;
                    }
                    SeekFrom::Current(n) => {
                        self.pos = self.pos.wrapping_add(n as u64);
                    }
                    SeekFrom::End(n) => {
                        self.pos = u64::max_value().wrapping_add(n as u64);
                    }
                }
                Ok(self.pos)
            }
        }

        let mut reader = BufReader::with_capacity(5, PositionReader { pos: 0 });
        assert_eq!(reader.fill_buf().ok(), Some(&[0, 1, 2, 3, 4][..]));
        assert_eq!(reader.seek(SeekFrom::End(-5)).ok(), Some(u64::max_value()-5));
        assert_eq!(reader.fill_buf().ok().map(|s| s.len()), Some(5));
        // the following seek will require two underlying seeks
        let expected = 9223372036854775802;
        assert_eq!(reader.seek(SeekFrom::Current(i64::min_value())).ok(), Some(expected));
        assert_eq!(reader.fill_buf().ok().map(|s| s.len()), Some(5));
        // seeking to 0 should empty the buffer.
        assert_eq!(reader.seek(SeekFrom::Current(0)).ok(), Some(expected));
        assert_eq!(reader.get_ref().pos, expected);
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

    #[test]
    fn test_buffered_writer_seek() {
        let mut w = BufWriter::with_capacity(3, io::Cursor::new(Vec::new()));
        w.write_all(&[0, 1, 2, 3, 4, 5]).unwrap();
        w.write_all(&[6, 7]).unwrap();
        assert_eq!(w.seek(SeekFrom::Current(0)).ok(), Some(8));
        assert_eq!(&w.get_ref().get_ref()[..], &[0, 1, 2, 3, 4, 5, 6, 7][..]);
        assert_eq!(w.seek(SeekFrom::Start(2)).ok(), Some(2));
        w.write_all(&[8, 9]).unwrap();
        assert_eq!(&w.into_inner().unwrap().into_inner()[..], &[0, 1, 8, 9, 4, 5, 6, 7]);
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
    fn test_line_buffer_fail_flush() {
        // Issue #32085
        struct FailFlushWriter<'a>(&'a mut Vec<u8>);

        impl<'a> Write for FailFlushWriter<'a> {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::Other, "flush failed"))
            }
        }

        let mut buf = Vec::new();
        {
            let mut writer = LineWriter::new(FailFlushWriter(&mut buf));
            let to_write = b"abc\ndef";
            if let Ok(written) = writer.write(to_write) {
                assert!(written < to_write.len(), "didn't flush on new line");
                // PASS
                return;
            }
        }
        assert!(buf.is_empty(), "write returned an error but wrote data");
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
        let in_buf: &[u8] = b"a\nb\nc";
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
        let in_buf: &[u8] = b"a\nb\nc";
        let reader = BufReader::with_capacity(2, in_buf);
        let mut it = reader.lines();
        assert_eq!(it.next().unwrap().unwrap(), "a".to_string());
        assert_eq!(it.next().unwrap().unwrap(), "b".to_string());
        assert_eq!(it.next().unwrap().unwrap(), "c".to_string());
        assert!(it.next().is_none());
    }

    #[test]
    fn test_short_reads() {
        let inner = ShortReader{lengths: vec![0, 1, 2, 0, 1, 0]};
        let mut reader = BufReader::new(inner);
        let mut buf = [0, 0];
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
        assert_eq!(reader.read(&mut buf).unwrap(), 1);
        assert_eq!(reader.read(&mut buf).unwrap(), 2);
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
        assert_eq!(reader.read(&mut buf).unwrap(), 1);
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn read_char_buffered() {
        let buf = [195, 159];
        let reader = BufReader::with_capacity(1, &buf[..]);
        assert_eq!(reader.chars().next().unwrap().unwrap(), 'ß');
    }

    #[test]
    fn test_chars() {
        let buf = [195, 159, b'a'];
        let reader = BufReader::with_capacity(1, &buf[..]);
        let mut it = reader.chars();
        assert_eq!(it.next().unwrap().unwrap(), 'ß');
        assert_eq!(it.next().unwrap().unwrap(), 'a');
        assert!(it.next().is_none());
    }

    #[test]
    #[should_panic]
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

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)]
    fn panic_in_write_doesnt_flush_in_drop() {
        static WRITES: AtomicUsize = AtomicUsize::new(0);

        struct PanicWriter;

        impl Write for PanicWriter {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                WRITES.fetch_add(1, Ordering::SeqCst);
                panic!();
            }
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }

        thread::spawn(|| {
            let mut writer = BufWriter::new(PanicWriter);
            let _ = writer.write(b"hello world");
            let _ = writer.flush();
        }).join().unwrap_err();

        assert_eq!(WRITES.load(Ordering::SeqCst), 1);
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

    struct AcceptOneThenFail {
        written: bool,
        flushed: bool,
    }

    impl Write for AcceptOneThenFail {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if !self.written {
                assert_eq!(data, b"a\nb\n");
                self.written = true;
                Ok(data.len())
            } else {
                Err(io::Error::new(io::ErrorKind::NotFound, "test"))
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            assert!(self.written);
            assert!(!self.flushed);
            self.flushed = true;
            Err(io::Error::new(io::ErrorKind::Other, "test"))
        }
    }

    #[test]
    fn erroneous_flush_retried() {
        let a = AcceptOneThenFail {
            written: false,
            flushed: false,
        };

        let mut l = LineWriter::new(a);
        assert_eq!(l.write(b"a\nb\na").unwrap(), 4);
        assert!(l.get_ref().written);
        assert!(l.get_ref().flushed);
        l.get_mut().flushed = false;

        assert_eq!(l.write(b"a").unwrap_err().kind(), io::ErrorKind::Other)
    }
}
