//! Buffering wrappers for I/O traits

#[cfg(test)]
mod tests;

use crate::io::prelude::*;

use crate::cmp;
use crate::error;
use crate::fmt;
use crate::io::{
    self, Error, ErrorKind, Initializer, IoSlice, IoSliceMut, SeekFrom, DEFAULT_BUF_SIZE,
};
use crate::memchr;

/// The `BufReader<R>` struct adds buffering to any reader.
///
/// It can be excessively inefficient to work directly with a [`Read`] instance.
/// For example, every call to [`read`][`TcpStream::read`] on [`TcpStream`]
/// results in a system call. A `BufReader<R>` performs large, infrequent reads on
/// the underlying [`Read`] and maintains an in-memory buffer of the results.
///
/// `BufReader<R>` can improve the speed of programs that make *small* and
/// *repeated* read calls to the same file or network socket. It does not
/// help when reading very large amounts at once, or reading just one or a few
/// times. It also provides no advantage when reading from a source that is
/// already in memory, like a [`Vec`]`<u8>`.
///
/// When the `BufReader<R>` is dropped, the contents of its buffer will be
/// discarded. Creating multiple instances of a `BufReader<R>` on the same
/// stream can cause data loss. Reading from the underlying reader after
/// unwrapping the `BufReader<R>` with [`BufReader::into_inner`] can also cause
/// data loss.
///
/// [`TcpStream::read`]: Read::read
/// [`TcpStream`]: crate::net::TcpStream
///
/// # Examples
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::io::BufReader;
/// use std::fs::File;
///
/// fn main() -> std::io::Result<()> {
///     let f = File::open("log.txt")?;
///     let mut reader = BufReader::new(f);
///
///     let mut line = String::new();
///     let len = reader.read_line(&mut line)?;
///     println!("First line is {} bytes long", len);
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BufReader<R> {
    inner: R,
    buf: Box<[u8]>,
    pos: usize,
    cap: usize,
}

impl<R: Read> BufReader<R> {
    /// Creates a new `BufReader<R>` with a default buffer capacity. The default is currently 8 KB,
    /// but may change in the future.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let reader = BufReader::new(f);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new(inner: R) -> BufReader<R> {
        BufReader::with_capacity(DEFAULT_BUF_SIZE, inner)
    }

    /// Creates a new `BufReader<R>` with the specified buffer capacity.
    ///
    /// # Examples
    ///
    /// Creating a buffer with ten bytes of capacity:
    ///
    /// ```no_run
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let reader = BufReader::with_capacity(10, f);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize, inner: R) -> BufReader<R> {
        unsafe {
            let mut buffer = Vec::with_capacity(capacity);
            buffer.set_len(capacity);
            inner.initializer().initialize(&mut buffer);
            BufReader { inner, buf: buffer.into_boxed_slice(), pos: 0, cap: 0 }
        }
    }
}

impl<R> BufReader<R> {
    /// Gets a reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f1 = File::open("log.txt")?;
    ///     let reader = BufReader::new(f1);
    ///
    ///     let f2 = reader.get_ref();
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// It is inadvisable to directly read from the underlying reader.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f1 = File::open("log.txt")?;
    ///     let mut reader = BufReader::new(f1);
    ///
    ///     let f2 = reader.get_mut();
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Returns a reference to the internally buffered data.
    ///
    /// Unlike [`fill_buf`], this will not attempt to fill the buffer if it is empty.
    ///
    /// [`fill_buf`]: BufRead::fill_buf
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::{BufReader, BufRead};
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let mut reader = BufReader::new(f);
    ///     assert!(reader.buffer().is_empty());
    ///
    ///     if reader.fill_buf()?.len() > 0 {
    ///         assert!(!reader.buffer().is_empty());
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "bufreader_buffer", since = "1.37.0")]
    pub fn buffer(&self) -> &[u8] {
        &self.buf[self.pos..self.cap]
    }

    /// Returns the number of bytes the internal buffer can hold at once.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::{BufReader, BufRead};
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f = File::open("log.txt")?;
    ///     let mut reader = BufReader::new(f);
    ///
    ///     let capacity = reader.capacity();
    ///     let buffer = reader.fill_buf()?;
    ///     assert!(buffer.len() <= capacity);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "buffered_io_capacity", since = "1.46.0")]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Unwraps this `BufReader<R>`, returning the underlying reader.
    ///
    /// Note that any leftover data in the internal buffer is lost. Therefore,
    /// a following read from the underlying reader may lead to data loss.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufReader;
    /// use std::fs::File;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let f1 = File::open("log.txt")?;
    ///     let reader = BufReader::new(f1);
    ///
    ///     let f2 = reader.into_inner();
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Invalidates all data in the internal buffer.
    #[inline]
    fn discard_buffer(&mut self) {
        self.pos = 0;
        self.cap = 0;
    }
}

impl<R: Seek> BufReader<R> {
    /// Seeks relative to the current position. If the new position lies within the buffer,
    /// the buffer will not be flushed, allowing for more efficient seeks.
    /// This method does not return the location of the underlying reader, so the caller
    /// must track this information themselves if it is required.
    #[unstable(feature = "bufreader_seek_relative", issue = "31100")]
    pub fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        let pos = self.pos as u64;
        if offset < 0 {
            if let Some(new_pos) = pos.checked_sub((-offset) as u64) {
                self.pos = new_pos as usize;
                return Ok(());
            }
        } else {
            if let Some(new_pos) = pos.checked_add(offset as u64) {
                if new_pos <= self.cap as u64 {
                    self.pos = new_pos as usize;
                    return Ok(());
                }
            }
        }
        self.seek(SeekFrom::Current(offset)).map(drop)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: Read> Read for BufReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.pos == self.cap && buf.len() >= self.buf.len() {
            self.discard_buffer();
            return self.inner.read(buf);
        }
        let nread = {
            let mut rem = self.fill_buf()?;
            rem.read(buf)?
        };
        self.consume(nread);
        Ok(nread)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum::<usize>();
        if self.pos == self.cap && total_len >= self.buf.len() {
            self.discard_buffer();
            return self.inner.read_vectored(bufs);
        }
        let nread = {
            let mut rem = self.fill_buf()?;
            rem.read_vectored(bufs)?
        };
        self.consume(nread);
        Ok(nread)
    }

    fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }

    // we can't skip unconditionally because of the large buffer case in read.
    unsafe fn initializer(&self) -> Initializer {
        self.inner.initializer()
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
impl<R> fmt::Debug for BufReader<R>
where
    R: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    /// The position used for seeking with [`SeekFrom::Current`]`(_)` is the
    /// position the underlying reader would be at if the `BufReader<R>` had no
    /// internal buffer.
    ///
    /// Seeking always discards the internal buffer, even if the seek position
    /// would otherwise fall within it. This guarantees that calling
    /// [`BufReader::into_inner()`] immediately after a seek yields the underlying reader
    /// at the same position.
    ///
    /// To seek without discarding the internal buffer, use [`BufReader::seek_relative`].
    ///
    /// See [`std::io::Seek`] for more details.
    ///
    /// Note: In the edge case where you're seeking with [`SeekFrom::Current`]`(n)`
    /// where `n` minus the internal buffer length overflows an `i64`, two
    /// seeks will be performed instead of one. If the second seek returns
    /// [`Err`], the underlying reader will be left at the same position it would
    /// have if you called `seek` with [`SeekFrom::Current`]`(0)`.
    ///
    /// [`std::io::Seek`]: Seek
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let result: u64;
        if let SeekFrom::Current(n) = pos {
            let remainder = (self.cap - self.pos) as i64;
            // it should be safe to assume that remainder fits within an i64 as the alternative
            // means we managed to allocate 8 exbibytes and that's absurd.
            // But it's not out of the realm of possibility for some weird underlying reader to
            // support seeking by i64::MIN so we need to handle underflow when subtracting
            // remainder.
            if let Some(offset) = n.checked_sub(remainder) {
                result = self.inner.seek(SeekFrom::Current(offset))?;
            } else {
                // seek backwards by our remainder, and then by the offset
                self.inner.seek(SeekFrom::Current(-remainder))?;
                self.discard_buffer();
                result = self.inner.seek(SeekFrom::Current(n))?;
            }
        } else {
            // Seeking with Start/End doesn't care about our buffer length.
            result = self.inner.seek(pos)?;
        }
        self.discard_buffer();
        Ok(result)
    }
}

/// Wraps a writer and buffers its output.
///
/// It can be excessively inefficient to work directly with something that
/// implements [`Write`]. For example, every call to
/// [`write`][`TcpStream::write`] on [`TcpStream`] results in a system call. A
/// `BufWriter<W>` keeps an in-memory buffer of data and writes it to an underlying
/// writer in large, infrequent batches.
///
/// `BufWriter<W>` can improve the speed of programs that make *small* and
/// *repeated* write calls to the same file or network socket. It does not
/// help when writing very large amounts at once, or writing just one or a few
/// times. It also provides no advantage when writing to a destination that is
/// in memory, like a [`Vec`]<u8>`.
///
/// It is critical to call [`flush`] before `BufWriter<W>` is dropped. Though
/// dropping will attempt to flush the contents of the buffer, any errors
/// that happen in the process of dropping will be ignored. Calling [`flush`]
/// ensures that the buffer is empty and thus dropping will not even attempt
/// file operations.
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
/// for i in 0..10 {
///     stream.write(&[i+1]).unwrap();
/// }
/// ```
///
/// Because we're not buffering, we write each one in turn, incurring the
/// overhead of a system call per byte written. We can fix this with a
/// `BufWriter<W>`:
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::io::BufWriter;
/// use std::net::TcpStream;
///
/// let mut stream = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
///
/// for i in 0..10 {
///     stream.write(&[i+1]).unwrap();
/// }
/// stream.flush().unwrap();
/// ```
///
/// By wrapping the stream with a `BufWriter<W>`, these ten writes are all grouped
/// together by the buffer and will all be written out in one system call when
/// the `stream` is flushed.
///
/// [`TcpStream::write`]: Write::write
/// [`TcpStream`]: crate::net::TcpStream
/// [`flush`]: Write::flush
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BufWriter<W: Write> {
    inner: Option<W>,
    buf: Vec<u8>,
    // #30888: If the inner writer panics in a call to write, we don't want to
    // write the buffered data a second time in BufWriter's destructor. This
    // flag tells the Drop impl if it should skip the flush.
    panicked: bool,
}

/// An error returned by [`BufWriter::into_inner`] which combines an error that
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
    /// Creates a new `BufWriter<W>` with a default buffer capacity. The default is currently 8 KB,
    /// but may change in the future.
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

    /// Creates a new `BufWriter<W>` with the specified buffer capacity.
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
    pub fn with_capacity(capacity: usize, inner: W) -> BufWriter<W> {
        BufWriter { inner: Some(inner), buf: Vec::with_capacity(capacity), panicked: false }
    }

    /// Send data in our local buffer into the inner writer, looping as
    /// necessary until either it's all been sent or an error occurs.
    ///
    /// Because all the data in the buffer has been reported to our owner as
    /// "successfully written" (by returning nonzero success values from
    /// `write`), any 0-length writes from `inner` must be reported as i/o
    /// errors from this method.
    fn flush_buf(&mut self) -> io::Result<()> {
        /// Helper struct to ensure the buffer is updated after all the writes
        /// are complete. It tracks the number of written bytes and drains them
        /// all from the front of the buffer when dropped.
        struct BufGuard<'a> {
            buffer: &'a mut Vec<u8>,
            written: usize,
        }

        impl<'a> BufGuard<'a> {
            fn new(buffer: &'a mut Vec<u8>) -> Self {
                Self { buffer, written: 0 }
            }

            /// The unwritten part of the buffer
            fn remaining(&self) -> &[u8] {
                &self.buffer[self.written..]
            }

            /// Flag some bytes as removed from the front of the buffer
            fn consume(&mut self, amt: usize) {
                self.written += amt;
            }

            /// true if all of the bytes have been written
            fn done(&self) -> bool {
                self.written >= self.buffer.len()
            }
        }

        impl Drop for BufGuard<'_> {
            fn drop(&mut self) {
                if self.written > 0 {
                    self.buffer.drain(..self.written);
                }
            }
        }

        let mut guard = BufGuard::new(&mut self.buf);
        let inner = self.inner.as_mut().unwrap();
        while !guard.done() {
            self.panicked = true;
            let r = inner.write(guard.remaining());
            self.panicked = false;

            match r {
                Ok(0) => {
                    return Err(Error::new(
                        ErrorKind::WriteZero,
                        "failed to write the buffered data",
                    ));
                }
                Ok(n) => guard.consume(n),
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Buffer some data without flushing it, regardless of the size of the
    /// data. Writes as much as possible without exceeding capacity. Returns
    /// the number of bytes written.
    fn write_to_buf(&mut self, buf: &[u8]) -> usize {
        let available = self.buf.capacity() - self.buf.len();
        let amt_to_buffer = available.min(buf.len());
        self.buf.extend_from_slice(&buf[..amt_to_buffer]);
        amt_to_buffer
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
    pub fn get_ref(&self) -> &W {
        self.inner.as_ref().unwrap()
    }

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
    pub fn get_mut(&mut self) -> &mut W {
        self.inner.as_mut().unwrap()
    }

    /// Returns a reference to the internally buffered data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let buf_writer = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // See how many bytes are currently buffered
    /// let bytes_buffered = buf_writer.buffer().len();
    /// ```
    #[stable(feature = "bufreader_buffer", since = "1.37.0")]
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }

    /// Returns the number of bytes the internal buffer can hold without flushing.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::io::BufWriter;
    /// use std::net::TcpStream;
    ///
    /// let buf_writer = BufWriter::new(TcpStream::connect("127.0.0.1:34254").unwrap());
    ///
    /// // Check the capacity of the inner buffer
    /// let capacity = buf_writer.capacity();
    /// // Calculate how many bytes can be written without flushing
    /// let without_flush = capacity - buf_writer.buffer().len();
    /// ```
    #[stable(feature = "buffered_io_capacity", since = "1.46.0")]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Unwraps this `BufWriter<W>`, returning the underlying writer.
    ///
    /// The buffer is written out before returning the writer.
    ///
    /// # Errors
    ///
    /// An [`Err`] will be returned if an error occurs while flushing the buffer.
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
            Ok(()) => Ok(self.inner.take().unwrap()),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> Write for BufWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.buf.len() + buf.len() > self.buf.capacity() {
            self.flush_buf()?;
        }
        // FIXME: Why no len > capacity? Why not buffer len == capacity? #72919
        if buf.len() >= self.buf.capacity() {
            self.panicked = true;
            let r = self.get_mut().write(buf);
            self.panicked = false;
            r
        } else {
            self.buf.extend_from_slice(buf);
            Ok(buf.len())
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        // Normally, `write_all` just calls `write` in a loop. We can do better
        // by calling `self.get_mut().write_all()` directly, which avoids
        // round trips through the buffer in the event of a series of partial
        // writes in some circumstances.
        if self.buf.len() + buf.len() > self.buf.capacity() {
            self.flush_buf()?;
        }
        // FIXME: Why no len > capacity? Why not buffer len == capacity? #72919
        if buf.len() >= self.buf.capacity() {
            self.panicked = true;
            let r = self.get_mut().write_all(buf);
            self.panicked = false;
            r
        } else {
            self.buf.extend_from_slice(buf);
            Ok(())
        }
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum::<usize>();
        if self.buf.len() + total_len > self.buf.capacity() {
            self.flush_buf()?;
        }
        // FIXME: Why no len > capacity? Why not buffer len == capacity? #72919
        if total_len >= self.buf.capacity() {
            self.panicked = true;
            let r = self.get_mut().write_vectored(bufs);
            self.panicked = false;
            r
        } else {
            bufs.iter().for_each(|b| self.buf.extend_from_slice(b));
            Ok(total_len)
        }
    }

    fn is_write_vectored(&self) -> bool {
        self.get_ref().is_write_vectored()
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buf().and_then(|()| self.get_mut().flush())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> fmt::Debug for BufWriter<W>
where
    W: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        self.flush_buf()?;
        self.get_mut().seek(pos)
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
    /// Returns the error which caused the call to [`BufWriter::into_inner()`]
    /// to fail.
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
    pub fn error(&self) -> &Error {
        &self.1
    }

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
    pub fn into_inner(self) -> W {
        self.0
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W> From<IntoInnerError<W>> for Error {
    fn from(iie: IntoInnerError<W>) -> Error {
        iie.1
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Send + fmt::Debug> error::Error for IntoInnerError<W> {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        error::Error::description(self.error())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W> fmt::Display for IntoInnerError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error().fmt(f)
    }
}

/// Private helper struct for implementing the line-buffered writing logic.
/// This shim temporarily wraps a BufWriter, and uses its internals to
/// implement a line-buffered writer (specifically by using the internal
/// methods like write_to_buf and flush_buf). In this way, a more
/// efficient abstraction can be created than one that only had access to
/// `write` and `flush`, without needlessly duplicating a lot of the
/// implementation details of BufWriter. This also allows existing
/// `BufWriters` to be temporarily given line-buffering logic; this is what
/// enables Stdout to be alternately in line-buffered or block-buffered mode.
#[derive(Debug)]
pub(super) struct LineWriterShim<'a, W: Write> {
    buffer: &'a mut BufWriter<W>,
}

impl<'a, W: Write> LineWriterShim<'a, W> {
    pub fn new(buffer: &'a mut BufWriter<W>) -> Self {
        Self { buffer }
    }

    /// Get a mutable reference to the inner writer (that is, the writer
    /// wrapped by the BufWriter). Be careful with this writer, as writes to
    /// it will bypass the buffer.
    fn inner_mut(&mut self) -> &mut W {
        self.buffer.get_mut()
    }

    /// Get the content currently buffered in self.buffer
    fn buffered(&self) -> &[u8] {
        self.buffer.buffer()
    }

    /// Flush the buffer iff the last byte is a newline (indicating that an
    /// earlier write only succeeded partially, and we want to retry flushing
    /// the buffered line before continuing with a subsequent write)
    fn flush_if_completed_line(&mut self) -> io::Result<()> {
        match self.buffered().last().copied() {
            Some(b'\n') => self.buffer.flush_buf(),
            _ => Ok(()),
        }
    }
}

impl<'a, W: Write> Write for LineWriterShim<'a, W> {
    /// Write some data into this BufReader with line buffering. This means
    /// that, if any newlines are present in the data, the data up to the last
    /// newline is sent directly to the underlying writer, and data after it
    /// is buffered. Returns the number of bytes written.
    ///
    /// This function operates on a "best effort basis"; in keeping with the
    /// convention of `Write::write`, it makes at most one attempt to write
    /// new data to the underlying writer. If that write only reports a partial
    /// success, the remaining data will be buffered.
    ///
    /// Because this function attempts to send completed lines to the underlying
    /// writer, it will also flush the existing buffer if it ends with a
    /// newline, even if the incoming data does not contain any newlines.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let newline_idx = match memchr::memrchr(b'\n', buf) {
            // If there are no new newlines (that is, if this write is less than
            // one line), just do a regular buffered write (which may flush if
            // we exceed the inner buffer's size)
            None => {
                self.flush_if_completed_line()?;
                return self.buffer.write(buf);
            }
            // Otherwise, arrange for the lines to be written directly to the
            // inner writer.
            Some(newline_idx) => newline_idx + 1,
        };

        // Flush existing content to prepare for our write. We have to do this
        // before attempting to write `buf` in order to maintain consistency;
        // if we add `buf` to the buffer then try to flush it all at once,
        // we're obligated to return Ok(), which would mean suppressing any
        // errors that occur during flush.
        self.buffer.flush_buf()?;

        // This is what we're going to try to write directly to the inner
        // writer. The rest will be buffered, if nothing goes wrong.
        let lines = &buf[..newline_idx];

        // Write `lines` directly to the inner writer. In keeping with the
        // `write` convention, make at most one attempt to add new (unbuffered)
        // data. Because this write doesn't touch the BufWriter state directly,
        // and the buffer is known to be empty, we don't need to worry about
        // self.buffer.panicked here.
        let flushed = self.inner_mut().write(lines)?;

        // If buffer returns Ok(0), propagate that to the caller without
        // doing additional buffering; otherwise we're just guaranteeing
        // an "ErrorKind::WriteZero" later.
        if flushed == 0 {
            return Ok(0);
        }

        // Now that the write has succeeded, buffer the rest (or as much of
        // the rest as possible). If there were any unwritten newlines, we
        // only buffer out to the last unwritten newline that fits in the
        // buffer; this helps prevent flushing partial lines on subsequent
        // calls to LineWriterShim::write.

        // Handle the cases in order of most-common to least-common, under
        // the presumption that most writes succeed in totality, and that most
        // writes are smaller than the buffer.
        // - Is this a partial line (ie, no newlines left in the unwritten tail)
        // - If not, does the data out to the last unwritten newline fit in
        //   the buffer?
        // - If not, scan for the last newline that *does* fit in the buffer
        let tail = if flushed >= newline_idx {
            &buf[flushed..]
        } else if newline_idx - flushed <= self.buffer.capacity() {
            &buf[flushed..newline_idx]
        } else {
            let scan_area = &buf[flushed..];
            let scan_area = &scan_area[..self.buffer.capacity()];
            match memchr::memrchr(b'\n', scan_area) {
                Some(newline_idx) => &scan_area[..newline_idx + 1],
                None => scan_area,
            }
        };

        let buffered = self.buffer.write_to_buf(tail);
        Ok(flushed + buffered)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buffer.flush()
    }

    /// Write some vectored data into this BufReader with line buffering. This
    /// means that, if any newlines are present in the data, the data up to
    /// and including the buffer containing the last newline is sent directly
    /// to the inner writer, and the data after it is buffered. Returns the
    /// number of bytes written.
    ///
    /// This function operates on a "best effort basis"; in keeping with the
    /// convention of `Write::write`, it makes at most one attempt to write
    /// new data to the underlying writer.
    ///
    /// Because this function attempts to send completed lines to the underlying
    /// writer, it will also flush the existing buffer if it contains any
    /// newlines.
    ///
    /// Because sorting through an array of `IoSlice` can be a bit convoluted,
    /// This method differs from write in the following ways:
    ///
    /// - It attempts to write the full content of all the buffers up to and
    ///   including the one containing the last newline. This means that it
    ///   may attempt to write a partial line, that buffer has data past the
    ///   newline.
    /// - If the write only reports partial success, it does not attempt to
    ///   find the precise location of the written bytes and buffer the rest.
    ///
    /// If the underlying vector doesn't support vectored writing, we instead
    /// simply write the first non-empty buffer with `write`. This way, we
    /// get the benefits of more granular partial-line handling without losing
    /// anything in efficiency
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        // If there's no specialized behavior for write_vectored, just use
        // write. This has the benefit of more granular partial-line handling.
        if !self.is_write_vectored() {
            return match bufs.iter().find(|buf| !buf.is_empty()) {
                Some(buf) => self.write(buf),
                None => Ok(0),
            };
        }

        // Find the buffer containing the last newline
        let last_newline_buf_idx = bufs
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, buf)| memchr::memchr(b'\n', buf).map(|_| i));

        // If there are no new newlines (that is, if this write is less than
        // one line), just do a regular buffered write
        let last_newline_buf_idx = match last_newline_buf_idx {
            // No newlines; just do a normal buffered write
            None => {
                self.flush_if_completed_line()?;
                return self.buffer.write_vectored(bufs);
            }
            Some(i) => i,
        };

        // Flush existing content to prepare for our write
        self.buffer.flush_buf()?;

        // This is what we're going to try to write directly to the inner
        // writer. The rest will be buffered, if nothing goes wrong.
        let (lines, tail) = bufs.split_at(last_newline_buf_idx + 1);

        // Write `lines` directly to the inner writer. In keeping with the
        // `write` convention, make at most one attempt to add new (unbuffered)
        // data. Because this write doesn't touch the BufWriter state directly,
        // and the buffer is known to be empty, we don't need to worry about
        // self.panicked here.
        let flushed = self.inner_mut().write_vectored(lines)?;

        // If inner returns Ok(0), propagate that to the caller without
        // doing additional buffering; otherwise we're just guaranteeing
        // an "ErrorKind::WriteZero" later.
        if flushed == 0 {
            return Ok(0);
        }

        // Don't try to reconstruct the exact amount written; just bail
        // in the event of a partial write
        let lines_len = lines.iter().map(|buf| buf.len()).sum();
        if flushed < lines_len {
            return Ok(flushed);
        }

        // Now that the write has succeeded, buffer the rest (or as much of the
        // rest as possible)
        let buffered: usize = tail
            .iter()
            .filter(|buf| !buf.is_empty())
            .map(|buf| self.buffer.write_to_buf(buf))
            .take_while(|&n| n > 0)
            .sum();

        Ok(flushed + buffered)
    }

    fn is_write_vectored(&self) -> bool {
        self.buffer.is_write_vectored()
    }

    /// Write some data into this BufReader with line buffering. This means
    /// that, if any newlines are present in the data, the data up to the last
    /// newline is sent directly to the underlying writer, and data after it
    /// is buffered.
    ///
    /// Because this function attempts to send completed lines to the underlying
    /// writer, it will also flush the existing buffer if it contains any
    /// newlines, even if the incoming data does not contain any newlines.
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match memchr::memrchr(b'\n', buf) {
            // If there are no new newlines (that is, if this write is less than
            // one line), just do a regular buffered write (which may flush if
            // we exceed the inner buffer's size)
            None => {
                self.flush_if_completed_line()?;
                self.buffer.write_all(buf)
            }
            Some(newline_idx) => {
                let (lines, tail) = buf.split_at(newline_idx + 1);

                if self.buffered().is_empty() {
                    self.inner_mut().write_all(lines)?;
                } else {
                    // If there is any buffered data, we add the incoming lines
                    // to that buffer before flushing, which saves us at least
                    // one write call. We can't really do this with `write`,
                    // since we can't do this *and* not suppress errors *and*
                    // report a consistent state to the caller in a return
                    // value, but here in write_all it's fine.
                    self.buffer.write_all(lines)?;
                    self.buffer.flush_buf()?;
                }

                self.buffer.write_all(tail)
            }
        }
    }
}

/// Wraps a writer and buffers output to it, flushing whenever a newline
/// (`0x0a`, `'\n'`) is detected.
///
/// The [`BufWriter`] struct wraps a writer and buffers its output.
/// But it only does this batched write when it goes out of scope, or when the
/// internal buffer is full. Sometimes, you'd prefer to write each line as it's
/// completed, rather than the entire buffer at once. Enter `LineWriter`. It
/// does exactly that.
///
/// Like [`BufWriter`], a `LineWriter`’s buffer will also be flushed when the
/// `LineWriter` goes out of scope or when its internal buffer is full.
///
/// If there's still a partial line in the buffer when the `LineWriter` is
/// dropped, it will flush those contents.
///
/// # Examples
///
/// We can use `LineWriter` to write one line at a time, significantly
/// reducing the number of actual writes to the file.
///
/// ```no_run
/// use std::fs::{self, File};
/// use std::io::prelude::*;
/// use std::io::LineWriter;
///
/// fn main() -> std::io::Result<()> {
///     let road_not_taken = b"I shall be telling this with a sigh
/// Somewhere ages and ages hence:
/// Two roads diverged in a wood, and I -
/// I took the one less traveled by,
/// And that has made all the difference.";
///
///     let file = File::create("poem.txt")?;
///     let mut file = LineWriter::new(file);
///
///     file.write_all(b"I shall be telling this with a sigh")?;
///
///     // No bytes are written until a newline is encountered (or
///     // the internal buffer is filled).
///     assert_eq!(fs::read_to_string("poem.txt")?, "");
///     file.write_all(b"\n")?;
///     assert_eq!(
///         fs::read_to_string("poem.txt")?,
///         "I shall be telling this with a sigh\n",
///     );
///
///     // Write the rest of the poem.
///     file.write_all(b"Somewhere ages and ages hence:
/// Two roads diverged in a wood, and I -
/// I took the one less traveled by,
/// And that has made all the difference.")?;
///
///     // The last line of the poem doesn't end in a newline, so
///     // we have to flush or drop the `LineWriter` to finish
///     // writing.
///     file.flush()?;
///
///     // Confirm the whole poem was written.
///     assert_eq!(fs::read("poem.txt")?, &road_not_taken[..]);
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LineWriter<W: Write> {
    inner: BufWriter<W>,
}

impl<W: Write> LineWriter<W> {
    /// Creates a new `LineWriter`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let file = File::create("poem.txt")?;
    ///     let file = LineWriter::new(file);
    ///     Ok(())
    /// }
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
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let file = File::create("poem.txt")?;
    ///     let file = LineWriter::with_capacity(100, file);
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize, inner: W) -> LineWriter<W> {
        LineWriter { inner: BufWriter::with_capacity(capacity, inner) }
    }

    /// Gets a reference to the underlying writer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let file = File::create("poem.txt")?;
    ///     let file = LineWriter::new(file);
    ///
    ///     let reference = file.get_ref();
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_ref(&self) -> &W {
        self.inner.get_ref()
    }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// Caution must be taken when calling methods on the mutable reference
    /// returned as extra writes could corrupt the output stream.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let file = File::create("poem.txt")?;
    ///     let mut file = LineWriter::new(file);
    ///
    ///     // we can use reference just like file
    ///     let reference = file.get_mut();
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut W {
        self.inner.get_mut()
    }

    /// Unwraps this `LineWriter`, returning the underlying writer.
    ///
    /// The internal buffer is written out before returning the writer.
    ///
    /// # Errors
    ///
    /// An [`Err`] will be returned if an error occurs while flushing the buffer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use std::io::LineWriter;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let file = File::create("poem.txt")?;
    ///
    ///     let writer: LineWriter<File> = LineWriter::new(file);
    ///
    ///     let file: File = writer.into_inner()?;
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(self) -> Result<W, IntoInnerError<LineWriter<W>>> {
        self.inner
            .into_inner()
            .map_err(|IntoInnerError(buf, e)| IntoInnerError(LineWriter { inner: buf }, e))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> Write for LineWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        LineWriterShim::new(&mut self.inner).write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        LineWriterShim::new(&mut self.inner).write_vectored(bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        LineWriterShim::new(&mut self.inner).write_all(buf)
    }

    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        LineWriterShim::new(&mut self.inner).write_all_vectored(bufs)
    }

    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        LineWriterShim::new(&mut self.inner).write_fmt(fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> fmt::Debug for LineWriter<W>
where
    W: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("LineWriter")
            .field("writer", &self.inner.inner)
            .field(
                "buffer",
                &format_args!("{}/{}", self.inner.buf.len(), self.inner.buf.capacity()),
            )
            .finish()
    }
}
