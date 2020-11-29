use crate::fmt;
use crate::io::{
    self, Error, ErrorKind, IntoInnerError, IoSlice, Seek, SeekFrom, Write, DEFAULT_BUF_SIZE,
};
use crate::iter::FusedIterator;

/// Helper macro for a common write pattern. Write a buffer using the given
/// function call, then use the returned usize to get the unwritten tail of
/// the buffer.
///
/// Example:
///
/// ```
/// // Use a ? for an i/o operation
/// let tail = tail!(self.flush_buf_vectored(buf)?);
///
/// // omit the ? for an infallible operation
/// let tail = tail!(self.write_to_buffer(buf));
/// ```
macro_rules! tail {
    ($($write:ident).+ ($buf:expr)) => {{
        let buf = $buf;
        let written = $($write).+ (buf);
        &buf[written..]
    }};

    ($($write:ident).+ ($buf:expr) ? ) => {{
        let buf = $buf;
        let written = $($write).+ (buf)?;
        &buf[written..]
    }};
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
/// in memory, like a [`Vec`]`<u8>`.
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

/// Helper struct for BufWriter::flush_buf to ensure the buffer is updated
/// after all the writes are complete. It tracks the number of written bytes
/// and drains them all from the front of the buffer when dropped.
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

    /// Used in vectored flush mode; reports how many *extra* bytes after
    /// `buffer` (ie, new bytes from the caller) were written
    fn extra_written(&self) -> Option<usize> {
        self.written.checked_sub(self.buffer.len())
    }
}

impl Drop for BufGuard<'_> {
    fn drop(&mut self) {
        if self.written >= self.buffer.len() {
            self.buffer.clear();
        } else if self.written > 0 {
            self.buffer.drain(..self.written);
        }
    }
}

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
    pub(super) fn flush_buf(&mut self) -> io::Result<()> {
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

    /// Same as flush_buf, but uses vector operations to attempt to *also*
    /// flush an incoming buffer. The returned usize is the number of bytes
    /// successfully written from the *new* buf. This method will loop until
    /// the entire *current* buffer is flushed, even if that means 0 bytes
    /// from the new buffer were written.
    pub(super) fn flush_buf_vectored(&mut self, buf: &[u8]) -> io::Result<usize> {
        let inner = self.inner.as_mut().unwrap();

        if !inner.is_write_vectored() {
            self.flush_buf()?;
            return Ok(0);
        }

        let mut guard = BufGuard::new(&mut self.buf);

        // Continue looping only as long as there is unwritten content in self.buf
        loop {
            match guard.extra_written() {
                None => {
                    let buffers = [IoSlice::new(guard.remaining()), IoSlice::new(buf)];
                    self.panicked = true;
                    let r = inner.write_vectored(&buffers);
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
                Some(extra) => return Ok(extra),
            }
        }
    }

    /// Buffer some data without flushing it, regardless of the size of the
    /// data. Writes as much as possible without exceeding capacity. Returns
    /// the number of bytes written.
    pub(super) fn write_to_buf(&mut self, buf: &[u8]) -> usize {
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

    /// Returns the unused buffer capacity.
    fn available(&self) -> usize {
        self.capacity() - self.buf.len()
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
            Err(e) => Err(IntoInnerError::new(self, e)),
            Ok(()) => Ok(self.inner.take().unwrap()),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> Write for BufWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // We assume that callers of `write` prefer to avoid split writes where
        // possible, so if the incoming buf doesn't fit in remaining available
        // buffer, we pre-flush rather than doing a partial write to fill it.
        //
        // During the pre-flush, though, we attempt a vectored write of both
        // the buffered bytes and the new bytes. In the worst case, this will
        // be the same as a typical pre-flush, since by default vectored
        // writes just do a normal write of the first buffer. In the best case,
        // we were able to do some additional writing during a single syscall.
        let written = match buf.len() > self.available() {
            true => self.flush_buf_vectored(buf)?,
            false => 0,
        };
        let tail = &buf[written..];

        // If the incoming buf doesn't fit in our buffer, even after we flushed
        // it to make room, we should forward it directly (via inner.write).
        // However, if the vectored flush successfully wrote some of `buf`,
        // we're now obligated to return Ok(..) before trying any more
        // fallible i/o operations.
        let tail_written = if tail.len() < self.capacity() {
            self.write_to_buf(tail)
        } else if written > 0 {
            0
        } else {
            // It's guaranteed at this point that the buffer is empty, because
            // if wasn't, it would have been flushed earlier in this function.
            self.get_mut().write(tail)?
        };

        Ok(written + tail_written)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        // Unlike with `write`, we assume that a caller of `write_all` is
        // interested in minimizing system calls even if the buffer is split.
        // This method tries to fill up the buffer as much as possible before
        // flushing, whereas `write` prefers not split incoming bufs.

        // Bypass the buffer if the the incoming write is larger than the
        // whole buffer. Use a vectored write to attempt to write the new
        // data and the existing buffer in a single operation
        let buf = match buf.len() >= self.capacity() {
            true => match tail!(self.flush_buf_vectored(buf)?) {
                // If the vectored write flushed everything at once, we're done!
                [] => return Ok(()),

                // If what's left after the vector flush is *still* larger than
                // the buffer, bypass the buffer and forward it directly
                tail if tail.len() >= self.capacity() => return self.get_mut().write_all(tail),

                // Otherwise, we're going to buffer whatever's left of the user input
                tail => tail,
            },
            false => buf,
        };

        // In order to reduce net writes in aggregate, we buffer as much as
        // possible, then forward, then buffer the rest
        let buf = tail!(self.write_to_buf(buf));
        if !buf.is_empty() {
            let buf = tail!(self.flush_buf_vectored(buf)?);

            // At this point, because we know that buf.len() < self.buf.len(),
            // and that the buffer has been flushed we know that this will
            // succeed in totality
            self.write_to_buf(buf);
        }

        // If, at this point, the buffer is full, we may as well eagerly
        // attempt to flush, so that the next write will have an empty
        // buffer.
        if self.available() == 0 {
            self.flush_buf()?;
        }

        Ok(())
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        if let Some(buf) = only_one(bufs, |b| !b.is_empty()) {
            // If there's exactly 1 incoming buffer, `Self::write` can make
            // use of self.inner.write_vectored to attempt to combine flushing
            // the existing buffer with writing the new one.
            self.write(buf)
        } else if self.get_ref().is_write_vectored() {
            let total_len: usize = bufs.iter().map(|buf| buf.len()).sum();

            if total_len > self.available() {
                self.flush_buf()?;
            }

            if total_len >= self.capacity() {
                self.get_mut().write_vectored(bufs)
            } else {
                // Correctness note: we've already verified that none of these
                // will overflow the buffer, because total_len < capacity
                bufs.iter().for_each(|buf| self.buf.extend_from_slice(buf));
                Ok(total_len)
            }
        } else {
            // Because the inner writer doesn't have native vectored write
            // support, we should take care of buffering together the individual
            // incoming bufs, even if the *total* length is larger than our
            // buffer. We only want to skip our buffer if an *individual* write
            // exceeds our buffer capacity.
            let mut total_buffered = 0;

            for buf in bufs {
                if total_buffered == 0 {
                    if buf.len() > self.available() {
                        // If an individual write would overflow our remaining
                        // capacity and we haven't buffered anything yet,
                        // pre-flush before buffering (same as with regular
                        // write()).
                        self.flush_buf()?;
                    }

                    if buf.len() >= self.capacity() && self.buf.is_empty() {
                        // If an individual buffer exceeds our *total* capacity
                        // and we haven't buffered anything yet, just forward
                        // it to the underlying device
                        return self.get_mut().write(buf);
                    }
                }

                // Buffer as much as possible until we reach full capacity.
                // Once we've buffered at least 1 byte, we're obligated to
                // return Ok(..) before attempting any fallible i/o operations,
                // so once the buffer is full we immediately return.
                total_buffered += self.write_to_buf(buf);
                if self.available() == 0 {
                    break;
                }
            }

            Ok(total_buffered)
        }
    }

    fn is_write_vectored(&self) -> bool {
        true
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buf()?;
        self.get_mut().flush()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write> fmt::Debug for BufWriter<W>
where
    W: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("BufWriter")
            .field("writer", self.get_ref())
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

/// Similar to iter.find, this method searches an iterator for an item
/// matching a predicate, but returns it only if it is the *only* item
/// matching that predicate. Used to check if there is exactly one non-empty
/// buffer in a list input to write_vectored.
///
/// FIXME: delete this function and replace it with slice::trim if that becomes
/// a thing (https://github.com/rust-lang/rfcs/issues/2547)
fn only_one<I>(iter: I, filter: impl Fn(&I::Item) -> bool) -> Option<I::Item>
where
    I: IntoIterator,
    I::IntoIter: FusedIterator,
{
    let mut iter = iter.into_iter().filter(filter);
    match (iter.next(), iter.next()) {
        (Some(item), None) => Some(item),
        _ => None,
    }
}
