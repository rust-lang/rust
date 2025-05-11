mod buffer;

use buffer::Buffer;

use crate::fmt;
use crate::io::{
    self, BorrowedCursor, BufRead, DEFAULT_BUF_SIZE, IoSliceMut, Read, Seek, SeekFrom, SizeHint,
    SpecReadByte, uninlined_slow_read_byte,
};

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
/// already in memory, like a <code>[Vec]\<u8></code>.
///
/// When the `BufReader<R>` is dropped, the contents of its buffer will be
/// discarded. Creating multiple instances of a `BufReader<R>` on the same
/// stream can cause data loss. Reading from the underlying reader after
/// unwrapping the `BufReader<R>` with [`BufReader::into_inner`] can also cause
/// data loss.
///
/// [`TcpStream::read`]: crate::net::TcpStream::read
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
///     println!("First line is {len} bytes long");
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BufReader<R: ?Sized> {
    buf: Buffer,
    inner: R,
}

impl<R: Read> BufReader<R> {
    /// Creates a new `BufReader<R>` with a default buffer capacity. The default is currently 8 KiB,
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

    pub(crate) fn try_new_buffer() -> io::Result<Buffer> {
        Buffer::try_with_capacity(DEFAULT_BUF_SIZE)
    }

    pub(crate) fn with_buffer(inner: R, buf: Buffer) -> Self {
        Self { inner, buf }
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
        BufReader { inner, buf: Buffer::with_capacity(capacity) }
    }
}

impl<R: Read + ?Sized> BufReader<R> {
    /// Attempt to look ahead `n` bytes.
    ///
    /// `n` must be less than or equal to `capacity`.
    ///
    /// The returned slice may be less than `n` bytes long if
    /// end of file is reached.
    ///
    /// After calling this method, you may call [`consume`](BufRead::consume)
    /// with a value less than or equal to `n` to advance over some or all of
    /// the returned bytes.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// #![feature(bufreader_peek)]
    /// use std::io::{Read, BufReader};
    ///
    /// let mut bytes = &b"oh, hello there"[..];
    /// let mut rdr = BufReader::with_capacity(6, &mut bytes);
    /// assert_eq!(rdr.peek(2).unwrap(), b"oh");
    /// let mut buf = [0; 4];
    /// rdr.read(&mut buf[..]).unwrap();
    /// assert_eq!(&buf, b"oh, ");
    /// assert_eq!(rdr.peek(5).unwrap(), b"hello");
    /// let mut s = String::new();
    /// rdr.read_to_string(&mut s).unwrap();
    /// assert_eq!(&s, "hello there");
    /// assert_eq!(rdr.peek(1).unwrap().len(), 0);
    /// ```
    #[unstable(feature = "bufreader_peek", issue = "128405")]
    pub fn peek(&mut self, n: usize) -> io::Result<&[u8]> {
        assert!(n <= self.capacity());
        while n > self.buf.buffer().len() {
            if self.buf.pos() > 0 {
                self.buf.backshift();
            }
            let new = self.buf.read_more(&mut self.inner)?;
            if new == 0 {
                // end of file, no more bytes to read
                return Ok(&self.buf.buffer()[..]);
            }
            debug_assert_eq!(self.buf.pos(), 0);
        }
        Ok(&self.buf.buffer()[..n])
    }
}

impl<R: ?Sized> BufReader<R> {
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
        self.buf.buffer()
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
        self.buf.capacity()
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
    pub fn into_inner(self) -> R
    where
        R: Sized,
    {
        self.inner
    }

    /// Invalidates all data in the internal buffer.
    #[inline]
    pub(in crate::io) fn discard_buffer(&mut self) {
        self.buf.discard_buffer()
    }
}

// This is only used by a test which asserts that the initialization-tracking is correct.
#[cfg(test)]
impl<R: ?Sized> BufReader<R> {
    #[allow(missing_docs)]
    pub fn initialized(&self) -> usize {
        self.buf.initialized()
    }
}

impl<R: ?Sized + Seek> BufReader<R> {
    /// Seeks relative to the current position. If the new position lies within the buffer,
    /// the buffer will not be flushed, allowing for more efficient seeks.
    /// This method does not return the location of the underlying reader, so the caller
    /// must track this information themselves if it is required.
    #[stable(feature = "bufreader_seek_relative", since = "1.53.0")]
    pub fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        let pos = self.buf.pos() as u64;
        if offset < 0 {
            if let Some(_) = pos.checked_sub((-offset) as u64) {
                self.buf.unconsume((-offset) as usize);
                return Ok(());
            }
        } else if let Some(new_pos) = pos.checked_add(offset as u64) {
            if new_pos <= self.buf.filled() as u64 {
                self.buf.consume(offset as usize);
                return Ok(());
            }
        }

        self.seek(SeekFrom::Current(offset)).map(drop)
    }
}

impl<R> SpecReadByte for BufReader<R>
where
    Self: Read,
{
    #[inline]
    fn spec_read_byte(&mut self) -> Option<io::Result<u8>> {
        let mut byte = 0;
        if self.buf.consume_with(1, |claimed| byte = claimed[0]) {
            return Some(Ok(byte));
        }

        // Fallback case, only reached once per buffer refill.
        uninlined_slow_read_byte(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: ?Sized + Read> Read for BufReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.buf.pos() == self.buf.filled() && buf.len() >= self.capacity() {
            self.discard_buffer();
            return self.inner.read(buf);
        }
        let mut rem = self.fill_buf()?;
        let nread = rem.read(buf)?;
        self.consume(nread);
        Ok(nread)
    }

    fn read_buf(&mut self, mut cursor: BorrowedCursor<'_>) -> io::Result<()> {
        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.buf.pos() == self.buf.filled() && cursor.capacity() >= self.capacity() {
            self.discard_buffer();
            return self.inner.read_buf(cursor);
        }

        let prev = cursor.written();

        let mut rem = self.fill_buf()?;
        rem.read_buf(cursor.reborrow())?; // actually never fails

        self.consume(cursor.written() - prev); //slice impl of read_buf known to never unfill buf

        Ok(())
    }

    // Small read_exacts from a BufReader are extremely common when used with a deserializer.
    // The default implementation calls read in a loop, which results in surprisingly poor code
    // generation for the common path where the buffer has enough bytes to fill the passed-in
    // buffer.
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.buf.consume_with(buf.len(), |claimed| buf.copy_from_slice(claimed)) {
            return Ok(());
        }

        crate::io::default_read_exact(self, buf)
    }

    fn read_buf_exact(&mut self, mut cursor: BorrowedCursor<'_>) -> io::Result<()> {
        if self.buf.consume_with(cursor.capacity(), |claimed| cursor.append(claimed)) {
            return Ok(());
        }

        crate::io::default_read_buf_exact(self, cursor)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum::<usize>();
        if self.buf.pos() == self.buf.filled() && total_len >= self.capacity() {
            self.discard_buffer();
            return self.inner.read_vectored(bufs);
        }
        let mut rem = self.fill_buf()?;
        let nread = rem.read_vectored(bufs)?;

        self.consume(nread);
        Ok(nread)
    }

    fn is_read_vectored(&self) -> bool {
        self.inner.is_read_vectored()
    }

    // The inner reader might have an optimized `read_to_end`. Drain our buffer and then
    // delegate to the inner implementation.
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let inner_buf = self.buffer();
        buf.try_reserve(inner_buf.len())?;
        buf.extend_from_slice(inner_buf);
        let nread = inner_buf.len();
        self.discard_buffer();
        Ok(nread + self.inner.read_to_end(buf)?)
    }

    // The inner reader might have an optimized `read_to_end`. Drain our buffer and then
    // delegate to the inner implementation.
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        // In the general `else` case below we must read bytes into a side buffer, check
        // that they are valid UTF-8, and then append them to `buf`. This requires a
        // potentially large memcpy.
        //
        // If `buf` is empty--the most common case--we can leverage `append_to_string`
        // to read directly into `buf`'s internal byte buffer, saving an allocation and
        // a memcpy.
        if buf.is_empty() {
            // `append_to_string`'s safety relies on the buffer only being appended to since
            // it only checks the UTF-8 validity of new data. If there were existing content in
            // `buf` then an untrustworthy reader (i.e. `self.inner`) could not only append
            // bytes but also modify existing bytes and render them invalid. On the other hand,
            // if `buf` is empty then by definition any writes must be appends and
            // `append_to_string` will validate all of the new bytes.
            unsafe { crate::io::append_to_string(buf, |b| self.read_to_end(b)) }
        } else {
            // We cannot append our byte buffer directly onto the `buf` String as there could
            // be an incomplete UTF-8 sequence that has only been partially read. We must read
            // everything into a side buffer first and then call `from_utf8` on the complete
            // buffer.
            let mut bytes = Vec::new();
            self.read_to_end(&mut bytes)?;
            let string = crate::str::from_utf8(&bytes).map_err(|_| io::Error::INVALID_UTF8)?;
            *buf += string;
            Ok(string.len())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: ?Sized + Read> BufRead for BufReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.buf.fill_buf(&mut self.inner)
    }

    fn consume(&mut self, amt: usize) {
        self.buf.consume(amt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R> fmt::Debug for BufReader<R>
where
    R: ?Sized + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("BufReader")
            .field("reader", &&self.inner)
            .field(
                "buffer",
                &format_args!("{}/{}", self.buf.filled() - self.buf.pos(), self.capacity()),
            )
            .finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<R: ?Sized + Seek> Seek for BufReader<R> {
    /// Seek to an offset, in bytes, in the underlying reader.
    ///
    /// The position used for seeking with <code>[SeekFrom::Current]\(_)</code> is the
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
    /// Note: In the edge case where you're seeking with <code>[SeekFrom::Current]\(n)</code>
    /// where `n` minus the internal buffer length overflows an `i64`, two
    /// seeks will be performed instead of one. If the second seek returns
    /// [`Err`], the underlying reader will be left at the same position it would
    /// have if you called `seek` with <code>[SeekFrom::Current]\(0)</code>.
    ///
    /// [`std::io::Seek`]: Seek
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let result: u64;
        if let SeekFrom::Current(n) = pos {
            let remainder = (self.buf.filled() - self.buf.pos()) as i64;
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

    /// Returns the current seek position from the start of the stream.
    ///
    /// The value returned is equivalent to `self.seek(SeekFrom::Current(0))`
    /// but does not flush the internal buffer. Due to this optimization the
    /// function does not guarantee that calling `.into_inner()` immediately
    /// afterwards will yield the underlying reader at the same position. Use
    /// [`BufReader::seek`] instead if you require that guarantee.
    ///
    /// # Panics
    ///
    /// This function will panic if the position of the inner reader is smaller
    /// than the amount of buffered data. That can happen if the inner reader
    /// has an incorrect implementation of [`Seek::stream_position`], or if the
    /// position has gone out of sync due to calling [`Seek::seek`] directly on
    /// the underlying reader.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::{
    ///     io::{self, BufRead, BufReader, Seek},
    ///     fs::File,
    /// };
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut f = BufReader::new(File::open("foo.txt")?);
    ///
    ///     let before = f.stream_position()?;
    ///     f.read_line(&mut String::new())?;
    ///     let after = f.stream_position()?;
    ///
    ///     println!("The first line was {} bytes long", after - before);
    ///     Ok(())
    /// }
    /// ```
    fn stream_position(&mut self) -> io::Result<u64> {
        let remainder = (self.buf.filled() - self.buf.pos()) as u64;
        self.inner.stream_position().map(|pos| {
            pos.checked_sub(remainder).expect(
                "overflow when subtracting remaining buffer size from inner stream position",
            )
        })
    }

    /// Seeks relative to the current position.
    ///
    /// If the new position lies within the buffer, the buffer will not be
    /// flushed, allowing for more efficient seeks. This method does not return
    /// the location of the underlying reader, so the caller must track this
    /// information themselves if it is required.
    fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        self.seek_relative(offset)
    }
}

impl<T: ?Sized> SizeHint for BufReader<T> {
    #[inline]
    fn lower_bound(&self) -> usize {
        SizeHint::lower_bound(self.get_ref()) + self.buffer().len()
    }

    #[inline]
    fn upper_bound(&self) -> Option<usize> {
        SizeHint::upper_bound(self.get_ref()).and_then(|up| self.buffer().len().checked_add(up))
    }
}
