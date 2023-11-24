use super::{BorrowedBuf, BufReader, BufWriter, Read, Result, Write, DEFAULT_BUF_SIZE};
use crate::alloc::Allocator;
use crate::cmp;
use crate::cmp::min;
use crate::collections::VecDeque;
use crate::io::IoSlice;
use crate::mem::MaybeUninit;

#[cfg(test)]
mod tests;

/// Copies the entire contents of a reader into a writer.
///
/// This function will continuously read data from `reader` and then
/// write it into `writer` in a streaming fashion until `reader`
/// returns EOF.
///
/// On success, the total number of bytes that were copied from
/// `reader` to `writer` is returned.
///
/// If you want to copy the contents of one file to another and you’re
/// working with filesystem paths, see the [`fs::copy`] function.
///
/// [`fs::copy`]: crate::fs::copy
///
/// # Errors
///
/// This function will return an error immediately if any call to [`read`] or
/// [`write`] returns an error. All instances of [`ErrorKind::Interrupted`] are
/// handled by this function and the underlying operation is retried.
///
/// [`read`]: Read::read
/// [`write`]: Write::write
/// [`ErrorKind::Interrupted`]: crate::io::ErrorKind::Interrupted
///
/// # Examples
///
/// ```
/// use std::io;
///
/// fn main() -> io::Result<()> {
///     let mut reader: &[u8] = b"hello";
///     let mut writer: Vec<u8> = vec![];
///
///     io::copy(&mut reader, &mut writer)?;
///
///     assert_eq!(&b"hello"[..], &writer[..]);
///     Ok(())
/// }
/// ```
///
/// # Platform-specific behavior
///
/// On Linux (including Android), this function uses `copy_file_range(2)`,
/// `sendfile(2)` or `splice(2)` syscalls to move data directly between file
/// descriptors if possible.
///
/// Note that platform-specific behavior [may change in the future][changes].
///
/// [changes]: crate::io#platform-specific-behavior
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> Result<u64>
where
    R: Read,
    W: Write,
{
    cfg_if::cfg_if! {
        if #[cfg(any(target_os = "linux", target_os = "android"))] {
            crate::sys::kernel_copy::copy_spec(reader, writer)
        } else {
            generic_copy(reader, writer)
        }
    }
}

/// The userspace read-write-loop implementation of `io::copy` that is used when
/// OS-specific specializations for copy offloading are not available or not applicable.
pub(crate) fn generic_copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> Result<u64>
where
    R: Read,
    W: Write,
{
    let read_buf = BufferedReaderSpec::buffer_size(reader);
    let write_buf = BufferedWriterSpec::buffer_size(writer);

    if read_buf >= DEFAULT_BUF_SIZE && read_buf >= write_buf {
        return BufferedReaderSpec::copy_to(reader, writer);
    }

    BufferedWriterSpec::copy_from(writer, reader)
}

/// Specialization of the read-write loop that reuses the internal
/// buffer of a BufReader. If there's no buffer then the writer side
/// should be used instead.
trait BufferedReaderSpec {
    fn buffer_size(&self) -> usize;

    fn copy_to(&mut self, to: &mut (impl Write + ?Sized)) -> Result<u64>;
}

impl<T> BufferedReaderSpec for T
where
    Self: Read,
    T: ?Sized,
{
    #[inline]
    default fn buffer_size(&self) -> usize {
        0
    }

    default fn copy_to(&mut self, _to: &mut (impl Write + ?Sized)) -> Result<u64> {
        unreachable!("only called from specializations")
    }
}

impl BufferedReaderSpec for &[u8] {
    fn buffer_size(&self) -> usize {
        // prefer this specialization since the source "buffer" is all we'll ever need,
        // even if it's small
        usize::MAX
    }

    fn copy_to(&mut self, to: &mut (impl Write + ?Sized)) -> Result<u64> {
        let len = self.len();
        to.write_all(self)?;
        *self = &self[len..];
        Ok(len as u64)
    }
}

impl<A: Allocator> BufferedReaderSpec for VecDeque<u8, A> {
    fn buffer_size(&self) -> usize {
        // prefer this specialization since the source "buffer" is all we'll ever need,
        // even if it's small
        usize::MAX
    }

    fn copy_to(&mut self, to: &mut (impl Write + ?Sized)) -> Result<u64> {
        let len = self.len();
        let (front, back) = self.as_slices();
        let bufs = &mut [IoSlice::new(front), IoSlice::new(back)];
        to.write_all_vectored(bufs)?;
        self.clear();
        Ok(len as u64)
    }
}

impl<I> BufferedReaderSpec for BufReader<I>
where
    Self: Read,
    I: ?Sized,
{
    fn buffer_size(&self) -> usize {
        self.capacity()
    }

    fn copy_to(&mut self, to: &mut (impl Write + ?Sized)) -> Result<u64> {
        let mut len = 0;

        loop {
            // Hack: this relies on `impl Read for BufReader` always calling fill_buf
            // if the buffer is empty, even for empty slices.
            // It can't be called directly here since specialization prevents us
            // from adding I: Read
            match self.read(&mut []) {
                Ok(_) => {}
                Err(e) if e.is_interrupted() => continue,
                Err(e) => return Err(e),
            }
            let buf = self.buffer();
            if self.buffer().len() == 0 {
                return Ok(len);
            }

            // In case the writer side is a BufWriter then its write_all
            // implements an optimization that passes through large
            // buffers to the underlying writer. That code path is #[cold]
            // but we're still avoiding redundant memcopies when doing
            // a copy between buffered inputs and outputs.
            to.write_all(buf)?;
            len += buf.len() as u64;
            self.discard_buffer();
        }
    }
}

/// Specialization of the read-write loop that either uses a stack buffer
/// or reuses the internal buffer of a BufWriter
trait BufferedWriterSpec: Write {
    fn buffer_size(&self) -> usize;

    fn copy_from<R: Read + ?Sized>(&mut self, reader: &mut R) -> Result<u64>;
}

impl<W: Write + ?Sized> BufferedWriterSpec for W {
    #[inline]
    default fn buffer_size(&self) -> usize {
        0
    }

    default fn copy_from<R: Read + ?Sized>(&mut self, reader: &mut R) -> Result<u64> {
        stack_buffer_copy(reader, self)
    }
}

impl<I: Write + ?Sized> BufferedWriterSpec for BufWriter<I> {
    fn buffer_size(&self) -> usize {
        self.capacity()
    }

    fn copy_from<R: Read + ?Sized>(&mut self, reader: &mut R) -> Result<u64> {
        if self.capacity() < DEFAULT_BUF_SIZE {
            return stack_buffer_copy(reader, self);
        }

        let mut len = 0;
        let mut init = 0;

        loop {
            let buf = self.buffer_mut();
            let mut read_buf: BorrowedBuf<'_> = buf.spare_capacity_mut().into();

            unsafe {
                // SAFETY: init is either 0 or the init_len from the previous iteration.
                read_buf.set_init(init);
            }

            if read_buf.capacity() >= DEFAULT_BUF_SIZE {
                let mut cursor = read_buf.unfilled();
                match reader.read_buf(cursor.reborrow()) {
                    Ok(()) => {
                        let bytes_read = cursor.written();

                        if bytes_read == 0 {
                            return Ok(len);
                        }

                        init = read_buf.init_len() - bytes_read;
                        len += bytes_read as u64;

                        // SAFETY: BorrowedBuf guarantees all of its filled bytes are init
                        unsafe { buf.set_len(buf.len() + bytes_read) };

                        // Read again if the buffer still has enough capacity, as BufWriter itself would do
                        // This will occur if the reader returns short reads
                    }
                    Err(ref e) if e.is_interrupted() => {}
                    Err(e) => return Err(e),
                }
            } else {
                self.flush_buf()?;
                init = 0;
            }
        }
    }
}

impl<A: Allocator> BufferedWriterSpec for Vec<u8, A> {
    fn buffer_size(&self) -> usize {
        cmp::max(DEFAULT_BUF_SIZE, self.capacity() - self.len())
    }

    fn copy_from<R: Read + ?Sized>(&mut self, reader: &mut R) -> Result<u64> {
        let mut bytes = 0;

        // avoid inflating empty/small vecs before we have determined that there's anything to read
        if self.capacity() < DEFAULT_BUF_SIZE {
            let stack_read_limit = DEFAULT_BUF_SIZE as u64;
            bytes = stack_buffer_copy(&mut reader.take(stack_read_limit), self)?;
            // fewer bytes than requested -> EOF reached
            if bytes < stack_read_limit {
                return Ok(bytes);
            }
        }

        // don't immediately offer the vec's whole spare capacity, otherwise
        // we might have to fully initialize it if the reader doesn't have a custom read_buf() impl
        let mut max_read_size = DEFAULT_BUF_SIZE;

        loop {
            self.reserve(DEFAULT_BUF_SIZE);
            let mut initialized_spare_capacity = 0;

            loop {
                let buf = self.spare_capacity_mut();
                let read_size = min(max_read_size, buf.len());
                let mut buf = BorrowedBuf::from(&mut buf[..read_size]);
                // SAFETY: init is either 0 or the init_len from the previous iteration.
                unsafe {
                    buf.set_init(initialized_spare_capacity);
                }
                match reader.read_buf(buf.unfilled()) {
                    Ok(()) => {
                        let bytes_read = buf.len();

                        // EOF
                        if bytes_read == 0 {
                            return Ok(bytes);
                        }

                        // the reader is returning short reads but it doesn't call ensure_init()
                        if buf.init_len() < buf.capacity() {
                            max_read_size = usize::MAX;
                        }
                        // the reader hasn't returned short reads so far
                        if bytes_read == buf.capacity() {
                            max_read_size *= 2;
                        }

                        initialized_spare_capacity = buf.init_len() - bytes_read;
                        bytes += bytes_read as u64;
                        // SAFETY: BorrowedBuf guarantees all of its filled bytes are init
                        // and the number of read bytes can't exceed the spare capacity since
                        // that's what the buffer is borrowing from.
                        unsafe { self.set_len(self.len() + bytes_read) };

                        // spare capacity full, reserve more
                        if self.len() == self.capacity() {
                            break;
                        }
                    }
                    Err(e) if e.is_interrupted() => continue,
                    Err(e) => return Err(e),
                }
            }
        }
    }
}

fn stack_buffer_copy<R: Read + ?Sized, W: Write + ?Sized>(
    reader: &mut R,
    writer: &mut W,
) -> Result<u64> {
    let buf: &mut [_] = &mut [MaybeUninit::uninit(); DEFAULT_BUF_SIZE];
    let mut buf: BorrowedBuf<'_> = buf.into();

    let mut len = 0;

    loop {
        match reader.read_buf(buf.unfilled()) {
            Ok(()) => {}
            Err(e) if e.is_interrupted() => continue,
            Err(e) => return Err(e),
        };

        if buf.filled().is_empty() {
            break;
        }

        len += buf.filled().len() as u64;
        writer.write_all(buf.filled())?;
        buf.clear();
    }

    Ok(len)
}
