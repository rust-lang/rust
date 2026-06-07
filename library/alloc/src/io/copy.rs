use core::cmp;
use core::mem::MaybeUninit;

#[cfg(not(no_global_oom_handling))]
use crate::collections::VecDeque;
use crate::io::{BorrowedBuf, BufReader, BufWriter, DEFAULT_BUF_SIZE, Read, Result, Write};
use crate::vec::Vec;
#[cfg_attr(
    no_global_oom_handling,
    expect(unused_imports, reason = "only required for VecDeque specialization")
)]
use crate::{alloc::Allocator, io::IoSlice};

/// The userspace read-write-loop implementation of `io::copy` that is used when
/// OS-specific specializations for copy offloading are not available or not applicable.
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn generic_copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> Result<u64>
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

#[cfg(not(no_global_oom_handling))]
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
        let mut init = false;

        loop {
            let buf = self.buffer_mut();
            let mut read_buf: BorrowedBuf<'_> = buf.spare_capacity_mut().into();

            if init {
                // SAFETY: `init` is only true after `reader` initializes
                // `read_buf`. See the comment about `flush_buf` below.
                unsafe { read_buf.set_init() };
            }

            if read_buf.capacity() >= DEFAULT_BUF_SIZE {
                let mut cursor = read_buf.unfilled();
                match reader.read_buf(cursor.reborrow()) {
                    Ok(()) => {
                        let bytes_read = cursor.written();

                        if bytes_read == 0 {
                            return Ok(len);
                        }

                        init = read_buf.is_init();
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
                // SAFETY: `flush_buf` will not de-initialize any elements of
                // the spare capacity so we can remember `init` across this.
                self.flush_buf()?;
            }
        }
    }
}

impl BufferedWriterSpec for Vec<u8> {
    fn buffer_size(&self) -> usize {
        cmp::max(DEFAULT_BUF_SIZE, self.capacity() - self.len())
    }

    fn copy_from<R: Read + ?Sized>(&mut self, reader: &mut R) -> Result<u64> {
        reader.read_to_end(self).map(|bytes| u64::try_from(bytes).expect("usize overflowed u64"))
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
