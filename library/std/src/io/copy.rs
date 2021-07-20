use super::{BufWriter, ErrorKind, Read, ReadBuf, Result, Write, DEFAULT_BUF_SIZE};
use crate::mem::MaybeUninit;

/// Copies the entire contents of a reader into a writer.
///
/// This function will continuously read data from `reader` and then
/// write it into `writer` in a streaming fashion until `reader`
/// returns EOF.
///
/// On success, the total number of bytes that were copied from
/// `reader` to `writer` is returned.
///
/// If you’re wanting to copy the contents of one file to another and you’re
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
    BufferedCopySpec::copy_to(reader, writer)
}

/// Specialization of the read-write loop that either uses a stack buffer
/// or reuses the internal buffer of a BufWriter
trait BufferedCopySpec: Write {
    fn copy_to<R: Read + ?Sized>(reader: &mut R, writer: &mut Self) -> Result<u64>;
}

impl<W: Write + ?Sized> BufferedCopySpec for W {
    default fn copy_to<R: Read + ?Sized>(reader: &mut R, writer: &mut Self) -> Result<u64> {
        stack_buffer_copy(reader, writer)
    }
}

impl<I: Write> BufferedCopySpec for BufWriter<I> {
    fn copy_to<R: Read + ?Sized>(reader: &mut R, writer: &mut Self) -> Result<u64> {
        if writer.capacity() < DEFAULT_BUF_SIZE {
            return stack_buffer_copy(reader, writer);
        }

        let mut len = 0;
        let mut init = 0;

        loop {
            let buf = writer.buffer_mut();
            let mut read_buf = ReadBuf::uninit(buf.spare_capacity_mut());

            // SAFETY: init is either 0 or the initialized_len of the previous iteration
            unsafe {
                read_buf.assume_init(init);
            }

            if read_buf.capacity() >= DEFAULT_BUF_SIZE {
                match reader.read_buf(&mut read_buf) {
                    Ok(()) => {
                        let bytes_read = read_buf.filled_len();

                        if bytes_read == 0 {
                            return Ok(len);
                        }

                        init = read_buf.initialized_len() - bytes_read;

                        // SAFETY: ReadBuf guarantees all of its filled bytes are init
                        unsafe { buf.set_len(buf.len() + bytes_read) };
                        len += bytes_read as u64;
                        // Read again if the buffer still has enough capacity, as BufWriter itself would do
                        // This will occur if the reader returns short reads
                        continue;
                    }
                    Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }

            writer.flush_buf()?;
        }
    }
}

fn stack_buffer_copy<R: Read + ?Sized, W: Write + ?Sized>(
    reader: &mut R,
    writer: &mut W,
) -> Result<u64> {
    let mut buf = [MaybeUninit::uninit(); DEFAULT_BUF_SIZE];
    let mut buf = ReadBuf::uninit(&mut buf);

    let mut len = 0;

    loop {
        match reader.read_buf(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
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
