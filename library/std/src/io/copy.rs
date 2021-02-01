use super::{BufWriter, ErrorKind, Read, Result, Write, DEFAULT_BUF_SIZE};
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

        // FIXME: #42788
        //
        //   - This creates a (mut) reference to a slice of
        //     _uninitialized_ integers, which is **undefined behavior**
        //
        //   - Only the standard library gets to soundly "ignore" this,
        //     based on its privileged knowledge of unstable rustc
        //     internals;
        unsafe {
            let spare_cap = writer.buffer_mut().spare_capacity_mut();
            reader.initializer().initialize(MaybeUninit::slice_assume_init_mut(spare_cap));
        }

        let mut len = 0;

        loop {
            let buf = writer.buffer_mut();
            let spare_cap = buf.spare_capacity_mut();

            if spare_cap.len() >= DEFAULT_BUF_SIZE {
                match reader.read(unsafe { MaybeUninit::slice_assume_init_mut(spare_cap) }) {
                    Ok(0) => return Ok(len), // EOF reached
                    Ok(bytes_read) => {
                        assert!(bytes_read <= spare_cap.len());
                        // Safety: The initializer contract guarantees that either it or `read`
                        // will have initialized these bytes. And we just checked that the number
                        // of bytes is within the buffer capacity.
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
    let mut buf = MaybeUninit::<[u8; DEFAULT_BUF_SIZE]>::uninit();
    // FIXME: #42788
    //
    //   - This creates a (mut) reference to a slice of
    //     _uninitialized_ integers, which is **undefined behavior**
    //
    //   - Only the standard library gets to soundly "ignore" this,
    //     based on its privileged knowledge of unstable rustc
    //     internals;
    unsafe {
        reader.initializer().initialize(buf.assume_init_mut());
    }

    let mut written = 0;
    loop {
        let len = match reader.read(unsafe { buf.assume_init_mut() }) {
            Ok(0) => return Ok(written),
            Ok(len) => len,
            Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        writer.write_all(unsafe { &buf.assume_init_ref()[..len] })?;
        written += len as u64;
    }
}
