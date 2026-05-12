mod generic;
mod specialization;

use self::generic::generic_copy;
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use self::specialization::SpecCopy;
use self::specialization::specialized_copy;
use crate::io::{Read, Result, Write};

#[derive(Debug)]
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub enum CopyState {
    Ended(u64),
    Fallback(u64),
}

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
// FIXME(#74481): Hard-links required to link from `alloc` to `std`
/// [`fs::copy`]: ../../std/fs/fn.copy.html
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
/// Note that platform-specific behavior may change in the future.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn copy<R: ?Sized, W: ?Sized>(reader: &mut R, writer: &mut W) -> Result<u64>
where
    R: Read,
    W: Write,
{
    match specialized_copy(reader, writer)? {
        CopyState::Ended(copied) => Ok(copied),
        CopyState::Fallback(copied) => {
            generic_copy(reader, writer).map(|additional| copied + additional)
        }
    }
}
