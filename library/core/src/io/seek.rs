use crate::io::Result;

/// The `Seek` trait provides a cursor which can be moved within a stream of
/// bytes.
///
/// The stream typically has a fixed size, allowing seeking relative to either
/// end or the current offset.
///
/// # Examples
///
/// ```no_run
/// # use core as std;
/// use std::io::{self, Cursor, SeekFrom};
/// use std::io::prelude::*;
///
/// fn main() -> io::Result<()> {
///     let mut buff = Cursor::new([0; 128]);
///
///     // move the cursor 42 bytes after the start of the buffer
///     buff.seek(SeekFrom::Start(42))?;
///
///     assert_eq!(buff.stream_position()?, 42);
///
///     // move the cursor 28 bytes before the end of the buffer
///     buff.seek(SeekFrom::End(-28))?;
///
///     assert_eq!(buff.stream_position()?, 100);
///
///     Ok(())
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "IoSeek")]
pub trait Seek {
    /// Seek to an offset, in bytes, in a stream.
    ///
    /// A seek beyond the end of a stream is allowed, but behavior is defined
    /// by the implementation.
    ///
    /// If the seek operation completed successfully,
    /// this method returns the new position from the start of the stream.
    /// That position can be used later with [`SeekFrom::Start`].
    ///
    /// # Errors
    ///
    /// Seeking can fail, for example because it might involve flushing a buffer.
    ///
    /// Seeking to a negative offset is considered an error.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn seek(&mut self, pos: SeekFrom) -> Result<u64>;

    /// Rewind to the beginning of a stream.
    ///
    /// This is a convenience method, equivalent to `seek(SeekFrom::Start(0))`.
    ///
    /// # Errors
    ///
    /// Rewinding can fail, for example because it might involve flushing a buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use core as std;
    /// use std::io::{Cursor, Seek, Write};
    ///
    /// let mut buffer = [0; 5];
    /// let mut cursor = Cursor::new(&mut buffer);
    ///
    /// cursor.write_all(&[1, 2, 3])?;
    ///
    /// cursor.rewind()?;
    ///
    /// cursor.write_all(&[4, 5, 6])?;
    ///
    /// assert_eq!(&buffer, &[4, 5, 6, 0, 0]);
    /// # std::io::Result::Ok(())
    /// ```
    #[stable(feature = "seek_rewind", since = "1.55.0")]
    fn rewind(&mut self) -> Result<()> {
        self.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Returns the length of this stream (in bytes).
    ///
    /// The default implementation uses up to three seek operations. If this
    /// method returns successfully, the seek position is unchanged (i.e. the
    /// position before calling this method is the same as afterwards).
    /// However, if this method returns an error, the seek position is
    /// unspecified.
    ///
    /// If you need to obtain the length of *many* streams and you don't care
    /// about the seek position afterwards, you can reduce the number of seek
    /// operations by simply calling `seek(SeekFrom::End(0))` and using its
    /// return value (it is also the stream length).
    ///
    /// Note that length of a stream can change over time (for example, when
    /// data is appended to a file). So calling this method multiple times does
    /// not necessarily return the same length each time.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(seek_stream_len)]
    /// # use core as std;
    /// use std::io::{Cursor, Seek};
    ///
    /// let mut cursor = Cursor::new([0; 42]);
    /// assert_eq!(cursor.stream_len()?, 42);
    /// # std::io::Result::Ok(())
    /// ```
    #[unstable(feature = "seek_stream_len", issue = "59359")]
    fn stream_len(&mut self) -> Result<u64> {
        stream_len_default(self)
    }

    /// Returns the current seek position from the start of the stream.
    ///
    /// This is equivalent to `self.seek(SeekFrom::Current(0))`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use core as std;
    /// use std::io::{self, Cursor, SeekFrom};
    /// use std::io::prelude::*;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut buff = Cursor::new([0; 128]);
    ///
    ///     // move the cursor 42 bytes after the start of the buffer
    ///     buff.seek(SeekFrom::Start(42))?;
    ///
    ///     assert_eq!(buff.stream_position()?, 42);
    ///
    ///     // move the cursor 28 bytes before the end of the buffer
    ///     buff.seek(SeekFrom::End(-28))?;
    ///
    ///     assert_eq!(buff.stream_position()?, 100);
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "seek_convenience", since = "1.51.0")]
    fn stream_position(&mut self) -> Result<u64> {
        self.seek(SeekFrom::Current(0))
    }

    /// Seeks relative to the current position.
    ///
    /// This is equivalent to `self.seek(SeekFrom::Current(offset))` but
    /// doesn't return the new position which can allow some implementations
    /// such as `BufReader` to perform more efficient seeks.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use core as std;
    /// use std::io::{self, Cursor, SeekFrom};
    /// use std::io::prelude::*;
    ///
    /// fn main() -> io::Result<()> {
    ///     let mut buff = Cursor::new([0; 64]);
    ///
    ///     // move the cursor forward 42 bytes
    ///     buff.seek_relative(42)?;
    ///
    ///     assert_eq!(buff.stream_position()?, 42);
    ///
    ///     // move the cursor backwards 28 bytes
    ///     buff.seek_relative(-28)?;
    ///
    ///     assert_eq!(buff.stream_position()?, 14);
    ///
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "seek_seek_relative", since = "1.80.0")]
    fn seek_relative(&mut self, offset: i64) -> Result<()> {
        self.seek(SeekFrom::Current(offset))?;
        Ok(())
    }
}

/// The default implementation of [`Seek::stream_len`].
/// This may be desirable in `libstd` where the default implementation is desirable,
/// but additional work needs to be done before or after.
#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub fn stream_len_default<T: Seek + ?Sized>(self_: &mut T) -> Result<u64> {
    let old_pos = self_.stream_position()?;
    let len = self_.seek(SeekFrom::End(0))?;

    // Avoid seeking a third time when we were already at the end of the
    // stream. The branch is usually way cheaper than a seek operation.
    if old_pos != len {
        self_.seek(SeekFrom::Start(old_pos))?;
    }

    Ok(len)
}

/// Enumeration of possible methods to seek within an I/O object.
///
/// It is used by the [`Seek`] trait.
#[derive(Copy, PartialEq, Eq, Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "SeekFrom")]
pub enum SeekFrom {
    /// Sets the offset to the provided number of bytes.
    #[stable(feature = "rust1", since = "1.0.0")]
    Start(#[stable(feature = "rust1", since = "1.0.0")] u64),

    /// Sets the offset to the size of this object plus the specified number of
    /// bytes.
    ///
    /// It is possible to seek beyond the end of an object, but it's an error to
    /// seek before byte 0.
    #[stable(feature = "rust1", since = "1.0.0")]
    End(#[stable(feature = "rust1", since = "1.0.0")] i64),

    /// Sets the offset to the current position plus the specified number of
    /// bytes.
    ///
    /// It is possible to seek beyond the end of an object, but it's an error to
    /// seek before byte 0.
    #[stable(feature = "rust1", since = "1.0.0")]
    Current(#[stable(feature = "rust1", since = "1.0.0")] i64),
}
