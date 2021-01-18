#![allow(missing_copy_implementations)]

#[cfg(test)]
mod tests;

use crate::fmt;
use crate::io::{
    self, BufRead, IoSlice, IoSliceMut, Read, ReadBuf, Seek, SeekFrom, SizeHint, Write,
};

/// A reader which is always at EOF.
///
/// This struct is generally created by calling [`empty()`]. Please see
/// the documentation of [`empty()`] for more details.
#[stable(feature = "rust1", since = "1.0.0")]
#[non_exhaustive]
#[derive(Copy, Clone, Default)]
pub struct Empty;

/// Constructs a new handle to an empty reader.
///
/// All reads from the returned reader will return <code>[Ok]\(0)</code>.
///
/// # Examples
///
/// A slightly sad example of not reading anything into a buffer:
///
/// ```
/// use std::io::{self, Read};
///
/// let mut buffer = String::new();
/// io::empty().read_to_string(&mut buffer).unwrap();
/// assert!(buffer.is_empty());
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_io_structs", issue = "78812")]
pub const fn empty() -> Empty {
    Empty
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Empty {
    #[inline]
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Ok(0)
    }

    #[inline]
    fn read_buf(&mut self, _buf: &mut ReadBuf<'_>) -> io::Result<()> {
        Ok(())
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl BufRead for Empty {
    #[inline]
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(&[])
    }
    #[inline]
    fn consume(&mut self, _n: usize) {}
}

#[stable(feature = "empty_seek", since = "1.51.0")]
impl Seek for Empty {
    fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
        Ok(0)
    }

    fn stream_len(&mut self) -> io::Result<u64> {
        Ok(0)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(0)
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Empty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Empty").finish_non_exhaustive()
    }
}

impl SizeHint for Empty {
    #[inline]
    fn upper_bound(&self) -> Option<usize> {
        Some(0)
    }
}

/// A reader which yields one byte over and over and over and over and over and...
///
/// This struct is generally created by calling [`repeat()`]. Please
/// see the documentation of [`repeat()`] for more details.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Repeat {
    byte: u8,
}

/// Creates an instance of a reader that infinitely repeats one byte.
///
/// All reads from this reader will succeed by filling the specified buffer with
/// the given byte.
///
/// # Examples
///
/// ```
/// use std::io::{self, Read};
///
/// let mut buffer = [0; 3];
/// io::repeat(0b101).read_exact(&mut buffer).unwrap();
/// assert_eq!(buffer, [0b101, 0b101, 0b101]);
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_io_structs", issue = "78812")]
pub const fn repeat(byte: u8) -> Repeat {
    Repeat { byte }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Read for Repeat {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        for slot in &mut *buf {
            *slot = self.byte;
        }
        Ok(buf.len())
    }

    fn read_buf(&mut self, buf: &mut ReadBuf<'_>) -> io::Result<()> {
        for slot in unsafe { buf.unfilled_mut() } {
            slot.write(self.byte);
        }

        let remaining = buf.remaining();

        unsafe {
            buf.assume_init(remaining);
        }

        buf.add_filled(remaining);

        Ok(())
    }

    #[inline]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut nwritten = 0;
        for buf in bufs {
            nwritten += self.read(buf)?;
        }
        Ok(nwritten)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        true
    }
}

impl SizeHint for Repeat {
    #[inline]
    fn lower_bound(&self) -> usize {
        usize::MAX
    }

    #[inline]
    fn upper_bound(&self) -> Option<usize> {
        None
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Repeat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Repeat").finish_non_exhaustive()
    }
}

/// A writer which will move data into the void.
///
/// This struct is generally created by calling [`sink`]. Please
/// see the documentation of [`sink()`] for more details.
#[stable(feature = "rust1", since = "1.0.0")]
#[non_exhaustive]
#[derive(Copy, Clone, Default)]
pub struct Sink;

/// Creates an instance of a writer which will successfully consume all data.
///
/// All calls to [`write`] on the returned instance will return `Ok(buf.len())`
/// and the contents of the buffer will not be inspected.
///
/// [`write`]: Write::write
///
/// # Examples
///
/// ```rust
/// use std::io::{self, Write};
///
/// let buffer = vec![1, 2, 3, 5, 8];
/// let num_bytes = io::sink().write(&buffer).unwrap();
/// assert_eq!(num_bytes, 5);
/// ```
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_io_structs", issue = "78812")]
pub const fn sink() -> Sink {
    Sink
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Sink {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum();
        Ok(total_len)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "write_mt", since = "1.48.0")]
impl Write for &Sink {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let total_len = bufs.iter().map(|b| b.len()).sum();
        Ok(total_len)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for Sink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sink").finish_non_exhaustive()
    }
}
