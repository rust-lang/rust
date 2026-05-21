use crate::alloc::Allocator;
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::collections::VecDeque;
use crate::fmt;
use crate::io::{self, IoSlice, Seek, SeekFrom, SizeHint, Write};
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
use crate::sync::Arc;
use crate::vec::Vec;

// =============================================================================
// Forwarding implementations

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
impl<T> SizeHint for Box<T> {
    #[inline]
    fn lower_bound(&self) -> usize {
        SizeHint::lower_bound(&**self)
    }

    #[inline]
    fn upper_bound(&self) -> Option<usize> {
        SizeHint::upper_bound(&**self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: Write + ?Sized> Write for Box<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (**self).write(buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (**self).write_vectored(bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        (**self).is_write_vectored()
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        (**self).flush()
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        (**self).write_all(buf)
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        (**self).write_all_vectored(bufs)
    }

    #[inline]
    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        (**self).write_fmt(fmt)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<S: Seek + ?Sized> Seek for Box<S> {
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        (**self).seek(pos)
    }

    #[inline]
    fn rewind(&mut self) -> io::Result<()> {
        (**self).rewind()
    }

    #[inline]
    fn stream_len(&mut self) -> io::Result<u64> {
        (**self).stream_len()
    }

    #[inline]
    fn stream_position(&mut self) -> io::Result<u64> {
        (**self).stream_position()
    }

    #[inline]
    fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        (**self).seek_relative(offset)
    }
}

// =============================================================================
// In-memory buffer implementations

/// Write is implemented for `Vec<u8>` by appending to the vector.
/// The vector will grow as needed.
#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Allocator> Write for Vec<u8, A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        <Self as Write>::write_all(self, buf)?;
        Ok(buf.len())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let len = bufs.iter().map(|b| b.len()).sum();
        cfg_select! {
            no_global_oom_handling => {
                self.try_reserve(len)?;
            }
            _ => {
                self.reserve(len);
            }
        }
        for buf in bufs {
            <Self as Write>::write_all(self, buf)?;
        }
        Ok(len)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        cfg_select! {
            no_global_oom_handling => {
                let n = buf.len();
                self.try_reserve(n)?;
                // SAFETY:
                // * self and buf are non-overlapping
                // * self[..len] is already initialized
                // * self[len..len + n] is initialized by copy_nonoverlapping
                // * len + n is within the capacity of self based on the reservation completed above
                unsafe {
                    let len = self.len();
                    let src = buf.as_ptr();
                    let dst = self.as_mut_ptr().add(len);
                    core::ptr::copy_nonoverlapping(src, dst, n);
                    self.set_len(len + n);
                }
            }
            _ => {
                self.extend_from_slice(buf);
            }
        }
        Ok(())
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        self.write_vectored(bufs)?;
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Write is implemented for `VecDeque<u8>` by appending to the `VecDeque`, growing it as needed.
#[cfg(not(no_global_oom_handling))]
#[stable(feature = "vecdeque_read_write", since = "1.63.0")]
impl<A: Allocator> Write for VecDeque<u8, A> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.extend(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let len = bufs.iter().map(|b| b.len()).sum();
        self.reserve(len);
        for buf in bufs {
            self.extend(&**buf);
        }
        Ok(len)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.extend(buf);
        Ok(())
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        self.write_vectored(bufs)?;
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
#[stable(feature = "io_traits_arc", since = "1.73.0")]
impl<W: Write + ?Sized> Write for Arc<W>
where
    for<'a> &'a W: Write,
    W: crate::io::IoHandle,
{
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&**self).write(buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (&**self).write_vectored(bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        (&**self).is_write_vectored()
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        (&**self).flush()
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        (&**self).write_all(buf)
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        (&**self).write_all_vectored(bufs)
    }

    #[inline]
    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        (&**self).write_fmt(fmt)
    }
}
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
#[stable(feature = "io_traits_arc", since = "1.73.0")]
impl<S: Seek + ?Sized> Seek for Arc<S>
where
    for<'a> &'a S: Seek,
    S: crate::io::IoHandle,
{
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        (&**self).seek(pos)
    }

    #[inline]
    fn rewind(&mut self) -> io::Result<()> {
        (&**self).rewind()
    }

    #[inline]
    fn stream_len(&mut self) -> io::Result<u64> {
        (&**self).stream_len()
    }

    #[inline]
    fn stream_position(&mut self) -> io::Result<u64> {
        (&**self).stream_position()
    }

    #[inline]
    fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        (&**self).seek_relative(offset)
    }
}
