use crate::boxed::Box;
use crate::io::{self, Seek, SeekFrom, SizeHint};
#[cfg(all(not(no_rc), not(no_sync), target_has_atomic = "ptr"))]
use crate::sync::Arc;

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
