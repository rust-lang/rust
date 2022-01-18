//! Forwarding implementations for Rc

use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut, Read, ReadBuf, Seek, SeekFrom, Write};
use alloc::sync::Arc;

#[stable(feature = "io_delegation_rc", since = "1.60.0")]
impl<T> Read for Arc<T>
where
    for<'a> &'a T: Read,
{
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (self as &T).read(buf)
    }

    #[inline]
    fn read_buf(&mut self, buf: &mut ReadBuf<'_>) -> io::Result<()> {
        (self as &T).read_buf(buf)
    }

    #[inline]
    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        (self as &T).read_vectored(bufs)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        (self as &T).is_read_vectored()
    }

    #[inline]
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        (self as &T).read_to_end(buf)
    }

    #[inline]
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        (self as &T).read_to_string(buf)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        (self as &T).read_exact(buf)
    }
}
#[stable(feature = "io_delegation_rc", since = "1.60.0")]
impl<T> Write for Arc<T>
where
    for<'a> &'a T: Write,
{
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (self as &T).write(buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        (self as &T).write_vectored(bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        (self as &T).is_write_vectored()
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        (self as &T).flush()
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        (self as &T).write_all(buf)
    }

    #[inline]
    fn write_fmt(&mut self, fmt: fmt::Arguments<'_>) -> io::Result<()> {
        (self as &T).write_fmt(fmt)
    }
}
#[stable(feature = "io_delegation_rc", since = "1.60.0")]
impl<T> Seek for Arc<T>
where
    for<'a> &'a T: Seek,
{
    #[inline]
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        (self as &T).seek(pos)
    }

    #[inline]
    fn stream_position(&mut self) -> io::Result<u64> {
        (self as &T).stream_position()
    }
}
