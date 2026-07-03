#[cfg(test)]
mod tests;

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::io::Cursor;

use alloc_crate::io::{
    slice_write, slice_write_all, slice_write_all_vectored, slice_write_vectored, vec_write_all,
    vec_write_all_vectored,
};

use crate::alloc::Allocator;
use crate::io::prelude::*;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Read for Cursor<T>
where
    T: AsRef<[u8]>,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = Read::read(&mut Cursor::split(self).1, buf)?;
        self.set_position(self.position() + n as u64);
        Ok(n)
    }

    fn read_buf(&mut self, mut cursor: BorrowedCursor<'_, u8>) -> io::Result<()> {
        let prev_written = cursor.written();

        Read::read_buf(&mut Cursor::split(self).1, cursor.reborrow())?;

        self.set_position(self.position() + (cursor.written() - prev_written) as u64);

        Ok(())
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        let mut nread = 0;
        for buf in bufs {
            let n = self.read(buf)?;
            nread += n;
            if n < buf.len() {
                break;
            }
        }
        Ok(nread)
    }

    fn is_read_vectored(&self) -> bool {
        true
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let result = Read::read_exact(&mut Cursor::split(self).1, buf);

        match result {
            Ok(_) => self.set_position(self.position() + buf.len() as u64),
            // The only possible error condition is EOF, so place the cursor at "EOF"
            Err(_) => self.set_position(self.get_ref().as_ref().len() as u64),
        }

        result
    }

    fn read_buf_exact(&mut self, mut cursor: BorrowedCursor<'_, u8>) -> io::Result<()> {
        let prev_written = cursor.written();

        let result = Read::read_buf_exact(&mut Cursor::split(self).1, cursor.reborrow());
        self.set_position(self.position() + (cursor.written() - prev_written) as u64);

        result
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let content = Cursor::split(self).1;
        let len = content.len();
        buf.try_reserve(len)?;
        buf.extend_from_slice(content);
        self.set_position(self.position() + len as u64);

        Ok(len)
    }

    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        let content =
            crate::str::from_utf8(Cursor::split(self).1).map_err(|_| io::Error::INVALID_UTF8)?;
        let len = content.len();
        buf.try_reserve(len)?;
        buf.push_str(content);
        self.set_position(self.position() + len as u64);

        Ok(len)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> BufRead for Cursor<T>
where
    T: AsRef<[u8]>,
{
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(Cursor::split(self).1)
    }
    fn consume(&mut self, amt: usize) {
        self.set_position(self.position() + amt as u64);
    }
}

/// Trait used to allow indirect implementation of `Write` for `Cursor<Self>`.
/// Since [`Cursor`] is not a foundational type, it is not possible to implement
/// `Write` for `Cursor<T>` if `Write` is defined in `libcore` and `T` is in a
/// downstream crate (e.g., `liballoc` or `libstd`).
///
/// Methods are identical in purpose and meaning to their `Write` namesakes.
trait WriteThroughCursor: Sized {
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize>;
    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize>;
    fn is_write_vectored(this: &Cursor<Self>) -> bool;
    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()>;
    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()>;
    fn flush(this: &mut Cursor<Self>) -> io::Result<()>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<W: WriteThroughCursor> Write for Cursor<W> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        WriteThroughCursor::write(self, buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        WriteThroughCursor::write_vectored(self, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        WriteThroughCursor::is_write_vectored(self)
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        WriteThroughCursor::write_all(self, buf)
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        WriteThroughCursor::write_all_vectored(self, bufs)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        WriteThroughCursor::flush(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Cursor<&mut [u8]> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = self.into_parts_mut();
        slice_write(pos, inner, buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = self.into_parts_mut();
        slice_write_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = self.into_parts_mut();
        slice_write_all(pos, inner, buf)
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = self.into_parts_mut();
        slice_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "cursor_mut_vec", since = "1.25.0")]
impl<A> WriteThroughCursor for &mut Vec<u8, A>
where
    A: Allocator,
{
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)
    }

    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(_this: &Cursor<Self>) -> bool {
        true
    }

    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)?;
        Ok(())
    }

    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)?;
        Ok(())
    }

    #[inline]
    fn flush(_this: &mut Cursor<Self>) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> WriteThroughCursor for Vec<u8, A>
where
    A: Allocator,
{
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)
    }

    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(_this: &Cursor<Self>) -> bool {
        true
    }

    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all(pos, inner, buf)?;
        Ok(())
    }

    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        vec_write_all_vectored(pos, inner, bufs)?;
        Ok(())
    }

    #[inline]
    fn flush(_this: &mut Cursor<Self>) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "cursor_box_slice", since = "1.5.0")]
impl<A> WriteThroughCursor for Box<[u8], A>
where
    A: Allocator,
{
    #[inline]
    fn write(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        slice_write(pos, inner, buf)
    }

    #[inline]
    fn write_vectored(this: &mut Cursor<Self>, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = this.into_parts_mut();
        slice_write_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(_this: &Cursor<Self>) -> bool {
        true
    }

    #[inline]
    fn write_all(this: &mut Cursor<Self>, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        slice_write_all(pos, inner, buf)
    }

    #[inline]
    fn write_all_vectored(this: &mut Cursor<Self>, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = this.into_parts_mut();
        slice_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn flush(_this: &mut Cursor<Self>) -> io::Result<()> {
        Ok(())
    }
}

#[stable(feature = "cursor_array", since = "1.61.0")]
impl<const N: usize> Write for Cursor<[u8; N]> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let (pos, inner) = self.into_parts_mut();
        slice_write(pos, inner, buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let (pos, inner) = self.into_parts_mut();
        slice_write_vectored(pos, inner, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let (pos, inner) = self.into_parts_mut();
        slice_write_all(pos, inner, buf)
    }

    #[inline]
    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        let (pos, inner) = self.into_parts_mut();
        slice_write_all_vectored(pos, inner, bufs)
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
