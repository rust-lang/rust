#[cfg(test)]
mod tests;

use crate::alloc::Allocator;
use crate::cmp;
use crate::io::prelude::*;
use crate::io::{self, BorrowedCursor, ErrorKind, IoSlice, IoSliceMut, SeekFrom};

/// A `Cursor` wraps an in-memory buffer and provides it with a
/// [`Seek`] implementation.
///
/// `Cursor`s are used with in-memory buffers, anything implementing
/// <code>[AsRef]<\[u8]></code>, to allow them to implement [`Read`] and/or [`Write`],
/// allowing these buffers to be used anywhere you might use a reader or writer
/// that does actual I/O.
///
/// The standard library implements some I/O traits on various types which
/// are commonly used as a buffer, like <code>Cursor<[Vec]\<u8>></code> and
/// <code>Cursor<[&\[u8\]][bytes]></code>.
///
/// # Examples
///
/// We may want to write bytes to a [`File`] in our production
/// code, but use an in-memory buffer in our tests. We can do this with
/// `Cursor`:
///
/// [bytes]: crate::slice "slice"
/// [`File`]: crate::fs::File
///
/// ```no_run
/// use std::io::prelude::*;
/// use std::io::{self, SeekFrom};
/// use std::fs::File;
///
/// // a library function we've written
/// fn write_ten_bytes_at_end<W: Write + Seek>(mut writer: W) -> io::Result<()> {
///     writer.seek(SeekFrom::End(-10))?;
///
///     for i in 0..10 {
///         writer.write(&[i])?;
///     }
///
///     // all went well
///     Ok(())
/// }
///
/// # fn foo() -> io::Result<()> {
/// // Here's some code that uses this library function.
/// //
/// // We might want to use a BufReader here for efficiency, but let's
/// // keep this example focused.
/// let mut file = File::create("foo.txt")?;
/// // First, we need to allocate 10 bytes to be able to write into.
/// file.set_len(10)?;
///
/// write_ten_bytes_at_end(&mut file)?;
/// # Ok(())
/// # }
///
/// // now let's write a test
/// #[test]
/// fn test_writes_bytes() {
///     // setting up a real File is much slower than an in-memory buffer,
///     // let's use a cursor instead
///     use std::io::Cursor;
///     let mut buff = Cursor::new(vec![0; 15]);
///
///     write_ten_bytes_at_end(&mut buff).unwrap();
///
///     assert_eq!(&buff.get_ref()[5..15], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug, Default, Eq, PartialEq)]
pub struct Cursor<T> {
    inner: T,
    pos: u64,
}

impl<T> Cursor<T> {
    /// Creates a new cursor wrapping the provided underlying in-memory buffer.
    ///
    /// Cursor initial position is `0` even if underlying buffer (e.g., [`Vec`])
    /// is not empty. So writing to cursor starts with overwriting [`Vec`]
    /// content, not with appending to it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    ///
    /// let buff = Cursor::new(Vec::new());
    /// # fn force_inference(_: &Cursor<Vec<u8>>) {}
    /// # force_inference(&buff);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_io_structs", since = "1.79.0")]
    pub const fn new(inner: T) -> Cursor<T> {
        Cursor { pos: 0, inner }
    }

    /// Consumes this cursor, returning the underlying value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    ///
    /// let buff = Cursor::new(Vec::new());
    /// # fn force_inference(_: &Cursor<Vec<u8>>) {}
    /// # force_inference(&buff);
    ///
    /// let vec = buff.into_inner();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Gets a reference to the underlying value in this cursor.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    ///
    /// let buff = Cursor::new(Vec::new());
    /// # fn force_inference(_: &Cursor<Vec<u8>>) {}
    /// # force_inference(&buff);
    ///
    /// let reference = buff.get_ref();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_io_structs", since = "1.79.0")]
    pub const fn get_ref(&self) -> &T {
        &self.inner
    }

    /// Gets a mutable reference to the underlying value in this cursor.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the
    /// underlying value as it may corrupt this cursor's position.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    ///
    /// let mut buff = Cursor::new(Vec::new());
    /// # fn force_inference(_: &Cursor<Vec<u8>>) {}
    /// # force_inference(&buff);
    ///
    /// let reference = buff.get_mut();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_mut_cursor", since = "1.86.0")]
    pub const fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Returns the current position of this cursor.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    /// use std::io::prelude::*;
    /// use std::io::SeekFrom;
    ///
    /// let mut buff = Cursor::new(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(buff.position(), 0);
    ///
    /// buff.seek(SeekFrom::Current(2)).unwrap();
    /// assert_eq!(buff.position(), 2);
    ///
    /// buff.seek(SeekFrom::Current(-1)).unwrap();
    /// assert_eq!(buff.position(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_io_structs", since = "1.79.0")]
    pub const fn position(&self) -> u64 {
        self.pos
    }

    /// Sets the position of this cursor.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    ///
    /// let mut buff = Cursor::new(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(buff.position(), 0);
    ///
    /// buff.set_position(2);
    /// assert_eq!(buff.position(), 2);
    ///
    /// buff.set_position(4);
    /// assert_eq!(buff.position(), 4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_mut_cursor", since = "1.86.0")]
    pub const fn set_position(&mut self, pos: u64) {
        self.pos = pos;
    }
}

impl<T> Cursor<T>
where
    T: AsRef<[u8]>,
{
    /// Splits the underlying slice at the cursor position and returns them.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(cursor_split)]
    /// use std::io::Cursor;
    ///
    /// let mut buff = Cursor::new(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(buff.split(), ([].as_slice(), [1, 2, 3, 4, 5].as_slice()));
    ///
    /// buff.set_position(2);
    /// assert_eq!(buff.split(), ([1, 2].as_slice(), [3, 4, 5].as_slice()));
    ///
    /// buff.set_position(6);
    /// assert_eq!(buff.split(), ([1, 2, 3, 4, 5].as_slice(), [].as_slice()));
    /// ```
    #[unstable(feature = "cursor_split", issue = "86369")]
    pub fn split(&self) -> (&[u8], &[u8]) {
        let slice = self.inner.as_ref();
        let pos = self.pos.min(slice.len() as u64);
        slice.split_at(pos as usize)
    }
}

impl<T> Cursor<T>
where
    T: AsMut<[u8]>,
{
    /// Splits the underlying slice at the cursor position and returns them
    /// mutably.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(cursor_split)]
    /// use std::io::Cursor;
    ///
    /// let mut buff = Cursor::new(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(buff.split_mut(), ([].as_mut_slice(), [1, 2, 3, 4, 5].as_mut_slice()));
    ///
    /// buff.set_position(2);
    /// assert_eq!(buff.split_mut(), ([1, 2].as_mut_slice(), [3, 4, 5].as_mut_slice()));
    ///
    /// buff.set_position(6);
    /// assert_eq!(buff.split_mut(), ([1, 2, 3, 4, 5].as_mut_slice(), [].as_mut_slice()));
    /// ```
    #[unstable(feature = "cursor_split", issue = "86369")]
    pub fn split_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        let slice = self.inner.as_mut();
        let pos = self.pos.min(slice.len() as u64);
        slice.split_at_mut(pos as usize)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Cursor<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Cursor { inner: self.inner.clone(), pos: self.pos }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.inner.clone_from(&other.inner);
        self.pos = other.pos;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> io::Seek for Cursor<T>
where
    T: AsRef<[u8]>,
{
    fn seek(&mut self, style: SeekFrom) -> io::Result<u64> {
        let (base_pos, offset) = match style {
            SeekFrom::Start(n) => {
                self.pos = n;
                return Ok(n);
            }
            SeekFrom::End(n) => (self.inner.as_ref().len() as u64, n),
            SeekFrom::Current(n) => (self.pos, n),
        };
        match base_pos.checked_add_signed(offset) {
            Some(n) => {
                self.pos = n;
                Ok(self.pos)
            }
            None => Err(io::const_error!(
                ErrorKind::InvalidInput,
                "invalid seek to a negative or overflowing position",
            )),
        }
    }

    fn stream_len(&mut self) -> io::Result<u64> {
        Ok(self.inner.as_ref().len() as u64)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.pos)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Read for Cursor<T>
where
    T: AsRef<[u8]>,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = Read::read(&mut Cursor::split(self).1, buf)?;
        self.pos += n as u64;
        Ok(n)
    }

    fn read_buf(&mut self, mut cursor: BorrowedCursor<'_>) -> io::Result<()> {
        let prev_written = cursor.written();

        Read::read_buf(&mut Cursor::split(self).1, cursor.reborrow())?;

        self.pos += (cursor.written() - prev_written) as u64;

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
            Ok(_) => self.pos += buf.len() as u64,
            // The only possible error condition is EOF, so place the cursor at "EOF"
            Err(_) => self.pos = self.inner.as_ref().len() as u64,
        }

        result
    }

    fn read_buf_exact(&mut self, mut cursor: BorrowedCursor<'_>) -> io::Result<()> {
        let prev_written = cursor.written();

        let result = Read::read_buf_exact(&mut Cursor::split(self).1, cursor.reborrow());
        self.pos += (cursor.written() - prev_written) as u64;

        result
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let content = Cursor::split(self).1;
        let len = content.len();
        buf.try_reserve(len)?;
        buf.extend_from_slice(content);
        self.pos += len as u64;

        Ok(len)
    }

    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        let content =
            crate::str::from_utf8(Cursor::split(self).1).map_err(|_| io::Error::INVALID_UTF8)?;
        let len = content.len();
        buf.try_reserve(len)?;
        buf.push_str(content);
        self.pos += len as u64;

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
        self.pos += amt as u64;
    }
}

// Non-resizing write implementation
#[inline]
fn slice_write(pos_mut: &mut u64, slice: &mut [u8], buf: &[u8]) -> io::Result<usize> {
    let pos = cmp::min(*pos_mut, slice.len() as u64);
    let amt = (&mut slice[(pos as usize)..]).write(buf)?;
    *pos_mut += amt as u64;
    Ok(amt)
}

#[inline]
fn slice_write_vectored(
    pos_mut: &mut u64,
    slice: &mut [u8],
    bufs: &[IoSlice<'_>],
) -> io::Result<usize> {
    let mut nwritten = 0;
    for buf in bufs {
        let n = slice_write(pos_mut, slice, buf)?;
        nwritten += n;
        if n < buf.len() {
            break;
        }
    }
    Ok(nwritten)
}

/// Reserves the required space, and pads the vec with 0s if necessary.
fn reserve_and_pad<A: Allocator>(
    pos_mut: &mut u64,
    vec: &mut Vec<u8, A>,
    buf_len: usize,
) -> io::Result<usize> {
    let pos: usize = (*pos_mut).try_into().map_err(|_| {
        io::const_error!(
            ErrorKind::InvalidInput,
            "cursor position exceeds maximum possible vector length",
        )
    })?;

    // For safety reasons, we don't want these numbers to overflow
    // otherwise our allocation won't be enough
    let desired_cap = pos.saturating_add(buf_len);
    if desired_cap > vec.capacity() {
        // We want our vec's total capacity
        // to have room for (pos+buf_len) bytes. Reserve allocates
        // based on additional elements from the length, so we need to
        // reserve the difference
        vec.reserve(desired_cap - vec.len());
    }
    // Pad if pos is above the current len.
    if pos > vec.len() {
        let diff = pos - vec.len();
        // Unfortunately, `resize()` would suffice but the optimiser does not
        // realise the `reserve` it does can be eliminated. So we do it manually
        // to eliminate that extra branch
        let spare = vec.spare_capacity_mut();
        debug_assert!(spare.len() >= diff);
        // Safety: we have allocated enough capacity for this.
        // And we are only writing, not reading
        unsafe {
            spare.get_unchecked_mut(..diff).fill(core::mem::MaybeUninit::new(0));
            vec.set_len(pos);
        }
    }

    Ok(pos)
}

/// Writes the slice to the vec without allocating
/// # Safety: vec must have buf.len() spare capacity
unsafe fn vec_write_unchecked<A>(pos: usize, vec: &mut Vec<u8, A>, buf: &[u8]) -> usize
where
    A: Allocator,
{
    debug_assert!(vec.capacity() >= pos + buf.len());
    unsafe { vec.as_mut_ptr().add(pos).copy_from(buf.as_ptr(), buf.len()) };
    pos + buf.len()
}

/// Resizing write implementation for [`Cursor`]
///
/// Cursor is allowed to have a pre-allocated and initialised
/// vector body, but with a position of 0. This means the [`Write`]
/// will overwrite the contents of the vec.
///
/// This also allows for the vec body to be empty, but with a position of N.
/// This means that [`Write`] will pad the vec with 0 initially,
/// before writing anything from that point
fn vec_write<A>(pos_mut: &mut u64, vec: &mut Vec<u8, A>, buf: &[u8]) -> io::Result<usize>
where
    A: Allocator,
{
    let buf_len = buf.len();
    let mut pos = reserve_and_pad(pos_mut, vec, buf_len)?;

    // Write the buf then progress the vec forward if necessary
    // Safety: we have ensured that the capacity is available
    // and that all bytes get written up to pos
    unsafe {
        pos = vec_write_unchecked(pos, vec, buf);
        if pos > vec.len() {
            vec.set_len(pos);
        }
    };

    // Bump us forward
    *pos_mut += buf_len as u64;
    Ok(buf_len)
}

/// Resizing write_vectored implementation for [`Cursor`]
///
/// Cursor is allowed to have a pre-allocated and initialised
/// vector body, but with a position of 0. This means the [`Write`]
/// will overwrite the contents of the vec.
///
/// This also allows for the vec body to be empty, but with a position of N.
/// This means that [`Write`] will pad the vec with 0 initially,
/// before writing anything from that point
fn vec_write_vectored<A>(
    pos_mut: &mut u64,
    vec: &mut Vec<u8, A>,
    bufs: &[IoSlice<'_>],
) -> io::Result<usize>
where
    A: Allocator,
{
    // For safety reasons, we don't want this sum to overflow ever.
    // If this saturates, the reserve should panic to avoid any unsound writing.
    let buf_len = bufs.iter().fold(0usize, |a, b| a.saturating_add(b.len()));
    let mut pos = reserve_and_pad(pos_mut, vec, buf_len)?;

    // Write the buf then progress the vec forward if necessary
    // Safety: we have ensured that the capacity is available
    // and that all bytes get written up to the last pos
    unsafe {
        for buf in bufs {
            pos = vec_write_unchecked(pos, vec, buf);
        }
        if pos > vec.len() {
            vec.set_len(pos);
        }
    }

    // Bump us forward
    *pos_mut += buf_len as u64;
    Ok(buf_len)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Write for Cursor<&mut [u8]> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        slice_write(&mut self.pos, self.inner, buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        slice_write_vectored(&mut self.pos, self.inner, bufs)
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

#[stable(feature = "cursor_mut_vec", since = "1.25.0")]
impl<A> Write for Cursor<&mut Vec<u8, A>>
where
    A: Allocator,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        vec_write(&mut self.pos, self.inner, buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        vec_write_vectored(&mut self.pos, self.inner, bufs)
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Write for Cursor<Vec<u8, A>>
where
    A: Allocator,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        vec_write(&mut self.pos, &mut self.inner, buf)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        vec_write_vectored(&mut self.pos, &mut self.inner, bufs)
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

#[stable(feature = "cursor_box_slice", since = "1.5.0")]
impl<A> Write for Cursor<Box<[u8], A>>
where
    A: Allocator,
{
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        slice_write(&mut self.pos, &mut self.inner, buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        slice_write_vectored(&mut self.pos, &mut self.inner, bufs)
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

#[stable(feature = "cursor_array", since = "1.61.0")]
impl<const N: usize> Write for Cursor<[u8; N]> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        slice_write(&mut self.pos, &mut self.inner, buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        slice_write_vectored(&mut self.pos, &mut self.inner, bufs)
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
