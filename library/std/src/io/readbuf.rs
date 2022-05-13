#![unstable(feature = "read_buf", issue = "78485")]

#[cfg(test)]
mod tests;

use crate::cmp;
use crate::fmt::{self, Debug, Formatter};
use crate::mem::MaybeUninit;

// TODO docs
/// A wrapper around a byte buffer that is incrementally filled and initialized.
///
/// This type is a sort of "double cursor". It tracks three regions in the buffer: a region at the beginning of the
/// buffer that has been logically filled with data, a region that has been initialized at some point but not yet
/// logically filled, and a region at the end that is fully uninitialized. The filled region is guaranteed to be a
/// subset of the initialized region.
///
/// In summary, the contents of the buffer can be visualized as:
/// ```not_rust
/// [             capacity              ]
/// [ filled |         unfilled         ]
/// [    initialized    | uninitialized ]
/// ```
pub struct BorrowBuf<'a> {
    buf: &'a mut [MaybeUninit<u8>],
    filled: usize,
    initialized: usize,
}

impl Debug for BorrowBuf<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadBuf")
            .field("init", &self.initialized)
            .field("filled", &self.filled)
            .field("capacity", &self.capacity())
            .finish()
    }
}

/// Creates a new `BorrowBuf` from a fully initialized slice.
impl<'a> From<&'a mut [u8]> for BorrowBuf<'a> {
    #[inline]
    fn from(slice: &'a mut [u8]) -> BorrowBuf<'a> {
        let len = slice.len();

        BorrowBuf {
            //SAFETY: initialized data never becoming uninitialized is an invariant of BorrowBuf
            buf: unsafe { (slice as *mut [u8]).as_uninit_slice_mut().unwrap() },
            filled: 0,
            initialized: len,
        }
    }
}

/// Creates a new `BorrowBuf` from a fully uninitialized buffer.
///
/// Use `set_init` if part of the buffer is known to be already initialized.
impl<'a> From<&'a mut [MaybeUninit<u8>]> for BorrowBuf<'a> {
    #[inline]
    fn from(buf: &'a mut [MaybeUninit<u8>]) -> BorrowBuf<'a> {
        BorrowBuf { buf, filled: 0, initialized: 0 }
    }
}

impl<'a> BorrowBuf<'a> {
    /// Returns the total capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Returns the length of the filled part of the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.filled
    }

    /// Returns the length of the initialized part of the buffer.
    #[inline]
    pub fn init_len(&self) -> usize {
        self.initialized
    }

    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        //SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buf[0..self.filled]) }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled<'b>(&'b mut self) -> BorrowCursor<'a, 'b> {
        BorrowCursor { start: self.filled, buf: self }
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The number of initialized bytes is not changed, and the contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) -> &mut Self {
        self.filled = 0;
        self
    }

    /// Asserts that the first `n` bytes of the buffer are initialized.
    ///
    /// `BorrowBuf` assumes that bytes are never de-initialized, so this method does nothing when called with fewer
    /// bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` unfilled bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.initialized = cmp::max(self.initialized, n);
        self
    }
}

/// A cursor view of a [`BorrowBuf`](BorrowBuf).
///
/// Provides mutable access to the unfilled portion (both initialised and uninitialised data) from
/// the buffer.
#[derive(Debug)]
pub struct BorrowCursor<'a, 'b> {
    buf: &'b mut BorrowBuf<'a>,
    start: usize,
}

impl<'a, 'b> BorrowCursor<'a, 'b> {
    /// Clone this cursor.
    #[inline]
    pub fn clone<'c>(&'c mut self) -> BorrowCursor<'a, 'c> {
        BorrowCursor { buf: self.buf, start: self.start }
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity() - self.buf.filled
    }

    /// Returns the number of bytes written to this cursor.
    // TODO check for reuse uses
    #[inline]
    pub fn written(&self) -> usize {
        self.buf.filled - self.start
    }

    /// Returns a shared reference to the initialized portion of the buffer.
    #[inline]
    pub fn init_ref(&self) -> &[u8] {
        //SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe {
            MaybeUninit::slice_assume_init_ref(&self.buf.buf[self.buf.filled..self.buf.initialized])
        }
    }

    /// Returns a mutable reference to the initialized portion of the buffer.
    #[inline]
    pub fn init_mut(&mut self) -> &mut [u8] {
        //SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe {
            MaybeUninit::slice_assume_init_mut(
                &mut self.buf.buf[self.buf.filled..self.buf.initialized],
            )
        }
    }

    /// Returns a mutable reference to the uninitialized part of the buffer.
    ///
    /// It is safe to uninitialize any of these bytes.
    #[inline]
    pub fn uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buf.buf[self.buf.initialized..]
    }

    /// A view of the cursor as a mutable slice of `MaybeUninit<u8>`.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buf.buf[self.buf.filled..]
    }

    /// Increases the size of the filled region of the buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` elements of the cursor have been properly
    /// initialised.
    #[inline]
    pub unsafe fn advance(&mut self, n: usize) -> &mut Self {
        self.buf.filled += n;
        self.buf.initialized = cmp::max(self.buf.initialized, self.buf.filled);
        self
    }

    /// Initialised all bytes in the cursor.
    #[inline]
    pub fn ensure_init(&mut self) -> &mut Self {
        for byte in self.uninit_mut() {
            byte.write(0);
        }
        self.buf.initialized = self.buf.capacity();

        self
    }

    /// Asserts that the first `n` unfilled bytes of the cursor are initialized.
    ///
    /// `BorrowBuf` assumes that bytes are never de-initialized, so this method does nothing when called with fewer
    /// bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.buf.initialized = cmp::max(self.buf.initialized, self.buf.filled + n);
        self
    }

    /// Appends data to the cursor, advancing the position within its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `buf.len()`.
    #[inline]
    pub fn append(&mut self, buf: &[u8]) {
        assert!(self.capacity() >= buf.len());

        // SAFETY: we do not de-initialize any of the elements of the slice
        unsafe {
            MaybeUninit::write_slice(&mut self.as_mut()[..buf.len()], buf);
        }

        // SAFETY: We just added the entire contents of buf to the filled section.
        unsafe {

            self.set_init(buf.len());
        }
        self.buf.filled += buf.len();
    }
}
