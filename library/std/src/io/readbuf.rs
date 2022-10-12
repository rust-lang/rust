#![unstable(feature = "read_buf", issue = "78485")]

#[cfg(test)]
mod tests;

use crate::cmp;
use crate::fmt::{self, Debug, Formatter};
use crate::io::{Result, Write};
use crate::marker::PhantomData;
use crate::mem::{self, MaybeUninit};
use crate::ops::{Deref, DerefMut};
use crate::slice;

/// A borrowed byte buffer which is incrementally filled and initialized.
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
///
/// A `BorrowedBuf` is created around some existing data (or capacity for data) via a unique reference
/// (`&mut`). The `BorrowedBuf` can be configured (e.g., using `clear` or `set_init`), but cannot be
/// directly written. To write into the buffer, use `unfilled` to create a `BorrowedCursor`. The cursor
/// has write-only access to the unfilled portion of the buffer (you can think of it as a
/// write-only iterator).
///
/// The lifetime `'data` is a bound on the lifetime of the underlying data.
pub struct BorrowedBuf<'data> {
    /// The buffer's underlying data.
    buf: &'data mut [MaybeUninit<u8>],
    /// The length of `self.buf` which is known to be filled.
    filled: usize,
    /// The length of `self.buf` which is known to be initialized.
    init: usize,
}

impl Debug for BorrowedBuf<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedBuf")
            .field("init", &self.init)
            .field("filled", &self.filled)
            .field("capacity", &self.capacity())
            .finish()
    }
}

/// Create a new `BorrowedBuf` from a fully initialized slice.
impl<'data> From<&'data mut [u8]> for BorrowedBuf<'data> {
    #[inline]
    fn from(slice: &'data mut [u8]) -> BorrowedBuf<'data> {
        let len = slice.len();

        BorrowedBuf {
            // SAFETY: initialized data never becoming uninitialized is an invariant of BorrowedBuf
            buf: unsafe { (slice as *mut [u8]).as_uninit_slice_mut().unwrap() },
            filled: 0,
            init: len,
        }
    }
}

/// Create a new `BorrowedBuf` from an uninitialized buffer.
///
/// Use `set_init` if part of the buffer is known to be already initialized.
impl<'data> From<&'data mut [MaybeUninit<u8>]> for BorrowedBuf<'data> {
    #[inline]
    fn from(buf: &'data mut [MaybeUninit<u8>]) -> BorrowedBuf<'data> {
        BorrowedBuf { buf, filled: 0, init: 0 }
    }
}

impl<'data> BorrowedBuf<'data> {
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
        self.init
    }

    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buf[0..self.filled]) }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled<'this>(&'this mut self) -> BorrowedCursor<'this> {
        BorrowedCursor {
            start: self.filled,
            // SAFETY: we never assign into `BorrowedCursor::buf`, so treating its
            // lifetime covariantly is safe.
            buf: unsafe {
                mem::transmute::<&'this mut BorrowedBuf<'data>, &'this mut BorrowedBuf<'this>>(self)
            },
        }
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
    /// `BorrowedBuf` assumes that bytes are never de-initialized, so this method does nothing when called with fewer
    /// bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` unfilled bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.init = cmp::max(self.init, n);
        self
    }
}

/// A writeable view of the unfilled portion of a [`BorrowedBuf`](BorrowedBuf).
///
/// Provides access to the initialized and uninitialized parts of the underlying `BorrowedBuf`.
/// Data can be written directly to the cursor by using [`append`](BorrowedCursor::append) or
/// indirectly by getting a slice of part or all of the cursor and writing into the slice. In the
/// indirect case, the caller must call [`advance`](BorrowedCursor::advance) after writing to inform
/// the cursor how many bytes have been written.
///
/// Once data is written to the cursor, it becomes part of the filled portion of the underlying
/// `BorrowedBuf` and can no longer be accessed or re-written by the cursor. I.e., the cursor tracks
/// the unfilled part of the underlying `BorrowedBuf`.
///
/// The lifetime `'a` is a bound on the lifetime of the underlying buffer (which means it is a bound
/// on the data in that buffer by transitivity).
#[derive(Debug)]
pub struct BorrowedCursor<'a> {
    /// The underlying buffer.
    // Safety invariant: we treat the type of buf as covariant in the lifetime of `BorrowedBuf` when
    // we create a `BorrowedCursor`. This is only safe if we never replace `buf` by assigning into
    // it, so don't do that!
    buf: &'a mut BorrowedBuf<'a>,
    /// The length of the filled portion of the underlying buffer at the time of the cursor's
    /// creation.
    start: usize,
}

impl<'a> BorrowedCursor<'a> {
    /// Reborrow this cursor by cloning it with a smaller lifetime.
    ///
    /// Since a cursor maintains unique access to its underlying buffer, the borrowed cursor is
    /// not accessible while the new cursor exists.
    #[inline]
    pub fn reborrow<'this>(&'this mut self) -> BorrowedCursor<'this> {
        BorrowedCursor {
            // SAFETY: we never assign into `BorrowedCursor::buf`, so treating its
            // lifetime covariantly is safe.
            buf: unsafe {
                mem::transmute::<&'this mut BorrowedBuf<'a>, &'this mut BorrowedBuf<'this>>(
                    self.buf,
                )
            },
            start: self.start,
        }
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity() - self.buf.filled
    }

    /// Returns the number of bytes written to this cursor since it was created from a `BorrowedBuf`.
    ///
    /// Note that if this cursor is a reborrowed clone of another, then the count returned is the
    /// count written via either cursor, not the count since the cursor was reborrowed.
    #[inline]
    pub fn written(&self) -> usize {
        self.buf.filled - self.start
    }

    /// Returns a shared reference to the initialized portion of the cursor.
    #[inline]
    pub fn init_ref(&self) -> &[u8] {
        // SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buf.buf[self.buf.filled..self.buf.init]) }
    }

    /// Returns a mutable reference to the initialized portion of the cursor.
    #[inline]
    pub fn init_mut(&mut self) -> &mut [u8] {
        // SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe {
            MaybeUninit::slice_assume_init_mut(&mut self.buf.buf[self.buf.filled..self.buf.init])
        }
    }

    /// Returns a mutable reference to the uninitialized part of the cursor.
    ///
    /// It is safe to uninitialize any of these bytes.
    #[inline]
    pub fn uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buf.buf[self.buf.init..]
    }

    /// Returns a mutable reference to the whole cursor.
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any bytes in the initialized portion of the cursor.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buf.buf[self.buf.filled..]
    }

    /// Advance the cursor by asserting that `n` bytes have been filled.
    ///
    /// After advancing, the `n` bytes are no longer accessible via the cursor and can only be
    /// accessed via the underlying buffer. I.e., the buffer's filled portion grows by `n` elements
    /// and its unfilled portion (and the capacity of this cursor) shrinks by `n` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the cursor have been properly
    /// initialised.
    #[inline]
    pub unsafe fn advance(&mut self, n: usize) -> &mut Self {
        self.buf.filled += n;
        self.buf.init = cmp::max(self.buf.init, self.buf.filled);
        self
    }

    /// Initializes all bytes in the cursor.
    #[inline]
    pub fn ensure_init(&mut self) -> &mut Self {
        for byte in self.uninit_mut() {
            byte.write(0);
        }
        self.buf.init = self.buf.capacity();

        self
    }

    /// Asserts that the first `n` unfilled bytes of the cursor are initialized.
    ///
    /// `BorrowedBuf` assumes that bytes are never de-initialized, so this method does nothing when
    /// called with fewer bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.buf.init = cmp::max(self.buf.init, self.buf.filled + n);
        self
    }

    /// Appends data to the cursor, advancing position within its buffer.
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

impl<'a> Write for BorrowedCursor<'a> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.append(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

use libc::{c_void, iovec};

// TODO non-unix versions too
/// A buffer type used with `Read::read_buf_vectored`. Unlike `IoSliceMut`, there is no guarantee
/// that its memory has been initialised.
///
/// It is semantically a wrapper around an &mut [MaybeUninit<u8>], but is guaranteed to be ABI
/// compatible with the `iovec` type on Unix platforms and WSABUF on Windows.
#[repr(transparent)]
#[derive(Clone)]
pub struct IoSliceMaybeUninit<'a> {
    vec: iovec,
    _p: PhantomData<&'a mut [MaybeUninit<u8>]>,
}

impl Debug for IoSliceMaybeUninit<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoSliceMaybeUninit").field("len", &self.vec.iov_len).finish()
    }
}

/// Create a new `IoSliceMaybeUninit` from a fully initialized slice.
impl<'a> From<&'a mut [u8]> for IoSliceMaybeUninit<'a> {
    #[inline]
    fn from(slice: &'a mut [u8]) -> IoSliceMaybeUninit<'a> {
        IoSliceMaybeUninit {
            vec: iovec { iov_base: slice.as_mut_ptr() as *mut c_void, iov_len: slice.len() },
            _p: PhantomData,
        }
    }
}

/// Create a new `IoSliceMaybeUninit` from an uninitialized buffer.
impl<'a> From<&'a mut [MaybeUninit<u8>]> for IoSliceMaybeUninit<'a> {
    #[inline]
    fn from(buf: &'a mut [MaybeUninit<u8>]) -> IoSliceMaybeUninit<'a> {
        IoSliceMaybeUninit {
            vec: iovec { iov_base: buf.as_mut_ptr() as *mut c_void, iov_len: buf.len() },
            _p: PhantomData,
        }
    }
}

impl<'a> IoSliceMaybeUninit<'a> {
    /// Advance the internal cursor of the slice.
    #[inline]
    pub fn advance(&self, n: usize) -> Self {
        if self.vec.iov_len < n {
            panic!("advancing IoSliceMaybeUninit beyond its length");
        }

        IoSliceMaybeUninit {
            vec: iovec {
                iov_base: unsafe { self.vec.iov_base.add(n) },
                iov_len: self.vec.iov_len - n,
            },
            _p: PhantomData,
        }
    }

    /// View the slice as a slice of `u8`s.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all elements of the slice have been initialized.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        slice::from_raw_parts(self.vec.iov_base as *mut u8, self.vec.iov_len)
    }

    /// View the slice as a mutable slice of `u8`s.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all elements of the slice have been initialized.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        slice::from_raw_parts_mut(self.vec.iov_base as *mut u8, self.vec.iov_len)
    }

    /// View the slice as a mutable slice of `MaybeUninit<u8>`.
    #[inline]
    pub fn as_maybe_init_slice(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: Slice invariants follow from iovec. Lifetime of the returned type ensures
        // unique access.
        unsafe {
            slice::from_raw_parts_mut(self.vec.iov_base as *mut MaybeUninit<u8>, self.vec.iov_len)
        }
    }

    /// Returns the number of elements in the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.vec.iov_len
    }
}

/// A borrowed byte buffer, consisting of multiple underlying buffers, which is incrementally filled
/// and initialized. Primarily designed for vectored IO.
///
/// This type is a sort of "double cursor". It tracks three regions in the buffer: a region at the beginning of the
/// buffer that has been logically filled with data, a region that has been initialized at some point but not yet
/// logically filled, and a region at the end that is fully uninitialized. The filled region is guaranteed to be a
/// subset of the initialized region.
///
/// In summary, the contents of the buffer can be visualized as:
/// ```not_rust
/// [   |   |      |    |   |       |   ] Underlying buffers
/// [             capacity              ]
/// [ filled |         unfilled         ]
/// [    initialized    | uninitialized ]
/// ```
///
/// A `BorrowedSliceBuf` is created around some existing data (or capacity for data) via a unique reference
/// (`&mut`). The `BorrowedSliceBuf` can be configured (e.g., using `clear` or `set_init`), but otherwise
/// is read-only. To write into the buffer, use `unfilled` to create a `BorrowedSliceCursor`. The cursor
/// has write-only access to the unfilled portion of the buffer (you can think of it as a
/// write-only iterator).
///
/// The lifetime `'a` is a bound on the lifetime of the underlying data.
#[derive(Debug)]
pub struct BorrowedSliceBuf<'a> {
    // Tracks the initialized portion. The first part of the tuple is the index of the slice, the
    // second is the index within that slice.
    //
    // Invariant: init.0 == bufs.len() => init.1 == 0
    // Invariant: init.0 < bufs.len() => init.1 < bufs[init.0].len()
    // Note that these invariants are the same as those for filled.
    init: (usize, usize),

    // Tracks the filled portion. The first part of the tuple is the index of the slice, the
    // second is the index within that slice.
    //
    // Invariant: filled.0 == bufs.len() => filled.1 == 0
    // Invariant: filled.0 < bufs.len() => filled.1 < bufs[filled.0].len()
    // Note that these invariants are the same as those for init.
    filled: (usize, usize),
    // Tracks the number of bytes written by the currently or previously active cursor.
    written: usize,

    // The underlying buffers.
    // Safety invariant: we treat the type of bufs as covariant in the lifetime of `IoSliceMaybeUninit`
    // when we create a `BorrowedSliceBuf`. This is only safe if we never replace `bufs` by assigning
    // into it, so don't do that!
    bufs: &'a mut [IoSliceMaybeUninit<'a>],
}

impl<'a> BorrowedSliceBuf<'a> {
    /// Create a new `BorrowedSliceBuf` from a slice of possibly initialized io slices.
    #[inline]
    pub fn new<'b: 'a>(bufs: &'a mut [IoSliceMaybeUninit<'b>]) -> BorrowedSliceBuf<'a> {
        // Re-establish the init and filled invariants.
        let mut init = (0, 0);
        let mut filled = (0, 0);
        while init.0 < bufs.len() && init.1 >= bufs[init.0].len() {
            init.0 += 1;
            filled.0 += 1;
        }

        BorrowedSliceBuf {
            init,
            filled,
            written: 0,
            // SAFETY: we never assign into `BorrowedSliceBuf::bufs`, so treating its
            // lifetime covariantly is safe.
            bufs: unsafe {
                mem::transmute::<&'a mut [IoSliceMaybeUninit<'b>], &'a mut [IoSliceMaybeUninit<'a>]>(
                    bufs,
                )
            },
        }
    }

    /// Returns the length of the filled part of the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.iter_filled_slices().map(|s| s.len()).sum()
    }

    /// Returns the number of completely filled slices in the buffer.
    #[inline]
    pub fn len_filled_slices(&self) -> usize {
        self.filled.0
    }

    /// Returns the number of filled elements in any partially filled slice.
    ///
    /// If there are no partially filled slices, then this method returns `0`.
    #[inline]
    pub fn len_partial_filled_slice(&self) -> usize {
        self.filled.1
    }

    /// Iterate over the filled portion of the buffer.
    #[inline]
    pub fn iter_filled_slices(&self) -> FilledSliceIterator<'_, 'a> {
        FilledSliceIterator { bufs: self, next: 0 }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled<'this>(&'this mut self) -> BorrowedSliceCursor<'this> {
        self.written = 0;
        BorrowedSliceCursor {
            // SAFETY: we never assign into `BorrowedSliceCursor::bufs`, so treating its
            // lifetime covariantly is safe.
            bufs: unsafe {
                mem::transmute::<&'this mut BorrowedSliceBuf<'a>, &'this mut BorrowedSliceBuf<'this>>(
                    self,
                )
            },
        }
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The number of initialized bytes is not changed, and the contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) -> &mut Self {
        self.filled = (0, 0);

        // Re-establish the filled invariant.
        while self.filled.0 < self.bufs.len() && self.filled.1 >= self.bufs[self.filled.0].len() {
            self.filled.0 += 1;
        }

        self
    }

    /// Asserts that a prefix of the underlying buffers are initialized. The initialized prefix is
    /// all of the first `b - 1` buffers and the first `n` bytes of the `b`th buffer. In other words,
    /// `(b, n)` is the coordinates of the first uninitialized byte in the buffers.
    ///
    /// `BorrowedSliceBuf` assumes that bytes are never de-initialized, so this method does nothing when called with fewer
    /// bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all of the `(b, n)` prefix has already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, b: usize, n: usize) -> &mut Self {
        if b == self.init.0 {
            self.init.1 = cmp::max(self.init.1, n);
        } else if b > self.init.0 {
            self.init.0 = b;
            self.init.1 = n;
        }

        // Re-establish the init invariant.
        while self.init.0 < self.bufs.len() && self.init.1 >= self.bufs[self.init.0].len() {
            self.init.0 += 1;
        }

        self
    }
}

/// A writeable view of the unfilled portion of a [`BorrowedSliceBuf`](BorrowedSliceBuf).
///
/// Provides access to the initialized and uninitialized parts of the underlying `BorrowedSliceBuf`.
/// Data can be written directly to the cursor by using [`append`](BorrowedSliceCursor::append) or
/// indirectly by writing into a view of the cursor (obtained by calling `as_mut`, `next_init_mut`,
/// etc.) and then calling `advance`.
///
/// Once data is written to the cursor, it becomes part of the filled portion of the underlying
/// `BorrowedSliceBuf` and can no longer be accessed or re-written by the cursor. I.e., the cursor tracks
/// the unfilled part of the underlying `BorrowedSliceBuf`.
///
/// The lifetime `'a` is a bound on the lifetime of the underlying data.
#[derive(Debug)]
pub struct BorrowedSliceCursor<'a> {
    /// The underlying buffers.
    // Safety invariant: we treat the type of bufs as covariant in the lifetime of `BorrowedSliceBuf`
    // when we create a `BorrowedSliceCursor`. This is only safe if we never replace `bufs` by assigning
    // into it, so don't do that!
    bufs: &'a mut BorrowedSliceBuf<'a>,
}

impl<'a> BorrowedSliceCursor<'a> {
    /// Clone this cursor.
    ///
    /// Since a cursor maintains unique access to its underlying buffer, the cloned cursor is not
    /// accessible while the clone is alive.
    #[inline]
    pub fn reborrow<'this>(&'this mut self) -> BorrowedSliceCursor<'this> {
        BorrowedSliceCursor {
            // SAFETY: we never assign into `BorrowedSliceCursor::bufs`, so treating its
            // lifetime covariantly is safe.
            bufs: unsafe {
                mem::transmute::<&'this mut BorrowedSliceBuf<'a>, &'this mut BorrowedSliceBuf<'this>>(
                    self.bufs,
                )
            },
        }
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        if self.bufs.filled.0 >= self.bufs.bufs.len() {
            return 0;
        }

        let mut result = self.bufs.bufs[self.bufs.filled.0].len() - self.bufs.filled.1;

        for buf in &self.bufs.bufs[(self.bufs.filled.0 + 1)..] {
            result += buf.len();
        }

        result
    }

    /// Returns the number of bytes written to this cursor since it was created from a `BorrowBuf`.
    ///
    /// Note that if this cursor is a reborrow of another, then the count returned is the count written
    /// via either cursor, not the count since the cursor was reborrowed.
    #[inline]
    pub fn written(&self) -> usize {
        self.bufs.written
    }

    /// Returns a mutable reference to the whole cursor.
    ///
    /// Returns a guard type which dereferences to a `&mut [IoSliceMaybeUninit<'a>]`
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any bytes in the initialized portion of the cursor.
    #[inline]
    pub unsafe fn as_mut<'this>(&'this mut self) -> BorrowedSliceGuard<'this, 'a> {
        let prev = if self.bufs.filled.1 == 0 {
            None
        } else {
            let prev = Some(self.bufs.bufs[self.bufs.filled.0].clone());
            self.bufs.bufs[self.bufs.filled.0] =
                self.bufs.bufs[self.bufs.filled.0].advance(self.bufs.filled.1);
            prev
        };

        BorrowedSliceGuard { bufs: &mut self.bufs, prev }
    }

    /// Returns a shared reference to the initialized portion of the first (at least partially)
    /// initialised buffer in the cursor.
    ///
    /// Returns a reference to a slice of a single underlying buffer. That buffer will be the first
    /// unfilled buffer which is at least partially initialized. The returned slice is the part of
    /// that buffer which is initialized. If there is no part of any buffer which is both unfilled
    /// and initialised, then this method returns `None`.
    ///
    /// Does not iterate over buffers in any way. Calling this method multiple times will return
    /// the same slice unless data is either filled or initialized (e.g., by calling `advance`).
    #[inline]
    pub fn next_init_ref(&self) -> Option<&[u8]> {
        //SAFETY: We only slice the initialized part of the buffer, which is always valid
        Some(unsafe {
            if self.bufs.filled.0 == self.bufs.init.0 {
                &self.bufs.bufs.get(self.bufs.filled.0)?.as_slice()
                    [self.bufs.filled.1..self.bufs.init.1]
            } else {
                &self.bufs.bufs.get(self.bufs.filled.0)?.as_slice()[self.bufs.filled.1..]
            }
        })
    }

    /// Returns a mutable reference to the initialized portion of the first (at least partially)
    /// initialised buffer in the cursor.
    ///
    /// Returns a reference to a slice of a single underlying buffer. That buffer will be the first
    /// unfilled buffer which is at least partially initialized. The returned slice is the part of
    /// that buffer which is initialized. If there is no part of any buffer which is both unfilled
    /// and initialised, then this method returns `None`.
    ///
    /// Does not iterate over buffers in any way. Calling this method multiple times will return
    /// the same slice unless data is either filled or initialized (e.g., by calling `advance`).
    #[inline]
    pub fn next_init_mut(&mut self) -> Option<&mut [u8]> {
        //SAFETY: We only slice the initialized part of the buffer, which is always valid
        Some(unsafe {
            if self.bufs.filled.0 == self.bufs.init.0 {
                &mut self.bufs.bufs.get_mut(self.bufs.filled.0)?.as_mut_slice()
                    [self.bufs.filled.1..self.bufs.init.1]
            } else {
                &mut self.bufs.bufs.get_mut(self.bufs.filled.0)?.as_mut_slice()
                    [self.bufs.filled.1..]
            }
        })
    }

    /// Returns a mutable reference to the uninitialized portion of the first (at least partially)
    /// uninitialised buffer in the cursor.
    ///
    /// It is safe to uninitialize any of these bytes.
    ///
    /// Returns `None` if `self` is entirely initialized.
    ///
    /// Does not iterate over buffers in any way. Calling this method multiple times will return
    /// the same slice unless data is either filled or initialized (e.g., by calling `advance`).
    #[inline]
    pub fn next_uninit_mut(&mut self) -> Option<&mut [MaybeUninit<u8>]> {
        if self.bufs.init.0 != self.bufs.filled.0 {
            return Some(&mut []);
        }

        Some(
            &mut self.bufs.bufs.get_mut(self.bufs.init.0)?.as_maybe_init_slice()
                [self.bufs.init.1..],
        )
    }

    /// Returns a mutable reference to the first buffer in the cursor.
    ///
    /// Returns `None` if `self` is empty.
    ///
    /// Does not iterate over buffers in any way. Calling this method multiple times will return
    /// the same slice unless data is filled (e.g., by calling `advance`).
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any bytes in the initialized portion of the cursor.
    #[inline]
    pub unsafe fn next_mut(&mut self) -> Option<&mut [MaybeUninit<u8>]> {
        Some(
            &mut self.bufs.bufs.get_mut(self.bufs.filled.0)?.as_maybe_init_slice()
                [self.bufs.filled.1..],
        )
    }

    /// Initializes all bytes in the cursor.
    #[inline]
    pub fn ensure_init(&mut self) -> &mut Self {
        // Already initialized.
        if self.bufs.init.0 >= self.bufs.bufs.len() {
            return self;
        }

        // `first_uninit` is the index of the first wholly uninitialized buffer.
        let first_uninit = if self.bufs.init.1 > 0 {
            // Initialize any partially initialized buffer.
            for byte in
                &mut self.bufs.bufs[self.bufs.init.0].as_maybe_init_slice()[self.bufs.init.1..]
            {
                byte.write(0);
            }
            self.bufs.init.0 + 1
        } else {
            self.bufs.init.0
        };

        // Initialize any wholly uninitialized buffers.
        for buf in &mut self.bufs.bufs[first_uninit..] {
            for byte in buf.as_maybe_init_slice() {
                byte.write(0);
            }
        }

        // Record that we're fully initialized.
        self.bufs.init.0 = self.bufs.bufs.len();
        self.bufs.init.1 = 0;

        self
    }

    /// Initializes all bytes in the first (at least partially unfilled) buffer in the cursor.
    #[inline]
    pub fn ensure_next_init(&mut self) -> &mut Self {
        // The whole buffer is initialized.
        if self.bufs.init.0 >= self.bufs.bufs.len() {
            return self;
        }

        for byte in &mut self.bufs.bufs[self.bufs.init.0].as_maybe_init_slice()[self.bufs.init.1..]
        {
            byte.write(0);
        }

        self.bufs.init.0 += 1;
        self.bufs.init.1 = 0;

        // Re-establish the init invariant.
        while self.bufs.init.0 < self.bufs.bufs.len()
            && self.bufs.init.1 >= self.bufs.bufs[self.bufs.init.0].len()
        {
            self.bufs.init.0 += 1;
        }

        self
    }

    /// Advance the cursor by asserting that `n` bytes have been filled.
    ///
    /// After advancing, the `n` bytes are no longer accessible via the cursor and can only be
    /// accessed via the underlying `BorrowedSliceBuf`. I.e., the `BorrowedSliceBuf`'s filled portion
    /// grows by `n` elements and its unfilled portion (and the capacity of this cursor) shrinks by
    /// `n` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the cursor have been properly
    /// initialised.
    #[inline]
    pub unsafe fn advance(&mut self, mut n: usize) -> &mut Self {
        self.bufs.written += n;

        while n > 0 && self.bufs.filled.0 < self.bufs.bufs.len() {
            let buf = &mut self.bufs.bufs[self.bufs.filled.0];
            let capacity = buf.len() - self.bufs.filled.1;
            if n < capacity {
                self.bufs.filled.1 += n;
                n = 0;
            } else {
                n -= capacity;

                self.bufs.filled.0 += 1;
                self.bufs.filled.1 = 0;
            }
        }

        assert_eq!(0, n, "advancing borrowed buffers beyond their length");

        // Re-establish the filled invariant.
        while self.bufs.filled.0 < self.bufs.bufs.len()
            && self.bufs.filled.1 >= self.bufs.bufs[self.bufs.filled.0].len()
        {
            self.bufs.filled.0 += 1;
        }

        // SAFETY the filled region has grown by a maximum of `n` and the caller must
        // ensure those `n` bytes are initialized.
        self.update_init_to_filled();

        self
    }

    /// Appends data to the cursor, advancing position within its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `buf.len()`.
    #[inline]
    pub fn append(&mut self, mut buf: &[u8]) {
        self.bufs.written += buf.len();

        while !buf.is_empty() && self.bufs.filled.0 < self.bufs.bufs.len() {
            // SAFETY: we do not de-initialize any of the elements of the slice
            let next = unsafe { self.next_mut().unwrap() };
            if buf.len() < next.len() {
                MaybeUninit::write_slice(&mut next[..buf.len()], buf);
                self.bufs.filled.1 += buf.len();
                buf = &[];
            } else {
                MaybeUninit::write_slice(next, &buf[..next.len()]);
                buf = &buf[next.len()..];
                self.bufs.filled.0 += 1;
                self.bufs.filled.1 = 0;
            }
        }

        assert!(buf.is_empty(), "appending to borrowed buffers beyond their length");

        // Re-establish the filled invariant.
        while self.bufs.filled.0 < self.bufs.bufs.len()
            && self.bufs.filled.1 >= self.bufs.bufs[self.bufs.filled.0].len()
        {
            self.bufs.filled.0 += 1;
        }

        // SAFETY: we have filled (and thus initialized) the filled region in the loop above.
        unsafe {
            self.update_init_to_filled();
        }
    }

    /// Sets the initialized region to the minimum of the currently initialized region
    /// and the filled region.
    ///
    /// The caller must ensure all invariants for `self.bufs.filled` are satisfied (see the field definition)
    ///
    /// # Safety
    ///
    /// The caller must ensure that the filled region is entirely initialized.
    #[inline]
    unsafe fn update_init_to_filled(&mut self) {
        if self.bufs.init.0 == self.bufs.filled.0 {
            self.bufs.init.1 = cmp::max(self.bufs.init.1, self.bufs.filled.1);
        } else if self.bufs.init.0 < self.bufs.filled.0 {
            self.bufs.init.0 = self.bufs.filled.0;
            self.bufs.init.1 = self.bufs.filled.1;
        }
    }
}

impl<'a> Write for BorrowedSliceCursor<'a> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.append(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// An iterator over the filled slices of a `BorrowedSliceBuf`.
///
/// See `BorrowedSliceBuf::iter_filled`.
#[derive(Debug)]
pub struct FilledSliceIterator<'buf, 'data> {
    bufs: &'buf BorrowedSliceBuf<'data>,
    next: usize,
}

impl<'buf, 'data> Iterator for FilledSliceIterator<'buf, 'data> {
    type Item = &'buf [u8];

    fn next(&mut self) -> Option<&'buf [u8]> {
        if self.next > self.bufs.filled.0 || self.next >= self.bufs.bufs.len() {
            return None;
        }

        let mut result = unsafe { self.bufs.bufs[self.next].as_slice() };
        if self.next == self.bufs.filled.0 {
            result = &result[..self.bufs.filled.1];
        }

        self.next += 1;
        Some(result)
    }
}

/// Guard type used by `BorrowedSliceCursor::as_mut`.
///
/// Presents a view of the cursor containing only the filled data (via the `Deref` impls). Also
/// resets the state of the underlying BorrowedSliceBuf to a view of the complete
/// buffer when dropped.
#[derive(Debug)]
pub struct BorrowedSliceGuard<'buf, 'data> {
    bufs: &'buf mut BorrowedSliceBuf<'data>,
    // One of the io slices in self.bufs may have been sliced so as only to contain filled data. In
    // that case we keep a copy of the original so we can restore it.
    prev: Option<IoSliceMaybeUninit<'data>>,
}

impl<'buf, 'data> Drop for BorrowedSliceGuard<'buf, 'data> {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev {
            self.bufs.bufs[self.bufs.filled.0] = prev.clone();
        }
    }
}

impl<'buf, 'data> Deref for BorrowedSliceGuard<'buf, 'data> {
    type Target = [IoSliceMaybeUninit<'data>];

    fn deref(&self) -> &[IoSliceMaybeUninit<'data>] {
        &self.bufs.bufs[self.bufs.filled.0..]
    }
}

impl<'buf, 'data> DerefMut for BorrowedSliceGuard<'buf, 'data> {
    fn deref_mut(&mut self) -> &mut [IoSliceMaybeUninit<'data>] {
        &mut self.bufs.bufs[self.bufs.filled.0..]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_capacity() {
        // Empty buffers
        let mut buf = BorrowedSliceBuf::new(&mut []);
        let cursor = buf.unfilled();
        assert_eq!(0, cursor.capacity());

        let a: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let b: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b];
        let mut buf = BorrowedSliceBuf::new(slices);
        let cursor = buf.unfilled();
        assert_eq!(0, cursor.capacity());

        // Partially empty buffers
        let mut a = [MaybeUninit::uninit()];
        let a: IoSliceMaybeUninit<'_> = (&mut a as &mut [MaybeUninit<u8>]).into();
        let b: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b];
        let mut buf = BorrowedSliceBuf::new(slices);
        let cursor = buf.unfilled();
        assert_eq!(1, cursor.capacity());

        let a: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let mut b = [MaybeUninit::uninit(), MaybeUninit::uninit()];
        let b: IoSliceMaybeUninit<'_> = (&mut b as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b];
        let mut buf = BorrowedSliceBuf::new(slices);
        let cursor = buf.unfilled();
        assert_eq!(2, cursor.capacity());

        // Non-empty.
        let mut a = [MaybeUninit::uninit()];
        let a: IoSliceMaybeUninit<'_> = (&mut a as &mut [MaybeUninit<u8>]).into();
        let mut b = [MaybeUninit::uninit(), MaybeUninit::uninit()];
        let b: IoSliceMaybeUninit<'_> = (&mut b as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b];
        let mut buf = BorrowedSliceBuf::new(slices);
        let cursor = buf.unfilled();
        assert_eq!(3, cursor.capacity());

        // Filled and cleared
        buf.filled = (0, 1);
        let cursor = buf.unfilled();
        assert_eq!(2, cursor.capacity());
        buf.filled = (1, 0);
        let cursor = buf.unfilled();
        assert_eq!(2, cursor.capacity());
        buf.filled = (1, 1);
        let cursor = buf.unfilled();
        assert_eq!(1, cursor.capacity());
        buf.filled = (1, 2);
        let cursor = buf.unfilled();
        assert_eq!(0, cursor.capacity());
        buf.filled = (2, 0);
        let cursor = buf.unfilled();
        assert_eq!(0, cursor.capacity());
        buf.clear();
        let cursor = buf.unfilled();
        assert_eq!(3, cursor.capacity());

        // set_init does not affect capacity
        unsafe {
            buf.set_init(0, 1);
            let cursor = buf.unfilled();
            assert_eq!(3, cursor.capacity());
            buf.set_init(3, 42);
            let cursor = buf.unfilled();
            assert_eq!(3, cursor.capacity());
        }
    }

    fn as_mut_len(slices: &[IoSliceMaybeUninit<'_>]) -> usize {
        slices.iter().map(|s| s.len()).sum()
    }

    fn bufs_len(bufs: &BorrowedSliceBuf<'_>) -> usize {
        bufs.bufs.iter().map(|s| s.len()).sum()
    }

    #[test]
    fn test_as_mut() {
        // Empty buf
        let mut buf = BorrowedSliceBuf::new(&mut []);
        let mut cursor = buf.unfilled();
        let mut_view = unsafe { cursor.as_mut() };
        assert_eq!(0, mut_view.len());

        // Check data and modifications are preserved
        let a: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let mut b = [MaybeUninit::new(0u8), MaybeUninit::new(1)];
        let b: IoSliceMaybeUninit<'_> = (&mut b as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b];
        let mut buf = BorrowedSliceBuf::new(slices);
        let mut cursor = buf.unfilled();
        {
            let mut mut_view = unsafe { cursor.as_mut() };

            assert_eq!(2, as_mut_len(&mut_view));
            assert_eq!(0, unsafe { mut_view[0].as_slice()[0] });
            assert_eq!(1, unsafe { mut_view[0].as_slice()[1] });

            unsafe {
                mut_view[0].as_mut_slice()[0] = 42;
            }
        }
        unsafe {
            assert_eq!(42, slices[1].as_slice()[0]);
        }

        macro_rules! test_filled {
            ($buf: ident, $filled: expr, $mut_len: literal, $buf_len: literal) => {
                $buf.filled = $filled;
                {
                    let mut cursor = $buf.unfilled();
                    let mut_view = unsafe { cursor.as_mut() };
                    assert_eq!($mut_len, as_mut_len(&mut_view));
                }
                assert_eq!($buf_len, $buf.len());
                assert_eq!(6, bufs_len(&$buf));
            };
        }

        // Check that filled data is not recorded, and that the original buffer is not affected.
        let mut a = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let a: IoSliceMaybeUninit<'_> = (&mut a as &mut [MaybeUninit<u8>]).into();
        let mut b = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let b: IoSliceMaybeUninit<'_> = (&mut b as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b];
        let mut buf = BorrowedSliceBuf::new(slices);

        // Nothing filled
        test_filled!(buf, (0, 0), 6, 0);
        // All filled
        test_filled!(buf, (2, 0), 0, 6);
        test_filled!(buf, (1, 3), 0, 6);
        // One buffer filled
        test_filled!(buf, (1, 0), 3, 3);
        test_filled!(buf, (0, 3), 3, 3);
        // Part of buffer filled
        test_filled!(buf, (1, 1), 2, 4);
        test_filled!(buf, (0, 1), 5, 1);

        // With middle buffer empty
        let mut a = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let a: IoSliceMaybeUninit<'_> = (&mut a as &mut [MaybeUninit<u8>]).into();
        let b: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let mut c = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let c: IoSliceMaybeUninit<'_> = (&mut c as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b, c];
        let mut buf = BorrowedSliceBuf::new(slices);

        // Nothing filled
        test_filled!(buf, (0, 0), 6, 0);
        // All filled
        test_filled!(buf, (3, 0), 0, 6);
        test_filled!(buf, (2, 3), 0, 6);
        // One buffer filled
        test_filled!(buf, (0, 3), 3, 3);
        test_filled!(buf, (1, 0), 3, 3);
        test_filled!(buf, (2, 0), 3, 3);
        // Part of buffer filled
        test_filled!(buf, (0, 1), 5, 1);
        test_filled!(buf, (2, 1), 2, 4);

        // With first buffer empty
        let a: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let mut b = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let b: IoSliceMaybeUninit<'_> = (&mut b as &mut [MaybeUninit<u8>]).into();
        let mut c = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let c: IoSliceMaybeUninit<'_> = (&mut c as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b, c];
        let mut buf = BorrowedSliceBuf::new(slices);

        // Nothing filled
        test_filled!(buf, (0, 0), 6, 0);
        test_filled!(buf, (1, 0), 6, 0);
        // All filled
        test_filled!(buf, (3, 0), 0, 6);
        test_filled!(buf, (2, 3), 0, 6);
        // One buffer filled
        test_filled!(buf, (1, 3), 3, 3);
        test_filled!(buf, (2, 0), 3, 3);
        // Part of buffer filled
        test_filled!(buf, (1, 1), 5, 1);
        test_filled!(buf, (2, 1), 2, 4);

        // With last buffer empty
        let mut a = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let a: IoSliceMaybeUninit<'_> = (&mut a as &mut [MaybeUninit<u8>]).into();
        let mut b = [MaybeUninit::new(0u8), MaybeUninit::new(1), MaybeUninit::new(1)];
        let b: IoSliceMaybeUninit<'_> = (&mut b as &mut [MaybeUninit<u8>]).into();
        let c: IoSliceMaybeUninit<'_> = (&mut [] as &mut [MaybeUninit<u8>]).into();
        let slices = &mut [a, b, c];
        let mut buf = BorrowedSliceBuf::new(slices);

        // Nothing filled
        test_filled!(buf, (0, 0), 6, 0);
        // All filled
        test_filled!(buf, (2, 0), 0, 6);
        test_filled!(buf, (3, 0), 0, 6);
        test_filled!(buf, (1, 3), 0, 6);
        // One buffer filled
        test_filled!(buf, (1, 0), 3, 3);
        test_filled!(buf, (0, 3), 3, 3);
        // Part of buffer filled
        test_filled!(buf, (1, 1), 2, 4);
        test_filled!(buf, (0, 1), 5, 1);
    }
}
