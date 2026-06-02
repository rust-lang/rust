#![unstable(feature = "core_io_borrowed_buf", issue = "117693")]

use crate::fmt::{self, Debug, Formatter};
use crate::mem::{self, MaybeUninit};
use crate::ptr;

/// A borrowed buffer of initially uninitialized elements, which is incrementally filled.
///
/// This type makes it safer to work with `MaybeUninit` buffers, such as to read into a buffer
/// without having to initialize it first. It tracks the region of elements that have been filled
/// and whether the unfilled region was initialized.
///
/// In summary, the contents of the buffer can be visualized as:
/// ```not_rust
/// [                capacity                ]
/// [ filled | unfilled (may be initialized) ]
/// ```
///
/// A `BorrowedBuf` is created around some existing elements (or capacity for elements) via a unique
/// reference (`&mut`). The `BorrowedBuf` can be configured (e.g., using `clear` or `set_init`), but
/// cannot be directly written. To write into the buffer, use `unfilled` to create a
/// `BorrowedCursor`. The cursor has write-only access to the unfilled portion of the buffer (you
/// can think of it as a write-only iterator).
///
/// The lifetime `'data` is a bound on the lifetime of the underlying elements.
///
/// The type is most commonly used to manage bytes, but can manage any type of elements.
pub struct BorrowedBuf<'data, T> {
    /// The buffer's underlying elements.
    buf: &'data mut [MaybeUninit<T>],
    /// The number of elements of `self.buf` that are known to be filled.
    filled: usize,
    /// Whether the entire unfilled part of `self.buf` has explicitly been initialized.
    init: bool,
}

impl<T> Debug for BorrowedBuf<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedBuf")
            .field("init", &self.init)
            .field("filled", &self.filled)
            .field("capacity", &self.capacity())
            .finish()
    }
}

/// Creates a new `BorrowedBuf` from a fully initialized slice.
impl<'data, T: Copy> From<&'data mut [T]> for BorrowedBuf<'data, T> {
    #[inline]
    fn from(slice: &'data mut [T]) -> BorrowedBuf<'data, T> {
        BorrowedBuf {
            // SAFETY: no initialized element is ever uninitialized as per `BorrowedBuf`'s invariant
            buf: unsafe { &mut *(slice as *mut [T] as *mut [MaybeUninit<T>]) },
            filled: 0,
            init: true,
        }
    }
}

/// Creates a new `BorrowedBuf` from an uninitialized buffer.
impl<'data, T: Copy> From<&'data mut [MaybeUninit<T>]> for BorrowedBuf<'data, T> {
    #[inline]
    fn from(buf: &'data mut [MaybeUninit<T>]) -> BorrowedBuf<'data, T> {
        BorrowedBuf { buf, filled: 0, init: false }
    }
}

/// Creates a new `BorrowedBuf` from a cursor.
///
/// Use `BorrowedCursor::with_unfilled_buf` instead for a safer alternative.
impl<'data, T: Copy> From<BorrowedCursor<'data, T>> for BorrowedBuf<'data, T> {
    #[inline]
    fn from(buf: BorrowedCursor<'data, T>) -> BorrowedBuf<'data, T> {
        BorrowedBuf {
            // SAFETY: no initialized element is ever uninitialized as per `BorrowedBuf`'s invariant
            buf: unsafe { buf.buf.buf.get_unchecked_mut(buf.buf.filled..) },
            filled: 0,
            init: buf.buf.init,
        }
    }
}

impl<'data, T> BorrowedBuf<'data, T> {
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

    /// Returns `true` if the buffer is initialized.
    #[unstable(feature = "borrowed_buf_init", issue = "78485")]
    #[inline]
    pub fn is_init(&self) -> bool {
        self.init
    }
}

impl<'data, T: Copy> BorrowedBuf<'data, T> {
    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe {
            let buf = self.buf.get_unchecked(..self.filled);
            buf.assume_init_ref()
        }
    }

    /// Returns a mutable reference to the filled portion of the buffer.
    #[inline]
    pub fn filled_mut(&mut self) -> &mut [T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe {
            let buf = self.buf.get_unchecked_mut(..self.filled);
            buf.assume_init_mut()
        }
    }

    /// Returns a shared reference to the filled portion of the buffer with its original lifetime.
    #[inline]
    pub fn into_filled(self) -> &'data [T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe {
            let buf = self.buf.get_unchecked(..self.filled);
            buf.assume_init_ref()
        }
    }

    /// Returns a mutable reference to the filled portion of the buffer with its original lifetime.
    #[inline]
    pub fn into_filled_mut(self) -> &'data mut [T] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe {
            let buf = self.buf.get_unchecked_mut(..self.filled);
            buf.assume_init_mut()
        }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled<'this>(&'this mut self) -> BorrowedCursor<'this, T> {
        BorrowedCursor {
            // SAFETY: we never assign into `BorrowedCursor::buf`, so treating its
            // lifetime covariantly is safe.
            buf: unsafe {
                mem::transmute::<&'this mut BorrowedBuf<'data, T>, &'this mut BorrowedBuf<'this, T>>(
                    self,
                )
            },
        }
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) -> &mut Self {
        self.filled = 0;
        self
    }

    /// Asserts that the unfilled part of the buffer is initialized.
    ///
    /// # Safety
    ///
    /// All the elements of the buffer must be initialized.
    #[unstable(feature = "borrowed_buf_init", issue = "78485")]
    #[inline]
    pub unsafe fn set_init(&mut self) -> &mut Self {
        self.init = true;
        self
    }
}

/// A writeable view of the unfilled portion of a [`BorrowedBuf`].
///
/// The unfilled portion may be uninitialized; see [`BorrowedBuf`] for details.
///
/// Data can be written directly to the cursor by using [`append`](BorrowedCursor::append) or
/// indirectly by getting a slice of part or all of the cursor and writing into the slice. In the
/// indirect case, the caller must call [`advance`](BorrowedCursor::advance) after writing to inform
/// the cursor how many elements have been written.
///
/// Once elements are written to the cursor, they become part of the filled portion of the
/// underlying `BorrowedBuf` and can no longer be accessed or re-written by the cursor. In other
/// words, the cursor tracks the unfilled part of the underlying `BorrowedBuf`.
///
/// The lifetime `'a` is a bound on the lifetime of the underlying buffer (which means it is a bound
/// on the elements in that buffer by transitivity).
pub struct BorrowedCursor<'a, T> {
    /// The underlying buffer.
    // Safety invariant: we treat the type of buf as covariant in the lifetime of `BorrowedBuf` when
    // we create a `BorrowedCursor`. This is only safe if we never replace `buf` by assigning into
    // it, so don't do that!
    buf: &'a mut BorrowedBuf<'a, T>,
}

impl<T> Debug for BorrowedCursor<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BorrowedCursor").field("buf", &self.buf).finish()
    }
}

impl<'a, T: Copy> BorrowedCursor<'a, T> {
    /// Reborrows this cursor by cloning it with a smaller lifetime.
    ///
    /// Since a cursor maintains unique access to its underlying buffer, the borrowed cursor is
    /// not accessible while the new cursor exists.
    #[inline]
    pub fn reborrow<'this>(&'this mut self) -> BorrowedCursor<'this, T> {
        BorrowedCursor {
            // SAFETY: we never assign into `BorrowedCursor::buf`, so treating its
            // lifetime covariantly is safe.
            buf: unsafe {
                mem::transmute::<&'this mut BorrowedBuf<'a, T>, &'this mut BorrowedBuf<'this, T>>(
                    self.buf,
                )
            },
        }
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity() - self.buf.filled
    }

    /// Returns the number of elements written to the `BorrowedBuf` this cursor was created from.
    ///
    /// In particular, the count returned is shared by all reborrows of the cursor.
    #[inline]
    pub fn written(&self) -> usize {
        self.buf.filled
    }

    /// Returns `true` if the buffer is initialized.
    #[unstable(feature = "borrowed_buf_init", issue = "78485")]
    #[inline]
    pub fn is_init(&self) -> bool {
        self.buf.init
    }

    /// Set the buffer as fully initialized.
    ///
    /// # Safety
    ///
    /// All the elements of the cursor must be initialized.
    #[unstable(feature = "borrowed_buf_init", issue = "78485")]
    #[inline]
    pub unsafe fn set_init(&mut self) {
        self.buf.init = true;
    }

    /// Returns a mutable reference to the whole cursor.
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any elements of the cursor if it is initialized.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<T>] {
        // SAFETY: always in bounds
        unsafe { self.buf.buf.get_unchecked_mut(self.buf.filled..) }
    }

    /// Advances the cursor by asserting that `n` elements have been filled.
    ///
    /// After advancing, the `n` elements are no longer accessible via the cursor and can only be
    /// accessed via the underlying buffer. I.e., the buffer's filled portion grows by `n` elements
    /// and its unfilled portion (and the capacity of this cursor) shrinks by `n` elements.
    ///
    /// If less than `n` elements initialized (by the cursor's point of view), `set_init` should be
    /// called first.
    ///
    /// # Panics
    ///
    /// Panics if there are less than `n` elements initialized.
    #[unstable(feature = "borrowed_buf_init", issue = "78485")]
    #[inline]
    pub fn advance_checked(&mut self, n: usize) -> &mut Self {
        // The subtraction cannot underflow by invariant of this type.
        let init_unfilled = if self.buf.init { self.buf.buf.len() - self.buf.filled } else { 0 };
        assert!(n <= init_unfilled);

        self.buf.filled += n;
        self
    }

    /// Advances the cursor by asserting that `n` elements have been filled.
    ///
    /// After advancing, the `n` elements are no longer accessible via the cursor and can only be
    /// accessed via the underlying buffer. I.e., the buffer's filled portion grows by `n` elements
    /// and its unfilled portion (and the capacity of this cursor) shrinks by `n` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` elements of the cursor have been initialized.
    #[inline]
    pub unsafe fn advance(&mut self, n: usize) -> &mut Self {
        self.buf.filled += n;
        self
    }

    /// Append elements to the cursor, advancing position within its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `buf.len()`.
    #[inline]
    pub fn append(&mut self, buf: &[T]) {
        assert!(self.capacity() >= buf.len());

        // SAFETY: we do not de-initialize any of the elements of the slice
        unsafe {
            self.as_mut()[..buf.len()].write_copy_of_slice(buf);
        }

        self.buf.filled += buf.len();
    }

    /// Runs the given closure with a `BorrowedBuf` containing the unfilled part
    /// of the cursor.
    ///
    /// This enables inspecting what was written to the cursor.
    ///
    /// # Panics
    ///
    /// Panics if the `BorrowedBuf` given to the closure is replaced by another
    /// one.
    pub fn with_unfilled_buf<R>(&mut self, f: impl FnOnce(&mut BorrowedBuf<'_, T>) -> R) -> R {
        let mut buf = BorrowedBuf::from(self.reborrow());
        let prev_ptr = buf.buf as *const _;
        let res = f(&mut buf);

        // Check that the caller didn't replace the `BorrowedBuf`.
        // This is necessary for the safety of the code below: if the check wasn't
        // there, one could mark some elements as initialized even though they aren't.
        assert!(core::ptr::eq(prev_ptr, buf.buf));

        let filled = buf.filled;
        let init = buf.init;

        // Update `init` and `filled` fields with what was written to the buffer.
        // `self.buf.filled` was the starting length of the `BorrowedBuf`.
        //
        // SAFETY: These elements were initialized/filled in the `BorrowedBuf`, and therefore they
        // are initialized/filled in the cursor too, because the buffer wasn't replaced.
        self.buf.init = init;
        self.buf.filled += filled;

        res
    }
}

impl<'a> BorrowedCursor<'a, u8> {
    /// Initializes all bytes in the cursor and returns them.
    #[unstable(feature = "borrowed_buf_init", issue = "78485")]
    #[inline]
    pub fn ensure_init(&mut self) -> &mut [u8] {
        // SAFETY: always in bounds and we never uninitialize these bytes.
        let unfilled = unsafe { self.buf.buf.get_unchecked_mut(self.buf.filled..) };

        if !self.buf.init {
            // SAFETY: 0 is a valid value for MaybeUninit<u8> and the length matches the allocation
            // since it is comes from a slice reference.
            unsafe {
                ptr::write_bytes(unfilled.as_mut_ptr(), 0, unfilled.len());
            }
            self.buf.init = true;
        }

        // SAFETY: these bytes have just been initialized if they weren't before
        unsafe { unfilled.assume_init_mut() }
    }
}
