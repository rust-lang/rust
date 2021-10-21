#![unstable(feature = "read_buf", issue = "78485")]

#[cfg(test)]
mod tests;

use crate::cmp;
use crate::fmt::{self, Debug, Formatter};
use crate::mem::MaybeUninit;

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
pub struct ReadBuf<'a> {
    buf: &'a mut [MaybeUninit<u8>],
    filled: usize,
    initialized: usize,
}

impl Debug for ReadBuf<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadBuf")
            .field("init", &self.initialized())
            .field("filled", &self.filled)
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl<'a> ReadBuf<'a> {
    /// Creates a new `ReadBuf` from a fully initialized buffer.
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> ReadBuf<'a> {
        let len = buf.len();

        ReadBuf {
            //SAFETY: initialized data never becoming uninitialized is an invariant of ReadBuf
            buf: unsafe { (buf as *mut [u8]).as_uninit_slice_mut().unwrap() },
            filled: 0,
            initialized: len,
        }
    }

    /// Creates a new `ReadBuf` from a fully uninitialized buffer.
    ///
    /// Use `assume_init` if part of the buffer is known to be already inintialized.
    #[inline]
    pub fn uninit(buf: &'a mut [MaybeUninit<u8>]) -> ReadBuf<'a> {
        ReadBuf { buf, filled: 0, initialized: 0 }
    }

    /// Returns the total capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        //SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buf[0..self.filled]) }
    }

    /// Returns a mutable reference to the filled portion of the buffer.
    #[inline]
    pub fn filled_mut(&mut self) -> &mut [u8] {
        //SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf[0..self.filled]) }
    }

    /// Returns a shared reference to the initialized portion of the buffer.
    ///
    /// This includes the filled portion.
    #[inline]
    pub fn initialized(&self) -> &[u8] {
        //SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buf[0..self.initialized]) }
    }

    /// Returns a mutable reference to the initialized portion of the buffer.
    ///
    /// This includes the filled portion.
    #[inline]
    pub fn initialized_mut(&mut self) -> &mut [u8] {
        //SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf[0..self.initialized]) }
    }

    /// Returns a mutable reference to the unfilled part of the buffer without ensuring that it has been fully
    /// initialized.
    ///
    /// # Safety
    ///
    /// The caller must not de-initialize portions of the buffer that have already been initialized.
    #[inline]
    pub unsafe fn unfilled_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buf[self.filled..]
    }

    /// Returns a mutable reference to the uninitialized part of the buffer.
    ///
    /// It is safe to uninitialize any of these bytes.
    #[inline]
    pub fn uninitialized_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.buf[self.initialized..]
    }

    /// Returns a mutable reference to the unfilled part of the buffer, ensuring it is fully initialized.
    ///
    /// Since `ReadBuf` tracks the region of the buffer that has been initialized, this is effectively "free" after
    /// the first use.
    #[inline]
    pub fn initialize_unfilled(&mut self) -> &mut [u8] {
        // should optimize out the assertion
        self.initialize_unfilled_to(self.remaining())
    }

    /// Returns a mutable reference to the first `n` bytes of the unfilled part of the buffer, ensuring it is
    /// fully initialized.
    ///
    /// # Panics
    ///
    /// Panics if `self.remaining()` is less than `n`.
    #[inline]
    pub fn initialize_unfilled_to(&mut self, n: usize) -> &mut [u8] {
        assert!(self.remaining() >= n);

        let extra_init = self.initialized - self.filled;
        // If we dont have enough initialized, do zeroing
        if n > extra_init {
            let uninit = n - extra_init;
            let unfilled = &mut self.uninitialized_mut()[0..uninit];

            for byte in unfilled.iter_mut() {
                byte.write(0);
            }

            // SAFETY: we just inintialized uninit bytes, and the previous bytes were already init
            unsafe {
                self.assume_init(n);
            }
        }

        let filled = self.filled;

        &mut self.initialized_mut()[filled..filled + n]
    }

    /// Returns the number of bytes at the end of the slice that have not yet been filled.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity() - self.filled
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The number of initialized bytes is not changed, and the contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) {
        self.set_filled(0); // The assertion in `set_filled` is optimized out
    }

    /// Increases the size of the filled region of the buffer.
    ///
    /// The number of initialized bytes is not changed.
    ///
    /// # Panics
    ///
    /// Panics if the filled region of the buffer would become larger than the initialized region.
    #[inline]
    pub fn add_filled(&mut self, n: usize) {
        self.set_filled(self.filled + n);
    }

    /// Sets the size of the filled region of the buffer.
    ///
    /// The number of initialized bytes is not changed.
    ///
    /// Note that this can be used to *shrink* the filled region of the buffer in addition to growing it (for
    /// example, by a `Read` implementation that compresses data in-place).
    ///
    /// # Panics
    ///
    /// Panics if the filled region of the buffer would become larger than the initialized region.
    #[inline]
    pub fn set_filled(&mut self, n: usize) {
        assert!(n <= self.initialized);

        self.filled = n;
    }

    /// Asserts that the first `n` unfilled bytes of the buffer are initialized.
    ///
    /// `ReadBuf` assumes that bytes are never de-initialized, so this method does nothing when called with fewer
    /// bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` unfilled bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn assume_init(&mut self, n: usize) {
        self.initialized = cmp::max(self.initialized, self.filled + n);
    }

    /// Appends data to the buffer, advancing the written position and possibly also the initialized position.
    ///
    /// # Panics
    ///
    /// Panics if `self.remaining()` is less than `buf.len()`.
    #[inline]
    pub fn append(&mut self, buf: &[u8]) {
        assert!(self.remaining() >= buf.len());

        // SAFETY: we do not de-initialize any of the elements of the slice
        unsafe {
            MaybeUninit::write_slice(&mut self.unfilled_mut()[..buf.len()], buf);
        }

        // SAFETY: We just added the entire contents of buf to the filled section.
        unsafe { self.assume_init(buf.len()) }
        self.add_filled(buf.len());
    }

    /// Returns the amount of bytes that have been filled.
    #[inline]
    pub fn filled_len(&self) -> usize {
        self.filled
    }

    /// Returns the amount of bytes that have been initialized.
    #[inline]
    pub fn initialized_len(&self) -> usize {
        self.initialized
    }
}
