use crate::mem::{forget, replace, MaybeUninit};
use crate::ptr;

/// The internal-use drop guard for implementing array methods.
///
/// This is free to be changed whenever.  Its purpose is not to provide a
/// beautiful safe interface, but to make the unsafe details of `super`'s
/// other methods slightly more obvious and have reduced code duplication.
pub struct Guard<'a, T, const N: usize> {
    array_mut: &'a mut [MaybeUninit<T>; N],
    initialized: usize,
}

impl<'a, T, const N: usize> Guard<'a, T, N> {
    #[inline]
    pub fn new(buffer: &'a mut [MaybeUninit<T>; N]) -> Self {
        Self { array_mut: buffer, initialized: 0 }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.initialized
    }

    /// Initialize the next item
    ///
    /// # Safety
    ///
    /// Requires `self.len() < N`.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, value: T) {
        debug_assert!(self.len() < N);
        // SAFETY: The precondition means we have space
        unsafe {
            self.array_mut.get_unchecked_mut(self.initialized).write(value);
        }
        self.initialized += 1;
    }

    /// Initialize the next `CHUNK` item(s)
    ///
    /// # Safety
    ///
    /// Requires `self.len() + CHUNK <= N`.
    #[inline]
    pub unsafe fn push_chunk_unchecked<const CHUNK: usize>(&mut self, chunk: [T; CHUNK]) {
        assert!(CHUNK <= N);
        debug_assert!(N - self.len() >= CHUNK);
        // SAFETY: The precondition means we have space
        unsafe {
            // Since we're going to write multiple items, make sure not to do so
            // via a `&mut MaybeUninit<T>`, as that would violate stacked borrows.
            let first = self.array_mut.as_mut_ptr();
            let p = first.add(self.initialized).cast();
            ptr::write(p, chunk);
        }
        self.initialized += CHUNK;
    }

    /// Read the whole buffer as an initialized array.
    ///
    /// This always de-initializes the original buffer -- even if `T: Copy`.
    ///
    /// # Safety
    ///
    /// Requires `self.len() == N`.
    #[inline]
    pub unsafe fn into_array_unchecked(self) -> [T; N] {
        debug_assert_eq!(self.len(), N);

        // This tells LLVM and MIRI that we don't care about the buffer after,
        // and the extra `undef` write is trivial for it to optimize away.
        let buffer = replace(self.array_mut, MaybeUninit::uninit_array());

        // SAFETY: the condition above asserts that all elements are
        // initialized.
        let out = unsafe { MaybeUninit::array_assume_init(buffer) };

        forget(self);

        out
    }
}

impl<T, const N: usize> Drop for Guard<'_, T, N> {
    fn drop(&mut self) {
        debug_assert!(self.initialized <= N);

        // SAFETY: this slice will contain only initialized objects.
        unsafe {
            crate::ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(
                &mut self.array_mut.get_unchecked_mut(..self.initialized),
            ));
        }
    }
}
