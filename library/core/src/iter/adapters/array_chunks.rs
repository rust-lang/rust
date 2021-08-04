use core::iter::FusedIterator;
use core::mem::{self, MaybeUninit};
use core::ptr;

/// An iterator that yields the elements of another iterator in
/// chunks of size `N`.
///
/// This `struct` is created by the [`array_chunks`] method on [`Iterator`]. See
/// its documentation for more.
///
/// [`array_chunks`]: Iterator::array_chunks
#[unstable(feature = "iter_array_chunks", issue = "none")]
#[derive(Debug)]
pub struct ArrayChunks<I: Iterator, const N: usize> {
    iter: I,
    buffer: [MaybeUninit<I::Item>; N],
    init: usize,
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: Iterator, const N: usize> Drop for ArrayChunks<I, N> {
    fn drop(&mut self) {
        // SAFETY: This is safe: `remainder_mut` returns exactly the sub-slice
        // of elements that were initialized but not yielded and so have yet
        // to be dropped.
        unsafe {
            ptr::drop_in_place(self.remainder_mut());
        }
    }
}

impl<I: Iterator, const N: usize> ArrayChunks<I, N> {
    pub(in crate::iter) fn new(iter: I) -> Self {
        Self { iter, init: 0, buffer: MaybeUninit::uninit_array() }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder(&self) -> &[I::Item] {
        // SAFETY: We know that all elements before `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buffer[..self.init]) }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder_mut(&mut self) -> &mut [I::Item] {
        // SAFETY: We know that all elements before `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buffer[..self.init]) }
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: Iterator, const N: usize> Iterator for ArrayChunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        while self.init < N {
            self.buffer[self.init] = MaybeUninit::new(self.iter.next()?);
            self.init += 1;
        }
        self.init = 0;
        // SAFETY: This is safe: `MaybeUninit<T>` is guaranteed to have the same layout
        // as `T` and the entire array has just been initialized.
        unsafe { Some(mem::transmute_copy(&self.buffer)) }
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: FusedIterator, const N: usize> FusedIterator for ArrayChunks<I, N> {}

/// An iterator that yields the elements of another iterator in
/// chunks of size `N` starting from the end.
///
/// This `struct` is created by the [`array_rchunks`] method on [`Iterator`]. See
/// its documentation for more.
///
/// [`array_rchunks`]: Iterator::array_rchunks
#[unstable(feature = "iter_array_chunks", issue = "none")]
#[derive(Debug)]
pub struct ArrayRChunks<I: DoubleEndedIterator, const N: usize> {
    iter: I,
    buffer: [MaybeUninit<I::Item>; N],
    init: usize,
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: DoubleEndedIterator, const N: usize> Drop for ArrayRChunks<I, N> {
    fn drop(&mut self) {
        // SAFETY: This is safe: `remainder_mut` returns exactly the sub-slice
        // of elements that were initialized but not yielded and so have yet
        // to be dropped.
        unsafe {
            ptr::drop_in_place(self.remainder_mut());
        }
    }
}

impl<I: DoubleEndedIterator, const N: usize> ArrayRChunks<I, N> {
    pub(in crate::iter) fn new(iter: I) -> Self {
        Self { iter, init: 0, buffer: MaybeUninit::uninit_array() }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder(&self) -> &[I::Item] {
        // SAFETY: We know that all elements after `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buffer[(N - self.init)..]) }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder_mut(&mut self) -> &mut [I::Item] {
        // SAFETY: We know that all elements after `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buffer[(N - self.init)..]) }
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: DoubleEndedIterator, const N: usize> Iterator for ArrayRChunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        while self.init < N {
            self.buffer[N - self.init - 1] = MaybeUninit::new(self.iter.next_back()?);
            self.init += 1;
        }
        self.init = 0;
        // SAFETY: This is safe: `MaybeUninit<T>` is guaranteed to have the same layout
        // as `T` and the entire array has just been initialized.
        unsafe { Some(mem::transmute_copy(&self.buffer)) }
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I, const N: usize> FusedIterator for ArrayRChunks<I, N> where
    I: DoubleEndedIterator + FusedIterator
{
}
