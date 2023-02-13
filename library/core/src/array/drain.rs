use crate::iter::{TrustedLen, UncheckedIterator};
use crate::mem::ManuallyDrop;
use crate::ptr::drop_in_place;
use crate::slice;

/// A situationally-optimized version of `array.into_iter().for_each(func)`.
///
/// [`crate::array::IntoIter`]s are great when you need an owned iterator, but
/// storing the entire array *inside* the iterator like that can sometimes
/// pessimize code.  Notable, it can be more bytes than you really want to move
/// around, and because the array accesses index into it SRoA has a harder time
/// optimizing away the type than it does iterators that just hold a couple pointers.
///
/// Thus this function exists, which gives a way to get *moved* access to the
/// elements of an array using a small iterator -- no bigger than a slice iterator.
///
/// The function-taking-a-closure structure makes it safe, as it keeps callers
/// from looking at already-dropped elements.
pub(crate) fn drain_array_with<T, R, const N: usize>(
    array: [T; N],
    func: impl for<'a> FnOnce(Drain<'a, T>) -> R,
) -> R {
    let mut array = ManuallyDrop::new(array);
    // SAFETY: Now that the local won't drop it, it's ok to construct the `Drain` which will.
    let drain = Drain(array.iter_mut());
    func(drain)
}

/// See [`drain_array_with`] -- this is `pub(crate)` only so it's allowed to be
/// mentioned in the signature of that method.  (Otherwise it hits `E0446`.)
// INVARIANT: It's ok to drop the remainder of the inner iterator.
pub(crate) struct Drain<'a, T>(slice::IterMut<'a, T>);

impl<T> Drop for Drain<'_, T> {
    fn drop(&mut self) {
        // SAFETY: By the type invariant, we're allowed to drop all these.
        unsafe { drop_in_place(self.0.as_mut_slice()) }
    }
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        let p: *const T = self.0.next()?;
        // SAFETY: The iterator was already advanced, so we won't drop this later.
        Some(unsafe { p.read() })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
}

impl<T> ExactSizeIterator for Drain<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

// SAFETY: This is a 1:1 wrapper for a slice iterator, which is also `TrustedLen`.
unsafe impl<T> TrustedLen for Drain<'_, T> {}

impl<T> UncheckedIterator for Drain<'_, T> {
    unsafe fn next_unchecked(&mut self) -> T {
        // SAFETY: `Drain` is 1:1 with the inner iterator, so if the caller promised
        // that there's an element left, the inner iterator has one too.
        let p: *const T = unsafe { self.0.next_unchecked() };
        // SAFETY: The iterator was already advanced, so we won't drop this later.
        unsafe { p.read() }
    }
}
