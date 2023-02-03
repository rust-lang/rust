use crate::iter::TrustedLen;
use crate::mem::ManuallyDrop;
use crate::ptr::drop_in_place;
use crate::slice;

// INVARIANT: It's ok to drop the remainder of the inner iterator.
pub(crate) struct Drain<'a, T>(slice::IterMut<'a, T>);

pub(crate) fn drain_array_with<T, R, const N: usize>(
    array: [T; N],
    func: impl for<'a> FnOnce(Drain<'a, T>) -> R,
) -> R {
    let mut array = ManuallyDrop::new(array);
    // SAFETY: Now that the local won't drop it, it's ok to construct the `Drain` which will.
    let drain = Drain(array.iter_mut());
    func(drain)
}

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
