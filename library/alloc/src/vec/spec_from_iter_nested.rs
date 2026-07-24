use core::iter::TrustedLen;
use core::{cmp, ptr};

use super::{SpecExtend, Vec};
use crate::raw_vec::RawVec;

/// Another specialization trait for Vec::from_iter
/// necessary to manually prioritize overlapping specializations
/// see [`SpecFromIter`](super::SpecFromIter) for details.
pub(super) trait SpecFromIterNested<T, I> {
    fn from_iter(iter: I) -> Self;
}

impl<T, I> SpecFromIterNested<T, I> for Vec<T>
where
    I: Iterator<Item = T>,
{
    default fn from_iter(mut iterator: I) -> Self {
        // Unroll the first iteration, as the vector is going to be
        // expanded on this iteration in every case when the iterable is not
        // empty, but the loop in extend_desugared() is not going to see the
        // vector being full in the few subsequent loop iterations.
        // So we get better branch prediction.
        let (low, high) = iterator.size_hint();
        let Some(first) = iterator.next() else {
            return Vec::new();
        };
        // `push`'s growth strategy is (currently) to double the capacity if
        // there's no space available, so it can have up to 50% "wasted" space.
        // Thus if the upper-bound on the size_hint also wouldn't waste more
        // than that, just allocate it from the start. (After all, it's silly
        // to allocate 254 for a hint of `(254, Some(255)`.)
        let initial_capacity = {
            // This is written like this to not overflow on any well-behaved iterator,
            // even things like `repeat_n(val, isize::MAX as usize + 10)`
            // where `low * 2` would need checking.
            // A bad (but safe) iterator might have `low > high`, but if so it'll
            // produce a huge `extra` that'll probably fail the following check.
            let hint = if let Some(high) = high
                && let extra = high - low
                && extra < low
            {
                high
            } else {
                low
            };
            cmp::max(RawVec::<T>::MIN_NON_ZERO_CAP, hint)
        };
        let mut vector = Vec::with_capacity(initial_capacity);
        // SAFETY: We requested capacity at least MIN_NON_ZERO_CAP, which
        // is never zero, so there's space for at least one element.
        unsafe {
            ptr::write(vector.as_mut_ptr(), first);
            vector.set_len(1);
        }

        // must delegate to spec_extend() since extend() itself delegates
        // to spec_from for empty Vecs
        <Vec<T> as SpecExtend<T, I>>::spec_extend(&mut vector, iterator);
        vector
    }
}

impl<T, I> SpecFromIterNested<T, I> for Vec<T>
where
    I: TrustedLen<Item = T>,
{
    fn from_iter(iterator: I) -> Self {
        let mut vector = match iterator.size_hint() {
            (_, Some(upper)) => Vec::with_capacity(upper),
            // TrustedLen contract guarantees that `size_hint() == (_, None)` means that there
            // are more than `usize::MAX` elements.
            // Since the previous branch would eagerly panic if the capacity is too large
            // (via `with_capacity`) we do the same here.
            _ => panic!("capacity overflow"),
        };
        // reuse extend specialization for TrustedLen
        vector.spec_extend(iterator);
        vector
    }
}
