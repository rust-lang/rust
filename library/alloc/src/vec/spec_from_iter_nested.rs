use core::cmp;
use core::iter::TrustedLen;
use core::ptr;

use crate::{alloc::Global, raw_vec::RawVec};

use super::{SpecExtend, Vec, VecError};

/// Another specialization trait for Vec::from_iter
/// necessary to manually prioritize overlapping specializations
/// see [`SpecFromIter`](super::SpecFromIter) for details.
pub(super) trait SpecFromIterNested<T, I, TError: VecError>
where
    Self: Sized,
{
    fn from_iter(iter: I) -> Result<Self, TError>;
}

impl<T, I, TError: VecError> SpecFromIterNested<T, I, TError> for Vec<T>
where
    I: Iterator<Item = T>,
{
    default fn from_iter(mut iterator: I) -> Result<Self, TError> {
        // Unroll the first iteration, as the vector is going to be
        // expanded on this iteration in every case when the iterable is not
        // empty, but the loop in extend_desugared() is not going to see the
        // vector being full in the few subsequent loop iterations.
        // So we get better branch prediction.
        let mut vector = match iterator.next() {
            None => return Ok(Vec::new()),
            Some(element) => {
                let (lower, _) = iterator.size_hint();
                let initial_capacity =
                    cmp::max(RawVec::<T>::MIN_NON_ZERO_CAP, lower.saturating_add(1));
                let mut vector = Self::with_capacity_in_impl(initial_capacity, Global)?;
                unsafe {
                    // SAFETY: We requested capacity at least 1
                    ptr::write(vector.as_mut_ptr(), element);
                    vector.set_len(1);
                }
                vector
            }
        };
        // must delegate to spec_extend() since extend() itself delegates
        // to spec_from for empty Vecs
        <Vec<T> as SpecExtend<T, I>>::spec_extend(&mut vector, iterator)?;
        Ok(vector)
    }
}

impl<T, I, TError: VecError> SpecFromIterNested<T, I, TError> for Vec<T>
where
    I: TrustedLen<Item = T>,
{
    fn from_iter(iterator: I) -> Result<Self, TError> {
        let mut vector = match iterator.size_hint() {
            (_, Some(upper)) => Self::with_capacity_in_impl(upper, Global)?,
            // TrustedLen contract guarantees that `size_hint() == (_, None)` means that there
            // are more than `usize::MAX` elements.
            // Since the previous branch would eagerly panic if the capacity is too large
            // (via `with_capacity`) we do the same here.
            _ => panic!("capacity overflow"),
        };
        // reuse extend specialization for TrustedLen
        <Vec<T> as SpecExtend<T, I>>::spec_extend(&mut vector, iterator)?;
        Ok(vector)
    }
}
