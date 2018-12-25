//! Exposes the NonZero lang item which provides optimization hints.

use ops::{CoerceUnsized, DispatchFromDyn};
use marker::Freeze;

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[rustc_layout_scalar_valid_range_start(1)]
#[derive(Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub(crate) struct NonZero<T: Freeze>(pub(crate) T);

// Do not call `T::clone` as theoretically it could turn the field into `0`
// invalidating `NonZero`'s invariant.
impl<T: Copy + Freeze> Clone for NonZero<T> {
    fn clone(&self) -> Self {
        unsafe { NonZero(self.0) }
    }
}

impl<T: CoerceUnsized<U> + Freeze, U: Freeze> CoerceUnsized<NonZero<U>> for NonZero<T> {}

impl<T: DispatchFromDyn<U> + Freeze, U: Freeze> DispatchFromDyn<NonZero<U>> for NonZero<T> {}
