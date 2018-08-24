//! Exposes the NonZero lang item which provides optimization hints.

use ops::CoerceUnsized;

/// A wrapper type for raw pointers and integers that will never be
/// NULL or 0 that might allow certain optimizations.
#[lang = "non_zero"]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub(crate) struct NonZero<T>(pub(crate) T);

impl<T: CoerceUnsized<U>, U> CoerceUnsized<NonZero<U>> for NonZero<T> {}
