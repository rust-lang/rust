//! Definitions for `Ranged` type.

use crate::ops::{CoerceUnsized, DispatchFromDyn, RangeInclusive};

#[cfg_attr(not(bootstrap), lang = "ranged")]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
#[unstable(feature = "ranged_int", issue = "none")]
/// A type that can only represent types in the given range.
/// Layout optimizations will take.
pub struct Ranged<T, const RANGE: RangeInclusive<u128>>(T);

impl<T: CheckInRange, const RANGE: RangeInclusive<u128>> Ranged<T, RANGE> {
    /// Create a new `Ranged` value if the passed argument is within `RANGE`.
    #[unstable(feature = "ranged_int", issue = "none")]
    #[inline(always)]
    pub fn new(i: T) -> Option<Self> {
        i.in_range(RANGE).then(|| Self(i))
    }

    /// Create a new `Ranged` value.
    ///
    /// Safety: the passed argument must be within `RANGE`
    #[unstable(feature = "ranged_int", issue = "none")]
    #[rustc_const_unstable(feature = "ranged_int", issue = "none")]
    #[inline(always)]
    pub const unsafe fn new_unchecked(i: T) -> Self {
        Self(i)
    }

    /// Fetch the wrapped value.
    #[unstable(feature = "ranged_int", issue = "none")]
    #[rustc_const_unstable(feature = "ranged_int", issue = "none")]
    #[inline(always)]
    pub const fn get(self) -> T {
        self.0
    }
}

#[unstable(feature = "ranged_int", issue = "none")]
impl<T, const RANGE: RangeInclusive<u128>> crate::ops::Deref for Ranged<T, RANGE> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T, U, const RANGE: RangeInclusive<u128>> CoerceUnsized<Ranged<U, RANGE>> for Ranged<T, RANGE> where
    T: CoerceUnsized<U>
{
}

#[unstable(feature = "dispatch_from_dyn", issue = "none")]
impl<T, U, const RANGE: RangeInclusive<u128>> DispatchFromDyn<Ranged<U, RANGE>> for Ranged<T, RANGE> where
    T: DispatchFromDyn<U>
{
}

/// A helper trait until we can declare `struct Ranged<T, const RANGE: RangeInclusive<T>>(T);`.
/// This is not meant to ever be stabilized.
#[unstable(feature = "ranged_int", issue = "none")]
pub trait CheckInRange {
    /// Returns `true` if the range contains `self`.
    fn in_range(&self, range: RangeInclusive<u128>) -> bool;
}

macro_rules! check_in_range {
    ($(($ty:ty | $signed:ty)),*) => {
        $(
            #[unstable(feature = "ranged_int", issue = "none")]
            impl CheckInRange for $ty {
                #[inline(always)]
                fn in_range(&self, range: RangeInclusive<u128>) -> bool {
                    range.contains(&(*self as u128))
                }
            }
        )*
        $(
            #[unstable(feature = "ranged_int", issue = "none")]
            impl CheckInRange for $signed {
                #[inline(always)]
                fn in_range(&self, range: RangeInclusive<u128>) -> bool {
                    range.contains(&(*self as $ty as u128))
                }
            }
        )*
    };
}

check_in_range!((u8 | i8), (u16 | i16), (u32 | i32), (u64 | i64), (u128 | i128), (usize | isize));

#[unstable(feature = "ranged_int", issue = "none")]
impl<T: ?Sized> CheckInRange for *const T {
    #[inline(always)]
    fn in_range(&self, range: RangeInclusive<u128>) -> bool {
        range.contains(&((*self as *const ()).addr() as u128))
    }
}

#[unstable(feature = "ranged_int", issue = "none")]
impl<T: ?Sized> CheckInRange for *mut T {
    #[inline(always)]
    fn in_range(&self, range: RangeInclusive<u128>) -> bool {
        range.contains(&((*self as *mut ()).addr() as u128))
    }
}
