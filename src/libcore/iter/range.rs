// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mem;
use ops::{self, Add, Sub};
use usize;

use super::{FusedIterator, TrustedLen};

/// Objects that can be stepped over in both directions.
///
/// The `steps_between` function provides a way to efficiently compare
/// two `Step` objects.
#[unstable(feature = "step_trait",
           reason = "likely to be replaced by finer-grained traits",
           issue = "42168")]
pub trait Step: PartialOrd + Sized + Clone {
    /// Steps `self` if possible.
    fn step(&self, by: usize) -> Option<Self>;

    /// Returns the number of steps between two step objects. The count is
    /// inclusive of `start` and exclusive of `end`.
    ///
    /// Returns `None` if the resultant number of steps overflows usize.
    fn steps_between(start: &Self, end: &Self, by: usize) -> Option<usize>;

    /// Same as `steps_between`, but with a `by` of 1
    fn steps_between_by_one(start: &Self, end: &Self) -> Option<usize>;

    /// Tests whether this step is negative or not (going backwards)
    fn is_negative(&self) -> bool;

    /// Replaces this step with `1`, returning itself
    fn replace_one(&mut self) -> Self;

    /// Replaces this step with `0`, returning itself
    fn replace_zero(&mut self) -> Self;

    /// Adds one to this step, returning the result
    fn add_one(&self) -> Self;

    /// Subtracts one to this step, returning the result
    fn sub_one(&self) -> Self;
}

macro_rules! step_impl_unsigned {
    ($($t:ty)*) => ($(
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "42168")]
        impl Step for $t {
            #[inline]
            fn step(&self, by: usize) -> Option<$t> {
                // If casting usize to Self fails, this means overflow happened.
                Self::cast(by).ok().and_then(|by| (*self).checked_add(by))
            }

            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t, by: usize) -> Option<usize> {
                if by == 0 { return None; }
                Self::cast(by).ok().and_then(|by| if *start < *end {
                    let diff = *end - *start;
                    usize::cast(if diff % by > 0 {
                        diff / by + 1
                    } else {
                        diff / by
                    }).ok()
                } else {
                    Some(0)
                })
            }

            #[inline]
            fn is_negative(&self) -> bool {
                false
            }

            #[inline]
            fn replace_one(&mut self) -> Self {
                mem::replace(self, 1)
            }

            #[inline]
            fn replace_zero(&mut self) -> Self {
                mem::replace(self, 0)
            }

            #[inline]
            fn add_one(&self) -> Self {
                Add::add(*self, 1)
            }

            #[inline]
            fn sub_one(&self) -> Self {
                Sub::sub(*self, 1)
            }

            #[inline]
            fn steps_between_by_one(start: &Self, end: &Self) -> Option<usize> {
                Self::steps_between(start, end, 1)
            }
        }
    )*)
}
macro_rules! step_impl_signed {
    ($($t:ty: $s:ty,)*) => ($(
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "42168")]
        impl Step for $t {
            #[inline]
            fn step(&self, by: usize) -> Option<$t> {
                Self::cast(by).ok().and_then(|by| (*self).checked_add(by))
            }

            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t, by: usize) -> Option<usize> {
                if by == 0 { return None; }
                <$s>::cast(by).ok().and_then(|by| if *start < *end {
                    let diff = end.wrapping_sub(*start) as $s;
                    usize::cast(if diff % by > 0 {
                        diff / by + 1
                    } else {
                        diff / by
                    }).ok()
                } else {
                    Some(0)
                })
            }

            #[inline]
            fn is_negative(&self) -> bool {
                *self < 0
            }

            #[inline]
            fn replace_one(&mut self) -> Self {
                mem::replace(self, 1)
            }

            #[inline]
            fn replace_zero(&mut self) -> Self {
                mem::replace(self, 0)
            }

            #[inline]
            fn add_one(&self) -> Self {
                Add::add(*self, 1)
            }

            #[inline]
            fn sub_one(&self) -> Self {
                Sub::sub(*self, 1)
            }

            #[inline]
            fn steps_between_by_one(start: &Self, end: &Self) -> Option<usize> {
                Self::steps_between(start, end, 1)
            }
        }
    )*)
}

step_impl_unsigned!(usize u8 u16 u32 u64 u128);
step_impl_signed!(isize: usize, i8: u8, i16: u16, i32: u32, i64: u64, i128: u128,);

macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl ExactSizeIterator for ops::Range<$t> { }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[unstable(feature = "inclusive_range",
                   reason = "recently added, follows RFC",
                   issue = "28237")]
        impl ExactSizeIterator for ops::RangeInclusive<$t> { }
    )*)
}

macro_rules! range_trusted_len_impl {
    ($($t:ty)*) => ($(
        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl TrustedLen for ops::Range<$t> { }
    )*)
}

macro_rules! range_incl_trusted_len_impl {
    ($($t:ty)*) => ($(
        #[unstable(feature = "inclusive_range",
                   reason = "recently added, follows RFC",
                   issue = "28237")]
        unsafe impl TrustedLen for ops::RangeInclusive<$t> { }
    )*)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::Range<A> where
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.start < self.end {
            let mut n = self.start.add_one();
            mem::swap(&mut n, &mut self.start);
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match Step::steps_between_by_one(&self.start, &self.end) {
            Some(hint) => (hint, Some(hint)),
            None => (0, None)
        }
    }

    #[inline]
    fn count(self) -> usize {
        if let Some(x) = Step::steps_between_by_one(&self.start, &self.end) {
            x
        } else {
            panic!("accumulator overflowed while counting the elements")
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        Step::step(&self.start, n).and_then(|next| if next < self.end {
            self.start = next.add_one();
            Some(next)
        } else {
            None
        })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        <Self as SpecLast>::last(self)
    }
}

trait SpecLast {
    type Item;
    fn last(self) -> Option<Self::Item>;
}

impl<It> SpecLast for It
where It: Iterator {
    type Item = <Self as Iterator>::Item;
    #[inline]
    default fn last(self) -> Option<Self::Item> {
        <Self as Iterator>::last(self)
    }
}

impl<It> SpecLast for It
where It: DoubleEndedIterator {
    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        <Self as DoubleEndedIterator>::next_back(&mut self)
    }
}

// These macros generate `ExactSizeIterator` impls for various range types.
// Range<{u,i}64> and RangeInclusive<{u,i}{32,64,size}> are excluded
// because they cannot guarantee having a length <= usize::MAX, which is
// required by ExactSizeIterator.
range_exact_iter_impl!(usize u8 u16 u32 isize i8 i16 i32);
range_incl_exact_iter_impl!(u8 u16 i8 i16);

// These macros generate `TrustedLen` impls.
//
// They need to guarantee that .size_hint() is either exact, or that
// the upper bound is None when it does not fit the type limits.
range_trusted_len_impl!(usize isize u8 i8 u16 i16 u32 i32 i64 u64 i128 u128);
range_incl_trusted_len_impl!(usize isize u8 i8 u16 i16 u32 i32 i64 u64 i128 u128);

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> DoubleEndedIterator for ops::Range<A> where
    for<'a> &'a A: Add<&'a A, Output = A>,
    for<'a> &'a A: Sub<&'a A, Output = A>
{
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.start < self.end {
            self.end = self.end.sub_one();
            Some(self.end.clone())
        } else {
            None
        }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<A> FusedIterator for ops::Range<A>
    where A: Step, for<'a> &'a A: Add<&'a A, Output = A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::RangeFrom<A> where
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let mut n = self.start.add_one();
        mem::swap(&mut n, &mut self.start);
        Some(n)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    #[inline]
    fn count(self) -> usize {
        usize::MAX
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let next = Step::step(&self.start, n).unwrap();
        self.start = next.add_one();
        Some(next)
    }

    #[cold]
    fn last(self) -> Option<Self::Item> {
        panic!("Iterator::last on a `x..` will overflow")
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<A> FusedIterator for ops::RangeFrom<A>
    where A: Step, for<'a> &'a A: Add<&'a A, Output = A> {}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<A: Step> Iterator for ops::RangeInclusive<A> where
    for<'a> &'a A: Add<&'a A, Output = A>
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        use cmp::Ordering::*;

        match self.start.partial_cmp(&self.end) {
            Some(Less) => {
                let n = self.start.add_one();
                Some(mem::replace(&mut self.start, n))
            },
            Some(Equal) => {
                let last = self.start.replace_one();
                self.end.replace_zero();
                Some(last)
            },
            _ => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if !(self.start <= self.end) {
            return (0, Some(0));
        }

        match Step::steps_between_by_one(&self.start, &self.end) {
            Some(hint) => (hint.saturating_add(1), hint.checked_add(1)),
            None => (0, None),
        }
    }

    #[inline]
    fn count(self) -> usize {
        if self.start > self.end {
            0
        } else {
            if let Some(x) = Step::steps_between_by_one(&self.start, &self.end) {
                x.add_one()
            } else {
                panic!("accumulator overflowed while counting the elements")
            }
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        use cmp::Ordering::*;
        Step::step(&self.start, n).and_then(|next| match next.partial_cmp(&self.end) {
            Some(Less) => {
                self.start = next.add_one();
                Some(next)
            },
            Some(Equal) => {
                // Avoid overflow in the case `self.end` is a `max_value()`
                self.end.replace_zero();
                self.start.replace_one();
                Some(next)
            },
            _ => None
        })
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.start <= self.end {
            Some(self.end)
        } else {
            None
        }
    }
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<A: Step> DoubleEndedIterator for ops::RangeInclusive<A> where
    for<'a> &'a A: Add<&'a A, Output = A>,
    for<'a> &'a A: Sub<&'a A, Output = A>
{
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        use cmp::Ordering::*;

        match self.start.partial_cmp(&self.end) {
            Some(Less) => {
                let n = self.end.sub_one();
                Some(mem::replace(&mut self.end, n))
            },
            Some(Equal) => {
                let last = self.end.replace_zero();
                self.start.replace_one();
                Some(last)
            },
            _ => None,
        }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<A> FusedIterator for ops::RangeInclusive<A>
    where A: Step, for<'a> &'a A: Add<&'a A, Output = A> {}
