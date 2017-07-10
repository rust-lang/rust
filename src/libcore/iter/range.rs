// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use convert::TryFrom;
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
pub trait Step: Clone + PartialOrd + Sized {
    /// Returns the number of steps between two step objects. The count is
    /// inclusive of `start` and exclusive of `end`.
    ///
    /// Returns `None` if it is not possible to calculate `steps_between`
    /// without overflow.
    fn steps_between(start: &Self, end: &Self) -> Option<usize>;

    /// Replaces this step with `1`, returning itself
    fn replace_one(&mut self) -> Self;

    /// Replaces this step with `0`, returning itself
    fn replace_zero(&mut self) -> Self;

    /// Adds one to this step, returning the result
    fn add_one(&self) -> Self;

    /// Subtracts one to this step, returning the result
    fn sub_one(&self) -> Self;

    /// Add an usize, returning None on overflow
    fn add_usize(&self, n: usize) -> Option<Self>;
}

// These are still macro-generated because the integer literals resolve to different types.
macro_rules! step_identical_methods {
    () => {
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
    }
}

macro_rules! step_impl_unsigned {
    ($($t:ty)*) => ($(
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "42168")]
        impl Step for $t {
            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t) -> Option<usize> {
                if *start < *end {
                    // Note: We assume $t <= usize here
                    Some((*end - *start) as usize)
                } else {
                    Some(0)
                }
            }

            #[inline]
            fn add_usize(&self, n: usize) -> Option<Self> {
                match <$t>::try_from(n) {
                    Ok(n_as_t) => self.checked_add(n_as_t),
                    Err(_) => None,
                }
            }

            step_identical_methods!();
        }
    )*)
}
macro_rules! step_impl_signed {
    ($( [$t:ty : $unsigned:ty] )*) => ($(
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "42168")]
        impl Step for $t {
            #[inline]
            #[allow(trivial_numeric_casts)]
            fn steps_between(start: &$t, end: &$t) -> Option<usize> {
                if *start < *end {
                    // Note: We assume $t <= isize here
                    // Use .wrapping_sub and cast to usize to compute the
                    // difference that may not fit inside the range of isize.
                    Some((*end as isize).wrapping_sub(*start as isize) as usize)
                } else {
                    Some(0)
                }
            }

            #[inline]
            fn add_usize(&self, n: usize) -> Option<Self> {
                match <$unsigned>::try_from(n) {
                    Ok(n_as_unsigned) => {
                        // Wrapping in unsigned space handles cases like
                        // `-120_i8.add_usize(200) == Some(80_i8)`,
                        // even though 200_usize is out of range for i8.
                        let wrapped = (*self as $unsigned).wrapping_add(n_as_unsigned) as $t;
                        if wrapped >= *self {
                            Some(wrapped)
                        } else {
                            None  // Addition overflowed
                        }
                    }
                    Err(_) => None,
                }
            }

            step_identical_methods!();
        }
    )*)
}

macro_rules! step_impl_no_between {
    ($($t:ty)*) => ($(
        #[unstable(feature = "step_trait",
                   reason = "likely to be replaced by finer-grained traits",
                   issue = "42168")]
        impl Step for $t {
            #[inline]
            fn steps_between(_start: &Self, _end: &Self) -> Option<usize> {
                None
            }

            #[inline]
            fn add_usize(&self, n: usize) -> Option<Self> {
                self.checked_add(n as $t)
            }

            step_identical_methods!();
        }
    )*)
}

step_impl_unsigned!(usize u8 u16 u32);
step_impl_signed!([isize: usize] [i8: u8] [i16: u16] [i32: u32]);
#[cfg(target_pointer_width = "64")]
step_impl_unsigned!(u64);
#[cfg(target_pointer_width = "64")]
step_impl_signed!([i64: u64]);
// If the target pointer width is not 64-bits, we
// assume here that it is less than 64-bits.
#[cfg(not(target_pointer_width = "64"))]
step_impl_no_between!(u64 i64);
step_impl_no_between!(u128 i128);

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
impl<A: Step> Iterator for ops::Range<A> {
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
        match Step::steps_between(&self.start, &self.end) {
            Some(hint) => (hint, Some(hint)),
            None => (0, None)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        if let Some(plus_n) = self.start.add_usize(n) {
            if plus_n < self.end {
                self.start = plus_n.add_one();
                return Some(plus_n)
            }
        }

        self.start = self.end.clone();
        None
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
range_trusted_len_impl!(usize isize u8 i8 u16 i16 u32 i32 i64 u64);
range_incl_trusted_len_impl!(usize isize u8 i8 u16 i16 u32 i32 i64 u64);

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> DoubleEndedIterator for ops::Range<A> {
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
impl<A: Step> FusedIterator for ops::Range<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::RangeFrom<A> {
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
    fn nth(&mut self, n: usize) -> Option<A> {
        let plus_n = self.start.add_usize(n).expect("overflow in RangeFrom::nth");
        self.start = plus_n.add_one();
        Some(plus_n)
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<A: Step> FusedIterator for ops::RangeFrom<A> {}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<A: Step> Iterator for ops::RangeInclusive<A> {
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

        match Step::steps_between(&self.start, &self.end) {
            Some(hint) => (hint.saturating_add(1), hint.checked_add(1)),
            None => (0, None),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        if let Some(plus_n) = self.start.add_usize(n) {
            use cmp::Ordering::*;

            match plus_n.partial_cmp(&self.end) {
                Some(Less) => {
                    self.start = plus_n.add_one();
                    return Some(plus_n)
                }
                Some(Equal) => {
                    self.start.replace_one();
                    self.end.replace_zero();
                    return Some(plus_n)
                }
                _ => {}
            }
        }

        self.start.replace_one();
        self.end.replace_zero();
        None
    }
}

#[unstable(feature = "inclusive_range", reason = "recently added, follows RFC", issue = "28237")]
impl<A: Step> DoubleEndedIterator for ops::RangeInclusive<A> {
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
impl<A: Step> FusedIterator for ops::RangeInclusive<A> {}
