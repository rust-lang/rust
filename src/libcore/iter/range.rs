use crate::convert::TryFrom;
use crate::mem;
use crate::ops::{self, Add, Sub, Try};
use crate::usize;

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

    /// Replaces this step with `1`, returning itself.
    fn replace_one(&mut self) -> Self;

    /// Replaces this step with `0`, returning itself.
    fn replace_zero(&mut self) -> Self;

    /// Adds one to this step, returning the result.
    fn add_one(&self) -> Self;

    /// Subtracts one to this step, returning the result.
    fn sub_one(&self) -> Self;

    /// Adds a `usize`, returning `None` on overflow.
    fn add_usize(&self, n: usize) -> Option<Self>;

    /// Subtracts a `usize`, returning `None` on underflow.
    fn sub_usize(&self, n: usize) -> Option<Self> {
        // this default implementation makes the addition of `sub_usize` a non-breaking change
        let _ = n;
        unimplemented!()
    }
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
            fn steps_between(start: &$t, end: &$t) -> Option<usize> {
                if *start < *end {
                    usize::try_from(*end - *start).ok()
                } else {
                    Some(0)
                }
            }

            #[inline]
            #[allow(unreachable_patterns)]
            fn add_usize(&self, n: usize) -> Option<Self> {
                match <$t>::try_from(n) {
                    Ok(n_as_t) => self.checked_add(n_as_t),
                    Err(_) => None,
                }
            }

            #[inline]
            #[allow(unreachable_patterns)]
            fn sub_usize(&self, n: usize) -> Option<Self> {
                match <$t>::try_from(n) {
                    Ok(n_as_t) => self.checked_sub(n_as_t),
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
            fn steps_between(start: &$t, end: &$t) -> Option<usize> {
                if *start < *end {
                    // Use .wrapping_sub and cast to unsigned to compute the
                    // difference that may not fit inside the range of $t.
                    usize::try_from(end.wrapping_sub(*start) as $unsigned).ok()
                } else {
                    Some(0)
                }
            }

            #[inline]
            #[allow(unreachable_patterns)]
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

            #[inline]
            #[allow(unreachable_patterns)]
            fn sub_usize(&self, n: usize) -> Option<Self> {
                match <$unsigned>::try_from(n) {
                    Ok(n_as_unsigned) => {
                        // Wrapping in unsigned space handles cases like
                        // `80_i8.sub_usize(200) == Some(-120_i8)`,
                        // even though 200_usize is out of range for i8.
                        let wrapped = (*self as $unsigned).wrapping_sub(n_as_unsigned) as $t;
                        if wrapped <= *self {
                            Some(wrapped)
                        } else {
                            None  // Subtraction underflowed
                        }
                    }
                    Err(_) => None,
                }
            }

            step_identical_methods!();
        }
    )*)
}

step_impl_unsigned!(usize u8 u16 u32 u64 u128);
step_impl_signed!([isize: usize] [i8: u8] [i16: u16]);
step_impl_signed!([i32: u32] [i64: u64] [i128: u128]);

macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl ExactSizeIterator for ops::Range<$t> { }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "inclusive_range", since = "1.26.0")]
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
        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl TrustedLen for ops::RangeInclusive<$t> { }
    )*)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::Range<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.start < self.end {
            // We check for overflow here, even though it can't actually
            // happen. Adding this check does however help llvm vectorize loops
            // for some ranges that don't get vectorized otherwise,
            // and this won't actually result in an extra check in an optimized build.
            if let Some(mut n) = self.start.add_usize(1) {
                mem::swap(&mut n, &mut self.start);
                Some(n)
            } else {
                None
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match Step::steps_between(&self.start, &self.end) {
            Some(hint) => (hint, Some(hint)),
            None => (usize::MAX, None)
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

    #[inline]
    fn last(mut self) -> Option<A> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<A> {
        self.next()
    }

    #[inline]
    fn max(mut self) -> Option<A> {
        self.next_back()
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
range_trusted_len_impl!(usize isize u8 i8 u16 i16 u32 i32 u64 i64 u128 i128);
range_incl_trusted_len_impl!(usize isize u8 i8 u16 i16 u32 i32 u64 i64 u128 i128);

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

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        if let Some(minus_n) = self.end.sub_usize(n) {
            if minus_n > self.start {
                self.end = minus_n.sub_one();
                return Some(self.end.clone())
            }
        }

        self.end = self.start.clone();
        None
    }
}

#[stable(feature = "fused", since = "1.26.0")]
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

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::RangeFrom<A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: Step> TrustedLen for ops::RangeFrom<A> {}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> Iterator for ops::RangeInclusive<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.compute_is_empty();
        if self.is_empty.unwrap_or_default() {
            return None;
        }
        let is_iterating = self.start < self.end;
        self.is_empty = Some(!is_iterating);
        Some(if is_iterating {
            let n = self.start.add_one();
            mem::replace(&mut self.start, n)
        } else {
            self.start.clone()
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_empty() {
            return (0, Some(0));
        }

        match Step::steps_between(&self.start, &self.end) {
            Some(hint) => (hint.saturating_add(1), hint.checked_add(1)),
            None => (usize::MAX, None),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.compute_is_empty();
        if self.is_empty.unwrap_or_default() {
            return None;
        }

        if let Some(plus_n) = self.start.add_usize(n) {
            use crate::cmp::Ordering::*;

            match plus_n.partial_cmp(&self.end) {
                Some(Less) => {
                    self.is_empty = Some(false);
                    self.start = plus_n.add_one();
                    return Some(plus_n);
                }
                Some(Equal) => {
                    self.is_empty = Some(true);
                    return Some(plus_n);
                }
                _ => {}
            }
        }

        self.is_empty = Some(true);
        None
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
    {
        self.compute_is_empty();

        if self.is_empty() {
            return Try::from_ok(init);
        }

        let mut accum = init;

        while self.start < self.end {
            let n = self.start.add_one();
            let n = mem::replace(&mut self.start, n);
            accum = f(accum, n)?;
        }

        self.is_empty = Some(true);

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        Try::from_ok(accum)
    }

    #[inline]
    fn last(mut self) -> Option<A> {
        self.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<A> {
        self.next()
    }

    #[inline]
    fn max(mut self) -> Option<A> {
        self.next_back()
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> DoubleEndedIterator for ops::RangeInclusive<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.compute_is_empty();
        if self.is_empty.unwrap_or_default() {
            return None;
        }
        let is_iterating = self.start < self.end;
        self.is_empty = Some(!is_iterating);
        Some(if is_iterating {
            let n = self.end.sub_one();
            mem::replace(&mut self.end, n)
        } else {
            self.end.clone()
        })
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.compute_is_empty();
        if self.is_empty.unwrap_or_default() {
            return None;
        }

        if let Some(minus_n) = self.end.sub_usize(n) {
            use crate::cmp::Ordering::*;

            match minus_n.partial_cmp(&self.start) {
                Some(Greater) => {
                    self.is_empty = Some(false);
                    self.end = minus_n.sub_one();
                    return Some(minus_n);
                }
                Some(Equal) => {
                    self.is_empty = Some(true);
                    return Some(minus_n);
                }
                _ => {}
            }
        }

        self.is_empty = Some(true);
        None
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R where
        Self: Sized, F: FnMut(B, Self::Item) -> R, R: Try<Ok=B>
    {
        self.compute_is_empty();

        if self.is_empty() {
            return Try::from_ok(init);
        }

        let mut accum = init;

        while self.start < self.end {
            let n = self.end.sub_one();
            let n = mem::replace(&mut self.end, n);
            accum = f(accum, n)?;
        }

        self.is_empty = Some(true);

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        Try::from_ok(accum)
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::RangeInclusive<A> {}
