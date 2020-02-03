use crate::convert::TryFrom;
use crate::mem;
use crate::ops::{self, Add, Sub, Try};
use crate::usize;

use super::{FusedIterator, TrustedLen};

/// Objects that have a notion of *successor* and *predecessor*.
///
/// The *successor* operation moves towards values that compare greater.
/// The *predecessor* operation moves towards values that compare lesser.
///
/// # Safety
///
/// This trait is `unsafe` because its implementation must be correct for
/// the safety of `unsafe trait TrustedLen` implementations, and the results
/// of using this trait can otherwise be trusted by `unsafe` code.
#[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
pub unsafe trait Step: Clone + PartialOrd + Sized {
    /// Returns the number of *successor* steps required to get from `start` to `end`.
    ///
    /// Returns `None` if the number of steps would overflow `usize`
    /// (or is infinite, or if `end` would never be reached).
    ///
    /// # Invariants
    ///
    /// For any `a`, `b`, and `n`:
    ///
    /// * `steps_between(&a, &b) == Some(n)` if and only if `a.forward(n) == Some(b)`
    /// * `steps_between(&a, &b) == Some(n)` if and only if `b.backward(n) == Some(a)`
    /// * `steps_between(&a, &b) == Some(n)` only if `a <= b`
    ///   * Corrolary: `steps_between(&a, &b) == Some(0)` if and only if `a == b`
    ///   * Note that `a <= b` does _not_ imply `steps_between(&a, &b) != None`;
    ///     this is the case when it would require more than `usize::MAX` steps to get to `b`
    /// * `steps_between(&a, &b) == None` if `a > b`
    fn steps_between(start: &Self, end: &Self) -> Option<usize>;

    /// Returns the value that would be obtained by taking the *successor*
    /// of `self` `count` times.
    ///
    /// Return s`None` if this would overflow the range of values supported by `Self`.
    ///
    /// # Invariants
    ///
    /// For any `a`, `n`, and `m` where `n + m` does not overflow:
    ///
    /// * `a.forward(n).and_then(|x| x.forward(m)) == a.forward(n + m)`
    /// * `a.forward(n)` equals `Step::successor` applied to `a` `n` times
    ///   * Corollary: `a.forward(0) == Some(a)`
    fn forward(&self, count: usize) -> Option<Self>;

    /// Returns the *successor* of `self`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this method is allowed to panic or wrap. Suggested behavior is
    /// to panic when debug assertions are enabled, and wrap otherwise.
    ///
    /// # Invariants
    ///
    /// For any `a` where `a.successor()` does not overflow:
    ///
    /// * `a == a.successor().predecessor()`
    /// * `a.successor() == a.forward(1).unwrap()`
    /// * `a.successor() >= a`
    #[inline]
    #[unstable(feature = "step_trait_ext", reason = "recently added", issue = "42168")]
    fn successor(&self) -> Self {
        self.forward(1).expect("overflow in `Step::successor`")
    }

    /// Returns the *successor* of `self`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this method is defined to return the input value instead.
    ///
    /// # Invaraints
    ///
    /// For any `a` where `a.successor()` does not overflow:
    ///
    /// * `a.successor_saturating() == a.successor()`
    ///
    /// For any `a` where `a.successor()` does overflow:
    ///
    /// * `a.successor_saturating() == a`
    #[inline]
    #[unstable(feature = "step_trait_ext", reason = "recently added", issue = "42168")]
    fn successor_saturating(&self) -> Self {
        self.forward(1).unwrap_or_else(|| self.clone())
    }

    /// Returns the *successor* of `self`.
    ///
    /// # Safety
    ///
    /// It is undefined behavior if this operation exceeds the range of
    /// values supported by `Self`. If you cannot guarantee that this
    /// will not overflow, use `forward` or `successor` instead.
    ///
    /// For any `a`, if there exists `b` such that `b > a`,
    /// it is safe to call `a.successor_unchecked()`.
    #[inline]
    #[unstable(feature = "unchecked_math", reason = "niche optimization path", issue = "none")]
    unsafe fn successor_unchecked(&self) -> Self {
        self.successor()
    }

    /// Returns the value that would be obtained by taking the *predecessor*
    /// of `self` `count` times.
    ///
    /// Returns `None` if this would overflow the range of values supported by `Self`.
    ///
    /// # Invariants
    ///
    /// For any `a`, `n`, and `m` where `n + m` does not overflow:
    ///
    /// * `a.backward(n).and_then(|x| x.backward(m)) == a.backward (n + m)`
    /// * `a.backward(n)` equals `Step::predecessor` applied to `a` `n` times
    ///   * Corollary: `a.backward(0) == Some(a)`
    /// * `a.backward(n).unwrap() <= a`
    fn backward(&self, count: usize) -> Option<Self>;

    /// Returns the *predecessor* of `self`.
    ///
    /// If this would underflow the range of values supported by `Self`,
    /// this method is allowed to panic or wrap. Suggested behavior is
    /// to panic when debug assertions are enabled, and wrap otherwise.
    ///
    /// # Invariants
    ///
    /// For any `a` where `a.predecessor()` does not underflow:
    ///
    /// * `a == a.predecessor().successor()`
    /// * `a.predecessor() == a.backward(1).unwrap()`
    /// * `a.predecessor() <= a`
    #[inline]
    #[unstable(feature = "step_trait_ext", reason = "recently added", issue = "42168")]
    fn predecessor(&self) -> Self {
        self.backward(1).expect("overflow in `Step::predecessor`")
    }

    /// Returns the *predecessor* of `self`.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this method is defined to return the input value instead.
    ///
    /// # Invariants
    ///
    /// For any `a` where `a.predecessor()` does not overflow:
    ///
    /// * `a.predecessor_saturating() == a.predecessor()`
    ///
    /// For any `a` where `a.predecessor()` does overflow:
    ///
    /// * `a.predecessor_saturating() == a`
    #[inline]
    #[unstable(feature = "step_trait_ext", reason = "recently added", issue = "42168")]
    fn predecessor_saturating(&self) -> Self {
        self.backward(1).unwrap_or_else(|| self.clone())
    }

    /// Returns the *predecessor* of `self`.
    ///
    /// # Safety
    ///
    /// It is undefined behavior if this operation exceeds the range of
    /// values supported by `Self`. If you cannot guarantee that this
    /// will not overflow, use `backward` or `predecessor` instead.
    ///
    /// For any `a`, if there exists `b` such that `b < a`,
    /// it is safe to call `a.predecessor_unchecked()`.
    #[inline]
    #[unstable(feature = "unchecked_math", reason = "niche optimization path", issue = "none")]
    unsafe fn predecessor_unchecked(&self) -> Self {
        self.predecessor()
    }
}

// These are still macro-generated because the integer literals resolve to different types.
macro_rules! step_identical_methods {
    () => {
        #[inline]
        fn successor(&self) -> Self {
            Add::add(*self, 1)
        }

        #[inline]
        fn successor_saturating(&self) -> Self {
            Self::saturating_add(*self, 1)
        }

        #[inline]
        unsafe fn successor_unchecked(&self) -> Self {
            Self::unchecked_add(*self, 1)
        }

        #[inline]
        fn predecessor(&self) -> Self {
            Sub::sub(*self, 1)
        }

        #[inline]
        fn predecessor_saturating(&self) -> Self {
            Self::saturating_sub(*self, 1)
        }

        #[inline]
        unsafe fn predecessor_unchecked(&self) -> Self {
            Self::unchecked_sub(*self, 1)
        }
    }
}

macro_rules! step_integer_impls {
    (
        narrower than or same width as usize:
            $( [ $narrower_unsigned:ident $narrower_signed:ident ] ),+;
        wider than usize:
            $( [ $wider_unsigned:ident $wider_signed:ident ] ),+;
    ) => {
        $(
            #[unstable(
                feature = "step_trait",
                reason = "recently redesigned",
                issue = "42168"
            )]
            unsafe impl Step for $narrower_unsigned {
                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        // This relies on $narrower_unsigned <= usize
                        Some((*end - *start) as usize)
                    } else {
                        None
                    }
                }

                #[inline]
                #[allow(unreachable_patterns)]
                fn forward(&self, n: usize) -> Option<Self> {
                    match Self::try_from(n) {
                        Ok(n_converted) => self.checked_add(n_converted),
                        Err(_) => None,  // if n is out of range, `something_unsigned + n` is too
                    }
                }

                #[inline]
                #[allow(unreachable_patterns)]
                fn backward(&self, n: usize) -> Option<Self> {
                    match Self::try_from(n) {
                        Ok(n_converted) => self.checked_sub(n_converted),
                        Err(_) => None,  // if n is out of range, `something_in_range - n` is too
                    }
                }

                step_identical_methods!();
            }

            #[unstable(
                feature = "step_trait",
                reason = "recently redesigned",
                issue = "42168"
            )]
            unsafe impl Step for $narrower_signed {
                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        // This relies on $narrower_signed <= usize
                        //
                        // Casting to isize extends the width but preserves the sign.
                        // Use wrapping_sub in isize space and cast to usize
                        // to compute the difference that may not fit inside the range of isize.
                        Some((*end as isize).wrapping_sub(*start as isize) as usize)
                    } else {
                        None
                    }
                }

                #[inline]
                #[allow(unreachable_patterns)]
                fn forward(&self, n: usize) -> Option<Self> {
                    match <$narrower_unsigned>::try_from(n) {
                        Ok(n_unsigned) => {
                            // Wrapping in unsigned space handles cases like
                            // `-120_i8.forward(200) == Some(80_i8)`,
                            // even though 200_usize is out of range for i8.
                            let self_unsigned = *self as $narrower_unsigned;
                            let wrapped = self_unsigned.wrapping_add(n_unsigned) as Self;
                            if wrapped >= *self {
                                Some(wrapped)
                            } else {
                                None  // Addition overflowed
                            }
                        }
                        // If n is out of range of e.g. u8,
                        // then it is bigger than the entire range for i8 is wide
                        // so `any_i8 + n` would overflow i8.
                        Err(_) => None,
                    }
                }

                #[inline]
                #[allow(unreachable_patterns)]
                fn backward(&self, n: usize) -> Option<Self> {
                    match <$narrower_unsigned>::try_from(n) {
                        Ok(n_unsigned) => {
                            // Wrapping in unsigned space handles cases like
                            // `-120_i8.forward(200) == Some(80_i8)`,
                            // even though 200_usize is out of range for i8.
                            let self_unsigned = *self as $narrower_unsigned;
                            let wrapped = self_unsigned.wrapping_sub(n_unsigned) as Self;
                            if wrapped <= *self {
                                Some(wrapped)
                            } else {
                                None  // Subtraction underflowed
                            }
                        }
                        // If n is out of range of e.g. u8,
                        // then it is bigger than the entire range for i8 is wide
                        // so `any_i8 - n` would underflow i8.
                        Err(_) => None,
                    }
                }

                step_identical_methods!();
            }
        )+

        $(
            #[unstable(
                feature = "step_trait",
                reason = "recently redesigned",
                issue = "42168"
            )]
            unsafe impl Step for $wider_unsigned {
                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        usize::try_from(*end - *start).ok()
                    } else {
                        None
                    }
                }

                #[inline]
                fn forward(&self, n: usize) -> Option<Self> {
                    self.checked_add(n as Self)
                }

                #[inline]
                fn backward(&self, n: usize) -> Option<Self> {
                    self.checked_sub(n as Self)
                }

                step_identical_methods!();
            }

            #[unstable(
                feature = "step_trait",
                reason = "recently redesigned",
                issue = "42168"
            )]
            unsafe impl Step for $wider_signed {
                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        match end.checked_sub(*start) {
                            Some(diff) => usize::try_from(diff).ok(),
                            // If the difference is too big for e.g. i128,
                            // it’s also gonna be too big for usize with fewer bits.
                            None => None
                        }
                    } else {
                        None
                    }
                }

                #[inline]
                fn forward(&self, n: usize) -> Option<Self> {
                    self.checked_add(n as Self)
                }

                #[inline]
                fn backward(&self, n: usize) -> Option<Self> {
                    self.checked_sub(n as Self)
                }

                step_identical_methods!();
            }
        )+
    }
}

#[cfg(target_pointer_width = "64")]
step_integer_impls! {
    narrower than or same width as usize: [u8 i8], [u16 i16], [u32 i32], [u64 i64], [usize isize];
    wider than usize: [u128 i128];
}

#[cfg(target_pointer_width = "32")]
step_integer_impls! {
    narrower than or same width as usize: [u8 i8], [u16 i16], [u32 i32], [usize isize];
    wider than usize: [u64 i64], [u128 i128];
}

#[cfg(target_pointer_width = "16")]
step_integer_impls! {
    narrower than or same width as usize: [u8 i8], [u16 i16], [usize isize];
    wider than usize: [u32 i32], [u64 i64], [u128 i128];
}

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

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::Range<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        if self.start < self.end {
            // SAFETY: just checked precondition
            let mut n = unsafe { self.start.successor_unchecked() };
            mem::swap(&mut n, &mut self.start);
            Some(n)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start < self.end {
            let hint = Step::steps_between(&self.start, &self.end);
            (hint.unwrap_or(usize::MAX), hint)
        } else {
            (0, Some(0))
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        if let Some(plus_n) = self.start.forward(n) {
            if plus_n < self.end {
                // SAFETY: just checked precondition
                self.start = unsafe { plus_n.successor_unchecked() };
                return Some(plus_n);
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
//
// * `ExactSizeIterator::len` is required to always return an exact `usize`,
//   so no range can be longer than `usize::MAX`.
// * For integer types in `Range<_>` this is the case for types narrower than or as wide as `usize`.
//   For integer types in `RangeInclusive<_>`
//   this is the case for types *strictly narrower* than `usize`
//   since e.g. `(0..=u64::MAX).len()` would be `u64::MAX + 1`.
range_exact_iter_impl! {
    usize u8 u16
    isize i8 i16

    // These are incorect per the reasoning above,
    // but removing them would be a breaking change as they were stabilized in Rust 1.0.0.
    // So e.g. `(0..66_000_u32).len()` for example will compile without error or warnings
    // on 16-bit platforms, but continue to give a wrong result.
    u32
    i32
}
range_incl_exact_iter_impl! {
    u8
    i8

    // These are incorect per the reasoning above,
    // but removing them would be a breaking change as they were stabilized in Rust 1.26.0.
    // So e.g. `(0..=u16::MAX).len()` for example will compile without error or warnings
    // on 16-bit platforms, but continue to give a wrong result.
    u16
    i16
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> DoubleEndedIterator for ops::Range<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        if self.start < self.end {
            // SAFETY: just checked precondition
            self.end = unsafe { self.end.predecessor_unchecked() };
            Some(self.end.clone())
        } else {
            None
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        if let Some(minus_n) = self.end.backward(n) {
            if minus_n > self.start {
                // SAFETY: just checked precondition
                self.end = unsafe { minus_n.predecessor_unchecked() };
                return Some(self.end.clone());
            }
        }

        self.end = self.start.clone();
        None
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: Step> TrustedLen for ops::Range<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::Range<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::RangeFrom<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        // This case is tricky. Consider `RangeFrom<u8> { start: 255_u8 }`.
        // Ideally, we would return `255`, and then panic on the next call.
        // Unfortunately, this is impossible as we don't have anywhere to
        // store that information. This does debug-checked addition, so in
        // debug mode, we panic instead of yielding 255, and in release mode,
        // we wrap around the number space.
        let mut n = self.start.successor();
        mem::swap(&mut n, &mut self.start);
        Some(n)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        // If we would jump over the maximum value, just panic immediately.
        // This is consistent with behavior before the Step redesign,
        // even though it's inconsistent with n `next` calls.
        let plus_n = self.start.forward(n).expect("overflow in `RangeFrom::nth`");
        // Now we call `successor` to get debug-checked behavior for the final step.
        self.start = plus_n.successor();
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
            // SAFETY: just checked precondition
            let n = unsafe { self.start.successor_unchecked() };
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

        if let Some(plus_n) = self.start.forward(n) {
            use crate::cmp::Ordering::*;

            match plus_n.partial_cmp(&self.end) {
                Some(Less) => {
                    self.is_empty = Some(false);
                    // SAFETY: just checked precondition
                    self.start = unsafe { plus_n.successor_unchecked() };
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
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.compute_is_empty();

        if self.is_empty() {
            return Try::from_ok(init);
        }

        let mut accum = init;

        while self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { self.start.successor_unchecked() };
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
            // SAFETY: just checked precondition
            let n = unsafe { self.end.predecessor_unchecked() };
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

        if let Some(minus_n) = self.end.backward(n) {
            use crate::cmp::Ordering::*;

            match minus_n.partial_cmp(&self.start) {
                Some(Greater) => {
                    self.is_empty = Some(false);
                    // SAFETY: just checked precondition
                    self.end = unsafe { minus_n.predecessor_unchecked() };
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
    fn try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Ok = B>,
    {
        self.compute_is_empty();

        if self.is_empty() {
            return Try::from_ok(init);
        }

        let mut accum = init;

        while self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { self.end.predeccessor_unchecked() };
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

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: Step> TrustedLen for ops::RangeInclusive<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::RangeInclusive<A> {}
