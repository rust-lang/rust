use crate::char;
use crate::convert::TryFrom;
use crate::mem;
use crate::ops::{self, Try};

use super::{
    FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce, TrustedStep,
};

// Safety: All invariants are upheld.
macro_rules! unsafe_impl_trusted_step {
    ($($type:ty)*) => {$(
        #[unstable(feature = "trusted_step", issue = "85731")]
        unsafe impl TrustedStep for $type {}
    )*};
}
unsafe_impl_trusted_step![char i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize];

/// Objects that have a notion of *successor* and *predecessor* operations.
///
/// The *successor* operation moves towards values that compare greater.
/// The *predecessor* operation moves towards values that compare lesser.
#[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
pub trait Step: Clone + PartialOrd + Sized {
    /// Returns the number of *successor* steps required to get from `start` to `end`.
    ///
    /// Returns `None` if the number of steps would overflow `usize`
    /// (or is infinite, or if `end` would never be reached).
    ///
    /// # Invariants
    ///
    /// For any `a`, `b`, and `n`:
    ///
    /// * `steps_between(&a, &b) == Some(n)` if and only if `Step::forward_checked(&a, n) == Some(b)`
    /// * `steps_between(&a, &b) == Some(n)` if and only if `Step::backward_checked(&b, n) == Some(a)`
    /// * `steps_between(&a, &b) == Some(n)` only if `a <= b`
    ///   * Corollary: `steps_between(&a, &b) == Some(0)` if and only if `a == b`
    ///   * Note that `a <= b` does _not_ imply `steps_between(&a, &b) != None`;
    ///     this is the case when it would require more than `usize::MAX` steps to get to `b`
    /// * `steps_between(&a, &b) == None` if `a > b`
    fn steps_between(start: &Self, end: &Self) -> Option<usize>;

    /// Returns the value that would be obtained by taking the *successor*
    /// of `self` `count` times.
    ///
    /// If this would overflow the range of values supported by `Self`, returns `None`.
    ///
    /// # Invariants
    ///
    /// For any `a`, `n`, and `m`:
    ///
    /// * `Step::forward_checked(a, n).and_then(|x| Step::forward_checked(x, m)) == Step::forward_checked(a, m).and_then(|x| Step::forward_checked(x, n))`
    ///
    /// For any `a`, `n`, and `m` where `n + m` does not overflow:
    ///
    /// * `Step::forward_checked(a, n).and_then(|x| Step::forward_checked(x, m)) == Step::forward_checked(a, n + m)`
    ///
    /// For any `a` and `n`:
    ///
    /// * `Step::forward_checked(a, n) == (0..n).try_fold(a, |x, _| Step::forward_checked(&x, 1))`
    ///   * Corollary: `Step::forward_checked(&a, 0) == Some(a)`
    fn forward_checked(start: Self, count: usize) -> Option<Self>;

    /// Returns the value that would be obtained by taking the *successor*
    /// of `self` `count` times.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this function is allowed to panic, wrap, or saturate.
    /// The suggested behavior is to panic when debug assertions are enabled,
    /// and to wrap or saturate otherwise.
    ///
    /// Unsafe code should not rely on the correctness of behavior after overflow.
    ///
    /// # Invariants
    ///
    /// For any `a`, `n`, and `m`, where no overflow occurs:
    ///
    /// * `Step::forward(Step::forward(a, n), m) == Step::forward(a, n + m)`
    ///
    /// For any `a` and `n`, where no overflow occurs:
    ///
    /// * `Step::forward_checked(a, n) == Some(Step::forward(a, n))`
    /// * `Step::forward(a, n) == (0..n).fold(a, |x, _| Step::forward(x, 1))`
    ///   * Corollary: `Step::forward(a, 0) == a`
    /// * `Step::forward(a, n) >= a`
    /// * `Step::backward(Step::forward(a, n), n) == a`
    fn forward(start: Self, count: usize) -> Self {
        Step::forward_checked(start, count).expect("overflow in `Step::forward`")
    }

    /// Returns the value that would be obtained by taking the *successor*
    /// of `self` `count` times.
    ///
    /// # Safety
    ///
    /// It is undefined behavior for this operation to overflow the
    /// range of values supported by `Self`. If you cannot guarantee that this
    /// will not overflow, use `forward` or `forward_checked` instead.
    ///
    /// # Invariants
    ///
    /// For any `a`:
    ///
    /// * if there exists `b` such that `b > a`, it is safe to call `Step::forward_unchecked(a, 1)`
    /// * if there exists `b`, `n` such that `steps_between(&a, &b) == Some(n)`,
    ///   it is safe to call `Step::forward_unchecked(a, m)` for any `m <= n`.
    ///
    /// For any `a` and `n`, where no overflow occurs:
    ///
    /// * `Step::forward_unchecked(a, n)` is equivalent to `Step::forward(a, n)`
    unsafe fn forward_unchecked(start: Self, count: usize) -> Self {
        Step::forward(start, count)
    }

    /// Returns the value that would be obtained by taking the *predecessor*
    /// of `self` `count` times.
    ///
    /// If this would overflow the range of values supported by `Self`, returns `None`.
    ///
    /// # Invariants
    ///
    /// For any `a`, `n`, and `m`:
    ///
    /// * `Step::backward_checked(a, n).and_then(|x| Step::backward_checked(x, m)) == n.checked_add(m).and_then(|x| Step::backward_checked(a, x))`
    /// * `Step::backward_checked(a, n).and_then(|x| Step::backward_checked(x, m)) == try { Step::backward_checked(a, n.checked_add(m)?) }`
    ///
    /// For any `a` and `n`:
    ///
    /// * `Step::backward_checked(a, n) == (0..n).try_fold(a, |x, _| Step::backward_checked(&x, 1))`
    ///   * Corollary: `Step::backward_checked(&a, 0) == Some(a)`
    fn backward_checked(start: Self, count: usize) -> Option<Self>;

    /// Returns the value that would be obtained by taking the *predecessor*
    /// of `self` `count` times.
    ///
    /// If this would overflow the range of values supported by `Self`,
    /// this function is allowed to panic, wrap, or saturate.
    /// The suggested behavior is to panic when debug assertions are enabled,
    /// and to wrap or saturate otherwise.
    ///
    /// Unsafe code should not rely on the correctness of behavior after overflow.
    ///
    /// # Invariants
    ///
    /// For any `a`, `n`, and `m`, where no overflow occurs:
    ///
    /// * `Step::backward(Step::backward(a, n), m) == Step::backward(a, n + m)`
    ///
    /// For any `a` and `n`, where no overflow occurs:
    ///
    /// * `Step::backward_checked(a, n) == Some(Step::backward(a, n))`
    /// * `Step::backward(a, n) == (0..n).fold(a, |x, _| Step::backward(x, 1))`
    ///   * Corollary: `Step::backward(a, 0) == a`
    /// * `Step::backward(a, n) <= a`
    /// * `Step::forward(Step::backward(a, n), n) == a`
    fn backward(start: Self, count: usize) -> Self {
        Step::backward_checked(start, count).expect("overflow in `Step::backward`")
    }

    /// Returns the value that would be obtained by taking the *predecessor*
    /// of `self` `count` times.
    ///
    /// # Safety
    ///
    /// It is undefined behavior for this operation to overflow the
    /// range of values supported by `Self`. If you cannot guarantee that this
    /// will not overflow, use `backward` or `backward_checked` instead.
    ///
    /// # Invariants
    ///
    /// For any `a`:
    ///
    /// * if there exists `b` such that `b < a`, it is safe to call `Step::backward_unchecked(a, 1)`
    /// * if there exists `b`, `n` such that `steps_between(&b, &a) == Some(n)`,
    ///   it is safe to call `Step::backward_unchecked(a, m)` for any `m <= n`.
    ///
    /// For any `a` and `n`, where no overflow occurs:
    ///
    /// * `Step::backward_unchecked(a, n)` is equivalent to `Step::backward(a, n)`
    unsafe fn backward_unchecked(start: Self, count: usize) -> Self {
        Step::backward(start, count)
    }
}

// These are still macro-generated because the integer literals resolve to different types.
macro_rules! step_identical_methods {
    () => {
        #[inline]
        unsafe fn forward_unchecked(start: Self, n: usize) -> Self {
            // SAFETY: the caller has to guarantee that `start + n` doesn't overflow.
            unsafe { start.unchecked_add(n as Self) }
        }

        #[inline]
        unsafe fn backward_unchecked(start: Self, n: usize) -> Self {
            // SAFETY: the caller has to guarantee that `start - n` doesn't overflow.
            unsafe { start.unchecked_sub(n as Self) }
        }

        #[inline]
        #[allow(arithmetic_overflow)]
        #[rustc_inherit_overflow_checks]
        fn forward(start: Self, n: usize) -> Self {
            // In debug builds, trigger a panic on overflow.
            // This should optimize completely out in release builds.
            if Self::forward_checked(start, n).is_none() {
                let _ = Self::MAX + 1;
            }
            // Do wrapping math to allow e.g. `Step::forward(-128i8, 255)`.
            start.wrapping_add(n as Self)
        }

        #[inline]
        #[allow(arithmetic_overflow)]
        #[rustc_inherit_overflow_checks]
        fn backward(start: Self, n: usize) -> Self {
            // In debug builds, trigger a panic on overflow.
            // This should optimize completely out in release builds.
            if Self::backward_checked(start, n).is_none() {
                let _ = Self::MIN - 1;
            }
            // Do wrapping math to allow e.g. `Step::backward(127i8, 255)`.
            start.wrapping_sub(n as Self)
        }
    };
}

macro_rules! step_integer_impls {
    {
        narrower than or same width as usize:
            $( [ $u_narrower:ident $i_narrower:ident ] ),+;
        wider than usize:
            $( [ $u_wider:ident $i_wider:ident ] ),+;
    } => {
        $(
            #[allow(unreachable_patterns)]
            #[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
            impl Step for $u_narrower {
                step_identical_methods!();

                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        // This relies on $u_narrower <= usize
                        Some((*end - *start) as usize)
                    } else {
                        None
                    }
                }

                #[inline]
                fn forward_checked(start: Self, n: usize) -> Option<Self> {
                    match Self::try_from(n) {
                        Ok(n) => start.checked_add(n),
                        Err(_) => None, // if n is out of range, `unsigned_start + n` is too
                    }
                }

                #[inline]
                fn backward_checked(start: Self, n: usize) -> Option<Self> {
                    match Self::try_from(n) {
                        Ok(n) => start.checked_sub(n),
                        Err(_) => None, // if n is out of range, `unsigned_start - n` is too
                    }
                }
            }

            #[allow(unreachable_patterns)]
            #[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
            impl Step for $i_narrower {
                step_identical_methods!();

                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        // This relies on $i_narrower <= usize
                        //
                        // Casting to isize extends the width but preserves the sign.
                        // Use wrapping_sub in isize space and cast to usize to compute
                        // the difference that might not fit inside the range of isize.
                        Some((*end as isize).wrapping_sub(*start as isize) as usize)
                    } else {
                        None
                    }
                }

                #[inline]
                fn forward_checked(start: Self, n: usize) -> Option<Self> {
                    match $u_narrower::try_from(n) {
                        Ok(n) => {
                            // Wrapping handles cases like
                            // `Step::forward(-120_i8, 200) == Some(80_i8)`,
                            // even though 200 is out of range for i8.
                            let wrapped = start.wrapping_add(n as Self);
                            if wrapped >= start {
                                Some(wrapped)
                            } else {
                                None // Addition overflowed
                            }
                        }
                        // If n is out of range of e.g. u8,
                        // then it is bigger than the entire range for i8 is wide
                        // so `any_i8 + n` necessarily overflows i8.
                        Err(_) => None,
                    }
                }

                #[inline]
                fn backward_checked(start: Self, n: usize) -> Option<Self> {
                    match $u_narrower::try_from(n) {
                        Ok(n) => {
                            // Wrapping handles cases like
                            // `Step::forward(-120_i8, 200) == Some(80_i8)`,
                            // even though 200 is out of range for i8.
                            let wrapped = start.wrapping_sub(n as Self);
                            if wrapped <= start {
                                Some(wrapped)
                            } else {
                                None // Subtraction overflowed
                            }
                        }
                        // If n is out of range of e.g. u8,
                        // then it is bigger than the entire range for i8 is wide
                        // so `any_i8 - n` necessarily overflows i8.
                        Err(_) => None,
                    }
                }
            }
        )+

        $(
            #[allow(unreachable_patterns)]
            #[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
            impl Step for $u_wider {
                step_identical_methods!();

                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        usize::try_from(*end - *start).ok()
                    } else {
                        None
                    }
                }

                #[inline]
                fn forward_checked(start: Self, n: usize) -> Option<Self> {
                    start.checked_add(n as Self)
                }

                #[inline]
                fn backward_checked(start: Self, n: usize) -> Option<Self> {
                    start.checked_sub(n as Self)
                }
            }

            #[allow(unreachable_patterns)]
            #[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
            impl Step for $i_wider {
                step_identical_methods!();

                #[inline]
                fn steps_between(start: &Self, end: &Self) -> Option<usize> {
                    if *start <= *end {
                        match end.checked_sub(*start) {
                            Some(result) => usize::try_from(result).ok(),
                            // If the difference is too big for e.g. i128,
                            // it's also gonna be too big for usize with fewer bits.
                            None => None,
                        }
                    } else {
                        None
                    }
                }

                #[inline]
                fn forward_checked(start: Self, n: usize) -> Option<Self> {
                    start.checked_add(n as Self)
                }

                #[inline]
                fn backward_checked(start: Self, n: usize) -> Option<Self> {
                    start.checked_sub(n as Self)
                }
            }
        )+
    };
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

#[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
impl Step for char {
    #[inline]
    fn steps_between(&start: &char, &end: &char) -> Option<usize> {
        let start = start as u32;
        let end = end as u32;
        if start <= end {
            let count = end - start;
            if start < 0xD800 && 0xE000 <= end {
                usize::try_from(count - 0x800).ok()
            } else {
                usize::try_from(count).ok()
            }
        } else {
            None
        }
    }

    #[inline]
    fn forward_checked(start: char, count: usize) -> Option<char> {
        let start = start as u32;
        let mut res = Step::forward_checked(start, count)?;
        if start < 0xD800 && 0xD800 <= res {
            res = Step::forward_checked(res, 0x800)?;
        }
        if res <= char::MAX as u32 {
            // SAFETY: res is a valid unicode scalar
            // (below 0x110000 and not in 0xD800..0xE000)
            Some(unsafe { char::from_u32_unchecked(res) })
        } else {
            None
        }
    }

    #[inline]
    fn backward_checked(start: char, count: usize) -> Option<char> {
        let start = start as u32;
        let mut res = Step::backward_checked(start, count)?;
        if start >= 0xE000 && 0xE000 > res {
            res = Step::backward_checked(res, 0x800)?;
        }
        // SAFETY: res is a valid unicode scalar
        // (below 0x110000 and not in 0xD800..0xE000)
        Some(unsafe { char::from_u32_unchecked(res) })
    }

    #[inline]
    unsafe fn forward_unchecked(start: char, count: usize) -> char {
        let start = start as u32;
        // SAFETY: the caller must guarantee that this doesn't overflow
        // the range of values for a char.
        let mut res = unsafe { Step::forward_unchecked(start, count) };
        if start < 0xD800 && 0xD800 <= res {
            // SAFETY: the caller must guarantee that this doesn't overflow
            // the range of values for a char.
            res = unsafe { Step::forward_unchecked(res, 0x800) };
        }
        // SAFETY: because of the previous contract, this is guaranteed
        // by the caller to be a valid char.
        unsafe { char::from_u32_unchecked(res) }
    }

    #[inline]
    unsafe fn backward_unchecked(start: char, count: usize) -> char {
        let start = start as u32;
        // SAFETY: the caller must guarantee that this doesn't overflow
        // the range of values for a char.
        let mut res = unsafe { Step::backward_unchecked(start, count) };
        if start >= 0xE000 && 0xE000 > res {
            // SAFETY: the caller must guarantee that this doesn't overflow
            // the range of values for a char.
            res = unsafe { Step::backward_unchecked(res, 0x800) };
        }
        // SAFETY: because of the previous contract, this is guaranteed
        // by the caller to be a valid char.
        unsafe { char::from_u32_unchecked(res) }
    }
}

macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl ExactSizeIterator for ops::Range<$t> { }
    )*)
}

/// Safety: This macro must only be used on types that are `Copy` and result in ranges
/// which have an exact `size_hint()` where the upper bound must not be `None`.
macro_rules! unsafe_range_trusted_random_access_impl {
    ($($t:ty)*) => ($(
        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccess for ops::Range<$t> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccessNoCoerce for ops::Range<$t> {
            const MAY_HAVE_SIDE_EFFECT: bool = false;
        }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "inclusive_range", since = "1.26.0")]
        impl ExactSizeIterator for ops::RangeInclusive<$t> { }
    )*)
}

/// Specialization implementations for `Range`.
trait RangeIteratorImpl {
    type Item;

    // Iterator
    fn spec_next(&mut self) -> Option<Self::Item>;
    fn spec_nth(&mut self, n: usize) -> Option<Self::Item>;

    // DoubleEndedIterator
    fn spec_next_back(&mut self) -> Option<Self::Item>;
    fn spec_nth_back(&mut self, n: usize) -> Option<Self::Item>;
}

impl<A: Step> RangeIteratorImpl for ops::Range<A> {
    type Item = A;

    #[inline]
    default fn spec_next(&mut self) -> Option<A> {
        if self.start < self.end {
            let n =
                Step::forward_checked(self.start.clone(), 1).expect("`Step` invariants not upheld");
            Some(mem::replace(&mut self.start, n))
        } else {
            None
        }
    }

    #[inline]
    default fn spec_nth(&mut self, n: usize) -> Option<A> {
        if let Some(plus_n) = Step::forward_checked(self.start.clone(), n) {
            if plus_n < self.end {
                self.start =
                    Step::forward_checked(plus_n.clone(), 1).expect("`Step` invariants not upheld");
                return Some(plus_n);
            }
        }

        self.start = self.end.clone();
        None
    }

    #[inline]
    default fn spec_next_back(&mut self) -> Option<A> {
        if self.start < self.end {
            self.end =
                Step::backward_checked(self.end.clone(), 1).expect("`Step` invariants not upheld");
            Some(self.end.clone())
        } else {
            None
        }
    }

    #[inline]
    default fn spec_nth_back(&mut self, n: usize) -> Option<A> {
        if let Some(minus_n) = Step::backward_checked(self.end.clone(), n) {
            if minus_n > self.start {
                self.end =
                    Step::backward_checked(minus_n, 1).expect("`Step` invariants not upheld");
                return Some(self.end.clone());
            }
        }

        self.end = self.start.clone();
        None
    }
}

impl<T: TrustedStep> RangeIteratorImpl for ops::Range<T> {
    #[inline]
    fn spec_next(&mut self) -> Option<T> {
        if self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { Step::forward_unchecked(self.start.clone(), 1) };
            Some(mem::replace(&mut self.start, n))
        } else {
            None
        }
    }

    #[inline]
    fn spec_nth(&mut self, n: usize) -> Option<T> {
        if let Some(plus_n) = Step::forward_checked(self.start.clone(), n) {
            if plus_n < self.end {
                // SAFETY: just checked precondition
                self.start = unsafe { Step::forward_unchecked(plus_n.clone(), 1) };
                return Some(plus_n);
            }
        }

        self.start = self.end.clone();
        None
    }

    #[inline]
    fn spec_next_back(&mut self) -> Option<T> {
        if self.start < self.end {
            // SAFETY: just checked precondition
            self.end = unsafe { Step::backward_unchecked(self.end.clone(), 1) };
            Some(self.end.clone())
        } else {
            None
        }
    }

    #[inline]
    fn spec_nth_back(&mut self, n: usize) -> Option<T> {
        if let Some(minus_n) = Step::backward_checked(self.end.clone(), n) {
            if minus_n > self.start {
                // SAFETY: just checked precondition
                self.end = unsafe { Step::backward_unchecked(minus_n, 1) };
                return Some(self.end.clone());
            }
        }

        self.end = self.start.clone();
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::Range<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.spec_next()
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
        self.spec_nth(n)
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

    #[inline]
    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: The TrustedRandomAccess contract requires that callers only  pass an index
        // that is in bounds.
        // Additionally Self: TrustedRandomAccess is only implemented for Copy types
        // which means even repeated reads of the same index would be safe.
        unsafe { Step::forward_unchecked(self.start.clone(), idx) }
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

unsafe_range_trusted_random_access_impl! {
    usize u8 u16
    isize i8 i16
}

#[cfg(target_pointer_width = "32")]
unsafe_range_trusted_random_access_impl! {
    u32 i32
}

#[cfg(target_pointer_width = "64")]
unsafe_range_trusted_random_access_impl! {
    u32 i32
    u64 i64
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
        self.spec_next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.spec_nth_back(n)
    }
}

// Safety:
// The following invariants for `Step::steps_between` exist:
//
// > * `steps_between(&a, &b) == Some(n)` only if `a <= b`
// >   * Note that `a <= b` does _not_ imply `steps_between(&a, &b) != None`;
// >     this is the case when it would require more than `usize::MAX` steps to
// >     get to `b`
// > * `steps_between(&a, &b) == None` if `a > b`
//
// The first invariant is what is generally required for `TrustedLen` to be
// sound. The note addendum satisfies an additional `TrustedLen` invariant.
//
// > The upper bound must only be `None` if the actual iterator length is larger
// > than `usize::MAX`
//
// The second invariant logically follows the first so long as the `PartialOrd`
// implementation is correct; regardless it is explicitly stated. If `a < b`
// then `(0, Some(0))` is returned by `ops::Range<A: Step>::size_hint`. As such
// the second invariant is upheld.
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for ops::Range<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::Range<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for ops::RangeFrom<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let n = Step::forward(self.start.clone(), 1);
        Some(mem::replace(&mut self.start, n))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        let plus_n = Step::forward(self.start.clone(), n);
        self.start = Step::forward(plus_n.clone(), 1);
        Some(plus_n)
    }
}

// Safety: See above implementation for `ops::Range<A>`
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for ops::RangeFrom<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::RangeFrom<A> {}

trait RangeInclusiveIteratorImpl {
    type Item;

    // Iterator
    fn spec_next(&mut self) -> Option<Self::Item>;
    fn spec_try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>;

    // DoubleEndedIterator
    fn spec_next_back(&mut self) -> Option<Self::Item>;
    fn spec_try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>;
}

impl<A: Step> RangeInclusiveIteratorImpl for ops::RangeInclusive<A> {
    type Item = A;

    #[inline]
    default fn spec_next(&mut self) -> Option<A> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            let n =
                Step::forward_checked(self.start.clone(), 1).expect("`Step` invariants not upheld");
            mem::replace(&mut self.start, n)
        } else {
            self.exhausted = true;
            self.start.clone()
        })
    }

    #[inline]
    default fn spec_try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, A) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            let n =
                Step::forward_checked(self.start.clone(), 1).expect("`Step` invariants not upheld");
            let n = mem::replace(&mut self.start, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }

    #[inline]
    default fn spec_next_back(&mut self) -> Option<A> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            let n =
                Step::backward_checked(self.end.clone(), 1).expect("`Step` invariants not upheld");
            mem::replace(&mut self.end, n)
        } else {
            self.exhausted = true;
            self.end.clone()
        })
    }

    #[inline]
    default fn spec_try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, A) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            let n =
                Step::backward_checked(self.end.clone(), 1).expect("`Step` invariants not upheld");
            let n = mem::replace(&mut self.end, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }
}

impl<T: TrustedStep> RangeInclusiveIteratorImpl for ops::RangeInclusive<T> {
    #[inline]
    fn spec_next(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            // SAFETY: just checked precondition
            let n = unsafe { Step::forward_unchecked(self.start.clone(), 1) };
            mem::replace(&mut self.start, n)
        } else {
            self.exhausted = true;
            self.start.clone()
        })
    }

    #[inline]
    fn spec_try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, T) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { Step::forward_unchecked(self.start.clone(), 1) };
            let n = mem::replace(&mut self.start, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }

    #[inline]
    fn spec_next_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let is_iterating = self.start < self.end;
        Some(if is_iterating {
            // SAFETY: just checked precondition
            let n = unsafe { Step::backward_unchecked(self.end.clone(), 1) };
            mem::replace(&mut self.end, n)
        } else {
            self.exhausted = true;
            self.end.clone()
        })
    }

    #[inline]
    fn spec_try_rfold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, T) -> R,
        R: Try<Output = B>,
    {
        if self.is_empty() {
            return try { init };
        }

        let mut accum = init;

        while self.start < self.end {
            // SAFETY: just checked precondition
            let n = unsafe { Step::backward_unchecked(self.end.clone(), 1) };
            let n = mem::replace(&mut self.end, n);
            accum = f(accum, n)?;
        }

        self.exhausted = true;

        if self.start == self.end {
            accum = f(accum, self.start.clone())?;
        }

        try { accum }
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> Iterator for ops::RangeInclusive<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.spec_next()
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
        if self.is_empty() {
            return None;
        }

        if let Some(plus_n) = Step::forward_checked(self.start.clone(), n) {
            use crate::cmp::Ordering::*;

            match plus_n.partial_cmp(&self.end) {
                Some(Less) => {
                    self.start = Step::forward(plus_n.clone(), 1);
                    return Some(plus_n);
                }
                Some(Equal) => {
                    self.start = plus_n.clone();
                    self.exhausted = true;
                    return Some(plus_n);
                }
                _ => {}
            }
        }

        self.start = self.end.clone();
        self.exhausted = true;
        None
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.spec_try_fold(init, f)
    }

    #[inline]
    fn fold<B, F>(mut self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(f)).unwrap()
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
        self.spec_next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        if self.is_empty() {
            return None;
        }

        if let Some(minus_n) = Step::backward_checked(self.end.clone(), n) {
            use crate::cmp::Ordering::*;

            match minus_n.partial_cmp(&self.start) {
                Some(Greater) => {
                    self.end = Step::backward(minus_n.clone(), 1);
                    return Some(minus_n);
                }
                Some(Equal) => {
                    self.end = minus_n.clone();
                    self.exhausted = true;
                    return Some(minus_n);
                }
                _ => {}
            }
        }

        self.end = self.start.clone();
        self.exhausted = true;
        None
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.spec_try_rfold(init, f)
    }

    #[inline]
    fn rfold<B, F>(mut self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_rfold(init, ok(f)).unwrap()
    }
}

// Safety: See above implementation for `ops::Range<A>`
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for ops::RangeInclusive<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for ops::RangeInclusive<A> {}
