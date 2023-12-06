use crate::ascii::Char as AsciiChar;
use crate::cmp::Ordering;
use crate::convert::TryFrom;
use crate::net::{Ipv4Addr, Ipv6Addr};
use crate::num::NonZeroUsize;
use crate::ops::{self, Try};

use super::{
    FromIterator, FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce,
    TrustedStep,
};

mod legacy;

// Safety: All invariants are upheld.
macro_rules! unsafe_impl_trusted_step {
    ($($type:ty)*) => {$(
        #[unstable(feature = "trusted_step", issue = "85731")]
        unsafe impl TrustedStep for $type {}
    )*};
}
unsafe_impl_trusted_step![AsciiChar char i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize Ipv4Addr Ipv6Addr];

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

#[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
impl Step for AsciiChar {
    #[inline]
    fn steps_between(&start: &AsciiChar, &end: &AsciiChar) -> Option<usize> {
        Step::steps_between(&start.to_u8(), &end.to_u8())
    }

    #[inline]
    fn forward_checked(start: AsciiChar, count: usize) -> Option<AsciiChar> {
        let end = Step::forward_checked(start.to_u8(), count)?;
        AsciiChar::from_u8(end)
    }

    #[inline]
    fn backward_checked(start: AsciiChar, count: usize) -> Option<AsciiChar> {
        let end = Step::backward_checked(start.to_u8(), count)?;

        // SAFETY: Values below that of a valid ASCII character are also valid ASCII
        Some(unsafe { AsciiChar::from_u8_unchecked(end) })
    }

    #[inline]
    unsafe fn forward_unchecked(start: AsciiChar, count: usize) -> AsciiChar {
        // SAFETY: Caller asserts that result is a valid ASCII character,
        // and therefore it is a valid u8.
        let end = unsafe { Step::forward_unchecked(start.to_u8(), count) };

        // SAFETY: Caller asserts that result is a valid ASCII character.
        unsafe { AsciiChar::from_u8_unchecked(end) }
    }

    #[inline]
    unsafe fn backward_unchecked(start: AsciiChar, count: usize) -> AsciiChar {
        // SAFETY: Caller asserts that result is a valid ASCII character,
        // and therefore it is a valid u8.
        let end = unsafe { Step::backward_unchecked(start.to_u8(), count) };

        // SAFETY: Caller asserts that result is a valid ASCII character.
        unsafe { AsciiChar::from_u8_unchecked(end) }
    }
}

#[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
impl Step for Ipv4Addr {
    #[inline]
    fn steps_between(&start: &Ipv4Addr, &end: &Ipv4Addr) -> Option<usize> {
        u32::steps_between(&start.to_bits(), &end.to_bits())
    }

    #[inline]
    fn forward_checked(start: Ipv4Addr, count: usize) -> Option<Ipv4Addr> {
        u32::forward_checked(start.to_bits(), count).map(Ipv4Addr::from_bits)
    }

    #[inline]
    fn backward_checked(start: Ipv4Addr, count: usize) -> Option<Ipv4Addr> {
        u32::backward_checked(start.to_bits(), count).map(Ipv4Addr::from_bits)
    }

    #[inline]
    unsafe fn forward_unchecked(start: Ipv4Addr, count: usize) -> Ipv4Addr {
        // SAFETY: Since u32 and Ipv4Addr are losslessly convertible,
        //   this is as safe as the u32 version.
        Ipv4Addr::from_bits(unsafe { u32::forward_unchecked(start.to_bits(), count) })
    }

    #[inline]
    unsafe fn backward_unchecked(start: Ipv4Addr, count: usize) -> Ipv4Addr {
        // SAFETY: Since u32 and Ipv4Addr are losslessly convertible,
        //   this is as safe as the u32 version.
        Ipv4Addr::from_bits(unsafe { u32::backward_unchecked(start.to_bits(), count) })
    }
}

#[unstable(feature = "step_trait", reason = "recently redesigned", issue = "42168")]
impl Step for Ipv6Addr {
    #[inline]
    fn steps_between(&start: &Ipv6Addr, &end: &Ipv6Addr) -> Option<usize> {
        u128::steps_between(&start.to_bits(), &end.to_bits())
    }

    #[inline]
    fn forward_checked(start: Ipv6Addr, count: usize) -> Option<Ipv6Addr> {
        u128::forward_checked(start.to_bits(), count).map(Ipv6Addr::from_bits)
    }

    #[inline]
    fn backward_checked(start: Ipv6Addr, count: usize) -> Option<Ipv6Addr> {
        u128::backward_checked(start.to_bits(), count).map(Ipv6Addr::from_bits)
    }

    #[inline]
    unsafe fn forward_unchecked(start: Ipv6Addr, count: usize) -> Ipv6Addr {
        // SAFETY: Since u128 and Ipv6Addr are losslessly convertible,
        //   this is as safe as the u128 version.
        Ipv6Addr::from_bits(unsafe { u128::forward_unchecked(start.to_bits(), count) })
    }

    #[inline]
    unsafe fn backward_unchecked(start: Ipv6Addr, count: usize) -> Ipv6Addr {
        // SAFETY: Since u128 and Ipv6Addr are losslessly convertible,
        //   this is as safe as the u128 version.
        Ipv6Addr::from_bits(unsafe { u128::backward_unchecked(start.to_bits(), count) })
    }
}

macro_rules! range_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl ExactSizeIterator for RangeIter<$t> { }
    )*)
}

/// Safety: This macro must only be used on types that are `Copy` and result in ranges
/// which have an exact `size_hint()` where the upper bound must not be `None`.
macro_rules! unsafe_range_trusted_random_access_impl {
    ($($t:ty)*) => ($(
        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccess for RangeIter<$t> {}

        #[doc(hidden)]
        #[unstable(feature = "trusted_random_access", issue = "none")]
        unsafe impl TrustedRandomAccessNoCoerce for RangeIter<$t> {
            const MAY_HAVE_SIDE_EFFECT: bool = false;
        }
    )*)
}

macro_rules! range_incl_exact_iter_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "inclusive_range", since = "1.26.0")]
        impl ExactSizeIterator for RangeInclusiveIter<$t> { }
    )*)
}

/// Iterator type for [`ops::range::Range`]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct RangeIter<A> {
    pub(crate) inner: ops::range::legacy::Range<A>,
}

impl<A> RangeIter<A> {
    #[doc(hidden)]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn inner(&self) -> &ops::range::legacy::Range<A> {
        &self.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for RangeIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.inner.nth(n)
    }

    #[inline]
    fn last(mut self) -> Option<A> {
        self.inner.next_back()
    }

    #[inline]
    fn min(mut self) -> Option<A>
    where
        A: Ord,
    {
        self.inner.next()
    }

    #[inline]
    fn max(mut self) -> Option<A>
    where
        A: Ord,
    {
        self.inner.next_back()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        self.inner.is_sorted()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.inner.advance_by(n)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item
    where
        Self: TrustedRandomAccessNoCoerce,
    {
        // SAFETY: The TrustedRandomAccess contract requires that callers only pass an index
        // that is in bounds.
        // Additionally Self: TrustedRandomAccess is only implemented for Copy types
        // which means even repeated reads of the same index would be safe.
        unsafe { Step::forward_unchecked(self.inner.start.clone(), idx) }
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

    // These are incorrect per the reasoning above,
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

    // These are incorrect per the reasoning above,
    // but removing them would be a breaking change as they were stabilized in Rust 1.26.0.
    // So e.g. `(0..=u16::MAX).len()` for example will compile without error or warnings
    // on 16-bit platforms, but continue to give a wrong result.
    u16
    i16
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> DoubleEndedIterator for RangeIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.inner.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.inner.nth_back(n)
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.inner.advance_back_by(n)
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
// then `(0, Some(0))` is returned by `ops::range::Range<A: Step>::size_hint`. As such
// the second invariant is upheld.
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeIter<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for RangeIter<A> {}

#[stable(feature = "new_range", since = "1.0.0")]
impl<Idx: Step> IntoIterator for ops::range::Range<Idx> {
    type Item = Idx;
    type IntoIter = RangeIter<Idx>;

    fn into_iter(self) -> Self::IntoIter {
        RangeIter { inner: ops::range::legacy::Range { start: self.start, end: self.end } }
    }
}

#[stable(feature = "new_range", since = "1.0.0")]
impl<Idx: Step + Copy> IntoIterator for &ops::range::Range<Idx> {
    type Item = Idx;
    type IntoIter = RangeIter<Idx>;

    fn into_iter(self) -> Self::IntoIter {
        (*self).into_iter()
    }
}

#[stable(feature = "new_range", since = "1.0.0")]
impl<Idx: Step + Copy> IntoIterator for &mut ops::range::Range<Idx> {
    type Item = Idx;
    type IntoIter = RangeIter<Idx>;

    fn into_iter(self) -> Self::IntoIter {
        (*self).into_iter()
    }
}

impl<Idx: Step> ops::range::Range<Idx> {
    /// Returns and advances `start` unless the range is empty.
    ///
    /// This differs from `.into_iter().next()` because
    /// that copies the range before advancing the iterator
    /// but this modifies the range in place.
    #[stable(feature = "new_range", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "can cause subtle bugs")]
    pub fn next(&mut self) -> Option<Idx> {
        let mut iter = self.clone().into_iter();
        let out = iter.next();

        self.start = iter.inner.start;
        self.end = iter.inner.end;

        out
    }
}

/// Iterator type for [`ops::range::RangeFrom`]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct RangeFromIter<A> {
    inner: ops::range::legacy::RangeFrom<A>,
}

impl<A> RangeFromIter<A> {
    #[doc(hidden)]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn inner(&self) -> &ops::range::legacy::RangeFrom<A> {
        &self.inner
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Step> Iterator for RangeFromIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.inner.nth(n)
    }
}

// Safety: See above implementation for `ops::range::Range<A>`
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeFromIter<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for RangeFromIter<A> {}

#[stable(feature = "new_range", since = "1.0.0")]
impl<A: Step> IntoIterator for ops::range::RangeFrom<A> {
    type Item = A;
    type IntoIter = RangeFromIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeFromIter { inner: ops::range::legacy::RangeFrom { start: self.start } }
    }
}

#[stable(feature = "new_range", since = "1.0.0")]
impl<A: Step + Copy> IntoIterator for &ops::range::RangeFrom<A> {
    type Item = A;
    type IntoIter = RangeFromIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        (*self).into_iter()
    }
}

#[stable(feature = "new_range", since = "1.0.0")]
impl<A: Step + Copy> IntoIterator for &mut ops::range::RangeFrom<A> {
    type Item = A;
    type IntoIter = RangeFromIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        (*self).into_iter()
    }
}

impl<Idx: Step> ops::range::RangeFrom<Idx> {
    /// Returns and advances `start` unless the range is empty.
    ///
    /// This differs from `.into_iter().next()` because
    /// that copies the range before advancing the iterator
    /// but this modifies the range in place.
    #[stable(feature = "new_range", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "can cause subtle bugs")]
    pub fn next(&mut self) -> Option<Idx> {
        let mut iter = self.clone().into_iter();
        let out = iter.next();

        self.start = iter.inner.start;

        out
    }
}

/// Iterator type for [`ops::range::RangeInclusive`]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct RangeInclusiveIter<A> {
    pub(crate) inner: ops::range::legacy::RangeInclusive<A>,
}

impl<A> RangeInclusiveIter<A> {
    #[doc(hidden)]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn inner(&self) -> &ops::range::legacy::RangeInclusive<A> {
        &self.inner
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> Iterator for RangeInclusiveIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.inner.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<A> {
        self.inner.nth(n)
    }

    #[inline]
    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.inner.try_fold(init, f)
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, f)
    }

    #[inline]
    fn last(self) -> Option<A> {
        self.inner.last()
    }

    #[inline]
    fn min(self) -> Option<A>
    where
        A: Ord,
    {
        self.inner.min()
    }

    #[inline]
    fn max(self) -> Option<A>
    where
        A: Ord,
    {
        self.inner.max()
    }

    #[inline]
    fn is_sorted(self) -> bool {
        self.inner.is_sorted()
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
impl<A: Step> DoubleEndedIterator for RangeInclusiveIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.inner.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<A> {
        self.inner.nth_back(n)
    }

    #[inline]
    fn try_rfold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.inner.try_rfold(init, f)
    }

    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.rfold(init, f)
    }
}

// Safety: See above implementation for `ops::range::Range<A>`
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A: TrustedStep> TrustedLen for RangeInclusiveIter<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A: Step> FusedIterator for RangeInclusiveIter<A> {}

#[stable(feature = "new_range", since = "1.0.0")]
impl<A: Step> IntoIterator for ops::range::RangeInclusive<A> {
    type Item = A;
    type IntoIter = RangeInclusiveIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        RangeInclusiveIter {
            inner: ops::range::legacy::RangeInclusive {
                start: self.start,
                end: self.end,
                exhausted: false,
            },
        }
    }
}

#[stable(feature = "new_range", since = "1.0.0")]
impl<A: Step + Copy> IntoIterator for &ops::range::RangeInclusive<A> {
    type Item = A;
    type IntoIter = RangeInclusiveIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        (*self).into_iter()
    }
}

#[stable(feature = "new_range", since = "1.0.0")]
impl<A: Step + Copy> IntoIterator for &mut ops::range::RangeInclusive<A> {
    type Item = A;
    type IntoIter = RangeInclusiveIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        (*self).into_iter()
    }
}

impl<Idx: Step> ops::range::RangeInclusive<Idx> {
    /// Returns and advances `start` unless the range is empty.
    ///
    /// This differs from `.into_iter().next()` because
    /// that copies the range before advancing the iterator
    /// but this modifies the range in place.
    #[stable(feature = "new_range", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "can cause subtle bugs")]
    pub fn next(&mut self) -> Option<Idx> {
        let mut iter = self.clone().into_iter();
        let out = iter.next();

        if iter.inner.exhausted {
            // When exhausted, attempt to put end before start so the range is empty
            // If end is the minimum value (`start = end = 0`), set start past end
            if let Some(n) = Step::backward_checked(iter.inner.start.clone(), 1) {
                self.end = n;
                self.start = iter.inner.start;
            } else {
                self.start = Step::forward(iter.inner.end.clone(), 1);
                self.end = iter.inner.end;
            }
        } else {
            // Not exhausted, so just set new start and end
            self.start = iter.inner.start;
            self.end = iter.inner.end;
        }

        out
    }
}

macro_rules! iter_methods {
    ($($ty:ident),*) => {$(

impl<Idx: Step> ops::range::$ty<Idx> {
    /// Shorthand for `.into_iter().size_hint()`.
    ///
    /// See [`Iterator::size_hint`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn size_hint(&self) -> (usize, Option<usize>) {
        self.clone().into_iter().size_hint()
    }

    /// Shorthand for `.into_iter().count()`.
    ///
    /// See [`Iterator::count`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn count(self) -> usize {
        self.into_iter().count()
    }

    /// Shorthand for `.into_iter().last()`.
    ///
    /// See [`Iterator::last`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn last(self) -> Option<Idx> {
        self.into_iter().last()
    }

    /// Shorthand for `.into_iter().step_by(...)`.
    ///
    /// See [`Iterator::step_by`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn step_by(self, step: usize) -> crate::iter::StepBy<<Self as IntoIterator>::IntoIter> {
        self.into_iter().step_by(step)
    }

    /// Shorthand for `.into_iter().chain(...)`
    ///
    /// See [`Iterator::chain`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn chain<U>(self, other: U) -> crate::iter::Chain<<Self as IntoIterator>::IntoIter, U::IntoIter>
    where
        U: IntoIterator<Item = Idx>,
    {
        self.into_iter().chain(other)
    }

    /// Shorthand for `.into_iter().zip(...)`
    ///
    /// See [`Iterator::zip`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn zip<U>(self, other: U) -> crate::iter::Zip<<Self as IntoIterator>::IntoIter, U::IntoIter>
    where
        U: IntoIterator,
    {
        self.into_iter().zip(other)
    }

    /// Shorthand for `.into_iter().intersperse(...)`
    ///
    /// See [`Iterator::intersperse`]
    #[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
    pub fn intersperse(self, separator: Idx) -> crate::iter::Intersperse<<Self as IntoIterator>::IntoIter>
    where
        Idx: Clone,
    {
        self.into_iter().intersperse(separator)
    }

    /// Shorthand for `.into_iter().intersperse_with(...)`
    ///
    /// See [`Iterator::intersperse_with`]
    #[unstable(feature = "iter_intersperse", reason = "recently added", issue = "79524")]
    pub fn intersperse_with<G>(self, separator: G) -> crate::iter::IntersperseWith<<Self as IntoIterator>::IntoIter, G>
    where
        G: FnMut() -> Idx,
    {
        self.into_iter().intersperse_with(separator)
    }

    /// Shorthand for `.into_iter().map(...)`
    ///
    /// See [`Iterator::map`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn map<B, F>(self, f: F) -> crate::iter::Map<<Self as IntoIterator>::IntoIter, F>
    where
        F: FnMut(Idx) -> B,
    {
        self.into_iter().map(f)
    }

    /// Shorthand for `.into_iter().for_each(...)`
    ///
    /// See [`Iterator::for_each`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn for_each<F>(self, f: F)
    where
        F: FnMut(Idx),
    {
        self.into_iter().for_each(f)
    }

    /// Shorthand for `.into_iter().filter(...)`
    ///
    /// See [`Iterator::filter`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn filter<P>(self, predicate: P) -> crate::iter::Filter<<Self as IntoIterator>::IntoIter, P>
    where P:
        FnMut(&Idx) -> bool,
    {
        self.into_iter().filter(predicate)
    }

    /// Shorthand for `.into_iter().filter_map(...)`
    ///
    /// See [`Iterator::filter_map`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn filter_map<B, F>(self, f: F) -> crate::iter::FilterMap<<Self as IntoIterator>::IntoIter, F>
    where
        F: FnMut(Idx) -> Option<B>,
    {
        self.into_iter().filter_map(f)
    }

    /// Shorthand for `.into_iter().enumerate()`
    ///
    /// See [`Iterator::enumerate`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn enumerate(self) -> crate::iter::Enumerate<<Self as IntoIterator>::IntoIter> {
        self.into_iter().enumerate()
    }

    /// Shorthand for `.into_iter().peekable()`
    ///
    /// See [`Iterator::peekable`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn peekable(self) -> crate::iter::Peekable<<Self as IntoIterator>::IntoIter> {
        self.into_iter().peekable()
    }

    /// Shorthand for `.into_iter().filter_map(...)`
    ///
    /// See [`Iterator::filter_map`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn skip_while<P>(self, predicate: P) -> crate::iter::SkipWhile<<Self as IntoIterator>::IntoIter, P>
    where
        P: FnMut(&Idx) -> bool,
    {
        self.into_iter().skip_while(predicate)
    }

    /// Shorthand for `.into_iter().take_while(...)`
    ///
    /// See [`Iterator::take_while`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn take_while<P>(self, predicate: P) -> crate::iter::TakeWhile<<Self as IntoIterator>::IntoIter, P>
    where
        P: FnMut(&Idx) -> bool,
    {
        self.into_iter().take_while(predicate)
    }

    /// Shorthand for `.into_iter().map_while(...)`
    ///
    /// See [`Iterator::map_while`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn map_while<B, P>(self, predicate: P) -> crate::iter::MapWhile<<Self as IntoIterator>::IntoIter, P>
    where
        P: FnMut(Idx) -> Option<B>,
    {
        self.into_iter().map_while(predicate)
    }

    /// Shorthand for `.into_iter().skip(...)`
    ///
    /// See [`Iterator::skip`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn skip(self, n: usize) -> crate::iter::Skip<<Self as IntoIterator>::IntoIter> {
        self.into_iter().skip(n)
    }

    /// Shorthand for `.into_iter().take(...)`
    ///
    /// See [`Iterator::take`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn take(self, n: usize) -> crate::iter::Take<<Self as IntoIterator>::IntoIter> {
        self.into_iter().take(n)
    }

    /// Shorthand for `.into_iter().scan(...)`
    ///
    /// See [`Iterator::scan`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn scan<St, B, F>(self, initial_state: St, f: F) -> crate::iter::Scan<<Self as IntoIterator>::IntoIter, St, F>
    where
        F: FnMut(&mut St, Idx) -> Option<B>,
    {
        self.into_iter().scan(initial_state, f)
    }

    /// Shorthand for `.into_iter().flat_map(...)`
    ///
    /// See [`Iterator::flat_map`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn flat_map<U, F>(self, f: F) -> crate::iter::FlatMap<<Self as IntoIterator>::IntoIter, U, F>
    where
        U: IntoIterator,
        F: FnMut(Idx) -> U,
    {
        self.into_iter().flat_map(f)
    }

    /// Shorthand for `.into_iter().fuse()`
    ///
    /// See [`Iterator::fuse`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn fuse(self) -> crate::iter::Fuse<<Self as IntoIterator>::IntoIter> {
        self.into_iter().fuse()
    }

    /// Shorthand for `.into_iter().inspect(...)`
    ///
    /// See [`Iterator::inspect`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn inspect<F>(self, f: F) -> crate::iter::Inspect<<Self as IntoIterator>::IntoIter, F>
    where
        F: FnMut(&Idx),
    {
        self.into_iter().inspect(f)
    }

    /// Shorthand for `.into_iter().collect()`
    ///
    /// See [`Iterator::collect`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn collect<B: FromIterator<Idx>>(self) -> B {
        FromIterator::from_iter(self)
    }

    /// Shorthand for `.into_iter().partition(...)`
    ///
    /// See [`Iterator::partition`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn partition<B, F>(self, f: F) -> (B, B)
    where
        B: Default + Extend<Idx>,
        F: FnMut(&Idx) -> bool,
    {
        self.into_iter().partition(f)
    }

    /// Shorthand for `.into_iter().fold(...)`
    ///
    /// See [`Iterator::fold`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Idx) -> B,
    {
        self.into_iter().fold(init, f)
    }

    /// Shorthand for `.into_iter().reduce(...)`
    ///
    /// See [`Iterator::reduce`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn reduce<F>(self, f: F) -> Option<Idx>
    where
        F: FnMut(Idx, Idx) -> Idx,
    {
        self.into_iter().reduce(f)
    }

    /// Shorthand for `.into_iter().all(...)`
    ///
    /// One noticeable difference is that this takes the
    /// range by copy, rather than mutating it in place.
    ///
    /// See [`Iterator::all`]
    #[stable(feature = "new_range", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "can cause subtle bugs")]
    pub fn all<F>(self, f: F) -> bool
    where
        F: FnMut(Idx) -> bool,
    {
        self.into_iter().all(f)
    }

    /// Shorthand for `.into_iter().any(...)`
    ///
    /// One noticeable difference is that this takes the
    /// range by copy, rather than mutating it in place.
    ///
    /// See [`Iterator::any`]
    #[stable(feature = "new_range", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "can cause subtle bugs")]
    pub fn any<F>(self, f: F) -> bool
    where
        F: FnMut(Idx) -> bool,
    {
        self.into_iter().any(f)
    }

    /// Shorthand for `.into_iter().find(...)`
    ///
    /// One noticeable difference is that this takes the
    /// range by copy, rather than mutating it in place.
    ///
    /// See [`Iterator::find`]
    #[stable(feature = "new_range", since = "1.0.0")]
    #[deprecated(since = "1.0.0", note = "can cause subtle bugs")]
    pub fn find<P>(self, predicate: P) -> Option<Idx>
    where
        P: FnMut(&Idx) -> bool,
    {
        self.into_iter().find(predicate)
    }

    /// Shorthand for `.into_iter().max()`
    ///
    /// See [`Iterator::max`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn max(self) -> Option<Idx>
    where
        Idx: Ord,
    {
        self.into_iter().max()
    }

    /// Shorthand for `.into_iter().min()`
    ///
    /// See [`Iterator::min`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn min(self) -> Option<Idx>
    where
        Idx: Ord,
    {
        self.into_iter().min()
    }

    /// Shorthand for `.into_iter().max_by_key(...)`
    ///
    /// See [`Iterator::max_by_key`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn max_by_key<B: Ord, F>(self, f: F) -> Option<Idx>
    where
        F: FnMut(&Idx) -> B,
    {
        self.into_iter().max_by_key(f)
    }

    /// Shorthand for `.into_iter().max_by(...)`
    ///
    /// See [`Iterator::max_by`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn max_by<F>(self, f: F) -> Option<Idx>
    where
        F: FnMut(&Idx, &Idx) -> Ordering,
    {
        self.into_iter().max_by(f)
    }

    /// Shorthand for `.into_iter().min_by_key(...)`
    ///
    /// See [`Iterator::min_by_key`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn min_by_key<B: Ord, F>(self, f: F) -> Option<Idx>
    where
        F: FnMut(&Idx) -> B,
    {
        self.into_iter().min_by_key(f)
    }

    /// Shorthand for `.into_iter().min_by(...)`
    ///
    /// See [`Iterator::min_by`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn min_by<F>(self, f: F) -> Option<Idx>
    where
        F: FnMut(&Idx, &Idx) -> Ordering,
    {
        self.into_iter().min_by(f)
    }

    /// Shorthand for `.into_iter().rev()`
    ///
    /// See [`Iterator::rev`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn rev(self) -> crate::iter::Rev<<Self as IntoIterator>::IntoIter>
    where <Self as IntoIterator>::IntoIter: DoubleEndedIterator
    {
        self.into_iter().rev()
    }

    /// Shorthand for `.into_iter().cycle()`
    ///
    /// See [`Iterator::cycle`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn cycle(self) -> crate::iter::Cycle<<Self as IntoIterator>::IntoIter> {
        self.into_iter().cycle()
    }

    /// Shorthand for `.into_iter().array_chunks()`
    ///
    /// See [`Iterator::array_chunks`]
    #[unstable(feature = "iter_array_chunks", reason = "recently added", issue = "100450")]
    pub fn array_chunks<const N: usize>(self) -> crate::iter::ArrayChunks<<Self as IntoIterator>::IntoIter, N> {
        self.into_iter().array_chunks()
    }

    /// Shorthand for `.into_iter().sum()`
    ///
    /// See [`Iterator::sum`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn sum<S>(self) -> S
    where
        S: crate::iter::Sum<Idx>,
    {
        self.into_iter().sum()
    }

    /// Shorthand for `.into_iter().product()`
    ///
    /// See [`Iterator::product`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn product<P>(self) -> P
    where
        P: crate::iter::Product<Idx>,
    {
        self.into_iter().product()
    }

    /// Shorthand for `.into_iter().cmp(...)`
    ///
    /// See [`Iterator::cmp`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn cmp<I>(self, other: I) -> Ordering
    where
        I: IntoIterator<Item = Idx>,
        Idx: Ord,
    {
        self.into_iter().cmp(other)
    }

    /// Shorthand for `.into_iter().cmp_by(...)`
    ///
    /// See [`Iterator::cmp_by`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn cmp_by<I, F>(self, other: I, cmp: F) -> Ordering
    where
        I: IntoIterator,
        F: FnMut(Idx, I::Item) -> Ordering,
    {
        self.into_iter().cmp_by(other, cmp)
    }

    /// Shorthand for `.into_iter().partial_cmp(...)`
    ///
    /// See [`Iterator::partial_cmp`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn partial_cmp<I>(self, other: I) -> Option<Ordering>
    where
        I: IntoIterator<Item = Idx>,
        Idx: Ord,
    {
        self.into_iter().partial_cmp(other)
    }

    /// Shorthand for `.into_iter().partial_cmp_by(...)`
    ///
    /// See [`Iterator::partial_cmp_by`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn partial_cmp_by<I, F>(self, other: I, partial_cmp: F) -> Option<Ordering>
    where
        I: IntoIterator,
        F: FnMut(Idx, I::Item) -> Option<Ordering>,
    {
        self.into_iter().partial_cmp_by(other, partial_cmp)
    }

    /// Shorthand for `.into_iter().eq(...)`
    ///
    /// See [`Iterator::eq`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn eq<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Idx: PartialEq<I::Item>,
    {
        self.into_iter().eq(other)
    }

    /// Shorthand for `.into_iter().eq_by(...)`
    ///
    /// See [`Iterator::eq_by`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn eq_by<I, F>(self, other: I, eq: F) -> bool
    where
        I: IntoIterator,
        F: FnMut(Idx, I::Item) -> bool,
    {
        self.into_iter().eq_by(other, eq)
    }

    /// Shorthand for `.into_iter().ne(...)`
    ///
    /// See [`Iterator::ne`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn ne<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Idx: PartialEq<I::Item>,
    {
        self.into_iter().ne(other)
    }

    /// Shorthand for `.into_iter().lt(...)`
    ///
    /// See [`Iterator::lt`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn lt<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Idx: PartialOrd<I::Item>,
    {
        self.into_iter().lt(other)
    }

    /// Shorthand for `.into_iter().le(...)`
    ///
    /// See [`Iterator::le`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn le<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Idx: PartialOrd<I::Item>,
    {
        self.into_iter().le(other)
    }

    /// Shorthand for `.into_iter().gt(...)`
    ///
    /// See [`Iterator::gt`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn gt<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Idx: PartialOrd<I::Item>,
    {
        self.into_iter().gt(other)
    }

    /// Shorthand for `.into_iter().ge(...)`
    ///
    /// See [`Iterator::ge`]
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn ge<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Idx: PartialOrd<I::Item>,
    {
        self.into_iter().ge(other)
    }

    /// Shorthand for `.into_iter().is_sorted()`
    ///
    /// See [`Iterator::ge`]
    #[unstable(feature = "is_sorted", reason = "new API", issue = "53485")]
    pub fn is_sorted(self) -> bool
    where
        Idx: PartialOrd,
    {
        self.into_iter().is_sorted()
    }

    /// Returns the length of the `Range`.
    #[stable(feature = "new_range", since = "1.0.0")]
    pub fn len(&self) -> usize
    where <Self as IntoIterator>::IntoIter: ExactSizeIterator
    {
        ExactSizeIterator::len(&self.clone().into_iter())
    }
}

    )*}
}

iter_methods!(Range, RangeFrom, RangeInclusive);
