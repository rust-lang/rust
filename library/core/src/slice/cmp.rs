//! Comparison traits for `[T]`.

use super::{from_raw_parts, memchr};
use crate::cmp::{self, AlwaysApplicableOrd, BytewiseEq, Ordering, UnsignedBytewiseOrd};
use crate::intrinsics::compare_bytes;
use crate::mem;

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, U> PartialEq<[U]> for [T]
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U]) -> bool {
        SlicePartialEq::equal(self, other)
    }

    fn ne(&self, other: &[U]) -> bool {
        SlicePartialEq::not_equal(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialEq
trait SlicePartialEq<B>: Sized {
    fn equal(left: &[Self], right: &[B]) -> bool;

    fn not_equal(left: &[Self], right: &[B]) -> bool {
        !Self::equal(left, right)
    }
}

// Generic slice equality
impl<A, B> SlicePartialEq<B> for A
where
    A: PartialEq<B>,
{
    default fn equal(left: &[A], right: &[B]) -> bool {
        if left.len() != right.len() {
            return false;
        }

        // Implemented as explicit indexing rather
        // than zipped iterators for performance reasons.
        // See PR https://github.com/rust-lang/rust/pull/116846
        for idx in 0..left.len() {
            // bound checks are optimized away
            if left[idx] != right[idx] {
                return false;
            }
        }

        true
    }
}

// When each element can be compared byte-wise, we can compare all the bytes
// from the whole size in one call to the intrinsics.
impl<A, B> SlicePartialEq<B> for A
where
    A: BytewiseEq<B>,
{
    fn equal(left: &[A], right: &[B]) -> bool {
        if left.len() != right.len() {
            return false;
        }

        // SAFETY: `self` and `other` are references and are thus guaranteed to be valid.
        // The two slices have been checked to have the same size above.
        unsafe {
            let size = mem::size_of_val(left);
            compare_bytes(left.as_ptr() as *const u8, right.as_ptr() as *const u8, size) == 0
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for [T] {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, U> PartialOrd<[U]> for [T]
where
    T: PartialOrd<U>,
{
    fn partial_cmp(&self, other: &[U]) -> Option<Ordering> {
        SlicePartialOrd::partial_compare(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialOrd
trait SlicePartialOrd<B>: Sized {
    fn partial_compare(left: &[Self], right: &[B]) -> Option<Ordering>;
}

/// Implements comparison of slices [lexicographically](Ord#lexicographical-comparison).
impl<A, B> SlicePartialOrd<B> for A
where
    A: PartialOrd<B>,
{
    default fn partial_compare(left: &[A], right: &[B]) -> Option<Ordering> {
        let l = cmp::min(left.len(), right.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &left[..l];
        let rhs = &right[..l];

        for i in 0..l {
            match lhs[i].partial_cmp(&rhs[i]) {
                Some(Ordering::Equal) => (),
                non_eq => return non_eq,
            }
        }

        left.len().partial_cmp(&right.len())
    }
}

impl<A, B> SlicePartialOrd<B> for A
where
    A: AlwaysApplicableOrd<B>,
{
    fn partial_compare(left: &[A], right: &[B]) -> Option<Ordering> {
        Some(SliceOrd::compare(left, right))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Ord for [T]
where
    T: Ord,
{
    fn cmp(&self, other: &[T]) -> Ordering {
        SliceOrd::compare(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's Ord
trait SliceOrd<B>: Sized + Ord<B> {
    fn compare(left: &[Self], right: &[B]) -> Ordering;
}

/// Implements comparison of slices [lexicographically](Ord#lexicographical-comparison).
impl<A, B> SliceOrd<B> for A
where
    A: Ord<B>,
{
    default fn compare(left: &[A], right: &[B]) -> Ordering {
        let l = cmp::min(left.len(), right.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &left[..l];
        let rhs = &right[..l];

        for i in 0..l {
            match lhs[i].cmp(&rhs[i]) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            }
        }

        left.len().cmp(&right.len())
    }
}

// `compare_bytes` compares a sequence of unsigned bytes lexicographically, so
// use it if the requirements for `UnsignedBytewiseOrd` are fulfilled.
impl<A, B> SliceOrd<B> for A
where
    A: UnsignedBytewiseOrd<B>,
{
    #[inline]
    fn compare(left: &[A], right: &[B]) -> Ordering {
        // Since the length of a slice is always less than or equal to
        // isize::MAX, this never underflows.
        let diff = left.len() as isize - right.len() as isize;
        // This comparison gets optimized away (on x86_64 and ARM) because the
        // subtraction updates flags.
        let len = if left.len() < right.len() { left.len() } else { right.len() };
        let left = left.as_ptr().cast();
        let right = right.as_ptr().cast();
        // SAFETY: `left` and `right` are references and are thus guaranteed to
        // be valid. `UnsignedBytewiseOrd` is only implemented for types that
        // are valid u8s and can be compared the same way. We use the minimum
        // of both lengths which guarantees that both regions are valid for
        // reads in that interval.
        let mut order = unsafe { compare_bytes(left, right, len) as isize };
        if order == 0 {
            order = diff;
        }
        order.cmp(&0)
    }
}

// trait for specialization of `slice::contains`
pub(super) trait SliceContains: Sized {
    fn slice_contains(&self, x: &[Self]) -> bool;
}

impl<T> SliceContains for T
where
    T: PartialEq,
{
    default fn slice_contains(&self, x: &[Self]) -> bool {
        x.iter().any(|y| *y == *self)
    }
}

impl SliceContains for u8 {
    #[inline]
    fn slice_contains(&self, x: &[Self]) -> bool {
        memchr::memchr(*self, x).is_some()
    }
}

impl SliceContains for i8 {
    #[inline]
    fn slice_contains(&self, x: &[Self]) -> bool {
        let byte = *self as u8;
        // SAFETY: `i8` and `u8` have the same memory layout, thus casting `x.as_ptr()`
        // as `*const u8` is safe. The `x.as_ptr()` comes from a reference and is thus guaranteed
        // to be valid for reads for the length of the slice `x.len()`, which cannot be larger
        // than `isize::MAX`. The returned slice is never mutated.
        let bytes: &[u8] = unsafe { from_raw_parts(x.as_ptr() as *const u8, x.len()) };
        memchr::memchr(byte, bytes).is_some()
    }
}
