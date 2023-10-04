//! Comparison traits for `[T]`.

use crate::cmp::{self, BytewiseEq, Ordering};
use crate::intrinsics::compare_bytes;
use crate::mem;

use super::from_raw_parts;
use super::memchr;

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> PartialEq<[B]> for [A]
where
    A: PartialEq<B>,
{
    fn eq(&self, other: &[B]) -> bool {
        SlicePartialEq::equal(self, other)
    }

    fn ne(&self, other: &[B]) -> bool {
        SlicePartialEq::not_equal(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for [T] {}

/// Implements comparison of vectors [lexicographically](Ord#lexicographical-comparison).
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for [T] {
    fn cmp(&self, other: &[T]) -> Ordering {
        SliceOrd::compare(self, other)
    }
}

/// Implements comparison of vectors [lexicographically](Ord#lexicographical-comparison).
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for [T] {
    fn partial_cmp(&self, other: &[T]) -> Option<Ordering> {
        SlicePartialOrd::partial_compare(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialEq
trait SlicePartialEq<B> {
    fn equal(&self, other: &[B]) -> bool;

    fn not_equal(&self, other: &[B]) -> bool {
        !self.equal(other)
    }
}

// Generic slice equality
impl<A, B> SlicePartialEq<B> for [A]
where
    A: PartialEq<B>,
{
    #[inline]
    default fn equal(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // at least 8 items for unrolling to make sense (4 peeled + 4+ unrolled)
        if self.len() < 8 {
            return eq_small(self, other);
        }

        eq_unroll(self, other)
    }
}

#[inline]
fn eq_small<A, B>(a: &[A], b: &[B]) -> bool
where
    A: PartialEq<B>,
{
    a.iter().zip(b).all(|(a, b)| a == b)
}

fn eq_unroll<A, B>(a: &[A], b: &[B]) -> bool
where
    A: PartialEq<B>,
{
    let (mut chunks_a, residual_a) = a.as_chunks::<4>();
    let (mut chunks_b, residual_b) = b.as_chunks::<4>();
    let peeled_a = chunks_a.take_first().unwrap();
    let peeled_b = chunks_b.take_first().unwrap();

    // peel the first chunk and do a short-circuiting comparison to bail early on mismatches
    // in case comparisons are expensive
    let mut result = eq_small(peeled_a, peeled_b);

    // then check the residual, another chance to bail early
    result = result && eq_small(residual_a, residual_b);

    // iter.all short-circuits which means the backend can't unroll the loop due to early exits.
    // So we unroll it manually.
    result = result
        && chunks_a
            .iter()
            .zip(chunks_b)
            .all(|(a, b)| (a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2]) & (a[3] == b[3]));

    result
}

// When each element can be compared byte-wise, we can compare all the bytes
// from the whole size in one call to the intrinsics.
impl<A, B> SlicePartialEq<B> for [A]
where
    A: BytewiseEq<B>,
{
    fn equal(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // SAFETY: `self` and `other` are references and are thus guaranteed to be valid.
        // The two slices have been checked to have the same size above.
        unsafe {
            let size = mem::size_of_val(self);
            compare_bytes(self.as_ptr() as *const u8, other.as_ptr() as *const u8, size) == 0
        }
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialOrd
trait SlicePartialOrd: Sized {
    fn partial_compare(left: &[Self], right: &[Self]) -> Option<Ordering>;
}

impl<A: PartialOrd> SlicePartialOrd for A {
    default fn partial_compare(left: &[A], right: &[A]) -> Option<Ordering> {
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

// This is the impl that we would like to have. Unfortunately it's not sound.
// See `partial_ord_slice.rs`.
/*
impl<A> SlicePartialOrd for A
where
    A: Ord,
{
    default fn partial_compare(left: &[A], right: &[A]) -> Option<Ordering> {
        Some(SliceOrd::compare(left, right))
    }
}
*/

impl<A: AlwaysApplicableOrd> SlicePartialOrd for A {
    fn partial_compare(left: &[A], right: &[A]) -> Option<Ordering> {
        Some(SliceOrd::compare(left, right))
    }
}

#[rustc_specialization_trait]
trait AlwaysApplicableOrd: SliceOrd + Ord {}

macro_rules! always_applicable_ord {
    ($([$($p:tt)*] $t:ty,)*) => {
        $(impl<$($p)*> AlwaysApplicableOrd for $t {})*
    }
}

always_applicable_ord! {
    [] u8, [] u16, [] u32, [] u64, [] u128, [] usize,
    [] i8, [] i16, [] i32, [] i64, [] i128, [] isize,
    [] bool, [] char,
    [T: ?Sized] *const T, [T: ?Sized] *mut T,
    [T: AlwaysApplicableOrd] &T,
    [T: AlwaysApplicableOrd] &mut T,
    [T: AlwaysApplicableOrd] Option<T>,
}

#[doc(hidden)]
// intermediate trait for specialization of slice's Ord
trait SliceOrd: Sized {
    fn compare(left: &[Self], right: &[Self]) -> Ordering;
}

impl<A: Ord> SliceOrd for A {
    default fn compare(left: &[Self], right: &[Self]) -> Ordering {
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

// `compare_bytes` compares a sequence of unsigned bytes lexicographically.
// this matches the order we want for [u8], but no others (not even [i8]).
impl SliceOrd for u8 {
    #[inline]
    fn compare(left: &[Self], right: &[Self]) -> Ordering {
        // Since the length of a slice is always less than or equal to isize::MAX, this never underflows.
        let diff = left.len() as isize - right.len() as isize;
        // This comparison gets optimized away (on x86_64 and ARM) because the subtraction updates flags.
        let len = if left.len() < right.len() { left.len() } else { right.len() };
        // SAFETY: `left` and `right` are references and are thus guaranteed to be valid.
        // We use the minimum of both lengths which guarantees that both regions are
        // valid for reads in that interval.
        let mut order = unsafe { compare_bytes(left.as_ptr(), right.as_ptr(), len) as isize };
        if order == 0 {
            order = diff;
        }
        order.cmp(&0)
    }
}

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
