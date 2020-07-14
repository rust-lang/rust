//! Implementations of things like `Eq` for fixed-length arrays
//! up to a certain length. Eventually, we should be able to generalize
//! to all lengths.
//!
//! *[See also the array primitive type](../../std/primitive.array.html).*

#![stable(feature = "core_array", since = "1.36.0")]

use crate::borrow::{Borrow, BorrowMut};
use crate::cmp::Ordering;
use crate::convert::{Infallible, TryFrom};
use crate::fmt;
use crate::hash::{self, Hash};
use crate::marker::Unsize;
use crate::slice::{Iter, IterMut};

mod iter;

#[unstable(feature = "array_value_iter", issue = "65798")]
pub use iter::IntoIter;

/// Utility trait implemented only on arrays of fixed size
///
/// This trait can be used to implement other traits on fixed-size arrays
/// without causing much metadata bloat.
///
/// The trait is marked unsafe in order to restrict implementors to fixed-size
/// arrays. User of this trait can assume that implementors have the exact
/// layout in memory of a fixed size array (for example, for unsafe
/// initialization).
///
/// Note that the traits [`AsRef`] and [`AsMut`] provide similar methods for types that
/// may not be fixed-size arrays. Implementors should prefer those traits
/// instead.
///
/// [`AsRef`]: ../convert/trait.AsRef.html
/// [`AsMut`]: ../convert/trait.AsMut.html
#[unstable(feature = "fixed_size_array", issue = "27778")]
pub unsafe trait FixedSizeArray<T> {
    /// Converts the array to immutable slice
    #[unstable(feature = "fixed_size_array", issue = "27778")]
    fn as_slice(&self) -> &[T];
    /// Converts the array to mutable slice
    #[unstable(feature = "fixed_size_array", issue = "27778")]
    fn as_mut_slice(&mut self) -> &mut [T];
}

#[unstable(feature = "fixed_size_array", issue = "27778")]
unsafe impl<T, A: Unsize<[T]>> FixedSizeArray<T> for A {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

/// The error type returned when a conversion from a slice to an array fails.
#[stable(feature = "try_from", since = "1.34.0")]
#[derive(Debug, Copy, Clone)]
pub struct TryFromSliceError(());

#[stable(feature = "core_array", since = "1.36.0")]
impl fmt::Display for TryFromSliceError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.__description(), f)
    }
}

impl TryFromSliceError {
    #[unstable(
        feature = "array_error_internals",
        reason = "available through Error trait and this method should not \
                     be exposed publicly",
        issue = "none"
    )]
    #[inline]
    #[doc(hidden)]
    pub fn __description(&self) -> &str {
        "could not convert slice to array"
    }
}

#[stable(feature = "try_from_slice_error", since = "1.36.0")]
impl From<Infallible> for TryFromSliceError {
    fn from(x: Infallible) -> TryFromSliceError {
        match x {}
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, const N: usize> AsRef<[T]> for [T; N]
where
    [T; N]: LengthAtMost32,
{
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, const N: usize> AsMut<[T]> for [T; N]
where
    [T; N]: LengthAtMost32,
{
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

#[stable(feature = "array_borrow", since = "1.4.0")]
impl<T, const N: usize> Borrow<[T]> for [T; N]
where
    [T; N]: LengthAtMost32,
{
    fn borrow(&self) -> &[T] {
        self
    }
}

#[stable(feature = "array_borrow", since = "1.4.0")]
impl<T, const N: usize> BorrowMut<[T]> for [T; N]
where
    [T; N]: LengthAtMost32,
{
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl<T, const N: usize> TryFrom<&[T]> for [T; N]
where
    T: Copy,
    [T; N]: LengthAtMost32,
{
    type Error = TryFromSliceError;

    fn try_from(slice: &[T]) -> Result<[T; N], TryFromSliceError> {
        <&Self>::try_from(slice).map(|r| *r)
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl<'a, T, const N: usize> TryFrom<&'a [T]> for &'a [T; N]
where
    [T; N]: LengthAtMost32,
{
    type Error = TryFromSliceError;

    fn try_from(slice: &[T]) -> Result<&[T; N], TryFromSliceError> {
        if slice.len() == N {
            let ptr = slice.as_ptr() as *const [T; N];
            // SAFETY: ok because we just checked that the length fits
            unsafe { Ok(&*ptr) }
        } else {
            Err(TryFromSliceError(()))
        }
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl<'a, T, const N: usize> TryFrom<&'a mut [T]> for &'a mut [T; N]
where
    [T; N]: LengthAtMost32,
{
    type Error = TryFromSliceError;

    fn try_from(slice: &mut [T]) -> Result<&mut [T; N], TryFromSliceError> {
        if slice.len() == N {
            let ptr = slice.as_mut_ptr() as *mut [T; N];
            // SAFETY: ok because we just checked that the length fits
            unsafe { Ok(&mut *ptr) }
        } else {
            Err(TryFromSliceError(()))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Hash, const N: usize> Hash for [T; N]
where
    [T; N]: LengthAtMost32,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self[..], state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug, const N: usize> fmt::Debug for [T; N]
where
    [T; N]: LengthAtMost32,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&&self[..], f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, const N: usize> IntoIterator for &'a [T; N]
where
    [T; N]: LengthAtMost32,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, const N: usize> IntoIterator for &'a mut [T; N]
where
    [T; N]: LengthAtMost32,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[B; N]> for [A; N]
where
    A: PartialEq<B>,
    [A; N]: LengthAtMost32,
    [B; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &[B; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[B; N]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[B]> for [A; N]
where
    A: PartialEq<B>,
    [A; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &[B]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[B]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B, const N: usize> PartialEq<[A; N]> for [B]
where
    B: PartialEq<A>,
    [A; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, A, B, const N: usize> PartialEq<&'b [B]> for [A; N]
where
    A: PartialEq<B>,
    [A; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &&'b [B]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &&'b [B]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, A, B, const N: usize> PartialEq<[A; N]> for &'b [B]
where
    B: PartialEq<A>,
    [A; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, A, B, const N: usize> PartialEq<&'b mut [B]> for [A; N]
where
    A: PartialEq<B>,
    [A; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &&'b mut [B]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &&'b mut [B]) -> bool {
        self[..] != other[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'b, A, B, const N: usize> PartialEq<[A; N]> for &'b mut [B]
where
    B: PartialEq<A>,
    [A; N]: LengthAtMost32,
{
    #[inline]
    fn eq(&self, other: &[A; N]) -> bool {
        self[..] == other[..]
    }
    #[inline]
    fn ne(&self, other: &[A; N]) -> bool {
        self[..] != other[..]
    }
}

// NOTE: some less important impls are omitted to reduce code bloat
// __impl_slice_eq2! { [A; $N], &'b [B; $N] }
// __impl_slice_eq2! { [A; $N], &'b mut [B; $N] }

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq, const N: usize> Eq for [T; N] where [T; N]: LengthAtMost32 {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd, const N: usize> PartialOrd for [T; N]
where
    [T; N]: LengthAtMost32,
{
    #[inline]
    fn partial_cmp(&self, other: &[T; N]) -> Option<Ordering> {
        PartialOrd::partial_cmp(&&self[..], &&other[..])
    }
    #[inline]
    fn lt(&self, other: &[T; N]) -> bool {
        PartialOrd::lt(&&self[..], &&other[..])
    }
    #[inline]
    fn le(&self, other: &[T; N]) -> bool {
        PartialOrd::le(&&self[..], &&other[..])
    }
    #[inline]
    fn ge(&self, other: &[T; N]) -> bool {
        PartialOrd::ge(&&self[..], &&other[..])
    }
    #[inline]
    fn gt(&self, other: &[T; N]) -> bool {
        PartialOrd::gt(&&self[..], &&other[..])
    }
}

/// Implements comparison of arrays lexicographically.
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord, const N: usize> Ord for [T; N]
where
    [T; N]: LengthAtMost32,
{
    #[inline]
    fn cmp(&self, other: &[T; N]) -> Ordering {
        Ord::cmp(&&self[..], &&other[..])
    }
}

/// A trait implemented by all arrays which are either empty or contain a type implementing `Default`.
#[unstable(feature = "array_default_internals", reason = "implementation detail", issue = "none")]
#[marker]
pub trait ArrayDefault {}

#[unstable(feature = "array_default_internals", reason = "implementation detail", issue = "none")]
impl<T> ArrayDefault for [T; 0] {}

#[unstable(feature = "array_default_internals", reason = "implementation detail", issue = "none")]
impl<T: Default, const N: usize> ArrayDefault for [T; N] {}

trait DefaultHack {
    fn default_hack() -> Self;
}

impl<T> DefaultHack for T {
    default fn default_hack() -> Self {
        unreachable!();
    }
}

impl<T: Default> DefaultHack for T {
    #[inline(always)]
    fn default_hack() -> Self {
        Default::default()
    }
}

#[stable(since = "1.4.0", feature = "array_default")]
impl<T, const N: usize> Default for [T; N]
where
    [T; N]: ArrayDefault + LengthAtMost32,
{
    #[inline]
    fn default() -> [T; N] {
        use crate::mem::MaybeUninit;
        let mut data: MaybeUninit<[T; N]> = MaybeUninit::uninit();
        // Invariant: first `init` items are initialized
        struct Guard<T, const N: usize> {
            data: *mut T,
            init: usize,
        }

        impl<T, const N: usize> Drop for Guard<T, N> {
            #[inline]
            fn drop(&mut self) {
                debug_assert!(self.init <= N);

                let initialized_part = crate::ptr::slice_from_raw_parts_mut(self.data, self.init);
                // SAFETY: this raw slice will contain only initialized objects
                // that's why, it is allowed to drop it.
                unsafe {
                    crate::ptr::drop_in_place(initialized_part);
                }
            }
        }

        let mut w = Guard::<T, N> { data: data.as_mut_ptr() as *mut T, init: 0 };
        // SAFETY: in fact we go from &mut MaybeUninit<[T; N]> to &mut [MaybeUninit<T>; N].
        // it is always correct.
        let slots = unsafe { &mut *(data.as_mut_ptr() as *mut [MaybeUninit<T>; N]) };
        for slot in slots.iter_mut() {
            slot.write(T::default_hack());
            // now, when slot is filled with value, we can increment
            // `init`.
            w.init += 1;
        }

        // Prevent double-read in callee and Wrapper::drop
        crate::mem::forget(w);
        // SAFETY: at this point whole array is initialized.
        unsafe { data.assume_init() }
    }
}

/// Implemented for lengths where trait impls are allowed on arrays in core/std
#[rustc_on_unimplemented(message = "arrays only have std trait implementations for lengths 0..=32")]
#[unstable(
    feature = "const_generic_impls_guard",
    issue = "none",
    reason = "will never be stable, just a temporary step until const generics are stable"
)]
pub trait LengthAtMost32 {}

macro_rules! array_impls {
    ($($N:literal)+) => {
        $(
            #[unstable(feature = "const_generic_impls_guard", issue = "none")]
            impl<T> LengthAtMost32 for [T; $N] {}
        )+
    }
}

array_impls! {
     0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32
}
