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

/// Converts a reference to `T` into a reference to an array of length 1 (without copying).
#[unstable(feature = "array_from_ref", issue = "77101")]
pub fn from_ref<T>(s: &T) -> &[T; 1] {
    // SAFETY: Converting `&T` to `&[T; 1]` is sound.
    unsafe { &*(s as *const T).cast::<[T; 1]>() }
}

/// Converts a mutable reference to `T` into a mutable reference to an array of length 1 (without copying).
#[unstable(feature = "array_from_ref", issue = "77101")]
pub fn from_mut<T>(s: &mut T) -> &mut [T; 1] {
    // SAFETY: Converting `&mut T` to `&mut [T; 1]` is sound.
    unsafe { &mut *(s as *mut T).cast::<[T; 1]>() }
}

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
impl<T, const N: usize> AsRef<[T]> for [T; N] {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, const N: usize> AsMut<[T]> for [T; N] {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

#[stable(feature = "array_borrow", since = "1.4.0")]
impl<T, const N: usize> Borrow<[T]> for [T; N] {
    fn borrow(&self) -> &[T] {
        self
    }
}

#[stable(feature = "array_borrow", since = "1.4.0")]
impl<T, const N: usize> BorrowMut<[T]> for [T; N] {
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl<T, const N: usize> TryFrom<&[T]> for [T; N]
where
    T: Copy,
{
    type Error = TryFromSliceError;

    fn try_from(slice: &[T]) -> Result<[T; N], TryFromSliceError> {
        <&Self>::try_from(slice).map(|r| *r)
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl<'a, T, const N: usize> TryFrom<&'a [T]> for &'a [T; N] {
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
impl<'a, T, const N: usize> TryFrom<&'a mut [T]> for &'a mut [T; N] {
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
impl<T: Hash, const N: usize> Hash for [T; N] {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self[..], state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug, const N: usize> fmt::Debug for [T; N] {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&&self[..], f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, const N: usize> IntoIterator for &'a [T; N] {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, const N: usize> IntoIterator for &'a mut [T; N] {
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
impl<T: Eq, const N: usize> Eq for [T; N] {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd, const N: usize> PartialOrd for [T; N] {
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

/// Implements comparison of arrays [lexicographically](Ord#lexicographical-comparison).
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord, const N: usize> Ord for [T; N] {
    #[inline]
    fn cmp(&self, other: &[T; N]) -> Ordering {
        Ord::cmp(&&self[..], &&other[..])
    }
}

// The Default impls cannot be done with const generics because `[T; 0]` doesn't
// require Default to be implemented, and having different impl blocks for
// different numbers isn't supported yet.

macro_rules! array_impl_default {
    {$n:expr, $t:ident $($ts:ident)*} => {
        #[stable(since = "1.4.0", feature = "array_default")]
        impl<T> Default for [T; $n] where T: Default {
            fn default() -> [T; $n] {
                [$t::default(), $($ts::default()),*]
            }
        }
        array_impl_default!{($n - 1), $($ts)*}
    };
    {$n:expr,} => {
        #[stable(since = "1.4.0", feature = "array_default")]
        impl<T> Default for [T; $n] {
            fn default() -> [T; $n] { [] }
        }
    };
}

array_impl_default! {32, T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T}

#[lang = "array"]
impl<T, const N: usize> [T; N] {
    /// Returns an array of the same size as `self`, with function `f` applied to each element
    /// in order.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_map)]
    /// let x = [1, 2, 3];
    /// let y = x.map(|v| v + 1);
    /// assert_eq!(y, [2, 3, 4]);
    ///
    /// let x = [1, 2, 3];
    /// let mut temp = 0;
    /// let y = x.map(|v| { temp += 1; v * temp });
    /// assert_eq!(y, [1, 4, 9]);
    ///
    /// let x = ["Ferris", "Bueller's", "Day", "Off"];
    /// let y = x.map(|v| v.len());
    /// assert_eq!(y, [6, 9, 3, 3]);
    /// ```
    #[unstable(feature = "array_map", issue = "75243")]
    pub fn map<F, U>(self, mut f: F) -> [U; N]
    where
        F: FnMut(T) -> U,
    {
        use crate::mem::MaybeUninit;
        struct Guard<T, const N: usize> {
            dst: *mut T,
            initialized: usize,
        }

        impl<T, const N: usize> Drop for Guard<T, N> {
            fn drop(&mut self) {
                debug_assert!(self.initialized <= N);

                let initialized_part =
                    crate::ptr::slice_from_raw_parts_mut(self.dst, self.initialized);
                // SAFETY: this raw slice will contain only initialized objects
                // that's why, it is allowed to drop it.
                unsafe {
                    crate::ptr::drop_in_place(initialized_part);
                }
            }
        }
        let mut dst = MaybeUninit::uninit_array::<N>();
        let mut guard: Guard<U, N> =
            Guard { dst: MaybeUninit::slice_as_mut_ptr(&mut dst), initialized: 0 };
        for (src, dst) in IntoIter::new(self).zip(&mut dst) {
            dst.write(f(src));
            guard.initialized += 1;
        }
        // FIXME: Convert to crate::mem::transmute once it works with generics.
        // unsafe { crate::mem::transmute::<[MaybeUninit<U>; N], [U; N]>(dst) }
        crate::mem::forget(guard);
        // SAFETY: At this point we've properly initialized the whole array
        // and we just need to cast it to the correct type.
        unsafe { crate::mem::transmute_copy::<_, [U; N]>(&dst) }
    }

    /// Returns a slice containing the entire array. Equivalent to `&s[..]`.
    #[unstable(feature = "array_methods", issue = "76118")]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Returns a mutable slice containing the entire array. Equivalent to
    /// `&mut s[..]`.
    #[unstable(feature = "array_methods", issue = "76118")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}
