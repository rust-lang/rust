//! Utilities for the array primitive type.
//!
//! *[See also the array primitive type](array).*

#![stable(feature = "core_array", since = "1.36.0")]

use crate::borrow::{Borrow, BorrowMut};
use crate::cmp::Ordering;
use crate::convert::{Infallible, TryFrom};
#[cfg(not(bootstrap))]
use crate::error::Error;
use crate::fmt;
use crate::hash::{self, Hash};
use crate::iter::TrustedLen;
use crate::mem::{self, MaybeUninit};
use crate::ops::{
    ChangeOutputType, ControlFlow, FromResidual, Index, IndexMut, NeverShortCircuit, Residual, Try,
};
use crate::slice::{Iter, IterMut};

mod equality;
mod iter;

#[stable(feature = "array_value_iter", since = "1.51.0")]
pub use iter::IntoIter;

/// Creates an array `[T; N]` where each array element `T` is returned by the `cb` call.
///
/// # Arguments
///
/// * `cb`: Callback where the passed argument is the current array index.
///
/// # Example
///
/// ```rust
/// let array = core::array::from_fn(|i| i);
/// assert_eq!(array, [0, 1, 2, 3, 4]);
/// ```
#[inline]
#[stable(feature = "array_from_fn", since = "1.63.0")]
pub fn from_fn<T, const N: usize, F>(mut cb: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut idx = 0;
    [(); N].map(|_| {
        let res = cb(idx);
        idx += 1;
        res
    })
}

/// Creates an array `[T; N]` where each fallible array element `T` is returned by the `cb` call.
/// Unlike [`from_fn`], where the element creation can't fail, this version will return an error
/// if any element creation was unsuccessful.
///
/// The return type of this function depends on the return type of the closure.
/// If you return `Result<T, E>` from the closure, you'll get a `Result<[T; N]; E>`.
/// If you return `Option<T>` from the closure, you'll get an `Option<[T; N]>`.
///
/// # Arguments
///
/// * `cb`: Callback where the passed argument is the current array index.
///
/// # Example
///
/// ```rust
/// #![feature(array_try_from_fn)]
///
/// let array: Result<[u8; 5], _> = std::array::try_from_fn(|i| i.try_into());
/// assert_eq!(array, Ok([0, 1, 2, 3, 4]));
///
/// let array: Result<[i8; 200], _> = std::array::try_from_fn(|i| i.try_into());
/// assert!(array.is_err());
///
/// let array: Option<[_; 4]> = std::array::try_from_fn(|i| i.checked_add(100));
/// assert_eq!(array, Some([100, 101, 102, 103]));
///
/// let array: Option<[_; 4]> = std::array::try_from_fn(|i| i.checked_sub(100));
/// assert_eq!(array, None);
/// ```
#[inline]
#[unstable(feature = "array_try_from_fn", issue = "89379")]
pub fn try_from_fn<R, const N: usize, F>(cb: F) -> ChangeOutputType<R, [R::Output; N]>
where
    F: FnMut(usize) -> R,
    R: Try,
    R::Residual: Residual<[R::Output; N]>,
{
    // SAFETY: we know for certain that this iterator will yield exactly `N`
    // items.
    unsafe { try_collect_into_array_unchecked(&mut (0..N).map(cb)) }
}

/// Converts a reference to `T` into a reference to an array of length 1 (without copying).
#[stable(feature = "array_from_ref", since = "1.53.0")]
#[rustc_const_stable(feature = "const_array_from_ref_shared", since = "1.63.0")]
pub const fn from_ref<T>(s: &T) -> &[T; 1] {
    // SAFETY: Converting `&T` to `&[T; 1]` is sound.
    unsafe { &*(s as *const T).cast::<[T; 1]>() }
}

/// Converts a mutable reference to `T` into a mutable reference to an array of length 1 (without copying).
#[stable(feature = "array_from_ref", since = "1.53.0")]
#[rustc_const_unstable(feature = "const_array_from_ref", issue = "90206")]
pub const fn from_mut<T>(s: &mut T) -> &mut [T; 1] {
    // SAFETY: Converting `&mut T` to `&mut [T; 1]` is sound.
    unsafe { &mut *(s as *mut T).cast::<[T; 1]>() }
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

#[cfg(not(bootstrap))]
#[stable(feature = "try_from", since = "1.34.0")]
impl Error for TryFromSliceError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
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
#[rustc_const_unstable(feature = "const_convert", issue = "88674")]
impl const From<Infallible> for TryFromSliceError {
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
#[rustc_const_unstable(feature = "const_borrow", issue = "91522")]
impl<T, const N: usize> const Borrow<[T]> for [T; N] {
    fn borrow(&self) -> &[T] {
        self
    }
}

#[stable(feature = "array_borrow", since = "1.4.0")]
#[rustc_const_unstable(feature = "const_borrow", issue = "91522")]
impl<T, const N: usize> const BorrowMut<[T]> for [T; N] {
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

#[stable(feature = "try_from_mut_slice_to_array", since = "1.59.0")]
impl<T, const N: usize> TryFrom<&mut [T]> for [T; N]
where
    T: Copy,
{
    type Error = TryFromSliceError;

    fn try_from(slice: &mut [T]) -> Result<[T; N], TryFromSliceError> {
        <Self>::try_from(&*slice)
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

/// The hash of an array is the same as that of the corresponding slice,
/// as required by the `Borrow` implementation.
///
/// ```
/// #![feature(build_hasher_simple_hash_one)]
/// use std::hash::BuildHasher;
///
/// let b = std::collections::hash_map::RandomState::new();
/// let a: [u8; 3] = [0xa8, 0x3c, 0x09];
/// let s: &[u8] = &[0xa8, 0x3c, 0x09];
/// assert_eq!(b.hash_one(a), b.hash_one(s));
/// ```
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

#[stable(feature = "index_trait_on_arrays", since = "1.50.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
impl<T, I, const N: usize> const Index<I> for [T; N]
where
    [T]: ~const Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self as &[T], index)
    }
}

#[stable(feature = "index_trait_on_arrays", since = "1.50.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
impl<T, I, const N: usize> const IndexMut<I> for [T; N]
where
    [T]: ~const IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self as &mut [T], index)
    }
}

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

#[stable(feature = "copy_clone_array_lib", since = "1.58.0")]
impl<T: Copy, const N: usize> Copy for [T; N] {}

#[stable(feature = "copy_clone_array_lib", since = "1.58.0")]
impl<T: Clone, const N: usize> Clone for [T; N] {
    #[inline]
    fn clone(&self) -> Self {
        SpecArrayClone::clone(self)
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.clone_from_slice(other);
    }
}

trait SpecArrayClone: Clone {
    fn clone<const N: usize>(array: &[Self; N]) -> [Self; N];
}

impl<T: Clone> SpecArrayClone for T {
    #[inline]
    default fn clone<const N: usize>(array: &[T; N]) -> [T; N] {
        // SAFETY: we know for certain that this iterator will yield exactly `N`
        // items.
        unsafe { collect_into_array_unchecked(&mut array.iter().cloned()) }
    }
}

impl<T: Copy> SpecArrayClone for T {
    #[inline]
    fn clone<const N: usize>(array: &[T; N]) -> [T; N] {
        *array
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
        #[rustc_const_unstable(feature = "const_default_impls", issue = "87864")]
        impl<T> const Default for [T; $n] {
            fn default() -> [T; $n] { [] }
        }
    };
}

array_impl_default! {32, T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T T}

impl<T, const N: usize> [T; N] {
    /// Returns an array of the same size as `self`, with function `f` applied to each element
    /// in order.
    ///
    /// If you don't necessarily need a new fixed-size array, consider using
    /// [`Iterator::map`] instead.
    ///
    ///
    /// # Note on performance and stack usage
    ///
    /// Unfortunately, usages of this method are currently not always optimized
    /// as well as they could be. This mainly concerns large arrays, as mapping
    /// over small arrays seem to be optimized just fine. Also note that in
    /// debug mode (i.e. without any optimizations), this method can use a lot
    /// of stack space (a few times the size of the array or more).
    ///
    /// Therefore, in performance-critical code, try to avoid using this method
    /// on large arrays or check the emitted code. Also try to avoid chained
    /// maps (e.g. `arr.map(...).map(...)`).
    ///
    /// In many cases, you can instead use [`Iterator::map`] by calling `.iter()`
    /// or `.into_iter()` on your array. `[T; N]::map` is only necessary if you
    /// really need a new array of the same size as the result. Rust's lazy
    /// iterators tend to get optimized very well.
    ///
    ///
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "array_map", since = "1.55.0")]
    pub fn map<F, U>(self, f: F) -> [U; N]
    where
        F: FnMut(T) -> U,
    {
        // SAFETY: we know for certain that this iterator will yield exactly `N`
        // items.
        unsafe { collect_into_array_unchecked(&mut IntoIterator::into_iter(self).map(f)) }
    }

    /// A fallible function `f` applied to each element on array `self` in order to
    /// return an array the same size as `self` or the first error encountered.
    ///
    /// The return type of this function depends on the return type of the closure.
    /// If you return `Result<T, E>` from the closure, you'll get a `Result<[T; N]; E>`.
    /// If you return `Option<T>` from the closure, you'll get an `Option<[T; N]>`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_try_map)]
    /// let a = ["1", "2", "3"];
    /// let b = a.try_map(|v| v.parse::<u32>()).unwrap().map(|v| v + 1);
    /// assert_eq!(b, [2, 3, 4]);
    ///
    /// let a = ["1", "2a", "3"];
    /// let b = a.try_map(|v| v.parse::<u32>());
    /// assert!(b.is_err());
    ///
    /// use std::num::NonZeroU32;
    /// let z = [1, 2, 0, 3, 4];
    /// assert_eq!(z.try_map(NonZeroU32::new), None);
    /// let a = [1, 2, 3];
    /// let b = a.try_map(NonZeroU32::new);
    /// let c = b.map(|x| x.map(NonZeroU32::get));
    /// assert_eq!(c, Some(a));
    /// ```
    #[unstable(feature = "array_try_map", issue = "79711")]
    pub fn try_map<F, R>(self, f: F) -> ChangeOutputType<R, [R::Output; N]>
    where
        F: FnMut(T) -> R,
        R: Try,
        R::Residual: Residual<[R::Output; N]>,
    {
        // SAFETY: we know for certain that this iterator will yield exactly `N`
        // items.
        unsafe { try_collect_into_array_unchecked(&mut IntoIterator::into_iter(self).map(f)) }
    }

    /// 'Zips up' two arrays into a single array of pairs.
    ///
    /// `zip()` returns a new array where every element is a tuple where the
    /// first element comes from the first array, and the second element comes
    /// from the second array. In other words, it zips two arrays together,
    /// into a single one.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_zip)]
    /// let x = [1, 2, 3];
    /// let y = [4, 5, 6];
    /// let z = x.zip(y);
    /// assert_eq!(z, [(1, 4), (2, 5), (3, 6)]);
    /// ```
    #[unstable(feature = "array_zip", issue = "80094")]
    pub fn zip<U>(self, rhs: [U; N]) -> [(T, U); N] {
        let mut iter = IntoIterator::into_iter(self).zip(rhs);

        // SAFETY: we know for certain that this iterator will yield exactly `N`
        // items.
        unsafe { collect_into_array_unchecked(&mut iter) }
    }

    /// Returns a slice containing the entire array. Equivalent to `&s[..]`.
    #[stable(feature = "array_as_slice", since = "1.57.0")]
    #[rustc_const_stable(feature = "array_as_slice", since = "1.57.0")]
    pub const fn as_slice(&self) -> &[T] {
        self
    }

    /// Returns a mutable slice containing the entire array. Equivalent to
    /// `&mut s[..]`.
    #[stable(feature = "array_as_slice", since = "1.57.0")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Borrows each element and returns an array of references with the same
    /// size as `self`.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(array_methods)]
    ///
    /// let floats = [3.1, 2.7, -1.0];
    /// let float_refs: [&f64; 3] = floats.each_ref();
    /// assert_eq!(float_refs, [&3.1, &2.7, &-1.0]);
    /// ```
    ///
    /// This method is particularly useful if combined with other methods, like
    /// [`map`](#method.map). This way, you can avoid moving the original
    /// array if its elements are not [`Copy`].
    ///
    /// ```
    /// #![feature(array_methods)]
    ///
    /// let strings = ["Ferris".to_string(), "♥".to_string(), "Rust".to_string()];
    /// let is_ascii = strings.each_ref().map(|s| s.is_ascii());
    /// assert_eq!(is_ascii, [true, false, true]);
    ///
    /// // We can still access the original array: it has not been moved.
    /// assert_eq!(strings.len(), 3);
    /// ```
    #[unstable(feature = "array_methods", issue = "76118")]
    pub fn each_ref(&self) -> [&T; N] {
        // SAFETY: we know for certain that this iterator will yield exactly `N`
        // items.
        unsafe { collect_into_array_unchecked(&mut self.iter()) }
    }

    /// Borrows each element mutably and returns an array of mutable references
    /// with the same size as `self`.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(array_methods)]
    ///
    /// let mut floats = [3.1, 2.7, -1.0];
    /// let float_refs: [&mut f64; 3] = floats.each_mut();
    /// *float_refs[0] = 0.0;
    /// assert_eq!(float_refs, [&mut 0.0, &mut 2.7, &mut -1.0]);
    /// assert_eq!(floats, [0.0, 2.7, -1.0]);
    /// ```
    #[unstable(feature = "array_methods", issue = "76118")]
    pub fn each_mut(&mut self) -> [&mut T; N] {
        // SAFETY: we know for certain that this iterator will yield exactly `N`
        // items.
        unsafe { collect_into_array_unchecked(&mut self.iter_mut()) }
    }

    /// Divides one array reference into two at an index.
    ///
    /// The first will contain all indices from `[0, M)` (excluding
    /// the index `M` itself) and the second will contain all
    /// indices from `[M, N)` (excluding the index `N` itself).
    ///
    /// # Panics
    ///
    /// Panics if `M > N`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(split_array)]
    ///
    /// let v = [1, 2, 3, 4, 5, 6];
    ///
    /// {
    ///    let (left, right) = v.split_array_ref::<0>();
    ///    assert_eq!(left, &[]);
    ///    assert_eq!(right, &[1, 2, 3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_array_ref::<2>();
    ///     assert_eq!(left, &[1, 2]);
    ///     assert_eq!(right, &[3, 4, 5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.split_array_ref::<6>();
    ///     assert_eq!(left, &[1, 2, 3, 4, 5, 6]);
    ///     assert_eq!(right, &[]);
    /// }
    /// ```
    #[unstable(
        feature = "split_array",
        reason = "return type should have array as 2nd element",
        issue = "90091"
    )]
    #[inline]
    pub fn split_array_ref<const M: usize>(&self) -> (&[T; M], &[T]) {
        (&self[..]).split_array_ref::<M>()
    }

    /// Divides one mutable array reference into two at an index.
    ///
    /// The first will contain all indices from `[0, M)` (excluding
    /// the index `M` itself) and the second will contain all
    /// indices from `[M, N)` (excluding the index `N` itself).
    ///
    /// # Panics
    ///
    /// Panics if `M > N`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(split_array)]
    ///
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// let (left, right) = v.split_array_mut::<2>();
    /// assert_eq!(left, &mut [1, 0][..]);
    /// assert_eq!(right, &mut [3, 0, 5, 6]);
    /// left[1] = 2;
    /// right[1] = 4;
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[unstable(
        feature = "split_array",
        reason = "return type should have array as 2nd element",
        issue = "90091"
    )]
    #[inline]
    pub fn split_array_mut<const M: usize>(&mut self) -> (&mut [T; M], &mut [T]) {
        (&mut self[..]).split_array_mut::<M>()
    }

    /// Divides one array reference into two at an index from the end.
    ///
    /// The first will contain all indices from `[0, N - M)` (excluding
    /// the index `N - M` itself) and the second will contain all
    /// indices from `[N - M, N)` (excluding the index `N` itself).
    ///
    /// # Panics
    ///
    /// Panics if `M > N`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(split_array)]
    ///
    /// let v = [1, 2, 3, 4, 5, 6];
    ///
    /// {
    ///    let (left, right) = v.rsplit_array_ref::<0>();
    ///    assert_eq!(left, &[1, 2, 3, 4, 5, 6]);
    ///    assert_eq!(right, &[]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.rsplit_array_ref::<2>();
    ///     assert_eq!(left, &[1, 2, 3, 4]);
    ///     assert_eq!(right, &[5, 6]);
    /// }
    ///
    /// {
    ///     let (left, right) = v.rsplit_array_ref::<6>();
    ///     assert_eq!(left, &[]);
    ///     assert_eq!(right, &[1, 2, 3, 4, 5, 6]);
    /// }
    /// ```
    #[unstable(
        feature = "split_array",
        reason = "return type should have array as 2nd element",
        issue = "90091"
    )]
    #[inline]
    pub fn rsplit_array_ref<const M: usize>(&self) -> (&[T], &[T; M]) {
        (&self[..]).rsplit_array_ref::<M>()
    }

    /// Divides one mutable array reference into two at an index from the end.
    ///
    /// The first will contain all indices from `[0, N - M)` (excluding
    /// the index `N - M` itself) and the second will contain all
    /// indices from `[N - M, N)` (excluding the index `N` itself).
    ///
    /// # Panics
    ///
    /// Panics if `M > N`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(split_array)]
    ///
    /// let mut v = [1, 0, 3, 0, 5, 6];
    /// let (left, right) = v.rsplit_array_mut::<4>();
    /// assert_eq!(left, &mut [1, 0]);
    /// assert_eq!(right, &mut [3, 0, 5, 6][..]);
    /// left[1] = 2;
    /// right[1] = 4;
    /// assert_eq!(v, [1, 2, 3, 4, 5, 6]);
    /// ```
    #[unstable(
        feature = "split_array",
        reason = "return type should have array as 2nd element",
        issue = "90091"
    )]
    #[inline]
    pub fn rsplit_array_mut<const M: usize>(&mut self) -> (&mut [T], &mut [T; M]) {
        (&mut self[..]).rsplit_array_mut::<M>()
    }
}

/// Pulls `N` items from `iter` and returns them as an array. If the iterator
/// yields fewer than `N` items, this function exhibits undefined behavior.
///
/// See [`try_collect_into_array`] for more information.
///
///
/// # Safety
///
/// It is up to the caller to guarantee that `iter` yields at least `N` items.
/// Violating this condition causes undefined behavior.
unsafe fn try_collect_into_array_unchecked<I, T, R, const N: usize>(iter: &mut I) -> R::TryType
where
    // Note: `TrustedLen` here is somewhat of an experiment. This is just an
    // internal function, so feel free to remove if this bound turns out to be a
    // bad idea. In that case, remember to also remove the lower bound
    // `debug_assert!` below!
    I: Iterator + TrustedLen,
    I::Item: Try<Output = T, Residual = R>,
    R: Residual<[T; N]>,
{
    debug_assert!(N <= iter.size_hint().1.unwrap_or(usize::MAX));
    debug_assert!(N <= iter.size_hint().0);

    // SAFETY: covered by the function contract.
    unsafe { try_collect_into_array(iter).unwrap_unchecked() }
}

// Infallible version of `try_collect_into_array_unchecked`.
unsafe fn collect_into_array_unchecked<I, const N: usize>(iter: &mut I) -> [I::Item; N]
where
    I: Iterator + TrustedLen,
{
    let mut map = iter.map(NeverShortCircuit);

    // SAFETY: The same safety considerations w.r.t. the iterator length
    // apply for `try_collect_into_array_unchecked` as for
    // `collect_into_array_unchecked`
    match unsafe { try_collect_into_array_unchecked(&mut map) } {
        NeverShortCircuit(array) => array,
    }
}

/// Pulls `N` items from `iter` and returns them as an array. If the iterator
/// yields fewer than `N` items, `Err` is returned containing an iterator over
/// the already yielded items.
///
/// Since the iterator is passed as a mutable reference and this function calls
/// `next` at most `N` times, the iterator can still be used afterwards to
/// retrieve the remaining items.
///
/// If `iter.next()` panicks, all items already yielded by the iterator are
/// dropped.
#[inline]
fn try_collect_into_array<I, T, R, const N: usize>(
    iter: &mut I,
) -> Result<R::TryType, IntoIter<T, N>>
where
    I: Iterator,
    I::Item: Try<Output = T, Residual = R>,
    R: Residual<[T; N]>,
{
    if N == 0 {
        // SAFETY: An empty array is always inhabited and has no validity invariants.
        return Ok(Try::from_output(unsafe { mem::zeroed() }));
    }

    struct Guard<'a, T, const N: usize> {
        array_mut: &'a mut [MaybeUninit<T>; N],
        initialized: usize,
    }

    impl<T, const N: usize> Drop for Guard<'_, T, N> {
        fn drop(&mut self) {
            debug_assert!(self.initialized <= N);

            // SAFETY: this slice will contain only initialized objects.
            unsafe {
                crate::ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(
                    &mut self.array_mut.get_unchecked_mut(..self.initialized),
                ));
            }
        }
    }

    let mut array = MaybeUninit::uninit_array::<N>();
    let mut guard = Guard { array_mut: &mut array, initialized: 0 };

    for _ in 0..N {
        match iter.next() {
            Some(item_rslt) => {
                let item = match item_rslt.branch() {
                    ControlFlow::Break(r) => {
                        return Ok(FromResidual::from_residual(r));
                    }
                    ControlFlow::Continue(elem) => elem,
                };

                // SAFETY: `guard.initialized` starts at 0, is increased by one in the
                // loop and the loop is aborted once it reaches N (which is
                // `array.len()`).
                unsafe {
                    guard.array_mut.get_unchecked_mut(guard.initialized).write(item);
                }
                guard.initialized += 1;
            }
            None => {
                let alive = 0..guard.initialized;
                mem::forget(guard);
                // SAFETY: `array` was initialized with exactly `initialized`
                // number of elements.
                return Err(unsafe { IntoIter::new_unchecked(array, alive) });
            }
        }
    }

    mem::forget(guard);
    // SAFETY: All elements of the array were populated in the loop above.
    let output = unsafe { MaybeUninit::array_assume_init(array) };
    Ok(Try::from_output(output))
}

/// Returns the next chunk of `N` items from the iterator or errors with an
/// iterator over the remainder. Used for `Iterator::next_chunk`.
#[inline]
pub(crate) fn iter_next_chunk<I, const N: usize>(
    iter: &mut I,
) -> Result<[I::Item; N], IntoIter<I::Item, N>>
where
    I: Iterator,
{
    let mut map = iter.map(NeverShortCircuit);
    try_collect_into_array(&mut map).map(|NeverShortCircuit(arr)| arr)
}
