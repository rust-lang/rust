//! Helper functions and types for fixed-length arrays.
//!
//! *[See also the array primitive type](array).*

#![stable(feature = "core_array", since = "1.36.0")]

use crate::borrow::{Borrow, BorrowMut};
use crate::cmp::Ordering;
use crate::convert::{Infallible, TryFrom};
use crate::fmt;
use crate::hash::{self, Hash};
use crate::iter::TrustedLen;
use crate::mem::{self, MaybeUninit};
use crate::ops::{Index, IndexMut};
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
/// #![feature(array_from_fn)]
///
/// let array = core::array::from_fn(|i| i);
/// assert_eq!(array, [0, 1, 2, 3, 4]);
/// ```
#[inline]
#[unstable(feature = "array_from_fn", issue = "89379")]
pub fn from_fn<F, T, const N: usize>(mut cb: F) -> [T; N]
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
/// Unlike `core::array::from_fn`, where the element creation can't fail, this version will return an error
/// if any element creation was unsuccessful.
///
/// # Arguments
///
/// * `cb`: Callback where the passed argument is the current array index.
///
/// # Example
///
/// ```rust
/// #![feature(array_from_fn)]
///
/// #[derive(Debug, PartialEq)]
/// enum SomeError {
///     Foo,
/// }
///
/// let array = core::array::try_from_fn(|i| Ok::<_, SomeError>(i));
/// assert_eq!(array, Ok([0, 1, 2, 3, 4]));
///
/// let another_array = core::array::try_from_fn::<SomeError, _, (), 2>(|_| Err(SomeError::Foo));
/// assert_eq!(another_array, Err(SomeError::Foo));
/// ```
#[inline]
#[unstable(feature = "array_from_fn", issue = "89379")]
pub fn try_from_fn<E, F, T, const N: usize>(cb: F) -> Result<[T; N], E>
where
    F: FnMut(usize) -> Result<T, E>,
{
    // SAFETY: we know for certain that this iterator will yield exactly `N`
    // items.
    unsafe { collect_into_array_rslt_unchecked(&mut (0..N).map(cb)) }
}

/// Converts a reference to `T` into a reference to an array of length 1 (without copying).
#[stable(feature = "array_from_ref", since = "1.53.0")]
pub fn from_ref<T>(s: &T) -> &[T; 1] {
    // SAFETY: Converting `&T` to `&[T; 1]` is sound.
    unsafe { &*(s as *const T).cast::<[T; 1]>() }
}

/// Converts a mutable reference to `T` into a mutable reference to an array of length 1 (without copying).
#[stable(feature = "array_from_ref", since = "1.53.0")]
pub fn from_mut<T>(s: &mut T) -> &mut [T; 1] {
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

// Note: the `#[rustc_skip_array_during_method_dispatch]` on `trait IntoIterator`
// hides this implementation from explicit `.into_iter()` calls on editions < 2021,
// so those calls will still resolve to the slice implementation, by reference.
#[stable(feature = "array_into_iter_impl", since = "1.53.0")]
impl<T, const N: usize> IntoIterator for [T; N] {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the array (from start to end). The array cannot be used after calling
    /// this unless `T` implements `Copy`, so the whole array is copied.
    ///
    /// Arrays have special behavior when calling `.into_iter()` prior to the
    /// 2021 edition -- see the [array] Editions section for more information.
    ///
    /// [array]: prim@array
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
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
impl<T, I, const N: usize> Index<I> for [T; N]
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self as &[T], index)
    }
}

#[stable(feature = "index_trait_on_arrays", since = "1.50.0")]
impl<T, I, const N: usize> IndexMut<I> for [T; N]
where
    [T]: IndexMut<I>,
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

#[lang = "array"]
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
}

/// Pulls `N` items from `iter` and returns them as an array. If the iterator
/// yields fewer than `N` items, this function exhibits undefined behavior.
///
/// See [`collect_into_array`] for more information.
///
///
/// # Safety
///
/// It is up to the caller to guarantee that `iter` yields at least `N` items.
/// Violating this condition causes undefined behavior.
unsafe fn collect_into_array_rslt_unchecked<E, I, T, const N: usize>(
    iter: &mut I,
) -> Result<[T; N], E>
where
    // Note: `TrustedLen` here is somewhat of an experiment. This is just an
    // internal function, so feel free to remove if this bound turns out to be a
    // bad idea. In that case, remember to also remove the lower bound
    // `debug_assert!` below!
    I: Iterator<Item = Result<T, E>> + TrustedLen,
{
    debug_assert!(N <= iter.size_hint().1.unwrap_or(usize::MAX));
    debug_assert!(N <= iter.size_hint().0);

    // SAFETY: covered by the function contract.
    unsafe { collect_into_array(iter).unwrap_unchecked() }
}

// Infallible version of `collect_into_array_rslt_unchecked`.
unsafe fn collect_into_array_unchecked<I, const N: usize>(iter: &mut I) -> [I::Item; N]
where
    I: Iterator + TrustedLen,
{
    let mut map = iter.map(Ok::<_, Infallible>);

    // SAFETY: The same safety considerations w.r.t. the iterator length
    // apply for `collect_into_array_rslt_unchecked` as for
    // `collect_into_array_unchecked`
    match unsafe { collect_into_array_rslt_unchecked(&mut map) } {
        Ok(array) => array,
    }
}

/// Pulls `N` items from `iter` and returns them as an array. If the iterator
/// yields fewer than `N` items, `None` is returned and all already yielded
/// items are dropped.
///
/// Since the iterator is passed as a mutable reference and this function calls
/// `next` at most `N` times, the iterator can still be used afterwards to
/// retrieve the remaining items.
///
/// If `iter.next()` panicks, all items already yielded by the iterator are
/// dropped.
fn collect_into_array<E, I, T, const N: usize>(iter: &mut I) -> Option<Result<[T; N], E>>
where
    I: Iterator<Item = Result<T, E>>,
{
    if N == 0 {
        // SAFETY: An empty array is always inhabited and has no validity invariants.
        return unsafe { Some(Ok(mem::zeroed())) };
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

    while let Some(item_rslt) = iter.next() {
        let item = match item_rslt {
            Err(err) => {
                return Some(Err(err));
            }
            Ok(elem) => elem,
        };

        // SAFETY: `guard.initialized` starts at 0, is increased by one in the
        // loop and the loop is aborted once it reaches N (which is
        // `array.len()`).
        unsafe {
            guard.array_mut.get_unchecked_mut(guard.initialized).write(item);
        }
        guard.initialized += 1;

        // Check if the whole array was initialized.
        if guard.initialized == N {
            mem::forget(guard);

            // SAFETY: the condition above asserts that all elements are
            // initialized.
            let out = unsafe { MaybeUninit::array_assume_init(array) };
            return Some(Ok(out));
        }
    }

    // This is only reached if the iterator is exhausted before
    // `guard.initialized` reaches `N`. Also note that `guard` is dropped here,
    // dropping all already initialized elements.
    None
}
