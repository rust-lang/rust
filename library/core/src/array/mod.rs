//! Utilities for the array primitive type.
//!
//! *[See also the array primitive type](array).*

#![stable(feature = "core_array", since = "1.35.0")]

use crate::borrow::{Borrow, BorrowMut};
use crate::cmp::Ordering;
use crate::convert::Infallible;
use crate::error::Error;
use crate::fmt;
use crate::hash::{self, Hash};
use crate::intrinsics::transmute_unchecked;
use crate::iter::{UncheckedIterator, repeat_n};
use crate::mem::{self, MaybeUninit};
use crate::ops::{
    ChangeOutputType, ControlFlow, FromResidual, Index, IndexMut, NeverShortCircuit, Residual, Try,
};
use crate::ptr::{null, null_mut};
use crate::slice::{Iter, IterMut};

mod ascii;
mod drain;
mod equality;
mod iter;

pub(crate) use drain::drain_array_with;
#[stable(feature = "array_value_iter", since = "1.51.0")]
pub use iter::IntoIter;

/// Creates an array of type `[T; N]` by repeatedly cloning a value.
///
/// This is the same as `[val; N]`, but it also works for types that do not
/// implement [`Copy`].
///
/// The provided value will be used as an element of the resulting array and
/// will be cloned N - 1 times to fill up the rest. If N is zero, the value
/// will be dropped.
///
/// # Example
///
/// Creating multiple copies of a `String`:
/// ```rust
/// #![feature(array_repeat)]
///
/// use std::array;
///
/// let string = "Hello there!".to_string();
/// let strings = array::repeat(string);
/// assert_eq!(strings, ["Hello there!", "Hello there!"]);
/// ```
#[inline]
#[unstable(feature = "array_repeat", issue = "126695")]
pub fn repeat<T: Clone, const N: usize>(val: T) -> [T; N] {
    from_trusted_iterator(repeat_n(val, N))
}

/// Creates an array where each element is produced by calling `f` with
/// that element's index while walking forward through the array.
///
/// This is essentially the same as writing
/// ```text
/// [f(0), f(1), f(2), …, f(N - 2), f(N - 1)]
/// ```
/// and is similar to `(0..i).map(f)`, just for arrays not iterators.
///
/// If `N == 0`, this produces an empty array without ever calling `f`.
///
/// # Example
///
/// ```rust
/// // type inference is helping us here, the way `from_fn` knows how many
/// // elements to produce is the length of array down there: only arrays of
/// // equal lengths can be compared, so the const generic parameter `N` is
/// // inferred to be 5, thus creating array of 5 elements.
///
/// let array = core::array::from_fn(|i| i);
/// // indexes are:    0  1  2  3  4
/// assert_eq!(array, [0, 1, 2, 3, 4]);
///
/// let array2: [usize; 8] = core::array::from_fn(|i| i * 2);
/// // indexes are:     0  1  2  3  4  5   6   7
/// assert_eq!(array2, [0, 2, 4, 6, 8, 10, 12, 14]);
///
/// let bool_arr = core::array::from_fn::<_, 5, _>(|i| i % 2 == 0);
/// // indexes are:       0     1      2     3      4
/// assert_eq!(bool_arr, [true, false, true, false, true]);
/// ```
///
/// You can also capture things, for example to create an array full of clones
/// where you can't just use `[item; N]` because it's not `Copy`:
/// ```
/// # // TBH `array::repeat` would be better for this, but it's not stable yet.
/// let my_string = String::from("Hello");
/// let clones: [String; 42] = std::array::from_fn(|_| my_string.clone());
/// assert!(clones.iter().all(|x| *x == my_string));
/// ```
///
/// The array is generated in ascending index order, starting from the front
/// and going towards the back, so you can use closures with mutable state:
/// ```
/// let mut state = 1;
/// let a = std::array::from_fn(|_| { let x = state; state *= 2; x });
/// assert_eq!(a, [1, 2, 4, 8, 16, 32]);
/// ```
#[inline]
#[stable(feature = "array_from_fn", since = "1.63.0")]
pub fn from_fn<T, const N: usize, F>(f: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    try_from_fn(NeverShortCircuit::wrap_mut_1(f)).0
}

/// Creates an array `[T; N]` where each fallible array element `T` is returned by the `cb` call.
/// Unlike [`from_fn`], where the element creation can't fail, this version will return an error
/// if any element creation was unsuccessful.
///
/// The return type of this function depends on the return type of the closure.
/// If you return `Result<T, E>` from the closure, you'll get a `Result<[T; N], E>`.
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
    let mut array = [const { MaybeUninit::uninit() }; N];
    match try_from_fn_erased(&mut array, cb) {
        ControlFlow::Break(r) => FromResidual::from_residual(r),
        ControlFlow::Continue(()) => {
            // SAFETY: All elements of the array were populated.
            try { unsafe { MaybeUninit::array_assume_init(array) } }
        }
    }
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
#[rustc_const_stable(feature = "const_array_from_ref", since = "1.83.0")]
pub const fn from_mut<T>(s: &mut T) -> &mut [T; 1] {
    // SAFETY: Converting `&mut T` to `&mut [T; 1]` is sound.
    unsafe { &mut *(s as *mut T).cast::<[T; 1]>() }
}

/// The error type returned when a conversion from a slice to an array fails.
#[stable(feature = "try_from", since = "1.34.0")]
#[derive(Debug, Copy, Clone)]
pub struct TryFromSliceError(());

#[stable(feature = "core_array", since = "1.35.0")]
impl fmt::Display for TryFromSliceError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(deprecated)]
        self.description().fmt(f)
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl Error for TryFromSliceError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
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

/// Tries to create an array `[T; N]` by copying from a slice `&[T]`.
/// Succeeds if `slice.len() == N`.
///
/// ```
/// let bytes: [u8; 3] = [1, 0, 2];
///
/// let bytes_head: [u8; 2] = <[u8; 2]>::try_from(&bytes[0..2]).unwrap();
/// assert_eq!(1, u16::from_le_bytes(bytes_head));
///
/// let bytes_tail: [u8; 2] = bytes[1..3].try_into().unwrap();
/// assert_eq!(512, u16::from_le_bytes(bytes_tail));
/// ```
#[stable(feature = "try_from", since = "1.34.0")]
impl<T, const N: usize> TryFrom<&[T]> for [T; N]
where
    T: Copy,
{
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(slice: &[T]) -> Result<[T; N], TryFromSliceError> {
        <&Self>::try_from(slice).copied()
    }
}

/// Tries to create an array `[T; N]` by copying from a mutable slice `&mut [T]`.
/// Succeeds if `slice.len() == N`.
///
/// ```
/// let mut bytes: [u8; 3] = [1, 0, 2];
///
/// let bytes_head: [u8; 2] = <[u8; 2]>::try_from(&mut bytes[0..2]).unwrap();
/// assert_eq!(1, u16::from_le_bytes(bytes_head));
///
/// let bytes_tail: [u8; 2] = (&mut bytes[1..3]).try_into().unwrap();
/// assert_eq!(512, u16::from_le_bytes(bytes_tail));
/// ```
#[stable(feature = "try_from_mut_slice_to_array", since = "1.59.0")]
impl<T, const N: usize> TryFrom<&mut [T]> for [T; N]
where
    T: Copy,
{
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(slice: &mut [T]) -> Result<[T; N], TryFromSliceError> {
        <Self>::try_from(&*slice)
    }
}

/// Tries to create an array ref `&[T; N]` from a slice ref `&[T]`. Succeeds if
/// `slice.len() == N`.
///
/// ```
/// let bytes: [u8; 3] = [1, 0, 2];
///
/// let bytes_head: &[u8; 2] = <&[u8; 2]>::try_from(&bytes[0..2]).unwrap();
/// assert_eq!(1, u16::from_le_bytes(*bytes_head));
///
/// let bytes_tail: &[u8; 2] = bytes[1..3].try_into().unwrap();
/// assert_eq!(512, u16::from_le_bytes(*bytes_tail));
/// ```
#[stable(feature = "try_from", since = "1.34.0")]
impl<'a, T, const N: usize> TryFrom<&'a [T]> for &'a [T; N] {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(slice: &'a [T]) -> Result<&'a [T; N], TryFromSliceError> {
        slice.as_array().ok_or(TryFromSliceError(()))
    }
}

/// Tries to create a mutable array ref `&mut [T; N]` from a mutable slice ref
/// `&mut [T]`. Succeeds if `slice.len() == N`.
///
/// ```
/// let mut bytes: [u8; 3] = [1, 0, 2];
///
/// let bytes_head: &mut [u8; 2] = <&mut [u8; 2]>::try_from(&mut bytes[0..2]).unwrap();
/// assert_eq!(1, u16::from_le_bytes(*bytes_head));
///
/// let bytes_tail: &mut [u8; 2] = (&mut bytes[1..3]).try_into().unwrap();
/// assert_eq!(512, u16::from_le_bytes(*bytes_tail));
/// ```
#[stable(feature = "try_from", since = "1.34.0")]
impl<'a, T, const N: usize> TryFrom<&'a mut [T]> for &'a mut [T; N] {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(slice: &'a mut [T]) -> Result<&'a mut [T; N], TryFromSliceError> {
        slice.as_mut_array().ok_or(TryFromSliceError(()))
    }
}

/// The hash of an array is the same as that of the corresponding slice,
/// as required by the `Borrow` implementation.
///
/// ```
/// use std::hash::BuildHasher;
///
/// let b = std::hash::RandomState::new();
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

/// Implements comparison of arrays [lexicographically](Ord#lexicographical-comparison).
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
        from_trusted_iterator(array.iter().cloned())
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
        impl<T> Default for [T; $n] {
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
    #[must_use]
    #[stable(feature = "array_map", since = "1.55.0")]
    pub fn map<F, U>(self, f: F) -> [U; N]
    where
        F: FnMut(T) -> U,
    {
        self.try_map(NeverShortCircuit::wrap_mut_1(f)).0
    }

    /// A fallible function `f` applied to each element on array `self` in order to
    /// return an array the same size as `self` or the first error encountered.
    ///
    /// The return type of this function depends on the return type of the closure.
    /// If you return `Result<T, E>` from the closure, you'll get a `Result<[T; N], E>`.
    /// If you return `Option<T>` from the closure, you'll get an `Option<[T; N]>`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(array_try_map)]
    ///
    /// let a = ["1", "2", "3"];
    /// let b = a.try_map(|v| v.parse::<u32>()).unwrap().map(|v| v + 1);
    /// assert_eq!(b, [2, 3, 4]);
    ///
    /// let a = ["1", "2a", "3"];
    /// let b = a.try_map(|v| v.parse::<u32>());
    /// assert!(b.is_err());
    ///
    /// use std::num::NonZero;
    ///
    /// let z = [1, 2, 0, 3, 4];
    /// assert_eq!(z.try_map(NonZero::new), None);
    ///
    /// let a = [1, 2, 3];
    /// let b = a.try_map(NonZero::new);
    /// let c = b.map(|x| x.map(NonZero::get));
    /// assert_eq!(c, Some(a));
    /// ```
    #[unstable(feature = "array_try_map", issue = "79711")]
    pub fn try_map<R>(self, f: impl FnMut(T) -> R) -> ChangeOutputType<R, [R::Output; N]>
    where
        R: Try<Residual: Residual<[R::Output; N]>>,
    {
        drain_array_with(self, |iter| try_from_trusted_iterator(iter.map(f)))
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
    #[rustc_const_stable(feature = "const_array_as_mut_slice", since = "1.89.0")]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Borrows each element and returns an array of references with the same
    /// size as `self`.
    ///
    ///
    /// # Example
    ///
    /// ```
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
    /// let strings = ["Ferris".to_string(), "♥".to_string(), "Rust".to_string()];
    /// let is_ascii = strings.each_ref().map(|s| s.is_ascii());
    /// assert_eq!(is_ascii, [true, false, true]);
    ///
    /// // We can still access the original array: it has not been moved.
    /// assert_eq!(strings.len(), 3);
    /// ```
    #[stable(feature = "array_methods", since = "1.77.0")]
    #[rustc_const_unstable(feature = "const_array_each_ref", issue = "133289")]
    pub const fn each_ref(&self) -> [&T; N] {
        let mut buf = [null::<T>(); N];

        // FIXME(const-hack): We would like to simply use iterators for this (as in the original implementation), but this is not allowed in constant expressions.
        let mut i = 0;
        while i < N {
            buf[i] = &raw const self[i];

            i += 1;
        }

        // SAFETY: `*const T` has the same layout as `&T`, and we've also initialised each pointer as a valid reference.
        unsafe { transmute_unchecked(buf) }
    }

    /// Borrows each element mutably and returns an array of mutable references
    /// with the same size as `self`.
    ///
    ///
    /// # Example
    ///
    /// ```
    ///
    /// let mut floats = [3.1, 2.7, -1.0];
    /// let float_refs: [&mut f64; 3] = floats.each_mut();
    /// *float_refs[0] = 0.0;
    /// assert_eq!(float_refs, [&mut 0.0, &mut 2.7, &mut -1.0]);
    /// assert_eq!(floats, [0.0, 2.7, -1.0]);
    /// ```
    #[stable(feature = "array_methods", since = "1.77.0")]
    #[rustc_const_unstable(feature = "const_array_each_ref", issue = "133289")]
    pub const fn each_mut(&mut self) -> [&mut T; N] {
        let mut buf = [null_mut::<T>(); N];

        // FIXME(const-hack): We would like to simply use iterators for this (as in the original implementation), but this is not allowed in constant expressions.
        let mut i = 0;
        while i < N {
            buf[i] = &raw mut self[i];

            i += 1;
        }

        // SAFETY: `*mut T` has the same layout as `&mut T`, and we've also initialised each pointer as a valid reference.
        unsafe { transmute_unchecked(buf) }
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
        (&self[..]).split_first_chunk::<M>().unwrap()
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
        (&mut self[..]).split_first_chunk_mut::<M>().unwrap()
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
        (&self[..]).split_last_chunk::<M>().unwrap()
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
        (&mut self[..]).split_last_chunk_mut::<M>().unwrap()
    }
}

/// Populate an array from the first `N` elements of `iter`
///
/// # Panics
///
/// If the iterator doesn't actually have enough items.
///
/// By depending on `TrustedLen`, however, we can do that check up-front (where
/// it easily optimizes away) so it doesn't impact the loop that fills the array.
#[inline]
fn from_trusted_iterator<T, const N: usize>(iter: impl UncheckedIterator<Item = T>) -> [T; N] {
    try_from_trusted_iterator(iter.map(NeverShortCircuit)).0
}

#[inline]
fn try_from_trusted_iterator<T, R, const N: usize>(
    iter: impl UncheckedIterator<Item = R>,
) -> ChangeOutputType<R, [T; N]>
where
    R: Try<Output = T>,
    R::Residual: Residual<[T; N]>,
{
    assert!(iter.size_hint().0 >= N);
    fn next<T>(mut iter: impl UncheckedIterator<Item = T>) -> impl FnMut(usize) -> T {
        move |_| {
            // SAFETY: We know that `from_fn` will call this at most N times,
            // and we checked to ensure that we have at least that many items.
            unsafe { iter.next_unchecked() }
        }
    }

    try_from_fn(next(iter))
}

/// Version of [`try_from_fn`] using a passed-in slice in order to avoid
/// needing to monomorphize for every array length.
///
/// This takes a generator rather than an iterator so that *at the type level*
/// it never needs to worry about running out of items.  When combined with
/// an infallible `Try` type, that means the loop canonicalizes easily, allowing
/// it to optimize well.
///
/// It would be *possible* to unify this and [`iter_next_chunk_erased`] into one
/// function that does the union of both things, but last time it was that way
/// it resulted in poor codegen from the "are there enough source items?" checks
/// not optimizing away.  So if you give it a shot, make sure to watch what
/// happens in the codegen tests.
#[inline]
fn try_from_fn_erased<T, R>(
    buffer: &mut [MaybeUninit<T>],
    mut generator: impl FnMut(usize) -> R,
) -> ControlFlow<R::Residual>
where
    R: Try<Output = T>,
{
    let mut guard = Guard { array_mut: buffer, initialized: 0 };

    while guard.initialized < guard.array_mut.len() {
        let item = generator(guard.initialized).branch()?;

        // SAFETY: The loop condition ensures we have space to push the item
        unsafe { guard.push_unchecked(item) };
    }

    mem::forget(guard);
    ControlFlow::Continue(())
}

/// Panic guard for incremental initialization of arrays.
///
/// Disarm the guard with `mem::forget` once the array has been initialized.
///
/// # Safety
///
/// All write accesses to this structure are unsafe and must maintain a correct
/// count of `initialized` elements.
///
/// To minimize indirection fields are still pub but callers should at least use
/// `push_unchecked` to signal that something unsafe is going on.
struct Guard<'a, T> {
    /// The array to be initialized.
    pub array_mut: &'a mut [MaybeUninit<T>],
    /// The number of items that have been initialized so far.
    pub initialized: usize,
}

impl<T> Guard<'_, T> {
    /// Adds an item to the array and updates the initialized item counter.
    ///
    /// # Safety
    ///
    /// No more than N elements must be initialized.
    #[inline]
    pub(crate) unsafe fn push_unchecked(&mut self, item: T) {
        // SAFETY: If `initialized` was correct before and the caller does not
        // invoke this method more than N times then writes will be in-bounds
        // and slots will not be initialized more than once.
        unsafe {
            self.array_mut.get_unchecked_mut(self.initialized).write(item);
            self.initialized = self.initialized.unchecked_add(1);
        }
    }
}

impl<T> Drop for Guard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        debug_assert!(self.initialized <= self.array_mut.len());

        // SAFETY: this slice will contain only initialized objects.
        unsafe {
            self.array_mut.get_unchecked_mut(..self.initialized).assume_init_drop();
        }
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
///
/// Used for [`Iterator::next_chunk`].
#[inline]
pub(crate) fn iter_next_chunk<T, const N: usize>(
    iter: &mut impl Iterator<Item = T>,
) -> Result<[T; N], IntoIter<T, N>> {
    let mut array = [const { MaybeUninit::uninit() }; N];
    let r = iter_next_chunk_erased(&mut array, iter);
    match r {
        Ok(()) => {
            // SAFETY: All elements of `array` were populated.
            Ok(unsafe { MaybeUninit::array_assume_init(array) })
        }
        Err(initialized) => {
            // SAFETY: Only the first `initialized` elements were populated
            Err(unsafe { IntoIter::new_unchecked(array, 0..initialized) })
        }
    }
}

/// Version of [`iter_next_chunk`] using a passed-in slice in order to avoid
/// needing to monomorphize for every array length.
///
/// Unfortunately this loop has two exit conditions, the buffer filling up
/// or the iterator running out of items, making it tend to optimize poorly.
#[inline]
fn iter_next_chunk_erased<T>(
    buffer: &mut [MaybeUninit<T>],
    iter: &mut impl Iterator<Item = T>,
) -> Result<(), usize> {
    let mut guard = Guard { array_mut: buffer, initialized: 0 };
    while guard.initialized < guard.array_mut.len() {
        let Some(item) = iter.next() else {
            // Unlike `try_from_fn_erased`, we want to keep the partial results,
            // so we need to defuse the guard instead of using `?`.
            let initialized = guard.initialized;
            mem::forget(guard);
            return Err(initialized);
        };

        // SAFETY: The loop condition ensures we have space to push the item
        unsafe { guard.push_unchecked(item) };
    }

    mem::forget(guard);
    Ok(())
}
