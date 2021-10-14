//! Definitions of a bunch of iterators for `[T]`.

#[macro_use] // import iterator! and forward_iterator!
mod macros;

use crate::cmp;
use crate::cmp::Ordering;
use crate::fmt;
use crate::intrinsics::{assume, exact_div, unchecked_sub};
use crate::iter::{FusedIterator, TrustedLen, TrustedRandomAccess, TrustedRandomAccessNoCoerce};
use crate::marker::{PhantomData, Send, Sized, Sync};
use crate::mem;
use crate::num::NonZeroUsize;
use crate::ptr::NonNull;

use super::{from_raw_parts, from_raw_parts_mut};

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a [T] {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut [T] {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

// Macro helper functions
#[inline(always)]
fn size_from_ptr<T>(_: *const T) -> usize {
    mem::size_of::<T>()
}

/// Immutable slice iterator
///
/// This struct is created by the [`iter`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter` method to get the `Iter` struct (`&[usize]` here):
/// let slice = &[1, 2, 3];
///
/// // Then, we iterate over it:
/// for element in slice.iter() {
///     println!("{}", element);
/// }
/// ```
///
/// [`iter`]: slice::iter
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    ptr: NonNull<T>,
    end: *const T, // If T is a ZST, this is actually ptr+len.  This encoding is picked so that
    // ptr == end is a quick test for the Iterator being empty, that works
    // for both ZST and non-ZST.
    _marker: PhantomData<&'a T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.as_slice()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for Iter<'_, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Send for Iter<'_, T> {}

impl<'a, T> Iter<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T]) -> Self {
        let ptr = slice.as_ptr();
        // SAFETY: Similar to `IterMut::new`.
        unsafe {
            assume(!ptr.is_null());

            let end = if mem::size_of::<T>() == 0 {
                (ptr as *const u8).wrapping_add(slice.len()) as *const T
            } else {
                ptr.add(slice.len())
            };

            Self { ptr: NonNull::new_unchecked(ptr as *mut T), end, _marker: PhantomData }
        }
    }

    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has the `iter` method to get the `Iter`
    /// // struct (`&[usize]` here):
    /// let slice = &[1, 2, 3];
    ///
    /// // Then, we get the iterator:
    /// let mut iter = slice.iter();
    /// // So if we print what `as_slice` method returns here, we have "[1, 2, 3]":
    /// println!("{:?}", iter.as_slice());
    ///
    /// // Next, we move to the second element of the slice:
    /// iter.next();
    /// // Now `as_slice` returns "[2, 3]":
    /// println!("{:?}", iter.as_slice());
    /// ```
    #[must_use]
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn as_slice(&self) -> &'a [T] {
        self.make_slice()
    }
}

iterator! {struct Iter -> *const T, &'a T, const, {/* no mut */}, {
    fn is_sorted_by<F>(self, mut compare: F) -> bool
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
    {
        self.as_slice().windows(2).all(|w| {
            compare(&&w[0], &&w[1]).map(|o| o != Ordering::Greater).unwrap_or(false)
        })
    }
}}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { ptr: self.ptr, end: self.end, _marker: self._marker }
    }
}

#[stable(feature = "slice_iter_as_ref", since = "1.13.0")]
impl<T> AsRef<[T]> for Iter<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// Mutable slice iterator.
///
/// This struct is created by the [`iter_mut`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter_mut` method to get the `IterMut`
/// // struct (`&[usize]` here):
/// let mut slice = &mut [1, 2, 3];
///
/// // Then, we iterate over it and increment each element value:
/// for element in slice.iter_mut() {
///     *element += 1;
/// }
///
/// // We now have "[2, 3, 4]":
/// println!("{:?}", slice);
/// ```
///
/// [`iter_mut`]: slice::iter_mut
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    ptr: NonNull<T>,
    end: *mut T, // If T is a ZST, this is actually ptr+len.  This encoding is picked so that
    // ptr == end is a quick test for the Iterator being empty, that works
    // for both ZST and non-ZST.
    _marker: PhantomData<&'a mut T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IterMut").field(&self.make_slice()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for IterMut<'_, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for IterMut<'_, T> {}

impl<'a, T> IterMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice.as_mut_ptr();
        // SAFETY: There are several things here:
        //
        // `ptr` has been obtained by `slice.as_ptr()` where `slice` is a valid
        // reference thus it is non-NUL and safe to use and pass to
        // `NonNull::new_unchecked` .
        //
        // Adding `slice.len()` to the starting pointer gives a pointer
        // at the end of `slice`. `end` will never be dereferenced, only checked
        // for direct pointer equality with `ptr` to check if the iterator is
        // done.
        //
        // In the case of a ZST, the end pointer is just the start pointer plus
        // the length, to also allows for the fast `ptr == end` check.
        //
        // See the `next_unchecked!` and `is_empty!` macros as well as the
        // `post_inc_start` method for more informations.
        unsafe {
            assume(!ptr.is_null());

            let end = if mem::size_of::<T>() == 0 {
                (ptr as *mut u8).wrapping_add(slice.len()) as *mut T
            } else {
                ptr.add(slice.len())
            };

            Self { ptr: NonNull::new_unchecked(ptr), end, _marker: PhantomData }
        }
    }

    /// Views the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut` references that alias, this is forced
    /// to consume the iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has `iter_mut` method to get the `IterMut`
    /// // struct (`&[usize]` here):
    /// let mut slice = &mut [1, 2, 3];
    ///
    /// {
    ///     // Then, we get the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We move to next element:
    ///     iter.next();
    ///     // So if we print what `into_slice` method returns here, we have "[2, 3]":
    ///     println!("{:?}", iter.into_slice());
    /// }
    ///
    /// // Now let's modify a value of the slice:
    /// {
    ///     // First we get back the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We change the value of the first element of the slice returned by the `next` method:
    ///     *iter.next().unwrap() += 1;
    /// }
    /// // Now slice is "[2, 2, 3]":
    /// println!("{:?}", slice);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn into_slice(self) -> &'a mut [T] {
        // SAFETY: the iterator was created from a mutable slice with pointer
        // `self.ptr` and length `len!(self)`. This guarantees that all the prerequisites
        // for `from_raw_parts_mut` are fulfilled.
        unsafe { from_raw_parts_mut(self.ptr.as_ptr(), len!(self)) }
    }

    /// Views the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut [T]` references that alias, the returned slice
    /// borrows its lifetime from the iterator the method is applied on.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut slice: &mut [usize] = &mut [1, 2, 3];
    ///
    /// // First, we get the iterator:
    /// let mut iter = slice.iter_mut();
    /// // So if we check what the `as_slice` method returns here, we have "[1, 2, 3]":
    /// assert_eq!(iter.as_slice(), &[1, 2, 3]);
    ///
    /// // Next, we move to the second element of the slice:
    /// iter.next();
    /// // Now `as_slice` returns "[2, 3]":
    /// assert_eq!(iter.as_slice(), &[2, 3]);
    /// ```
    #[must_use]
    #[stable(feature = "slice_iter_mut_as_slice", since = "1.53.0")]
    pub fn as_slice(&self) -> &[T] {
        self.make_slice()
    }
}

#[stable(feature = "slice_iter_mut_as_slice", since = "1.53.0")]
impl<T> AsRef<[T]> for IterMut<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

iterator! {struct IterMut -> *mut T, &'a mut T, mut, {mut}, {}}

/// An internal abstraction over the splitting iterators, so that
/// splitn, splitn_mut etc can be implemented once.
#[doc(hidden)]
pub(super) trait SplitIter: DoubleEndedIterator {
    /// Marks the underlying iterator as complete, extracting the remaining
    /// portion of the slice.
    fn finish(&mut self) -> Option<Self::Item>;
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
///
/// This struct is created by the [`split`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = [10, 40, 33, 20];
/// let mut iter = slice.split(|num| num % 3 == 0);
/// ```
///
/// [`split`]: slice::split
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Split<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    // Used for `SplitWhitespace` and `SplitAsciiWhitespace` `as_str` methods
    pub(crate) v: &'a [T],
    pred: P,
    // Used for `SplitAsciiWhitespace` `as_str` method
    pub(crate) finished: bool,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> Split<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a [T], pred: P) -> Self {
        Self { v: slice, pred, finished: false }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for Split<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Split").field("v", &self.v).field("finished", &self.finished).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T, P> Clone for Split<'_, T, P>
where
    P: Clone + FnMut(&T) -> bool,
{
    fn clone(&self) -> Self {
        Split { v: self.v, pred: self.pred.clone(), finished: self.finished }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        match self.v.iter().position(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[..idx]);
                self.v = &self.v[idx + 1..];
                ret
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len() + 1` empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        match self.v.iter().rposition(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[idx + 1..]);
                self.v = &self.v[..idx];
                ret
            }
        }
    }
}

impl<'a, T, P> SplitIter for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(self.v)
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, P> FusedIterator for Split<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function. Unlike `Split`, it contains the matched part as a terminator
/// of the subslice.
///
/// This struct is created by the [`split_inclusive`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = [10, 40, 33, 20];
/// let mut iter = slice.split_inclusive(|num| num % 3 == 0);
/// ```
///
/// [`split_inclusive`]: slice::split_inclusive
/// [slices]: slice
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub struct SplitInclusive<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    v: &'a [T],
    pred: P,
    finished: bool,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> SplitInclusive<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a [T], pred: P) -> Self {
        Self { v: slice, pred, finished: false }
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitInclusive<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitInclusive")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<T, P> Clone for SplitInclusive<'_, T, P>
where
    P: Clone + FnMut(&T) -> bool,
{
    fn clone(&self) -> Self {
        SplitInclusive { v: self.v, pred: self.pred.clone(), finished: self.finished }
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, T, P> Iterator for SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        let idx =
            self.v.iter().position(|x| (self.pred)(x)).map(|idx| idx + 1).unwrap_or(self.v.len());
        if idx == self.v.len() {
            self.finished = true;
        }
        let ret = Some(&self.v[..idx]);
        self.v = &self.v[idx..];
        ret
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len()` one-element slices,
            // or a single empty slice.
            (1, Some(cmp::max(1, self.v.len())))
        }
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, T, P> DoubleEndedIterator for SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        // The last index of self.v is already checked and found to match
        // by the last iteration, so we start searching a new match
        // one index to the left.
        let remainder = if self.v.is_empty() { &[] } else { &self.v[..(self.v.len() - 1)] };
        let idx = remainder.iter().rposition(|x| (self.pred)(x)).map(|idx| idx + 1).unwrap_or(0);
        if idx == 0 {
            self.finished = true;
        }
        let ret = Some(&self.v[idx..]);
        self.v = &self.v[..idx];
        ret
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<T, P> FusedIterator for SplitInclusive<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the mutable subslices of the vector which are separated
/// by elements that match `pred`.
///
/// This struct is created by the [`split_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut v = [10, 40, 30, 20, 60, 50];
/// let iter = v.split_mut(|num| *num % 3 == 0);
/// ```
///
/// [`split_mut`]: slice::split_mut
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    v: &'a mut [T],
    pred: P,
    finished: bool,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> SplitMut<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], pred: P) -> Self {
        Self { v: slice, pred, finished: false }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitMut").field("v", &self.v).field("finished", &self.finished).finish()
    }
}

impl<'a, T, P> SplitIter for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(mem::replace(&mut self.v, &mut []))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().position(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = &mut tail[1..];
                Some(head)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len() + 1` empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().rposition(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = head;
                Some(&mut tail[1..])
            }
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, P> FusedIterator for SplitMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the mutable subslices of the vector which are separated
/// by elements that match `pred`. Unlike `SplitMut`, it contains the matched
/// parts in the ends of the subslices.
///
/// This struct is created by the [`split_inclusive_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut v = [10, 40, 30, 20, 60, 50];
/// let iter = v.split_inclusive_mut(|num| *num % 3 == 0);
/// ```
///
/// [`split_inclusive_mut`]: slice::split_inclusive_mut
/// [slices]: slice
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub struct SplitInclusiveMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    v: &'a mut [T],
    pred: P,
    finished: bool,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> SplitInclusiveMut<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], pred: P) -> Self {
        Self { v: slice, pred, finished: false }
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitInclusiveMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitInclusiveMut")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, T, P> Iterator for SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().position(|x| (*pred)(x))
        };
        let idx = idx_opt.map(|idx| idx + 1).unwrap_or(self.v.len());
        if idx == self.v.len() {
            self.finished = true;
        }
        let tmp = mem::replace(&mut self.v, &mut []);
        let (head, tail) = tmp.split_at_mut(idx);
        self.v = tail;
        Some(head)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // If the predicate doesn't match anything, we yield one slice.
            // If it matches every element, we yield `len()` one-element slices,
            // or a single empty slice.
            (1, Some(cmp::max(1, self.v.len())))
        }
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, T, P> DoubleEndedIterator for SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = if self.v.is_empty() {
            None
        } else {
            // work around borrowck limitations
            let pred = &mut self.pred;

            // The last index of self.v is already checked and found to match
            // by the last iteration, so we start searching a new match
            // one index to the left.
            let remainder = &self.v[..(self.v.len() - 1)];
            remainder.iter().rposition(|x| (*pred)(x))
        };
        let idx = idx_opt.map(|idx| idx + 1).unwrap_or(0);
        if idx == 0 {
            self.finished = true;
        }
        let tmp = mem::replace(&mut self.v, &mut []);
        let (head, tail) = tmp.split_at_mut(idx);
        self.v = head;
        Some(tail)
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<T, P> FusedIterator for SplitInclusiveMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = [11, 22, 33, 0, 44, 55];
/// let iter = slice.rsplit(|num| *num == 0);
/// ```
///
/// [`rsplit`]: slice::rsplit
/// [slices]: slice
#[stable(feature = "slice_rsplit", since = "1.27.0")]
#[derive(Clone)] // Is this correct, or does it incorrectly require `T: Clone`?
pub struct RSplit<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: Split<'a, T, P>,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> RSplit<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a [T], pred: P) -> Self {
        Self { inner: Split::new(slice, pred) }
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplit<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplit")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T, P> FusedIterator for RSplit<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`, starting from the end of the slice.
///
/// This struct is created by the [`rsplit_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = [11, 22, 33, 0, 44, 55];
/// let iter = slice.rsplit_mut(|num| *num == 0);
/// ```
///
/// [`rsplit_mut`]: slice::rsplit_mut
/// [slices]: slice
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub struct RSplitMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: SplitMut<'a, T, P>,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> RSplitMut<'a, T, P> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], pred: P) -> Self {
        Self { inner: SplitMut::new(slice, pred) }
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplitMut")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T, P> FusedIterator for RSplitMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
#[derive(Debug)]
struct GenericSplitN<I> {
    iter: I,
    count: usize,
}

impl<T, I: SplitIter<Item = T>> Iterator for GenericSplitN<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.count {
            0 => None,
            1 => {
                self.count -= 1;
                self.iter.finish()
            }
            _ => {
                self.count -= 1;
                self.iter.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper_opt) = self.iter.size_hint();
        (
            cmp::min(self.count, lower),
            Some(upper_opt.map_or(self.count, |upper| cmp::min(self.count, upper))),
        )
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = [10, 40, 30, 20, 60, 50];
/// let iter = slice.splitn(2, |num| *num % 3 == 0);
/// ```
///
/// [`splitn`]: slice::splitn
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitN<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<Split<'a, T, P>>,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> SplitN<'a, T, P> {
    #[inline]
    pub(super) fn new(s: Split<'a, T, P>, n: usize) -> Self {
        Self { inner: GenericSplitN { iter: s, count: n } }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitN<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitN").field("inner", &self.inner).finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = [10, 40, 30, 20, 60, 50];
/// let iter = slice.rsplitn(2, |num| *num % 3 == 0);
/// ```
///
/// [`rsplitn`]: slice::rsplitn
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitN<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<RSplit<'a, T, P>>,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> RSplitN<'a, T, P> {
    #[inline]
    pub(super) fn new(s: RSplit<'a, T, P>, n: usize) -> Self {
        Self { inner: GenericSplitN { iter: s, count: n } }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitN<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplitN").field("inner", &self.inner).finish()
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = [10, 40, 30, 20, 60, 50];
/// let iter = slice.splitn_mut(2, |num| *num % 3 == 0);
/// ```
///
/// [`splitn_mut`]: slice::splitn_mut
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitNMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<SplitMut<'a, T, P>>,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> SplitNMut<'a, T, P> {
    #[inline]
    pub(super) fn new(s: SplitMut<'a, T, P>, n: usize) -> Self {
        Self { inner: GenericSplitN { iter: s, count: n } }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitNMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitNMut").field("inner", &self.inner).finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = [10, 40, 30, 20, 60, 50];
/// let iter = slice.rsplitn_mut(2, |num| *num % 3 == 0);
/// ```
///
/// [`rsplitn_mut`]: slice::rsplitn_mut
/// [slices]: slice
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitNMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    inner: GenericSplitN<RSplitMut<'a, T, P>>,
}

impl<'a, T: 'a, P: FnMut(&T) -> bool> RSplitNMut<'a, T, P> {
    #[inline]
    pub(super) fn new(s: RSplitMut<'a, T, P>, n: usize) -> Self {
        Self { inner: GenericSplitN { iter: s, count: n } }
    }
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitNMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplitNMut").field("inner", &self.inner).finish()
    }
}

forward_iterator! { SplitN: T, &'a [T] }
forward_iterator! { RSplitN: T, &'a [T] }
forward_iterator! { SplitNMut: T, &'a mut [T] }
forward_iterator! { RSplitNMut: T, &'a mut [T] }

/// An iterator over overlapping subslices of length `size`.
///
/// This struct is created by the [`windows`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = ['r', 'u', 's', 't'];
/// let iter = slice.windows(2);
/// ```
///
/// [`windows`]: slice::windows
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Windows<'a, T: 'a> {
    v: &'a [T],
    size: NonZeroUsize,
}

impl<'a, T: 'a> Windows<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T], size: NonZeroUsize) -> Self {
        Self { v: slice, size }
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Windows<'_, T> {
    fn clone(&self) -> Self {
        Windows { v: self.v, size: self.size }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Windows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.size.get() > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[..self.size.get()]);
            self.v = &self.v[1..];
            ret
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size.get() > self.v.len() {
            (0, Some(0))
        } else {
            let size = self.v.len() - self.size.get() + 1;
            (size, Some(size))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = self.size.get().overflowing_add(n);
        if end > self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let nth = &self.v[n..end];
            self.v = &self.v[n + 1..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.size.get() > self.v.len() {
            None
        } else {
            let start = self.v.len() - self.size.get();
            Some(&self.v[start..])
        }
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: since the caller guarantees that `i` is in bounds,
        // which means that `i` cannot overflow an `isize`, and the
        // slice created by `from_raw_parts` is a subslice of `self.v`
        // thus is guaranteed to be valid for the lifetime `'a` of `self.v`.
        unsafe { from_raw_parts(self.v.as_ptr().add(idx), self.size.get()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Windows<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.size.get() > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[self.v.len() - self.size.get()..]);
            self.v = &self.v[..self.v.len() - 1];
            ret
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = self.v.len().overflowing_sub(n);
        if end < self.size.get() || overflow {
            self.v = &[];
            None
        } else {
            let ret = &self.v[end - self.size.get()..end];
            self.v = &self.v[..end - 1];
            Some(ret)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Windows<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Windows<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Windows<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for Windows<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for Windows<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.chunks(2);
/// ```
///
/// [`chunks`]: slice::chunks
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chunks<'a, T: 'a> {
    v: &'a [T],
    chunk_size: usize,
}

impl<'a, T: 'a> Chunks<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T], size: usize) -> Self {
        Self { v: slice, chunk_size: size }
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Chunks<'_, T> {
    fn clone(&self) -> Self {
        Chunks { v: self.v, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let nth = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&self.v[start..])
        }
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        // SAFETY: the caller guarantees that `i` is in bounds,
        // which means that `start` must be in bounds of the
        // underlying `self.v` slice, and we made sure that `len`
        // is also in bounds of `self.v`. Thus, `start` cannot overflow
        // an `isize`, and the slice constructed by `from_raw_parts`
        // is a subslice of `self.v` which is guaranteed to be valid
        // for the lifetime `'a` of `self.v`.
        unsafe {
            let len = cmp::min(self.v.len().unchecked_sub(start), self.chunk_size);
            from_raw_parts(self.v.as_ptr().add(start), len)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = match start.checked_add(self.chunk_size) {
                Some(res) => cmp::min(self.v.len(), res),
                None => self.v.len(),
            };
            let nth_back = &self.v[start..end];
            self.v = &self.v[..start];
            Some(nth_back)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Chunks<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Chunks<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Chunks<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for Chunks<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for Chunks<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.chunks_mut(2);
/// ```
///
/// [`chunks_mut`]: slice::chunks_mut
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ChunksMut<'a, T: 'a> {
    v: &'a mut [T],
    chunk_size: usize,
}

impl<'a, T: 'a> ChunksMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], size: usize) -> Self {
        Self { v: slice, chunk_size: size }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(end);
            let (_, nth) = head.split_at_mut(start);
            self.v = tail;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&mut self.v[start..])
        }
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        // SAFETY: see comments for `Chunks::__iterator_get_unchecked`.
        //
        // Also note that the caller also guarantees that we're never called
        // with the same index again, and that no other methods that will
        // access this subslice are called, so it is valid for the returned
        // slice to be mutable.
        unsafe {
            let len = cmp::min(self.v.len().unchecked_sub(start), self.chunk_size);
            from_raw_parts_mut(self.v.as_mut_ptr().add(start), len)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = match start.checked_add(self.chunk_size) {
                Some(res) => cmp::min(self.v.len(), res),
                None => self.v.len(),
            };
            let (temp, _tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (head, nth_back) = temp.split_at_mut(start);
            self.v = head;
            Some(nth_back)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for ChunksMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksMut<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for ChunksMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksMut<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for ChunksMut<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.chunks_exact(2);
/// ```
///
/// [`chunks_exact`]: slice::chunks_exact
/// [`remainder`]: ChunksExact::remainder
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub struct ChunksExact<'a, T: 'a> {
    v: &'a [T],
    rem: &'a [T],
    chunk_size: usize,
}

impl<'a, T> ChunksExact<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T], chunk_size: usize) -> Self {
        let rem = slice.len() % chunk_size;
        let fst_len = slice.len() - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        let (fst, snd) = unsafe { slice.split_at_unchecked(fst_len) };
        Self { v: fst, rem: snd, chunk_size }
    }

    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[must_use]
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> Clone for ChunksExact<'_, T> {
    fn clone(&self) -> Self {
        ChunksExact { v: self.v, rem: self.rem, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (_, snd) = self.v.split_at(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        // SAFETY: mostly identical to `Chunks::__iterator_get_unchecked`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for ChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = start + self.chunk_size;
            let nth_back = &self.v[start..end];
            self.v = &self.v[..start];
            Some(nth_back)
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> ExactSizeIterator for ChunksExact<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksExact<'_, T> {}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> FusedIterator for ChunksExact<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksExact<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for ChunksExact<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.chunks_exact_mut(2);
/// ```
///
/// [`chunks_exact_mut`]: slice::chunks_exact_mut
/// [`into_remainder`]: ChunksExactMut::into_remainder
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub struct ChunksExactMut<'a, T: 'a> {
    v: &'a mut [T],
    rem: &'a mut [T],
    chunk_size: usize,
}

impl<'a, T> ChunksExactMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], chunk_size: usize) -> Self {
        let rem = slice.len() % chunk_size;
        let fst_len = slice.len() - rem;
        // SAFETY: 0 <= fst_len <= slice.len() by construction above
        let (fst, snd) = unsafe { slice.split_at_mut_unchecked(fst_len) };
        Self { v: fst, rem: snd, chunk_size }
    }

    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> Iterator for ChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (_, snd) = tmp.split_at_mut(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        // SAFETY: see comments for `ChunksMut::__iterator_get_unchecked`.
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for ChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = start + self.chunk_size;
            let (temp, _tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (head, nth_back) = temp.split_at_mut(start);
            self.v = head;
            Some(nth_back)
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> ExactSizeIterator for ChunksExactMut<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksExactMut<'_, T> {}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> FusedIterator for ChunksExactMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksExactMut<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for ChunksExactMut<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// A windowed iterator over a slice in overlapping chunks (`N` elements at a
/// time), starting at the beginning of the slice
///
/// This struct is created by the [`array_windows`] method on [slices].
///
/// # Example
///
/// ```
/// #![feature(array_windows)]
///
/// let slice = [0, 1, 2, 3];
/// let iter = slice.array_windows::<2>();
/// ```
///
/// [`array_windows`]: slice::array_windows
/// [slices]: slice
#[derive(Debug, Clone, Copy)]
#[unstable(feature = "array_windows", issue = "75027")]
pub struct ArrayWindows<'a, T: 'a, const N: usize> {
    slice_head: *const T,
    num: usize,
    marker: PhantomData<&'a [T; N]>,
}

impl<'a, T: 'a, const N: usize> ArrayWindows<'a, T, N> {
    #[inline]
    pub(super) fn new(slice: &'a [T]) -> Self {
        let num_windows = slice.len().saturating_sub(N - 1);
        Self { slice_head: slice.as_ptr(), num: num_windows, marker: PhantomData }
    }
}

#[unstable(feature = "array_windows", issue = "75027")]
impl<'a, T, const N: usize> Iterator for ArrayWindows<'a, T, N> {
    type Item = &'a [T; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.num == 0 {
            return None;
        }
        // SAFETY:
        // This is safe because it's indexing into a slice guaranteed to be length > N.
        let ret = unsafe { &*self.slice_head.cast::<[T; N]>() };
        // SAFETY: Guaranteed that there are at least 1 item remaining otherwise
        // earlier branch would've been hit
        self.slice_head = unsafe { self.slice_head.add(1) };

        self.num -= 1;
        Some(ret)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num, Some(self.num))
    }

    #[inline]
    fn count(self) -> usize {
        self.num
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.num <= n {
            self.num = 0;
            return None;
        }
        // SAFETY:
        // This is safe because it's indexing into a slice guaranteed to be length > N.
        let ret = unsafe { &*self.slice_head.add(n).cast::<[T; N]>() };
        // SAFETY: Guaranteed that there are at least n items remaining
        self.slice_head = unsafe { self.slice_head.add(n + 1) };

        self.num -= n + 1;
        Some(ret)
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.nth(self.num.checked_sub(1)?)
    }
}

#[unstable(feature = "array_windows", issue = "75027")]
impl<'a, T, const N: usize> DoubleEndedIterator for ArrayWindows<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T; N]> {
        if self.num == 0 {
            return None;
        }
        // SAFETY: Guaranteed that there are n items remaining, n-1 for 0-indexing.
        let ret = unsafe { &*self.slice_head.add(self.num - 1).cast::<[T; N]>() };
        self.num -= 1;
        Some(ret)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<&'a [T; N]> {
        if self.num <= n {
            self.num = 0;
            return None;
        }
        // SAFETY: Guaranteed that there are n items remaining, n-1 for 0-indexing.
        let ret = unsafe { &*self.slice_head.add(self.num - (n + 1)).cast::<[T; N]>() };
        self.num -= n + 1;
        Some(ret)
    }
}

#[unstable(feature = "array_windows", issue = "75027")]
impl<T, const N: usize> ExactSizeIterator for ArrayWindows<'_, T, N> {
    fn is_empty(&self) -> bool {
        self.num == 0
    }
}

/// An iterator over a slice in (non-overlapping) chunks (`N` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `N-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`array_chunks`] method on [slices].
///
/// # Example
///
/// ```
/// #![feature(array_chunks)]
///
/// let slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.array_chunks::<2>();
/// ```
///
/// [`array_chunks`]: slice::array_chunks
/// [`remainder`]: ArrayChunks::remainder
/// [slices]: slice
#[derive(Debug)]
#[unstable(feature = "array_chunks", issue = "74985")]
pub struct ArrayChunks<'a, T: 'a, const N: usize> {
    iter: Iter<'a, [T; N]>,
    rem: &'a [T],
}

impl<'a, T, const N: usize> ArrayChunks<'a, T, N> {
    #[inline]
    pub(super) fn new(slice: &'a [T]) -> Self {
        let (array_slice, rem) = slice.as_chunks();
        Self { iter: array_slice.iter(), rem }
    }

    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `N-1`
    /// elements.
    #[must_use]
    #[unstable(feature = "array_chunks", issue = "74985")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> Clone for ArrayChunks<'_, T, N> {
    fn clone(&self) -> Self {
        ArrayChunks { iter: self.iter.clone(), rem: self.rem }
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<'a, T, const N: usize> Iterator for ArrayChunks<'a, T, N> {
    type Item = &'a [T; N];

    #[inline]
    fn next(&mut self) -> Option<&'a [T; N]> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.iter.last()
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, i: usize) -> &'a [T; N] {
        // SAFETY: The safety guarantees of `__iterator_get_unchecked` are
        // transferred to the caller.
        unsafe { self.iter.__iterator_get_unchecked(i) }
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<'a, T, const N: usize> DoubleEndedIterator for ArrayChunks<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T; N]> {
        self.iter.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n)
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> ExactSizeIterator for ArrayChunks<'_, T, N> {
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T, const N: usize> TrustedLen for ArrayChunks<'_, T, N> {}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> FusedIterator for ArrayChunks<'_, T, N> {}

#[doc(hidden)]
#[unstable(feature = "array_chunks", issue = "74985")]
unsafe impl<'a, T, const N: usize> TrustedRandomAccess for ArrayChunks<'a, T, N> {}

#[doc(hidden)]
#[unstable(feature = "array_chunks", issue = "74985")]
unsafe impl<'a, T, const N: usize> TrustedRandomAccessNoCoerce for ArrayChunks<'a, T, N> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`N` elements
/// at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `N-1` elements will be omitted but can be retrieved from
/// the [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`array_chunks_mut`] method on [slices].
///
/// # Example
///
/// ```
/// #![feature(array_chunks)]
///
/// let mut slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.array_chunks_mut::<2>();
/// ```
///
/// [`array_chunks_mut`]: slice::array_chunks_mut
/// [`into_remainder`]: ../../std/slice/struct.ArrayChunksMut.html#method.into_remainder
/// [slices]: slice
#[derive(Debug)]
#[unstable(feature = "array_chunks", issue = "74985")]
pub struct ArrayChunksMut<'a, T: 'a, const N: usize> {
    iter: IterMut<'a, [T; N]>,
    rem: &'a mut [T],
}

impl<'a, T, const N: usize> ArrayChunksMut<'a, T, N> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T]) -> Self {
        let (array_slice, rem) = slice.as_chunks_mut();
        Self { iter: array_slice.iter_mut(), rem }
    }

    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `N-1`
    /// elements.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[unstable(feature = "array_chunks", issue = "74985")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<'a, T, const N: usize> Iterator for ArrayChunksMut<'a, T, N> {
    type Item = &'a mut [T; N];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T; N]> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.iter.last()
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, i: usize) -> &'a mut [T; N] {
        // SAFETY: The safety guarantees of `__iterator_get_unchecked` are transferred to
        // the caller.
        unsafe { self.iter.__iterator_get_unchecked(i) }
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<'a, T, const N: usize> DoubleEndedIterator for ArrayChunksMut<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T; N]> {
        self.iter.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n)
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> ExactSizeIterator for ArrayChunksMut<'_, T, N> {
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T, const N: usize> TrustedLen for ArrayChunksMut<'_, T, N> {}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> FusedIterator for ArrayChunksMut<'_, T, N> {}

#[doc(hidden)]
#[unstable(feature = "array_chunks", issue = "74985")]
unsafe impl<'a, T, const N: usize> TrustedRandomAccess for ArrayChunksMut<'a, T, N> {}

#[doc(hidden)]
#[unstable(feature = "array_chunks", issue = "74985")]
unsafe impl<'a, T, const N: usize> TrustedRandomAccessNoCoerce for ArrayChunksMut<'a, T, N> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.rchunks(2);
/// ```
///
/// [`rchunks`]: slice::rchunks
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunks<'a, T: 'a> {
    v: &'a [T],
    chunk_size: usize,
}

impl<'a, T: 'a> RChunks<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T], size: usize) -> Self {
        Self { v: slice, chunk_size: size }
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> Clone for RChunks<'_, T> {
    fn clone(&self) -> Self {
        RChunks { v: self.v, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            // Can't underflow because of the check above
            let end = self.v.len() - end;
            let start = match end.checked_sub(self.chunk_size) {
                Some(sum) => sum,
                None => 0,
            };
            let nth = &self.v[start..end];
            self.v = &self.v[0..start];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let rem = self.v.len() % self.chunk_size;
            let end = if rem == 0 { self.chunk_size } else { rem };
            Some(&self.v[0..end])
        }
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = match end.checked_sub(self.chunk_size) {
            None => 0,
            Some(start) => start,
        };
        // SAFETY: mostly identical to `Chunks::__iterator_get_unchecked`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), end - start) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            // can't underflow because `n < len`
            let offset_from_end = (len - 1 - n) * self.chunk_size;
            let end = self.v.len() - offset_from_end;
            let start = end.saturating_sub(self.chunk_size);
            let nth_back = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> ExactSizeIterator for RChunks<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunks<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunks<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunks<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for RChunks<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.rchunks_mut(2);
/// ```
///
/// [`rchunks_mut`]: slice::rchunks_mut
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksMut<'a, T: 'a> {
    v: &'a mut [T],
    chunk_size: usize,
}

impl<'a, T: 'a> RChunksMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], size: usize) -> Self {
        Self { v: slice, chunk_size: size }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            // Can't underflow because of the check above
            let end = self.v.len() - end;
            let start = match end.checked_sub(self.chunk_size) {
                Some(sum) => sum,
                None => 0,
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(start);
            let (nth, _) = tail.split_at_mut(end - start);
            self.v = head;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let rem = self.v.len() % self.chunk_size;
            let end = if rem == 0 { self.chunk_size } else { rem };
            Some(&mut self.v[0..end])
        }
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = match end.checked_sub(self.chunk_size) {
            None => 0,
            Some(start) => start,
        };
        // SAFETY: see comments for `RChunks::__iterator_get_unchecked` and
        // `ChunksMut::__iterator_get_unchecked`
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), end - start) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            // can't underflow because `n < len`
            let offset_from_end = (len - 1 - n) * self.chunk_size;
            let end = self.v.len() - offset_from_end;
            let start = end.saturating_sub(self.chunk_size);
            let (tmp, tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (_, nth_back) = tmp.split_at_mut(start);
            self.v = tail;
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> ExactSizeIterator for RChunksMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunksMut<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunksMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksMut<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for RChunksMut<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact`] method on [slices].
///
/// # Example
///
/// ```
/// let slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.rchunks_exact(2);
/// ```
///
/// [`rchunks_exact`]: slice::rchunks_exact
/// [`remainder`]: ChunksExact::remainder
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksExact<'a, T: 'a> {
    v: &'a [T],
    rem: &'a [T],
    chunk_size: usize,
}

impl<'a, T> RChunksExact<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a [T], chunk_size: usize) -> Self {
        let rem = slice.len() % chunk_size;
        // SAFETY: 0 <= rem <= slice.len() by construction above
        let (fst, snd) = unsafe { slice.split_at_unchecked(rem) };
        Self { v: snd, rem: fst, chunk_size }
    }

    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[must_use]
    #[stable(feature = "rchunks", since = "1.31.0")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Clone for RChunksExact<'a, T> {
    fn clone(&self) -> RChunksExact<'a, T> {
        RChunksExact { v: self.v, rem: self.rem, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (fst, _) = self.v.split_at(self.v.len() - end);
            self.v = fst;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = end - self.chunk_size;
        // SAFETY:
        // SAFETY: mostmy identical to `Chunks::__iterator_get_unchecked`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            // now that we know that `n` corresponds to a chunk,
            // none of these operations can underflow/overflow
            let offset = (len - n) * self.chunk_size;
            let start = self.v.len() - offset;
            let end = start + self.chunk_size;
            let nth_back = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> ExactSizeIterator for RChunksExact<'a, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunksExact<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunksExact<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksExact<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for RChunksExact<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact_mut`] method on [slices].
///
/// # Example
///
/// ```
/// let mut slice = ['l', 'o', 'r', 'e', 'm'];
/// let iter = slice.rchunks_exact_mut(2);
/// ```
///
/// [`rchunks_exact_mut`]: slice::rchunks_exact_mut
/// [`into_remainder`]: ChunksExactMut::into_remainder
/// [slices]: slice
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksExactMut<'a, T: 'a> {
    v: &'a mut [T],
    rem: &'a mut [T],
    chunk_size: usize,
}

impl<'a, T> RChunksExactMut<'a, T> {
    #[inline]
    pub(super) fn new(slice: &'a mut [T], chunk_size: usize) -> Self {
        let rem = slice.len() % chunk_size;
        // SAFETY: 0 <= rem <= slice.len() by construction above
        let (fst, snd) = unsafe { slice.split_at_mut_unchecked(rem) };
        Self { v: snd, rem: fst, chunk_size }
    }

    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "rchunks", since = "1.31.0")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (fst, _) = tmp.split_at_mut(tmp_len - end);
            self.v = fst;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = end - self.chunk_size;
        // SAFETY: see comments for `RChunksMut::__iterator_get_unchecked`.
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            // now that we know that `n` corresponds to a chunk,
            // none of these operations can underflow/overflow
            let offset = (len - n) * self.chunk_size;
            let start = self.v.len() - offset;
            let end = start + self.chunk_size;
            let (tmp, tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (_, nth_back) = tmp.split_at_mut(start);
            self.v = tail;
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> ExactSizeIterator for RChunksExactMut<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunksExactMut<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunksExactMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksExactMut<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for RChunksExactMut<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for Iter<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for Iter<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for IterMut<'a, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccessNoCoerce for IterMut<'a, T> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// An iterator over slice in (non-overlapping) chunks separated by a predicate.
///
/// This struct is created by the [`group_by`] method on [slices].
///
/// [`group_by`]: slice::group_by
/// [slices]: slice
#[unstable(feature = "slice_group_by", issue = "80552")]
pub struct GroupBy<'a, T: 'a, P> {
    slice: &'a [T],
    predicate: P,
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> GroupBy<'a, T, P> {
    pub(super) fn new(slice: &'a [T], predicate: P) -> Self {
        GroupBy { slice, predicate }
    }
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> Iterator for GroupBy<'a, T, P>
where
    P: FnMut(&T, &T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.slice.windows(2);
            while let Some([l, r]) = iter.next() {
                if (self.predicate)(l, r) { len += 1 } else { break }
            }
            let (head, tail) = self.slice.split_at(len);
            self.slice = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() { (0, Some(0)) } else { (1, Some(self.slice.len())) }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> DoubleEndedIterator for GroupBy<'a, T, P>
where
    P: FnMut(&T, &T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.slice.windows(2);
            while let Some([l, r]) = iter.next_back() {
                if (self.predicate)(l, r) { len += 1 } else { break }
            }
            let (head, tail) = self.slice.split_at(self.slice.len() - len);
            self.slice = head;
            Some(tail)
        }
    }
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> FusedIterator for GroupBy<'a, T, P> where P: FnMut(&T, &T) -> bool {}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for GroupBy<'a, T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupBy").field("slice", &self.slice).finish()
    }
}

/// An iterator over slice in (non-overlapping) mutable chunks separated
/// by a predicate.
///
/// This struct is created by the [`group_by_mut`] method on [slices].
///
/// [`group_by_mut`]: slice::group_by_mut
/// [slices]: slice
#[unstable(feature = "slice_group_by", issue = "80552")]
pub struct GroupByMut<'a, T: 'a, P> {
    slice: &'a mut [T],
    predicate: P,
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> GroupByMut<'a, T, P> {
    pub(super) fn new(slice: &'a mut [T], predicate: P) -> Self {
        GroupByMut { slice, predicate }
    }
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> Iterator for GroupByMut<'a, T, P>
where
    P: FnMut(&T, &T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.slice.windows(2);
            while let Some([l, r]) = iter.next() {
                if (self.predicate)(l, r) { len += 1 } else { break }
            }
            let slice = mem::take(&mut self.slice);
            let (head, tail) = slice.split_at_mut(len);
            self.slice = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.slice.is_empty() { (0, Some(0)) } else { (1, Some(self.slice.len())) }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> DoubleEndedIterator for GroupByMut<'a, T, P>
where
    P: FnMut(&T, &T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() {
            None
        } else {
            let mut len = 1;
            let mut iter = self.slice.windows(2);
            while let Some([l, r]) = iter.next_back() {
                if (self.predicate)(l, r) { len += 1 } else { break }
            }
            let slice = mem::take(&mut self.slice);
            let (head, tail) = slice.split_at_mut(slice.len() - len);
            self.slice = head;
            Some(tail)
        }
    }
}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a, P> FusedIterator for GroupByMut<'a, T, P> where P: FnMut(&T, &T) -> bool {}

#[unstable(feature = "slice_group_by", issue = "80552")]
impl<'a, T: 'a + fmt::Debug, P> fmt::Debug for GroupByMut<'a, T, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupByMut").field("slice", &self.slice).finish()
    }
}
