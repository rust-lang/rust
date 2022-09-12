//! Indexing implementations for `[T]`.

use crate::intrinsics::assert_unsafe_precondition;
use crate::intrinsics::const_eval_select;
use crate::ops;
use crate::ptr;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
impl<T, I> const ops::Index<I> for [T]
where
    I: ~const SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
impl<T, I> const ops::IndexMut<I> for [T]
where
    I: ~const SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
const fn slice_start_index_len_fail(index: usize, len: usize) -> ! {
    // SAFETY: we are just panicking here
    unsafe {
        const_eval_select(
            (index, len),
            slice_start_index_len_fail_ct,
            slice_start_index_len_fail_rt,
        )
    }
}

// FIXME const-hack
#[track_caller]
fn slice_start_index_len_fail_rt(index: usize, len: usize) -> ! {
    panic!("range start index {index} out of range for slice of length {len}");
}

#[track_caller]
const fn slice_start_index_len_fail_ct(_: usize, _: usize) -> ! {
    panic!("slice start index is out of range for slice");
}

#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
const fn slice_end_index_len_fail(index: usize, len: usize) -> ! {
    // SAFETY: we are just panicking here
    unsafe {
        const_eval_select((index, len), slice_end_index_len_fail_ct, slice_end_index_len_fail_rt)
    }
}

// FIXME const-hack
#[track_caller]
fn slice_end_index_len_fail_rt(index: usize, len: usize) -> ! {
    panic!("range end index {index} out of range for slice of length {len}");
}

#[track_caller]
const fn slice_end_index_len_fail_ct(_: usize, _: usize) -> ! {
    panic!("slice end index is out of range for slice");
}

#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
const fn slice_index_order_fail(index: usize, end: usize) -> ! {
    // SAFETY: we are just panicking here
    unsafe { const_eval_select((index, end), slice_index_order_fail_ct, slice_index_order_fail_rt) }
}

// FIXME const-hack
#[track_caller]
fn slice_index_order_fail_rt(index: usize, end: usize) -> ! {
    panic!("slice index starts at {index} but ends at {end}");
}

#[track_caller]
const fn slice_index_order_fail_ct(_: usize, _: usize) -> ! {
    panic!("slice index start is larger than end");
}

#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
const fn slice_start_index_overflow_fail() -> ! {
    panic!("attempted to index slice from after maximum usize");
}

#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
const fn slice_end_index_overflow_fail() -> ! {
    panic!("attempted to index slice up to maximum usize");
}

mod private_slice_index {
    use super::ops;
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    pub trait Sealed {}

    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for usize {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::Range<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeTo<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeFrom<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeFull {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeInclusive<usize> {}
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    impl Sealed for ops::RangeToInclusive<usize> {}
    #[stable(feature = "slice_index_with_ops_bound_pair", since = "1.53.0")]
    impl Sealed for (ops::Bound<usize>, ops::Bound<usize>) {}
}

/// A helper trait used for indexing operations.
///
/// Implementations of this trait have to promise that if the argument
/// to `get_unchecked(_mut)` is a safe reference, then so is the result.
#[stable(feature = "slice_get_slice", since = "1.28.0")]
#[rustc_diagnostic_item = "SliceIndex"]
#[rustc_on_unimplemented(
    on(T = "str", label = "string indices are ranges of `usize`",),
    on(
        all(any(T = "str", T = "&str", T = "std::string::String"), _Self = "{integer}"),
        note = "you can use `.chars().nth()` or `.bytes().nth()`\n\
                for more information, see chapter 8 in The Book: \
                <https://doc.rust-lang.org/book/ch08-02-strings.html#indexing-into-strings>"
    ),
    message = "the type `{T}` cannot be indexed by `{Self}`",
    label = "slice indices are of type `usize` or ranges of `usize`"
)]
pub unsafe trait SliceIndex<T: ?Sized>: private_slice_index::Sealed {
    /// The output type returned by methods.
    #[stable(feature = "slice_get_slice", since = "1.28.0")]
    type Output: ?Sized;

    /// Returns a shared reference to the output at this location, if in
    /// bounds.
    #[unstable(feature = "slice_index_methods", issue = "none")]
    fn get(self, slice: &T) -> Option<&Self::Output>;

    /// Returns a mutable reference to the output at this location, if in
    /// bounds.
    #[unstable(feature = "slice_index_methods", issue = "none")]
    fn get_mut(self, slice: &mut T) -> Option<&mut Self::Output>;

    /// Returns a shared reference to the output at this location, without
    /// performing any bounds checking.
    /// Calling this method with an out-of-bounds index or a dangling `slice` pointer
    /// is *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[unstable(feature = "slice_index_methods", issue = "none")]
    unsafe fn get_unchecked(self, slice: *const T) -> *const Self::Output;

    /// Returns a mutable reference to the output at this location, without
    /// performing any bounds checking.
    /// Calling this method with an out-of-bounds index or a dangling `slice` pointer
    /// is *[undefined behavior]* even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    #[unstable(feature = "slice_index_methods", issue = "none")]
    unsafe fn get_unchecked_mut(self, slice: *mut T) -> *mut Self::Output;

    /// Returns a shared reference to the output at this location, panicking
    /// if out of bounds.
    #[unstable(feature = "slice_index_methods", issue = "none")]
    #[track_caller]
    fn index(self, slice: &T) -> &Self::Output;

    /// Returns a mutable reference to the output at this location, panicking
    /// if out of bounds.
    #[unstable(feature = "slice_index_methods", issue = "none")]
    #[track_caller]
    fn index_mut(self, slice: &mut T) -> &mut Self::Output;
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for usize {
    type Output = T;

    #[inline]
    fn get(self, slice: &[T]) -> Option<&T> {
        // SAFETY: `self` is checked to be in bounds.
        if self < slice.len() { unsafe { Some(&*self.get_unchecked(slice)) } } else { None }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut T> {
        // SAFETY: `self` is checked to be in bounds.
        if self < slice.len() { unsafe { Some(&mut *self.get_unchecked_mut(slice)) } } else { None }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const T {
        let this = self;
        // SAFETY: the caller guarantees that `slice` is not dangling, so it
        // cannot be longer than `isize::MAX`. They also guarantee that
        // `self` is in bounds of `slice` so `self` cannot overflow an `isize`,
        // so the call to `add` is safe.
        unsafe {
            assert_unsafe_precondition!([T](this: usize, slice: *const [T]) => this < slice.len());
            slice.as_ptr().add(self)
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut T {
        let this = self;
        // SAFETY: see comments for `get_unchecked` above.
        unsafe {
            assert_unsafe_precondition!([T](this: usize, slice: *mut [T]) => this < slice.len());
            slice.as_mut_ptr().add(self)
        }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &T {
        // N.B., use intrinsic indexing
        &(*slice)[self]
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut T {
        // N.B., use intrinsic indexing
        &mut (*slice)[self]
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for ops::Range<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        if self.start > self.end || self.end > slice.len() {
            None
        } else {
            // SAFETY: `self` is checked to be valid and in bounds above.
            unsafe { Some(&*self.get_unchecked(slice)) }
        }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        if self.start > self.end || self.end > slice.len() {
            None
        } else {
            // SAFETY: `self` is checked to be valid and in bounds above.
            unsafe { Some(&mut *self.get_unchecked_mut(slice)) }
        }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const [T] {
        let this = ops::Range { start: self.start, end: self.end };
        // SAFETY: the caller guarantees that `slice` is not dangling, so it
        // cannot be longer than `isize::MAX`. They also guarantee that
        // `self` is in bounds of `slice` so `self` cannot overflow an `isize`,
        // so the call to `add` is safe.

        unsafe {
            assert_unsafe_precondition!([T](this: ops::Range<usize>, slice: *const [T]) =>
            this.end >= this.start && this.end <= slice.len());
            ptr::slice_from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start)
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        let this = ops::Range { start: self.start, end: self.end };
        // SAFETY: see comments for `get_unchecked` above.
        unsafe {
            assert_unsafe_precondition!([T](this: ops::Range<usize>, slice: *mut [T]) =>
                this.end >= this.start && this.end <= slice.len());
            ptr::slice_from_raw_parts_mut(slice.as_mut_ptr().add(self.start), self.end - self.start)
        }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_end_index_len_fail(self.end, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { &*self.get_unchecked(slice) }
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        if self.start > self.end {
            slice_index_order_fail(self.start, self.end);
        } else if self.end > slice.len() {
            slice_end_index_len_fail(self.end, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { &mut *self.get_unchecked_mut(slice) }
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for ops::RangeTo<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0..self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0..self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked`.
        unsafe { (0..self.end).get_unchecked(slice) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked_mut`.
        unsafe { (0..self.end).get_unchecked_mut(slice) }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0..self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..self.end).index_mut(slice)
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for ops::RangeFrom<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (self.start..slice.len()).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (self.start..slice.len()).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked`.
        unsafe { (self.start..slice.len()).get_unchecked(slice) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked_mut`.
        unsafe { (self.start..slice.len()).get_unchecked_mut(slice) }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        if self.start > slice.len() {
            slice_start_index_len_fail(self.start, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { &*self.get_unchecked(slice) }
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        if self.start > slice.len() {
            slice_start_index_len_fail(self.start, slice.len());
        }
        // SAFETY: `self` is checked to be valid and in bounds above.
        unsafe { &mut *self.get_unchecked_mut(slice) }
    }
}

#[stable(feature = "slice_get_slice_impls", since = "1.15.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for ops::RangeFull {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        Some(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        Some(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const [T] {
        slice
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        slice
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        slice
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        slice
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for ops::RangeInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        if *self.end() == usize::MAX { None } else { self.into_slice_range().get(slice) }
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        if *self.end() == usize::MAX { None } else { self.into_slice_range().get_mut(slice) }
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked`.
        unsafe { self.into_slice_range().get_unchecked(slice) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked_mut`.
        unsafe { self.into_slice_range().get_unchecked_mut(slice) }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        if *self.end() == usize::MAX {
            slice_end_index_overflow_fail();
        }
        self.into_slice_range().index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        if *self.end() == usize::MAX {
            slice_end_index_overflow_fail();
        }
        self.into_slice_range().index_mut(slice)
    }
}

#[stable(feature = "inclusive_range", since = "1.26.0")]
#[rustc_const_unstable(feature = "const_slice_index", issue = "none")]
unsafe impl<T> const SliceIndex<[T]> for ops::RangeToInclusive<usize> {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&[T]> {
        (0..=self.end).get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut [T]> {
        (0..=self.end).get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked`.
        unsafe { (0..=self.end).get_unchecked(slice) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked_mut`.
        unsafe { (0..=self.end).get_unchecked_mut(slice) }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &[T] {
        (0..=self.end).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut [T] {
        (0..=self.end).index_mut(slice)
    }
}

/// Performs bounds-checking of a range.
///
/// This method is similar to [`Index::index`] for slices, but it returns a
/// [`Range`] equivalent to `range`. You can use this method to turn any range
/// into `start` and `end` values.
///
/// `bounds` is the range of the slice to use for bounds-checking. It should
/// be a [`RangeTo`] range that ends at the length of the slice.
///
/// The returned [`Range`] is safe to pass to [`slice::get_unchecked`] and
/// [`slice::get_unchecked_mut`] for slices with the given range.
///
/// [`Range`]: ops::Range
/// [`RangeTo`]: ops::RangeTo
/// [`slice::get_unchecked`]: slice::get_unchecked
/// [`slice::get_unchecked_mut`]: slice::get_unchecked_mut
///
/// # Panics
///
/// Panics if `range` would be out of bounds.
///
/// # Examples
///
/// ```
/// #![feature(slice_range)]
///
/// use std::slice;
///
/// let v = [10, 40, 30];
/// assert_eq!(1..2, slice::range(1..2, ..v.len()));
/// assert_eq!(0..2, slice::range(..2, ..v.len()));
/// assert_eq!(1..3, slice::range(1.., ..v.len()));
/// ```
///
/// Panics when [`Index::index`] would panic:
///
/// ```should_panic
/// #![feature(slice_range)]
///
/// use std::slice;
///
/// let _ = slice::range(2..1, ..3);
/// ```
///
/// ```should_panic
/// #![feature(slice_range)]
///
/// use std::slice;
///
/// let _ = slice::range(1..4, ..3);
/// ```
///
/// ```should_panic
/// #![feature(slice_range)]
///
/// use std::slice;
///
/// let _ = slice::range(1..=usize::MAX, ..3);
/// ```
///
/// [`Index::index`]: ops::Index::index
#[track_caller]
#[unstable(feature = "slice_range", issue = "76393")]
#[must_use]
pub fn range<R>(range: R, bounds: ops::RangeTo<usize>) -> ops::Range<usize>
where
    R: ops::RangeBounds<usize>,
{
    let len = bounds.end;

    let start: ops::Bound<&usize> = range.start_bound();
    let start = match start {
        ops::Bound::Included(&start) => start,
        ops::Bound::Excluded(start) => {
            start.checked_add(1).unwrap_or_else(|| slice_start_index_overflow_fail())
        }
        ops::Bound::Unbounded => 0,
    };

    let end: ops::Bound<&usize> = range.end_bound();
    let end = match end {
        ops::Bound::Included(end) => {
            end.checked_add(1).unwrap_or_else(|| slice_end_index_overflow_fail())
        }
        ops::Bound::Excluded(&end) => end,
        ops::Bound::Unbounded => len,
    };

    if start > end {
        slice_index_order_fail(start, end);
    }
    if end > len {
        slice_end_index_len_fail(end, len);
    }

    ops::Range { start, end }
}

/// Convert pair of `ops::Bound`s into `ops::Range` without performing any bounds checking and (in debug) overflow checking
fn into_range_unchecked(
    len: usize,
    (start, end): (ops::Bound<usize>, ops::Bound<usize>),
) -> ops::Range<usize> {
    use ops::Bound;
    let start = match start {
        Bound::Included(i) => i,
        Bound::Excluded(i) => i + 1,
        Bound::Unbounded => 0,
    };
    let end = match end {
        Bound::Included(i) => i + 1,
        Bound::Excluded(i) => i,
        Bound::Unbounded => len,
    };
    start..end
}

/// Convert pair of `ops::Bound`s into `ops::Range`.
/// Returns `None` on overflowing indices.
fn into_range(
    len: usize,
    (start, end): (ops::Bound<usize>, ops::Bound<usize>),
) -> Option<ops::Range<usize>> {
    use ops::Bound;
    let start = match start {
        Bound::Included(start) => start,
        Bound::Excluded(start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };

    let end = match end {
        Bound::Included(end) => end.checked_add(1)?,
        Bound::Excluded(end) => end,
        Bound::Unbounded => len,
    };

    // Don't bother with checking `start < end` and `end <= len`
    // since these checks are handled by `Range` impls

    Some(start..end)
}

/// Convert pair of `ops::Bound`s into `ops::Range`.
/// Panics on overflowing indices.
fn into_slice_range(
    len: usize,
    (start, end): (ops::Bound<usize>, ops::Bound<usize>),
) -> ops::Range<usize> {
    use ops::Bound;
    let start = match start {
        Bound::Included(start) => start,
        Bound::Excluded(start) => {
            start.checked_add(1).unwrap_or_else(|| slice_start_index_overflow_fail())
        }
        Bound::Unbounded => 0,
    };

    let end = match end {
        Bound::Included(end) => {
            end.checked_add(1).unwrap_or_else(|| slice_end_index_overflow_fail())
        }
        Bound::Excluded(end) => end,
        Bound::Unbounded => len,
    };

    // Don't bother with checking `start < end` and `end <= len`
    // since these checks are handled by `Range` impls

    start..end
}

#[stable(feature = "slice_index_with_ops_bound_pair", since = "1.53.0")]
unsafe impl<T> SliceIndex<[T]> for (ops::Bound<usize>, ops::Bound<usize>) {
    type Output = [T];

    #[inline]
    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        into_range(slice.len(), self)?.get(slice)
    }

    #[inline]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        into_range(slice.len(), self)?.get_mut(slice)
    }

    #[inline]
    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked`.
        unsafe { into_range_unchecked(slice.len(), self).get_unchecked(slice) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the caller has to uphold the safety contract for `get_unchecked_mut`.
        unsafe { into_range_unchecked(slice.len(), self).get_unchecked_mut(slice) }
    }

    #[inline]
    fn index(self, slice: &[T]) -> &Self::Output {
        into_slice_range(slice.len(), self).index(slice)
    }

    #[inline]
    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        into_slice_range(slice.len(), self).index_mut(slice)
    }
}
