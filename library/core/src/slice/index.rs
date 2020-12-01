//! Indexing implementations for `[T]`.

use crate::ops;
use crate::ptr;

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, I> ops::Index<I> for [T]
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, I> ops::IndexMut<I> for [T]
where
    I: SliceIndex<[T]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

#[inline(never)]
#[cold]
#[track_caller]
fn slice_start_index_len_fail(index: usize, len: usize) -> ! {
    panic!("range start index {} out of range for slice of length {}", index, len);
}

#[inline(never)]
#[cold]
#[track_caller]
pub(crate) fn slice_end_index_len_fail(index: usize, len: usize) -> ! {
    panic!("range end index {} out of range for slice of length {}", index, len);
}

#[inline(never)]
#[cold]
#[track_caller]
pub(crate) fn slice_index_order_fail(index: usize, end: usize) -> ! {
    panic!("slice index starts at {} but ends at {}", index, end);
}

#[inline(never)]
#[cold]
#[track_caller]
pub(crate) fn slice_start_index_overflow_fail() -> ! {
    panic!("attempted to index slice from after maximum usize");
}

#[inline(never)]
#[cold]
#[track_caller]
pub(crate) fn slice_end_index_overflow_fail() -> ! {
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
}

/// A helper trait used for indexing operations.
///
/// Implementations of this trait have to promise that if the argument
/// to `get_(mut_)unchecked` is a safe reference, then so is the result.
#[stable(feature = "slice_get_slice", since = "1.28.0")]
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
unsafe impl<T> SliceIndex<[T]> for usize {
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
        // SAFETY: the caller guarantees that `slice` is not dangling, so it
        // cannot be longer than `isize::MAX`. They also guarantee that
        // `self` is in bounds of `slice` so `self` cannot overflow an `isize`,
        // so the call to `add` is safe.
        unsafe { slice.as_ptr().add(self) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut T {
        // SAFETY: see comments for `get_unchecked` above.
        unsafe { slice.as_mut_ptr().add(self) }
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
unsafe impl<T> SliceIndex<[T]> for ops::Range<usize> {
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
        // SAFETY: the caller guarantees that `slice` is not dangling, so it
        // cannot be longer than `isize::MAX`. They also guarantee that
        // `self` is in bounds of `slice` so `self` cannot overflow an `isize`,
        // so the call to `add` is safe.
        unsafe { ptr::slice_from_raw_parts(slice.as_ptr().add(self.start), self.end - self.start) }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut [T] {
        // SAFETY: see comments for `get_unchecked` above.
        unsafe {
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
unsafe impl<T> SliceIndex<[T]> for ops::RangeTo<usize> {
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
unsafe impl<T> SliceIndex<[T]> for ops::RangeFrom<usize> {
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
unsafe impl<T> SliceIndex<[T]> for ops::RangeFull {
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
unsafe impl<T> SliceIndex<[T]> for ops::RangeInclusive<usize> {
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
unsafe impl<T> SliceIndex<[T]> for ops::RangeToInclusive<usize> {
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
