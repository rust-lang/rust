#![unstable(feature = "sliceindex_wrappers", issue = "146179")]

//! Helper types for indexing slices.

use crate::intrinsics::slice_get_unchecked;
use crate::slice::SliceIndex;
use crate::{cmp, ops, range};

/// Clamps an index, guaranteeing that it will only access valid elements of the slice.
///
/// # Examples
///
/// ```
/// #![feature(sliceindex_wrappers)]
///
/// use core::index::Clamp;
///
/// let s: &[usize] = &[0, 1, 2, 3];
///
/// assert_eq!(&3, &s[Clamp(6)]);
/// assert_eq!(&[1, 2, 3], &s[Clamp(1..6)]);
/// assert_eq!(&[] as &[usize], &s[Clamp(5..6)]);
/// assert_eq!(&[0, 1, 2, 3], &s[Clamp(..6)]);
/// assert_eq!(&[0, 1, 2, 3], &s[Clamp(..=6)]);
/// assert_eq!(&[] as &[usize], &s[Clamp(6..)]);
/// ```
#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
#[derive(Debug)]
pub struct Clamp<Idx>(pub Idx);

/// Always accesses the last element of the slice.
///
/// # Examples
///
/// ```
/// #![feature(sliceindex_wrappers)]
/// #![feature(slice_index_methods)]
///
/// use core::index::Last;
/// use core::slice::SliceIndex;
///
/// let s = &[0, 1, 2, 3];
///
/// assert_eq!(&3, &s[Last]);
/// assert_eq!(None, Last.get(&[] as &[usize]));
///
/// ```
#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
#[derive(Debug)]
pub struct Last;

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<usize> {
    type Output = T;

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        slice.get(cmp::min(self.0, slice.len() - 1))
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        slice.get_mut(cmp::min(self.0, slice.len() - 1))
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { slice_get_unchecked(slice, cmp::min(self.0, slice.len() - 1)) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { slice_get_unchecked(slice, cmp::min(self.0, slice.len() - 1)) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        &(*slice)[cmp::min(self.0, slice.len() - 1)]
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        &mut (*slice)[cmp::min(self.0, slice.len() - 1)]
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::Range<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        // SAFETY: a range ending before len is always valid
        unsafe { (start..end).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        // SAFETY: a range ending before len is always valid
        unsafe { (start..end).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<ops::Range<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        // SAFETY: a range ending before len is always valid
        unsafe { (start..end).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        // SAFETY: a range ending before len is always valid
        unsafe { (start..end).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        let start = cmp::min(self.0.start, slice.len());
        let end = cmp::min(self.0.end, slice.len());
        (start..end).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeInclusive<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.last, slice.len() - 1);
        (start..=end).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.last, slice.len() - 1);
        (start..=end).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.last, slice.len() - 1);
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (start..=end).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.last, slice.len() - 1);
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (start..=end).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.last, slice.len() - 1);
        (start..=end).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.last, slice.len() - 1);
        (start..=end).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<ops::RangeInclusive<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.end, slice.len() - 1);
        (start..=end).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.end, slice.len() - 1);
        (start..=end).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.end, slice.len() - 1);
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (start..=end).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.end, slice.len() - 1);
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (start..=end).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.end, slice.len() - 1);
        (start..=end).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        let start = cmp::min(self.0.start, slice.len() - 1);
        let end = cmp::min(self.0.end, slice.len() - 1);
        (start..=end).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeFrom<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (cmp::min(self.0.start, slice.len())..).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (cmp::min(self.0.start, slice.len())..).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: a range starting at len is valid
        unsafe { (cmp::min(self.0.start, slice.len())..).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: a range starting at len is valid
        unsafe { (cmp::min(self.0.start, slice.len())..).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (cmp::min(self.0.start, slice.len())..).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (cmp::min(self.0.start, slice.len())..).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<ops::RangeFrom<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (cmp::min(self.0.start, slice.len())..).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (cmp::min(self.0.start, slice.len())..).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: a range starting at len is valid
        unsafe { (cmp::min(self.0.start, slice.len())..).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: a range starting at len is valid
        unsafe { (cmp::min(self.0.start, slice.len())..).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (cmp::min(self.0.start, slice.len())..).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (cmp::min(self.0.start, slice.len())..).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeTo<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (..cmp::min(self.0.end, slice.len())).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (..cmp::min(self.0.end, slice.len())).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (..cmp::min(self.0.end, slice.len())).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (..cmp::min(self.0.end, slice.len())).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (..cmp::min(self.0.end, slice.len())).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (..cmp::min(self.0.end, slice.len())).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeToInclusive<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (..=cmp::min(self.0.last, slice.len() - 1)).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (..=cmp::min(self.0.last, slice.len() - 1)).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (..=cmp::min(self.0.last, slice.len() - 1)).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (..=cmp::min(self.0.last, slice.len() - 1)).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (..=cmp::min(self.0.last, slice.len() - 1)).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (..=cmp::min(self.0.last, slice.len() - 1)).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<ops::RangeToInclusive<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (..=cmp::min(self.0.end, slice.len() - 1)).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (..=cmp::min(self.0.end, slice.len() - 1)).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (..=cmp::min(self.0.end, slice.len() - 1)).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { (..=cmp::min(self.0.end, slice.len() - 1)).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (..=cmp::min(self.0.end, slice.len() - 1)).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (..=cmp::min(self.0.end, slice.len() - 1)).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeFull> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (..).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (..).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: RangeFull just returns `slice` here
        unsafe { (..).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: RangeFull just returns `slice` here
        unsafe { (..).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (..).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (..).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Last {
    type Output = T;

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        slice.last()
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        slice.last_mut()
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { slice_get_unchecked(slice, slice.len() - 1) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the caller ensures that the slice isn't empty
        unsafe { slice_get_unchecked(slice, slice.len() - 1) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        // N.B., use intrinsic indexing
        &(*slice)[slice.len() - 1]
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        // N.B., use intrinsic indexing
        &mut (*slice)[slice.len() - 1]
    }
}
