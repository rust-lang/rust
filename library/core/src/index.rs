use crate::intrinsics::slice_get_unchecked;
use crate::range;
use crate::slice::SliceIndex;

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
pub struct Clamp<Idx>(pub Idx);

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
pub struct Last;

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<usize> {
    type Output = T;

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        slice.get(self.0.min(slice.len() - 1))
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        slice.get_mut(self.0.min(slice.len() - 1))
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the index before the length is always valid
        unsafe { slice_get_unchecked(slice, self.0.min(slice.len() - 1)) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the index before the length is always valid
        unsafe { slice_get_unchecked(slice, self.0.min(slice.len() - 1)) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        &(*slice)[self.0.min(slice.len() - 1)]
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        &mut (*slice)[self.0.min(slice.len() - 1)]
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::Range<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (self.0.start..self.0.end.min(slice.len())).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (self.0.start..self.0.end.min(slice.len())).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (self.0.start..self.0.end.min(slice.len())).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (self.0.start..self.0.end.min(slice.len())).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (self.0.start..self.0.end.min(slice.len())).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (self.0.start..self.0.end.min(slice.len())).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeInclusive<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (self.0.start..=self.0.end.min(slice.len() - 1)).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (self.0.start..=self.0.end.min(slice.len() - 1)).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the index before the length is always valid
        unsafe { (self.0.start..=self.0.end.min(slice.len() - 1)).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the index before the length is always valid
        unsafe { (self.0.start..=self.0.end.min(slice.len() - 1)).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (self.0.start..=self.0.end.min(slice.len() - 1)).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (self.0.start..=self.0.end.min(slice.len() - 1)).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeFrom<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (self.0.start..slice.len()).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (self.0.start..slice.len()).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (self.0.start..slice.len()).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (self.0.start..slice.len()).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (self.0.start..slice.len()).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (self.0.start..slice.len()).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeTo<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (0..self.0.end.min(slice.len())).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (0..self.0.end.min(slice.len())).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (0..self.0.end.min(slice.len())).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: a range ending before len is always valid
        unsafe { (0..self.0.end.min(slice.len())).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (0..self.0.end.min(slice.len())).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (0..self.0.end.min(slice.len())).index_mut(slice)
    }
}

#[unstable(feature = "sliceindex_wrappers", issue = "146179")]
unsafe impl<T> SliceIndex<[T]> for Clamp<range::RangeToInclusive<usize>> {
    type Output = [T];

    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        (0..=self.0.end.min(slice.len() - 1)).get(slice)
    }

    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        (0..=self.0.end.min(slice.len() - 1)).get_mut(slice)
    }

    unsafe fn get_unchecked(self, slice: *const [T]) -> *const Self::Output {
        // SAFETY: the index before the length is always valid
        unsafe { (0..=self.0.end.min(slice.len() - 1)).get_unchecked(slice) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the index before the length is always valid
        unsafe { (0..=self.0.end.min(slice.len() - 1)).get_unchecked_mut(slice) }
    }

    fn index(self, slice: &[T]) -> &Self::Output {
        (0..=self.0.end.min(slice.len() - 1)).index(slice)
    }

    fn index_mut(self, slice: &mut [T]) -> &mut Self::Output {
        (0..=self.0.end.min(slice.len() - 1)).index_mut(slice)
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
        // SAFETY: the index before the length is always valid
        unsafe { slice_get_unchecked(slice, slice.len() - 1) }
    }

    unsafe fn get_unchecked_mut(self, slice: *mut [T]) -> *mut Self::Output {
        // SAFETY: the index before the length is always valid
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
