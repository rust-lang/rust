use core::ptr::{self};
use core::slice::{self};

// A helper struct for in-place iteration that drops the destination slice of iteration,
// i.e. the head. The source slice (the tail) is dropped by IntoIter.
pub(super) struct InPlaceDrop<T> {
    pub(super) inner: *mut T,
    pub(super) dst: *mut T,
}

impl<T> InPlaceDrop<T> {
    fn len(&self) -> usize {
        unsafe { self.dst.sub_ptr(self.inner) }
    }
}

impl<T> Drop for InPlaceDrop<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(slice::from_raw_parts_mut(self.inner, self.len()));
        }
    }
}

// A helper struct for in-place collection that drops the destination allocation and elements,
// to avoid leaking them if some other destructor panics.
pub(super) struct InPlaceDstBufDrop<T> {
    pub(super) ptr: *mut T,
    pub(super) len: usize,
    pub(super) cap: usize,
}

impl<T> Drop for InPlaceDstBufDrop<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe { super::Vec::from_raw_parts(self.ptr, self.len, self.cap) };
    }
}
