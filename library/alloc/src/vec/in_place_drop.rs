use core::marker::PhantomData;
use core::ptr::{self, NonNull, drop_in_place};
use core::slice::{self};

use crate::alloc::Global;
use crate::raw_vec::RawVec;

// A helper struct for in-place iteration that drops the destination slice of iteration,
// i.e. the head. The source slice (the tail) is dropped by IntoIter.
pub(super) struct InPlaceDrop<T> {
    pub(super) inner: *mut T,
    pub(super) dst: *mut T,
}

impl<T> InPlaceDrop<T> {
    fn len(&self) -> usize {
        unsafe { self.dst.offset_from_unsigned(self.inner) }
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

// A helper struct for in-place collection that drops the destination items together with
// the source allocation - i.e. before the reallocation happened - to avoid leaking them
// if some other destructor panics.
pub(super) struct InPlaceDstDataSrcBufDrop<Src, Dest> {
    pub(super) ptr: NonNull<Dest>,
    pub(super) len: usize,
    pub(super) src_cap: usize,
    pub(super) src: PhantomData<Src>,
}

impl<Src, Dest> Drop for InPlaceDstDataSrcBufDrop<Src, Dest> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let _drop_allocation =
                RawVec::<Src>::from_nonnull_in(self.ptr.cast::<Src>(), self.src_cap, Global);
            drop_in_place(core::ptr::slice_from_raw_parts_mut::<Dest>(self.ptr.as_ptr(), self.len));
        };
    }
}
