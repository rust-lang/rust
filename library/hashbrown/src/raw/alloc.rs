pub(crate) use self::inner::{do_alloc, Allocator, Global};

#[cfg(feature = "nightly")]
mod inner {
    use crate::alloc::alloc::Layout;
    pub use crate::alloc::alloc::{Allocator, Global};
    use core::ptr::NonNull;

    #[allow(clippy::map_err_ignore)]
    pub(crate) fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        match alloc.allocate(layout) {
            Ok(ptr) => Ok(ptr.as_non_null_ptr()),
            Err(_) => Err(()),
        }
    }

    #[cfg(feature = "bumpalo")]
    unsafe impl Allocator for crate::BumpWrapper<'_> {
        #[inline]
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, core::alloc::AllocError> {
            match self.0.try_alloc_layout(layout) {
                Ok(ptr) => Ok(NonNull::slice_from_raw_parts(ptr, layout.size())),
                Err(_) => Err(core::alloc::AllocError),
            }
        }
        #[inline]
        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    }
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::Layout;
    use core::ptr::NonNull;

    #[cfg(feature = "bumpalo")]
    use allocator_api2::alloc::AllocError;

    pub use allocator_api2::alloc::{Allocator, Global};

    pub(crate) fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        match alloc.allocate(layout) {
            Ok(ptr) => Ok(ptr.cast()),
            Err(_) => Err(()),
        }
    }

    #[cfg(feature = "bumpalo")]
    unsafe impl Allocator for crate::BumpWrapper<'_> {
        #[allow(clippy::map_err_ignore)]
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            match self.0.try_alloc_layout(layout) {
                Ok(ptr) => Ok(unsafe {
                    NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(
                        ptr.as_ptr(),
                        layout.size(),
                    ))
                }),
                Err(_) => Err(AllocError),
            }
        }
        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
    }
}
