pub use self::inner::*;

#[cfg(feature = "nightly")]
mod inner {
    use crate::alloc::alloc::Layout;
    pub use crate::alloc::alloc::{AllocError, Allocator, Global};
    use core::ptr::NonNull;

    #[allow(clippy::map_err_ignore)]
    pub fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        alloc
            .allocate(layout)
            .map(|ptr| ptr.as_non_null_ptr())
            .map_err(|_| ())
    }
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::{alloc, dealloc, Layout};
    use core::ptr::NonNull;

    pub struct AllocError;

    pub unsafe trait Allocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
    }

    #[derive(Copy, Clone)]
    pub struct Global;
    unsafe impl Allocator for Global {
        fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            unsafe { NonNull::new(alloc(layout)).ok_or(AllocError) }
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            dealloc(ptr.as_ptr(), layout)
        }
    }
    impl Default for Global {
        fn default() -> Self {
            Global
        }
    }

    #[allow(clippy::map_err_ignore)]
    pub fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        alloc.allocate(layout).map_err(|_| ())
    }
}
