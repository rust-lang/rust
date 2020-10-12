pub use self::inner::*;

#[cfg(feature = "nightly")]
mod inner {
    use crate::alloc::alloc::Layout;
    pub use crate::alloc::alloc::{AllocRef, Global};
    use core::ptr::NonNull;

    #[allow(clippy::map_err_ignore)]
    pub fn do_alloc<A: AllocRef>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        alloc
            .alloc(layout)
            .map(|ptr| ptr.as_non_null_ptr())
            .map_err(|_| ())
    }
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::{alloc, dealloc, Layout};
    use core::ptr::NonNull;

    pub struct AllocError;

    pub unsafe trait AllocRef {
        fn alloc(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;
        unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);
    }

    #[derive(Copy, Clone)]
    pub struct Global;
    unsafe impl AllocRef for Global {
        fn alloc(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
            unsafe { NonNull::new(alloc(layout)).ok_or(AllocError) }
        }
        unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
            dealloc(ptr.as_ptr(), layout)
        }
    }
    impl Default for Global {
        fn default() -> Self {
            Global
        }
    }

    #[allow(clippy::map_err_ignore)]
    pub fn do_alloc<A: AllocRef>(alloc: &A, layout: Layout) -> Result<NonNull<u8>, ()> {
        alloc.alloc(layout).map_err(|_| ())
    }
}
