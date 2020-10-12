pub use self::inner::*;

#[cfg(feature = "nightly")]
mod inner {
    pub use crate::alloc::alloc::{AllocError, AllocRef, Global};
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::{alloc, dealloc, Layout};
    use core::ptr;
    use core::ptr::NonNull;

    pub struct AllocError;

    pub unsafe trait AllocRef {
        fn alloc(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;
        unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);
    }

    #[derive(Copy, Clone)]
    pub struct Global;
    unsafe impl AllocRef for Global {
        fn alloc(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            unsafe {
                let ptr = alloc(layout);
                if ptr.is_null() {
                    return Err(AllocError);
                }
                let slice = ptr::slice_from_raw_parts_mut(ptr, layout.size());
                Ok(NonNull::new_unchecked(slice))
            }
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
}
