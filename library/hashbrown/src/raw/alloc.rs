pub use self::inner::*;

#[cfg(feature = "nightly")]
mod inner {
    pub use crate::alloc::alloc::{Alloc, Global};
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use core::ptr::NonNull;
    use crate::alloc::alloc::{Layout, alloc, dealloc};

    pub trait Alloc {
        unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()>;
        unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);
    }

    #[derive(Copy, Clone)]
    pub struct Global;
    impl Alloc for Global {
        unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
            Ok(NonNull::new_unchecked(alloc(layout)))
        }
        unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
            dealloc(ptr.as_ptr(), layout)
        }
    }
}

