pub use self::inner::*;

#[cfg(feature = "nightly")]
mod inner {
    pub use crate::alloc::alloc::{Alloc, Global};
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::{alloc, dealloc, Layout};
    use core::ptr::NonNull;

    pub trait Alloc {
        unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()>;
        unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);
    }

    #[derive(Copy, Clone)]
    pub struct Global;
    impl Alloc for Global {
        unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
            NonNull::new(alloc(layout)).ok_or(())
        }
        unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
            dealloc(ptr.as_ptr(), layout)
        }
    }
}
