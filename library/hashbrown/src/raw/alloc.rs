pub use self::inner::*;

#[cfg(feature = "nightly")]
mod inner {
    pub use crate::alloc::alloc::{AllocInit, AllocRef, Global, MemoryBlock};
}

#[cfg(not(feature = "nightly"))]
mod inner {
    use crate::alloc::alloc::{alloc, dealloc, Layout};
    use core::ptr::NonNull;

    #[derive(Debug, Copy, Clone)]
    pub enum AllocInit {
        Uninitialized,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct MemoryBlock {
        pub ptr: NonNull<u8>,
        pub size: usize,
    }

    pub trait AllocRef {
        unsafe fn alloc(&mut self, layout: Layout, init: AllocInit) -> Result<MemoryBlock, ()>;
        unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);
    }

    #[derive(Copy, Clone)]
    pub struct Global;
    impl AllocRef for Global {
        unsafe fn alloc(&mut self, layout: Layout, _init: AllocInit) -> Result<MemoryBlock, ()> {
            let ptr = NonNull::new(alloc(layout)).ok_or(())?;
            Ok(MemoryBlock {
                ptr,
                size: layout.size(),
            })
        }
        unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
            dealloc(ptr.as_ptr(), layout)
        }
    }
    impl Default for Global {
        fn default() -> Self {
            Global
        }
    }
}
