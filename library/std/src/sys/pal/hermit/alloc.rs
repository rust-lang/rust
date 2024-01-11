use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sys::hermit::abi;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        abi::malloc(layout.size(), layout.align())
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let addr = abi::malloc(layout.size(), layout.align());

        if !addr.is_null() {
            ptr::write_bytes(addr, 0x00, layout.size());
        }

        addr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        abi::free(ptr, layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        abi::realloc(ptr, layout.size(), layout.align(), new_size)
    }
}
