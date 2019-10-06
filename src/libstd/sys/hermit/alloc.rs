use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;

extern "C" {
    fn sys_malloc(size: usize, align: usize) -> *mut u8;
    fn sys_realloc(ptr: *mut u8, size: usize, align: usize, new_size: usize) -> *mut u8;
    fn sys_free(ptr: *mut u8, size: usize, align: usize);
}

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        sys_malloc(layout.size(), layout.align())
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let addr = sys_malloc(layout.size(), layout.align());

        if !addr.is_null() {
            ptr::write_bytes(
                addr,
                0x00,
                layout.size()
            );
        }

        addr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        sys_free(ptr, layout.size(), layout.align())
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        sys_realloc(ptr, layout.size(), layout.align(), new_size)
    }
}
