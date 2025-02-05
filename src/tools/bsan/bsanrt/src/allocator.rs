use core::alloc::{GlobalAlloc, Layout};
use core::mem;
pub struct BsanAlloc {}


unsafe impl GlobalAlloc for BsanAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        mem::transmute(jemalloc_sys::aligned_alloc(layout.align(), layout.size()))
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        jemalloc_sys::free(mem::transmute(ptr));
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let mut result: *mut u8 = mem::transmute(jemalloc_sys::calloc(0, layout.size()));
        if (layout.size() > 0 && !result.is_null()) {
            for i in 0..layout.size() {
                *result.add(i) = 0;
            }
        }
        result
    }

    unsafe fn realloc(&self, ptr: *mut u8, _layout: Layout, new_size: usize) -> *mut u8 {
        mem::transmute(jemalloc_sys::realloc(mem::transmute(ptr), new_size))
    }
}
