use crate::alloc::Layout;

#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    let size = layout.size();
    let align = layout.align();
    unsafe { hermit_abi::malloc(size, align) }
}

#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    let size = layout.size();
    let align = layout.align();
    unsafe {
        hermit_abi::free(ptr, size, align);
    }
}

#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    let size = layout.size();
    let align = layout.align();
    unsafe { hermit_abi::realloc(ptr, size, align, new_size) }
}
