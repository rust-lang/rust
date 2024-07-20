use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sys::common::alloc::{realloc_fallback, MIN_ALIGN};

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // jemalloc provides alignment less than MIN_ALIGN for small allocations.
        // So only rely on MIN_ALIGN if size >= align.
        // Also see <https://github.com/rust-lang/rust/issues/45955> and
        // <https://github.com/rust-lang/rust/issues/62251#issuecomment-507580914>.
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            unsafe { libc::malloc(layout.size()) as *mut u8 }
        } else {
            unsafe { aligned_malloc(&layout) }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // See the comment above in `alloc` for why this check looks the way it does.
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            unsafe { libc::calloc(layout.size(), 1) as *mut u8 }
        } else {
            let ptr = unsafe { self.alloc(layout) };
            if !ptr.is_null() {
                unsafe { ptr::write_bytes(ptr, 0, layout.size()) };
            }
            ptr
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { libc::free(ptr as *mut libc::c_void) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            unsafe { libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8 }
        } else {
            unsafe { realloc_fallback(self, ptr, layout, new_size) }
        }
    }
}

#[inline]
unsafe fn aligned_malloc(layout: &Layout) -> *mut u8 {
    let mut out = ptr::null_mut();
    // posix_memalign requires that the alignment be a multiple of `sizeof(void*)`.
    // Since these are all powers of 2, we can just use max.
    let align = layout.align().max(crate::mem::size_of::<usize>());
    let ret = unsafe { libc::posix_memalign(&mut out, align, layout.size()) };
    if ret != 0 { ptr::null_mut() } else { out as *mut u8 }
}
