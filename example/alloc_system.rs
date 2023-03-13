// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: The Rust Project Developers (see https://thanks.rust-lang.org)

#![no_std]

pub struct System;

#[cfg(any(windows, unix, target_os = "redox"))]
mod realloc_fallback {
    use core::alloc::{GlobalAlloc, Layout};
    use core::cmp;
    use core::ptr;
    impl super::System {
        pub(crate) unsafe fn realloc_fallback(
            &self,
            ptr: *mut u8,
            old_layout: Layout,
            new_size: usize,
        ) -> *mut u8 {
            // Docs for GlobalAlloc::realloc require this to be valid:
            let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());
            let new_ptr = GlobalAlloc::alloc(self, new_layout);
            if !new_ptr.is_null() {
                let size = cmp::min(old_layout.size(), new_size);
                ptr::copy_nonoverlapping(ptr, new_ptr, size);
                GlobalAlloc::dealloc(self, ptr, old_layout);
            }
            new_ptr
        }
    }
}
#[cfg(any(unix, target_os = "redox"))]
mod platform {
    use core::alloc::{GlobalAlloc, Layout};
    use core::ffi::c_void;
    use core::ptr;
    use System;
    extern "C" {
        fn posix_memalign(memptr: *mut *mut c_void, align: usize, size: usize) -> i32;
        fn free(p: *mut c_void);
    }
    unsafe impl GlobalAlloc for System {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            aligned_malloc(&layout)
        }
        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = self.alloc(layout.clone());
            if !ptr.is_null() {
                ptr::write_bytes(ptr, 0, layout.size());
            }
            ptr
        }
        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            free(ptr as *mut c_void)
        }
        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            self.realloc_fallback(ptr, layout, new_size)
        }
    }
    unsafe fn aligned_malloc(layout: &Layout) -> *mut u8 {
        let mut out = ptr::null_mut();
        let ret = posix_memalign(&mut out, layout.align(), layout.size());
        if ret != 0 { ptr::null_mut() } else { out as *mut u8 }
    }
}
#[cfg(windows)]
#[allow(nonstandard_style)]
mod platform {
    use core::alloc::{GlobalAlloc, Layout};
    use System;
    type LPVOID = *mut u8;
    type HANDLE = LPVOID;
    type SIZE_T = usize;
    type DWORD = u32;
    type BOOL = i32;
    extern "system" {
        fn GetProcessHeap() -> HANDLE;
        fn HeapAlloc(hHeap: HANDLE, dwFlags: DWORD, dwBytes: SIZE_T) -> LPVOID;
        fn HeapReAlloc(hHeap: HANDLE, dwFlags: DWORD, lpMem: LPVOID, dwBytes: SIZE_T) -> LPVOID;
        fn HeapFree(hHeap: HANDLE, dwFlags: DWORD, lpMem: LPVOID) -> BOOL;
        fn GetLastError() -> DWORD;
    }
    #[repr(C)]
    struct Header(*mut u8);
    const HEAP_ZERO_MEMORY: DWORD = 0x00000008;
    unsafe fn get_header<'a>(ptr: *mut u8) -> &'a mut Header {
        &mut *(ptr as *mut Header).sub(1)
    }
    unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
        let aligned = ptr.add(align - (ptr as usize & (align - 1)));
        *get_header(aligned) = Header(ptr);
        aligned
    }
    #[inline]
    unsafe fn allocate_with_flags(layout: Layout, flags: DWORD) -> *mut u8 {
        let size = layout.size() + layout.align();
        let ptr = HeapAlloc(GetProcessHeap(), flags, size);
        (if ptr.is_null() { ptr } else { align_ptr(ptr, layout.align()) }) as *mut u8
    }
    unsafe impl GlobalAlloc for System {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            allocate_with_flags(layout, 0)
        }
        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            allocate_with_flags(layout, HEAP_ZERO_MEMORY)
        }
        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            let header = get_header(ptr);
            let err = HeapFree(GetProcessHeap(), 0, header.0 as LPVOID);
            debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
        }
        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            self.realloc_fallback(ptr, layout, new_size)
        }
    }
}
