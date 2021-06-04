// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![no_std]
#![feature(allocator_api, rustc_private)]
#![cfg_attr(any(unix, target_os = "redox"), feature(libc))]

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[cfg(all(any(target_arch = "x86",
              target_arch = "arm",
              target_arch = "mips",
              target_arch = "powerpc",
              target_arch = "powerpc64")))]
const MIN_ALIGN: usize = 8;
#[cfg(all(any(target_arch = "x86_64",
              target_arch = "aarch64",
              target_arch = "mips64",
              target_arch = "s390x",
              target_arch = "sparc64")))]
const MIN_ALIGN: usize = 16;

pub struct System;
#[cfg(any(windows, unix, target_os = "redox"))]
mod realloc_fallback {
    use core::alloc::{GlobalAlloc, Layout};
    use core::cmp;
    use core::ptr;
    impl super::System {
        pub(crate) unsafe fn realloc_fallback(&self, ptr: *mut u8, old_layout: Layout,
                                              new_size: usize) -> *mut u8 {
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
    extern crate libc;
    use core::ptr;
    use MIN_ALIGN;
    use System;
    use core::alloc::{GlobalAlloc, Layout};
    unsafe impl GlobalAlloc for System {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
                libc::malloc(layout.size()) as *mut u8
            } else {
                #[cfg(target_os = "macos")]
                {
                    if layout.align() > (1 << 31) {
                        return ptr::null_mut()
                    }
                }
                aligned_malloc(&layout)
            }
        }
        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
                libc::calloc(layout.size(), 1) as *mut u8
            } else {
                let ptr = self.alloc(layout.clone());
                if !ptr.is_null() {
                    ptr::write_bytes(ptr, 0, layout.size());
                }
                ptr
            }
        }
        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            libc::free(ptr as *mut libc::c_void)
        }
        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
                libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8
            } else {
                self.realloc_fallback(ptr, layout, new_size)
            }
        }
    }
    #[cfg(any(target_os = "android",
              target_os = "hermit",
              target_os = "redox",
              target_os = "solaris"))]
    #[inline]
    unsafe fn aligned_malloc(layout: &Layout) -> *mut u8 {
        // On android we currently target API level 9 which unfortunately
        // doesn't have the `posix_memalign` API used below. Instead we use
        // `memalign`, but this unfortunately has the property on some systems
        // where the memory returned cannot be deallocated by `free`!
        //
        // Upon closer inspection, however, this appears to work just fine with
        // Android, so for this platform we should be fine to call `memalign`
        // (which is present in API level 9). Some helpful references could
        // possibly be chromium using memalign [1], attempts at documenting that
        // memalign + free is ok [2] [3], or the current source of chromium
        // which still uses memalign on android [4].
        //
        // [1]: https://codereview.chromium.org/10796020/
        // [2]: https://code.google.com/p/android/issues/detail?id=35391
        // [3]: https://bugs.chromium.org/p/chromium/issues/detail?id=138579
        // [4]: https://chromium.googlesource.com/chromium/src/base/+/master/
        //                                       /memory/aligned_memory.cc
        libc::memalign(layout.align(), layout.size()) as *mut u8
    }
    #[cfg(not(any(target_os = "android",
                  target_os = "hermit",
                  target_os = "redox",
                  target_os = "solaris")))]
    #[inline]
    unsafe fn aligned_malloc(layout: &Layout) -> *mut u8 {
        let mut out = ptr::null_mut();
        let ret = libc::posix_memalign(&mut out, layout.align(), layout.size());
        if ret != 0 {
            ptr::null_mut()
        } else {
            out as *mut u8
        }
    }
}
#[cfg(windows)]
#[allow(nonstandard_style)]
mod platform {
    use MIN_ALIGN;
    use System;
    use core::alloc::{GlobalAlloc, Layout};
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
        &mut *(ptr as *mut Header).offset(-1)
    }
    unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
        let aligned = ptr.add(align - (ptr as usize & (align - 1)));
        *get_header(aligned) = Header(ptr);
        aligned
    }
    #[inline]
    unsafe fn allocate_with_flags(layout: Layout, flags: DWORD) -> *mut u8 {
        let ptr = if layout.align() <= MIN_ALIGN {
            HeapAlloc(GetProcessHeap(), flags, layout.size())
        } else {
            let size = layout.size() + layout.align();
            let ptr = HeapAlloc(GetProcessHeap(), flags, size);
            if ptr.is_null() {
                ptr
            } else {
                align_ptr(ptr, layout.align())
            }
        };
        ptr as *mut u8
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
            if layout.align() <= MIN_ALIGN {
                let err = HeapFree(GetProcessHeap(), 0, ptr as LPVOID);
                debug_assert!(err != 0, "Failed to free heap memory: {}",
                              GetLastError());
            } else {
                let header = get_header(ptr);
                let err = HeapFree(GetProcessHeap(), 0, header.0 as LPVOID);
                debug_assert!(err != 0, "Failed to free heap memory: {}",
                              GetLastError());
            }
        }
        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if layout.align() <= MIN_ALIGN {
                HeapReAlloc(GetProcessHeap(), 0, ptr as LPVOID, new_size) as *mut u8
            } else {
                self.realloc_fallback(ptr, layout, new_size)
            }
        }
    }
}
