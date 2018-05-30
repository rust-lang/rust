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
#![allow(unused_attributes)]
#![unstable(feature = "alloc_system",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "32838")]
#![feature(global_allocator)]
#![feature(allocator_api)]
#![feature(core_intrinsics)]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![cfg_attr(any(unix, target_os = "cloudabi", target_os = "redox"), feature(libc))]
#![rustc_alloc_kind = "lib"]

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[cfg(all(any(target_arch = "x86",
              target_arch = "arm",
              target_arch = "mips",
              target_arch = "powerpc",
              target_arch = "powerpc64",
              target_arch = "asmjs",
              target_arch = "wasm32")))]
#[allow(dead_code)]
const MIN_ALIGN: usize = 8;
#[cfg(all(any(target_arch = "x86_64",
              target_arch = "aarch64",
              target_arch = "mips64",
              target_arch = "s390x",
              target_arch = "sparc64")))]
#[allow(dead_code)]
const MIN_ALIGN: usize = 16;

use core::alloc::{Alloc, GlobalAlloc, AllocErr, Layout, Opaque};
use core::ptr::NonNull;

#[unstable(feature = "allocator_api", issue = "32838")]
pub struct System;

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl Alloc for System {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
        GlobalAlloc::alloc(self, layout)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
        GlobalAlloc::alloc_zeroed(self, layout)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<Opaque>, layout: Layout) {
        GlobalAlloc::dealloc(self, ptr, layout)
    }

    #[inline]
    unsafe fn realloc(&mut self,
                      ptr: NonNull<Opaque>,
                      layout: Layout,
                      new_size: usize) -> Result<NonNull<Opaque>, AllocErr> {
        GlobalAlloc::realloc(self, ptr, layout, new_size)
    }
}

#[cfg(any(windows, unix, target_os = "cloudabi", target_os = "redox"))]
mod realloc_fallback {
    use core::alloc::{GlobalAlloc, Opaque, Layout, AllocErr};
    use core::cmp;
    use core::ptr::{self, NonNull};

    impl super::System {
        pub(crate) unsafe fn realloc_fallback(
            &self,
            ptr: NonNull<Opaque>,
            old_layout: Layout,
            new_size: usize,
        ) -> Result<NonNull<Opaque>, AllocErr> {
            // Docs for GlobalAlloc::realloc require this to be valid:
            let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());

            let new_ptr = GlobalAlloc::alloc(self, new_layout)?;
            let size = cmp::min(old_layout.size(), new_size);
            ptr::copy_nonoverlapping(ptr.as_ptr() as *mut u8, new_ptr.as_ptr() as *mut u8, size);
            GlobalAlloc::dealloc(self, ptr, old_layout);
            Ok(new_ptr)
        }
    }
}

#[cfg(any(unix, target_os = "cloudabi", target_os = "redox"))]
mod platform {
    extern crate libc;

    use core::ptr;

    use MIN_ALIGN;
    use System;
    use core::alloc::{GlobalAlloc, Layout, Opaque, AllocErr};
    use core::ptr::NonNull;

    #[unstable(feature = "allocator_api", issue = "32838")]
    unsafe impl GlobalAlloc for System {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
            if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
                NonNull::new(libc::malloc(layout.size()) as *mut Opaque).ok_or(AllocErr)
            } else {
                #[cfg(target_os = "macos")]
                {
                    if layout.align() > (1 << 31) {
                        return Err(AllocErr);
                    }
                }
                aligned_malloc(&layout)
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
            if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
                NonNull::new(libc::calloc(layout.size(), 1) as *mut Opaque).ok_or(AllocErr)
            } else {
                let ptr = self.alloc(layout.clone())?;
                ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, layout.size());
                Ok(ptr)
            }
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: NonNull<Opaque>, _layout: Layout) {
            libc::free(ptr.as_ptr() as *mut libc::c_void)
        }

        #[inline]
        unsafe fn realloc(
            &self,
            ptr: NonNull<Opaque>,
            layout: Layout,
            new_size: usize,
        ) -> Result<NonNull<Opaque>, AllocErr> {
            if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
                NonNull::new(
                    libc::realloc(ptr.as_ptr() as *mut libc::c_void, new_size) as *mut Opaque
                ).ok_or(AllocErr)
            } else {
                self.realloc_fallback(ptr, layout, new_size)
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "redox", target_os = "solaris"))]
    #[inline]
    unsafe fn aligned_malloc(layout: &Layout) -> Result<NonNull<Opaque>, AllocErr> {
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
        NonNull::new(libc::memalign(layout.align(), layout.size()) as *mut Opaque).ok_or(AllocErr)
    }

    #[cfg(not(any(target_os = "android", target_os = "redox", target_os = "solaris")))]
    #[inline]
    unsafe fn aligned_malloc(layout: &Layout) -> Result<NonNull<Opaque>, AllocErr> {
        let mut out = ptr::null_mut();
        let ret = libc::posix_memalign(&mut out, layout.align(), layout.size());
        if ret != 0 {
            Err(AllocErr)
        } else {
            Ok(NonNull::new_unchecked(out as *mut Opaque))
        }
    }
}

#[cfg(windows)]
#[allow(bad_style)]
mod platform {
    use MIN_ALIGN;
    use System;
    use core::alloc::{GlobalAlloc, Opaque, Layout, AllocErr};
    use core::ptr::NonNull;

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
        let aligned = ptr.offset((align - (ptr as usize & (align - 1))) as isize);
        *get_header(aligned) = Header(ptr);
        aligned
    }

    #[inline]
    unsafe fn allocate_with_flags(
        layout: Layout,
        flags: DWORD,
    ) -> Result<NonNull<Opaque>, AllocErr> {
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
        NonNull::new(ptr as *mut Opaque).ok_or(AllocErr)
    }

    #[unstable(feature = "allocator_api", issue = "32838")]
    unsafe impl GlobalAlloc for System {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
            allocate_with_flags(layout, 0)
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
            allocate_with_flags(layout, HEAP_ZERO_MEMORY)
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: NonNull<Opaque>, layout: Layout) {
            if layout.align() <= MIN_ALIGN {
                let err = HeapFree(GetProcessHeap(), 0, ptr.as_ptr() as LPVOID);
                debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
            } else {
                let header = get_header(ptr.as_ptr() as *mut u8);
                let err = HeapFree(GetProcessHeap(), 0, header.0 as LPVOID);
                debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
            }
        }

        #[inline]
        unsafe fn realloc(
            &self,
            ptr: NonNull<Opaque>,
            layout: Layout,
            new_size: usize,
        ) -> Result<NonNull<Opaque>, AllocErr> {
            if layout.align() <= MIN_ALIGN {
                NonNull::new(
                    HeapReAlloc(GetProcessHeap(), 0, ptr.as_ptr() as LPVOID, new_size)
                        as *mut Opaque
                ).ok_or(AllocErr)
            } else {
                self.realloc_fallback(ptr, layout, new_size)
            }
        }
    }
}

// This is an implementation of a global allocator on the wasm32 platform when
// emscripten is not in use. In that situation there's no actual runtime for us
// to lean on for allocation, so instead we provide our own!
//
// The wasm32 instruction set has two instructions for getting the current
// amount of memory and growing the amount of memory. These instructions are the
// foundation on which we're able to build an allocator, so we do so! Note that
// the instructions are also pretty "global" and this is the "global" allocator
// after all!
//
// The current allocator here is the `dlmalloc` crate which we've got included
// in the rust-lang/rust repository as a submodule. The crate is a port of
// dlmalloc.c from C to Rust and is basically just so we can have "pure Rust"
// for now which is currently technically required (can't link with C yet).
//
// The crate itself provides a global allocator which on wasm has no
// synchronization as there are no threads!
#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
mod platform {
    extern crate dlmalloc;

    use core::alloc::{GlobalAlloc, Layout, Opaque, AllocErr};
    use core::ptr::NonNull;
    use System;

    // No need for synchronization here as wasm is currently single-threaded
    static mut DLMALLOC: dlmalloc::Dlmalloc = dlmalloc::DLMALLOC_INIT;

    #[unstable(feature = "allocator_api", issue = "32838")]
    unsafe impl GlobalAlloc for System {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
            NonNull::new(DLMALLOC.malloc(layout.size(), layout.align()) as *mut Opaque)
                .ok_or(AllocErr)
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> Result<NonNull<Opaque>, AllocErr> {
            NonNull::new(DLMALLOC.calloc(layout.size(), layout.align()) as *mut Opaque)
                .ok_or(AllocErr)
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: NonNull<Opaque>, layout: Layout) {
            DLMALLOC.free(ptr.as_ptr() as *mut u8, layout.size(), layout.align())
        }

        #[inline]
        unsafe fn realloc(
            &self,
            ptr: NonNull<Opaque>,
            layout: Layout,
            new_size: usize,
        ) -> Result<NonNull<Opaque>, AllocErr> {
            NonNull::new(
                DLMALLOC.realloc(ptr.as_ptr() as *mut u8, layout.size(), layout.align(), new_size)
                    as *mut Opaque,
            ).ok_or(AllocErr)
        }
    }
}
