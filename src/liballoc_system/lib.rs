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
#![deny(warnings)]
#![unstable(feature = "alloc_system",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "32838")]
#![feature(global_allocator)]
#![feature(allocator_api)]
#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![cfg_attr(any(unix, target_os = "redox"), feature(libc))]
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

extern crate alloc;

use self::alloc::heap::{Alloc, AllocErr, Layout, Excess, CannotReallocInPlace};

#[unstable(feature = "allocator_api", issue = "32838")]
pub struct System;

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl Alloc for System {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        (&*self).alloc(layout)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout)
        -> Result<*mut u8, AllocErr>
    {
        (&*self).alloc_zeroed(layout)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        (&*self).dealloc(ptr, layout)
    }

    #[inline]
    unsafe fn realloc(&mut self,
                      ptr: *mut u8,
                      old_layout: Layout,
                      new_layout: Layout) -> Result<*mut u8, AllocErr> {
        (&*self).realloc(ptr, old_layout, new_layout)
    }

    fn oom(&mut self, err: AllocErr) -> ! {
        (&*self).oom(err)
    }

    #[inline]
    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        (&self).usable_size(layout)
    }

    #[inline]
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
        (&*self).alloc_excess(layout)
    }

    #[inline]
    unsafe fn realloc_excess(&mut self,
                             ptr: *mut u8,
                             layout: Layout,
                             new_layout: Layout) -> Result<Excess, AllocErr> {
        (&*self).realloc_excess(ptr, layout, new_layout)
    }

    #[inline]
    unsafe fn grow_in_place(&mut self,
                            ptr: *mut u8,
                            layout: Layout,
                            new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        (&*self).grow_in_place(ptr, layout, new_layout)
    }

    #[inline]
    unsafe fn shrink_in_place(&mut self,
                              ptr: *mut u8,
                              layout: Layout,
                              new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        (&*self).shrink_in_place(ptr, layout, new_layout)
    }
}

#[cfg(any(unix, target_os = "redox"))]
mod platform {
    extern crate libc;

    use core::cmp;
    use core::ptr;

    use MIN_ALIGN;
    use System;
    use alloc::heap::{Alloc, AllocErr, Layout};

    #[unstable(feature = "allocator_api", issue = "32838")]
    unsafe impl<'a> Alloc for &'a System {
        #[inline]
        unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
            let ptr = if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
                libc::malloc(layout.size()) as *mut u8
            } else {
                aligned_malloc(&layout)
            };
            if !ptr.is_null() {
                Ok(ptr)
            } else {
                Err(AllocErr::Exhausted { request: layout })
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&mut self, layout: Layout)
            -> Result<*mut u8, AllocErr>
        {
            if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
                let ptr = libc::calloc(layout.size(), 1) as *mut u8;
                if !ptr.is_null() {
                    Ok(ptr)
                } else {
                    Err(AllocErr::Exhausted { request: layout })
                }
            } else {
                let ret = self.alloc(layout.clone());
                if let Ok(ptr) = ret {
                    ptr::write_bytes(ptr, 0, layout.size());
                }
                ret
            }
        }

        #[inline]
        unsafe fn dealloc(&mut self, ptr: *mut u8, _layout: Layout) {
            libc::free(ptr as *mut libc::c_void)
        }

        #[inline]
        unsafe fn realloc(&mut self,
                          ptr: *mut u8,
                          old_layout: Layout,
                          new_layout: Layout) -> Result<*mut u8, AllocErr> {
            if old_layout.align() != new_layout.align() {
                return Err(AllocErr::Unsupported {
                    details: "cannot change alignment on `realloc`",
                })
            }

            if new_layout.align() <= MIN_ALIGN  && new_layout.align() <= new_layout.size(){
                let ptr = libc::realloc(ptr as *mut libc::c_void, new_layout.size());
                if !ptr.is_null() {
                    Ok(ptr as *mut u8)
                } else {
                    Err(AllocErr::Exhausted { request: new_layout })
                }
            } else {
                let res = self.alloc(new_layout.clone());
                if let Ok(new_ptr) = res {
                    let size = cmp::min(old_layout.size(), new_layout.size());
                    ptr::copy_nonoverlapping(ptr, new_ptr, size);
                    self.dealloc(ptr, old_layout);
                }
                res
            }
        }

        fn oom(&mut self, err: AllocErr) -> ! {
            use core::fmt::{self, Write};

            // Print a message to stderr before aborting to assist with
            // debugging. It is critical that this code does not allocate any
            // memory since we are in an OOM situation. Any errors are ignored
            // while printing since there's nothing we can do about them and we
            // are about to exit anyways.
            drop(writeln!(Stderr, "fatal runtime error: {}", err));
            unsafe {
                ::core::intrinsics::abort();
            }

            struct Stderr;

            impl Write for Stderr {
                #[cfg(target_os = "cloudabi")]
                fn write_str(&mut self, _: &str) -> fmt::Result {
                    // CloudABI does not have any reserved file descriptor
                    // numbers. We should not attempt to write to file
                    // descriptor #2, as it may be associated with any kind of
                    // resource.
                    Ok(())
                }

                #[cfg(not(target_os = "cloudabi"))]
                fn write_str(&mut self, s: &str) -> fmt::Result {
                    unsafe {
                        libc::write(libc::STDERR_FILENO,
                                    s.as_ptr() as *const libc::c_void,
                                    s.len());
                    }
                    Ok(())
                }
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "redox", target_os = "solaris"))]
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

    #[cfg(not(any(target_os = "android", target_os = "redox", target_os = "solaris")))]
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
#[allow(bad_style)]
mod platform {
    use core::cmp;
    use core::ptr;

    use MIN_ALIGN;
    use System;
    use alloc::heap::{Alloc, AllocErr, Layout, CannotReallocInPlace};

    type LPVOID = *mut u8;
    type HANDLE = LPVOID;
    type SIZE_T = usize;
    type DWORD = u32;
    type BOOL = i32;
    type LPDWORD = *mut DWORD;
    type LPOVERLAPPED = *mut u8;

    const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;

    extern "system" {
        fn GetProcessHeap() -> HANDLE;
        fn HeapAlloc(hHeap: HANDLE, dwFlags: DWORD, dwBytes: SIZE_T) -> LPVOID;
        fn HeapReAlloc(hHeap: HANDLE, dwFlags: DWORD, lpMem: LPVOID, dwBytes: SIZE_T) -> LPVOID;
        fn HeapFree(hHeap: HANDLE, dwFlags: DWORD, lpMem: LPVOID) -> BOOL;
        fn GetLastError() -> DWORD;
        fn WriteFile(hFile: HANDLE,
                     lpBuffer: LPVOID,
                     nNumberOfBytesToWrite: DWORD,
                     lpNumberOfBytesWritten: LPDWORD,
                     lpOverlapped: LPOVERLAPPED)
                     -> BOOL;
        fn GetStdHandle(which: DWORD) -> HANDLE;
    }

    #[repr(C)]
    struct Header(*mut u8);

    const HEAP_ZERO_MEMORY: DWORD = 0x00000008;
    const HEAP_REALLOC_IN_PLACE_ONLY: DWORD = 0x00000010;

    unsafe fn get_header<'a>(ptr: *mut u8) -> &'a mut Header {
        &mut *(ptr as *mut Header).offset(-1)
    }

    unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
        let aligned = ptr.offset((align - (ptr as usize & (align - 1))) as isize);
        *get_header(aligned) = Header(ptr);
        aligned
    }

    #[inline]
    unsafe fn allocate_with_flags(layout: Layout, flags: DWORD)
        -> Result<*mut u8, AllocErr>
    {
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
        if ptr.is_null() {
            Err(AllocErr::Exhausted { request: layout })
        } else {
            Ok(ptr as *mut u8)
        }
    }

    #[unstable(feature = "allocator_api", issue = "32838")]
    unsafe impl<'a> Alloc for &'a System {
        #[inline]
        unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
            allocate_with_flags(layout, 0)
        }

        #[inline]
        unsafe fn alloc_zeroed(&mut self, layout: Layout)
            -> Result<*mut u8, AllocErr>
        {
            allocate_with_flags(layout, HEAP_ZERO_MEMORY)
        }

        #[inline]
        unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
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
        unsafe fn realloc(&mut self,
                          ptr: *mut u8,
                          old_layout: Layout,
                          new_layout: Layout) -> Result<*mut u8, AllocErr> {
            if old_layout.align() != new_layout.align() {
                return Err(AllocErr::Unsupported {
                    details: "cannot change alignment on `realloc`",
                })
            }

            if new_layout.align() <= MIN_ALIGN {
                let ptr = HeapReAlloc(GetProcessHeap(),
                                      0,
                                      ptr as LPVOID,
                                      new_layout.size());
                if !ptr.is_null() {
                    Ok(ptr as *mut u8)
                } else {
                    Err(AllocErr::Exhausted { request: new_layout })
                }
            } else {
                let res = self.alloc(new_layout.clone());
                if let Ok(new_ptr) = res {
                    let size = cmp::min(old_layout.size(), new_layout.size());
                    ptr::copy_nonoverlapping(ptr, new_ptr, size);
                    self.dealloc(ptr, old_layout);
                }
                res
            }
        }

        #[inline]
        unsafe fn grow_in_place(&mut self,
                                ptr: *mut u8,
                                layout: Layout,
                                new_layout: Layout) -> Result<(), CannotReallocInPlace> {
            self.shrink_in_place(ptr, layout, new_layout)
        }

        #[inline]
        unsafe fn shrink_in_place(&mut self,
                                  ptr: *mut u8,
                                  old_layout: Layout,
                                  new_layout: Layout) -> Result<(), CannotReallocInPlace> {
            if old_layout.align() != new_layout.align() {
                return Err(CannotReallocInPlace)
            }

            let new = if new_layout.align() <= MIN_ALIGN {
                HeapReAlloc(GetProcessHeap(),
                            HEAP_REALLOC_IN_PLACE_ONLY,
                            ptr as LPVOID,
                            new_layout.size())
            } else {
                let header = get_header(ptr);
                HeapReAlloc(GetProcessHeap(),
                            HEAP_REALLOC_IN_PLACE_ONLY,
                            header.0 as LPVOID,
                            new_layout.size() + new_layout.align())
            };
            if new.is_null() {
                Err(CannotReallocInPlace)
            } else {
                Ok(())
            }
        }

        fn oom(&mut self, err: AllocErr) -> ! {
            use core::fmt::{self, Write};

            // Same as with unix we ignore all errors here
            drop(writeln!(Stderr, "fatal runtime error: {}", err));
            unsafe {
                ::core::intrinsics::abort();
            }

            struct Stderr;

            impl Write for Stderr {
                fn write_str(&mut self, s: &str) -> fmt::Result {
                    unsafe {
                        // WriteFile silently fails if it is passed an invalid
                        // handle, so there is no need to check the result of
                        // GetStdHandle.
                        WriteFile(GetStdHandle(STD_ERROR_HANDLE),
                                  s.as_ptr() as LPVOID,
                                  s.len() as DWORD,
                                  ptr::null_mut(),
                                  ptr::null_mut());
                    }
                    Ok(())
                }
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

    use alloc::heap::{Alloc, AllocErr, Layout, Excess, CannotReallocInPlace};
    use System;
    use self::dlmalloc::GlobalDlmalloc;

    #[unstable(feature = "allocator_api", issue = "32838")]
    unsafe impl<'a> Alloc for &'a System {
        #[inline]
        unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
            GlobalDlmalloc.alloc(layout)
        }

        #[inline]
        unsafe fn alloc_zeroed(&mut self, layout: Layout)
            -> Result<*mut u8, AllocErr>
        {
            GlobalDlmalloc.alloc_zeroed(layout)
        }

        #[inline]
        unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
            GlobalDlmalloc.dealloc(ptr, layout)
        }

        #[inline]
        unsafe fn realloc(&mut self,
                          ptr: *mut u8,
                          old_layout: Layout,
                          new_layout: Layout) -> Result<*mut u8, AllocErr> {
            GlobalDlmalloc.realloc(ptr, old_layout, new_layout)
        }

        #[inline]
        fn usable_size(&self, layout: &Layout) -> (usize, usize) {
            GlobalDlmalloc.usable_size(layout)
        }

        #[inline]
        unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
            GlobalDlmalloc.alloc_excess(layout)
        }

        #[inline]
        unsafe fn realloc_excess(&mut self,
                                 ptr: *mut u8,
                                 layout: Layout,
                                 new_layout: Layout) -> Result<Excess, AllocErr> {
            GlobalDlmalloc.realloc_excess(ptr, layout, new_layout)
        }

        #[inline]
        unsafe fn grow_in_place(&mut self,
                                ptr: *mut u8,
                                layout: Layout,
                                new_layout: Layout) -> Result<(), CannotReallocInPlace> {
            GlobalDlmalloc.grow_in_place(ptr, layout, new_layout)
        }

        #[inline]
        unsafe fn shrink_in_place(&mut self,
                                  ptr: *mut u8,
                                  layout: Layout,
                                  new_layout: Layout) -> Result<(), CannotReallocInPlace> {
            GlobalDlmalloc.shrink_in_place(ptr, layout, new_layout)
        }
    }
}
