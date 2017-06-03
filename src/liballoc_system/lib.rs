// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "alloc_system"]
#![crate_type = "rlib"]
#![no_std]
#![allocator]
#![deny(warnings)]
#![unstable(feature = "alloc_system",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "27783")]
#![feature(allocator)]
#![feature(staged_api)]
#![cfg_attr(any(unix, target_os = "redox"), feature(libc))]

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values. In practice, the alignment is a
// constant at the call site and the branch will be optimized out.
#[cfg(all(any(target_arch = "x86",
              target_arch = "arm",
              target_arch = "mips",
              target_arch = "powerpc",
              target_arch = "powerpc64",
              target_arch = "asmjs",
              target_arch = "wasm32")))]
const MIN_ALIGN: usize = 8;
#[cfg(all(any(target_arch = "x86_64",
              target_arch = "aarch64",
              target_arch = "mips64",
              target_arch = "s390x",
              target_arch = "sparc64")))]
const MIN_ALIGN: usize = 16;

#[no_mangle]
pub extern "C" fn __rust_allocate(size: usize, align: usize) -> *mut u8 {
    unsafe { imp::allocate(size, align) }
}

#[no_mangle]
pub extern "C" fn __rust_allocate_zeroed(size: usize, align: usize) -> *mut u8 {
    unsafe { imp::allocate_zeroed(size, align) }
}

#[no_mangle]
pub extern "C" fn __rust_deallocate(ptr: *mut u8, old_size: usize, align: usize) {
    unsafe { imp::deallocate(ptr, old_size, align) }
}

#[no_mangle]
pub extern "C" fn __rust_reallocate(ptr: *mut u8,
                                    old_size: usize,
                                    size: usize,
                                    align: usize)
                                    -> *mut u8 {
    unsafe { imp::reallocate(ptr, old_size, size, align) }
}

#[no_mangle]
pub extern "C" fn __rust_reallocate_inplace(ptr: *mut u8,
                                            old_size: usize,
                                            size: usize,
                                            align: usize)
                                            -> usize {
    unsafe { imp::reallocate_inplace(ptr, old_size, size, align) }
}

#[no_mangle]
pub extern "C" fn __rust_usable_size(size: usize, align: usize) -> usize {
    imp::usable_size(size, align)
}

#[cfg(any(unix, target_os = "redox"))]
mod imp {
    extern crate libc;

    use core::cmp;
    use core::ptr;
    use MIN_ALIGN;

    pub unsafe fn allocate(size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc::malloc(size as libc::size_t) as *mut u8
        } else {
            aligned_malloc(size, align)
        }
    }

    #[cfg(any(target_os = "android", target_os = "redox"))]
    unsafe fn aligned_malloc(size: usize, align: usize) -> *mut u8 {
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
        libc::memalign(align as libc::size_t, size as libc::size_t) as *mut u8
    }

    #[cfg(not(any(target_os = "android", target_os = "redox")))]
    unsafe fn aligned_malloc(size: usize, align: usize) -> *mut u8 {
        let mut out = ptr::null_mut();
        let ret = libc::posix_memalign(&mut out, align as libc::size_t, size as libc::size_t);
        if ret != 0 {
            ptr::null_mut()
        } else {
            out as *mut u8
        }
    }

    pub unsafe fn allocate_zeroed(size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc::calloc(size as libc::size_t, 1) as *mut u8
        } else {
            let ptr = aligned_malloc(size, align);
            if !ptr.is_null() {
                ptr::write_bytes(ptr, 0, size);
            }
            ptr
        }
    }

    pub unsafe fn reallocate(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc::realloc(ptr as *mut libc::c_void, size as libc::size_t) as *mut u8
        } else {
            let new_ptr = allocate(size, align);
            if !new_ptr.is_null() {
                ptr::copy(ptr, new_ptr, cmp::min(size, old_size));
                deallocate(ptr, old_size, align);
            }
            new_ptr
        }
    }

    pub unsafe fn reallocate_inplace(_ptr: *mut u8,
                                     old_size: usize,
                                     _size: usize,
                                     _align: usize)
                                     -> usize {
        old_size
    }

    pub unsafe fn deallocate(ptr: *mut u8, _old_size: usize, _align: usize) {
        libc::free(ptr as *mut libc::c_void)
    }

    pub fn usable_size(size: usize, _align: usize) -> usize {
        size
    }
}

#[cfg(windows)]
#[allow(bad_style)]
mod imp {
    use core::cmp::min;
    use core::ptr::copy_nonoverlapping;
    use MIN_ALIGN;

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
    unsafe fn allocate_with_flags(size: usize, align: usize, flags: DWORD) -> *mut u8 {
        if align <= MIN_ALIGN {
            HeapAlloc(GetProcessHeap(), flags, size as SIZE_T) as *mut u8
        } else {
            let ptr = HeapAlloc(GetProcessHeap(), flags, (size + align) as SIZE_T) as *mut u8;
            if ptr.is_null() {
                return ptr;
            }
            align_ptr(ptr, align)
        }
    }

    pub unsafe fn allocate(size: usize, align: usize) -> *mut u8 {
        allocate_with_flags(size, align, 0)
    }

    pub unsafe fn allocate_zeroed(size: usize, align: usize) -> *mut u8 {
        allocate_with_flags(size, align, HEAP_ZERO_MEMORY)
    }

    pub unsafe fn reallocate(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            HeapReAlloc(GetProcessHeap(), 0, ptr as LPVOID, size as SIZE_T) as *mut u8
        } else {
            let new = allocate(size, align);
            if !new.is_null() {
                copy_nonoverlapping(ptr, new, min(size, old_size));
                deallocate(ptr, old_size, align);
            }
            new
        }
    }

    pub unsafe fn reallocate_inplace(ptr: *mut u8,
                                     old_size: usize,
                                     size: usize,
                                     align: usize)
                                     -> usize {
        let new = if align <= MIN_ALIGN {
            HeapReAlloc(GetProcessHeap(),
                        HEAP_REALLOC_IN_PLACE_ONLY,
                        ptr as LPVOID,
                        size as SIZE_T) as *mut u8
        } else {
            let header = get_header(ptr);
            HeapReAlloc(GetProcessHeap(),
                        HEAP_REALLOC_IN_PLACE_ONLY,
                        header.0 as LPVOID,
                        size + align as SIZE_T) as *mut u8
        };
        if new.is_null() { old_size } else { size }
    }

    pub unsafe fn deallocate(ptr: *mut u8, _old_size: usize, align: usize) {
        if align <= MIN_ALIGN {
            let err = HeapFree(GetProcessHeap(), 0, ptr as LPVOID);
            debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
        } else {
            let header = get_header(ptr);
            let err = HeapFree(GetProcessHeap(), 0, header.0 as LPVOID);
            debug_assert!(err != 0, "Failed to free heap memory: {}", GetLastError());
        }
    }

    pub fn usable_size(size: usize, _align: usize) -> usize {
        size
    }
}
