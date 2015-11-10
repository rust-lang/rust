// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "alloc_system"]
#![crate_type = "rlib"]
#![staged_api]
#![no_std]
#![cfg_attr(not(stage0), allocator)]
#![cfg_attr(stage0, allow(improper_ctypes))]
#![unstable(feature = "alloc_system",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "27783")]
#![feature(allocator)]
#![feature(libc)]
#![feature(no_std)]
#![feature(staged_api)]

extern crate libc;

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values. In practice, the alignment is a
// constant at the call site and the branch will be optimized out.
#[cfg(all(any(target_arch = "arm",
              target_arch = "mips",
              target_arch = "mipsel",
              target_arch = "powerpc")))]
const MIN_ALIGN: usize = 8;
#[cfg(all(any(target_arch = "x86",
              target_arch = "x86_64",
              target_arch = "aarch64")))]
const MIN_ALIGN: usize = 16;

#[no_mangle]
pub extern "C" fn __rust_allocate(size: usize, align: usize) -> *mut u8 {
    unsafe { imp::allocate(size, align) }
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

#[cfg(unix)]
mod imp {
    use core::cmp;
    use core::ptr;
    use libc;
    use MIN_ALIGN;

    extern {
        // Apparently android doesn't have posix_memalign
        #[cfg(target_os = "android")]
        fn memalign(align: libc::size_t, size: libc::size_t) -> *mut libc::c_void;

        #[cfg(not(target_os = "android"))]
        fn posix_memalign(memptr: *mut *mut libc::c_void,
                          align: libc::size_t,
                          size: libc::size_t)
                          -> libc::c_int;
    }

    pub unsafe fn allocate(size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc::malloc(size as libc::size_t) as *mut u8
        } else {
            #[cfg(target_os = "android")]
            unsafe fn more_aligned_malloc(size: usize, align: usize) -> *mut u8 {
                memalign(align as libc::size_t, size as libc::size_t) as *mut u8
            }
            #[cfg(not(target_os = "android"))]
            unsafe fn more_aligned_malloc(size: usize, align: usize) -> *mut u8 {
                let mut out = ptr::null_mut();
                let ret = posix_memalign(&mut out, align as libc::size_t, size as libc::size_t);
                if ret != 0 {
                    ptr::null_mut()
                } else {
                    out as *mut u8
                }
            }
            more_aligned_malloc(size, align)
        }
    }

    pub unsafe fn reallocate(ptr: *mut u8, old_size: usize, size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc::realloc(ptr as *mut libc::c_void, size as libc::size_t) as *mut u8
        } else {
            let new_ptr = allocate(size, align);
            ptr::copy(ptr, new_ptr, cmp::min(size, old_size));
            deallocate(ptr, old_size, align);
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
    }

    #[repr(C)]
    struct Header(*mut u8);

    const HEAP_REALLOC_IN_PLACE_ONLY: DWORD = 0x00000010;

    unsafe fn get_header<'a>(ptr: *mut u8) -> &'a mut Header {
        &mut *(ptr as *mut Header).offset(-1)
    }

    unsafe fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
        let aligned = ptr.offset((align - (ptr as usize & (align - 1))) as isize);
        *get_header(aligned) = Header(ptr);
        aligned
    }

    pub unsafe fn allocate(size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            HeapAlloc(GetProcessHeap(), 0, size as SIZE_T) as *mut u8
        } else {
            let ptr = HeapAlloc(GetProcessHeap(), 0, (size + align) as SIZE_T) as *mut u8;
            if ptr.is_null() {
                return ptr
            }
            align_ptr(ptr, align)
        }
    }

    pub unsafe fn reallocate(ptr: *mut u8, _old_size: usize, size: usize, align: usize) -> *mut u8 {
        if align <= MIN_ALIGN {
            HeapReAlloc(GetProcessHeap(), 0, ptr as LPVOID, size as SIZE_T) as *mut u8
        } else {
            let header = get_header(ptr);
            let new = HeapReAlloc(GetProcessHeap(),
                                  0,
                                  header.0 as LPVOID,
                                  (size + align) as SIZE_T) as *mut u8;
            if new.is_null() {
                return new
            }
            align_ptr(new, align)
        }
    }

    pub unsafe fn reallocate_inplace(ptr: *mut u8,
                                     old_size: usize,
                                     size: usize,
                                     align: usize)
                                     -> usize {
        if align <= MIN_ALIGN {
            let new = HeapReAlloc(GetProcessHeap(),
                                  HEAP_REALLOC_IN_PLACE_ONLY,
                                  ptr as LPVOID,
                                  size as SIZE_T) as *mut u8;
            if new.is_null() {
                old_size
            } else {
                size
            }
        } else {
            old_size
        }
    }

    pub unsafe fn deallocate(ptr: *mut u8, _old_size: usize, align: usize) {
        if align <= MIN_ALIGN {
            let err = HeapFree(GetProcessHeap(), 0, ptr as LPVOID);
            debug_assert!(err != 0);
        } else {
            let header = get_header(ptr);
            let err = HeapFree(GetProcessHeap(), 0, header.0 as LPVOID);
            debug_assert!(err != 0);
        }
    }

    pub fn usable_size(size: usize, _align: usize) -> usize {
        size
    }
}
