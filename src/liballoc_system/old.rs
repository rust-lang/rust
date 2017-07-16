// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[no_mangle]
pub unsafe extern fn __rust_alloc(size: usize,
                                  align: usize,
                                  err: *mut u8) -> *mut u8 {
    let p = imp::allocate(size, align);
    if p.is_null() {
        __rust_oom(err);
    }
    p
}

#[no_mangle]
pub unsafe extern fn __rust_oom(_err: *const u8) -> ! {
    ::core::intrinsics::abort()
}

#[no_mangle]
pub unsafe extern fn __rust_dealloc(ptr: *mut u8,
                                    size: usize,
                                    align: usize) {
    imp::deallocate(ptr, size, align)
}

#[no_mangle]
pub unsafe extern fn __rust_usable_size(size: usize,
                                        _align: usize,
                                        min: *mut usize,
                                        max: *mut usize) {
    *min = size;
    *max = size;
}

#[no_mangle]
pub unsafe extern fn __rust_realloc(ptr: *mut u8,
                                    old_size: usize,
                                    old_align: usize,
                                    new_size: usize,
                                    new_align: usize,
                                    err: *mut u8) -> *mut u8 {
    if new_align != old_align {
        __rust_oom(err);
    }
    let p = imp::reallocate(ptr, old_size, new_size, new_align);
    if p.is_null() {
        __rust_oom(err);
    }
    p
}

#[no_mangle]
pub unsafe extern fn __rust_alloc_zeroed(size: usize,
                                         align: usize,
                                         err: *mut u8) -> *mut u8 {
    let p = imp::allocate_zeroed(size, align);
    if p.is_null() {
        __rust_oom(err);
    }
    p
}

#[no_mangle]
pub unsafe extern fn __rust_alloc_excess(_size: usize,
                                         _align: usize,
                                         _excess: *mut usize,
                                         err: *mut u8) -> *mut u8 {
    __rust_oom(err);
}

#[no_mangle]
pub unsafe extern fn __rust_realloc_excess(_ptr: *mut u8,
                                           _old_size: usize,
                                           _old_align: usize,
                                           _new_size: usize,
                                           _new_align: usize,
                                           _excess: *mut usize,
                                           err: *mut u8) -> *mut u8 {
    __rust_oom(err);
}

#[no_mangle]
pub unsafe extern fn __rust_grow_in_place(_ptr: *mut u8,
                                          _old_size: usize,
                                          _old_align: usize,
                                          _new_size: usize,
                                          _new_align: usize) -> u8 {
    0
}

#[no_mangle]
pub unsafe extern fn __rust_shrink_in_place(_ptr: *mut u8,
                                            _old_size: usize,
                                            _old_align: usize,
                                            _new_size: usize,
                                            _new_align: usize) -> u8 {
    0
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

    pub unsafe fn deallocate(ptr: *mut u8, _old_size: usize, _align: usize) {
        libc::free(ptr as *mut libc::c_void)
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
}
