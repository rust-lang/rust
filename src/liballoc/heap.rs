// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: #13994: port to the sized deallocation API when available
// FIXME: #13996: mark the `allocate` and `reallocate` return value as `noalias`
//                and `nonnull`

#[cfg(not(test))] use core::raw;
#[cfg(not(test))] use util;

/// Returns a pointer to `size` bytes of memory.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
#[inline]
pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
    imp::allocate(size, align)
}

/// Extends or shrinks the allocation referenced by `ptr` to `size` bytes of
/// memory.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `old_size` parameter may also
/// be the value returned by `usable_size` for the requested size.
#[inline]
pub unsafe fn reallocate(ptr: *mut u8, size: uint, align: uint,
                         old_size: uint) -> *mut u8 {
    imp::reallocate(ptr, size, align, old_size)
}

/// Extends or shrinks the allocation referenced by `ptr` to `size` bytes of
/// memory in-place.
///
/// Returns true if successful, otherwise false if the allocation was not
/// altered.
///
/// Behavior is undefined if the requested size is 0 or the alignment is not a
/// power of 2. The alignment must be no larger than the largest supported page
/// size on the platform.
///
/// The `old_size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `old_size` parameter may be
/// any value in range_inclusive(requested_size, usable_size).
#[inline]
pub unsafe fn reallocate_inplace(ptr: *mut u8, size: uint, align: uint,
                                 old_size: uint) -> bool {
    imp::reallocate_inplace(ptr, size, align, old_size)
}

/// Deallocates the memory referenced by `ptr`.
///
/// The `ptr` parameter must not be null.
///
/// The `size` and `align` parameters are the parameters that were used to
/// create the allocation referenced by `ptr`. The `size` parameter may also be
/// the value returned by `usable_size` for the requested size.
#[inline]
pub unsafe fn deallocate(ptr: *mut u8, size: uint, align: uint) {
    imp::deallocate(ptr, size, align)
}

/// Returns the usable size of an allocation created with the specified the
/// `size` and `align`.
#[inline]
pub fn usable_size(size: uint, align: uint) -> uint {
    imp::usable_size(size, align)
}

/// Prints implementation-defined allocator statistics.
///
/// These statistics may be inconsistent if other threads use the allocator
/// during the call.
#[unstable]
pub fn stats_print() {
    imp::stats_print();
}

// The compiler never calls `exchange_free` on Box<ZeroSizeType>, so zero-size
// allocations can point to this `static`. It would be incorrect to use a null
// pointer, due to enums assuming types like unique pointers are never null.
pub static mut EMPTY: uint = 12345;

/// The allocator for unique pointers.
#[cfg(not(test))]
#[lang="exchange_malloc"]
#[inline]
unsafe fn exchange_malloc(size: uint, align: uint) -> *mut u8 {
    if size == 0 {
        &EMPTY as *const uint as *mut u8
    } else {
        allocate(size, align)
    }
}

#[cfg(not(test))]
#[lang="exchange_free"]
#[inline]
unsafe fn exchange_free(ptr: *mut u8, size: uint, align: uint) {
    deallocate(ptr, size, align);
}

// FIXME: #7496
#[cfg(not(test))]
#[lang="closure_exchange_malloc"]
#[inline]
#[allow(deprecated)]
unsafe fn closure_exchange_malloc(drop_glue: fn(*mut u8), size: uint,
                                  align: uint) -> *mut u8 {
    let total_size = util::get_box_size(size, align);
    let p = allocate(total_size, 8);

    let alloc = p as *mut raw::Box<()>;
    (*alloc).drop_glue = drop_glue;

    alloc as *mut u8
}

#[cfg(jemalloc)]
mod imp {
    use core::option::{None, Option};
    use core::ptr::{RawPtr, mut_null, null};
    use core::num::Int;
    use libc::{c_char, c_int, c_void, size_t};

    #[link(name = "jemalloc", kind = "static")]
    #[cfg(not(test))]
    extern {}

    extern {
        fn je_mallocx(size: size_t, flags: c_int) -> *mut c_void;
        fn je_rallocx(ptr: *mut c_void, size: size_t,
                      flags: c_int) -> *mut c_void;
        fn je_xallocx(ptr: *mut c_void, size: size_t, extra: size_t,
                      flags: c_int) -> size_t;
        fn je_dallocx(ptr: *mut c_void, flags: c_int);
        fn je_nallocx(size: size_t, flags: c_int) -> size_t;
        fn je_malloc_stats_print(write_cb: Option<extern "C" fn(cbopaque: *mut c_void,
                                                                *const c_char)>,
                                 cbopaque: *mut c_void,
                                 opts: *const c_char);
    }

    // -lpthread needs to occur after -ljemalloc, the earlier argument isn't enough
    #[cfg(not(windows), not(target_os = "android"))]
    #[link(name = "pthread")]
    extern {}

    // MALLOCX_ALIGN(a) macro
    #[inline(always)]
    fn mallocx_align(a: uint) -> c_int { a.trailing_zeros() as c_int }

    #[inline]
    pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        let ptr = je_mallocx(size as size_t, mallocx_align(align)) as *mut u8;
        if ptr.is_null() {
            ::oom()
        }
        ptr
    }

    #[inline]
    pub unsafe fn reallocate(ptr: *mut u8, size: uint, align: uint,
                             _old_size: uint) -> *mut u8 {
        let ptr = je_rallocx(ptr as *mut c_void, size as size_t,
                             mallocx_align(align)) as *mut u8;
        if ptr.is_null() {
            ::oom()
        }
        ptr
    }

    #[inline]
    pub unsafe fn reallocate_inplace(ptr: *mut u8, size: uint, align: uint,
                                     _old_size: uint) -> bool {
        je_xallocx(ptr as *mut c_void, size as size_t, 0,
                   mallocx_align(align)) == size as size_t
    }

    #[inline]
    pub unsafe fn deallocate(ptr: *mut u8, _size: uint, align: uint) {
        je_dallocx(ptr as *mut c_void, mallocx_align(align))
    }

    #[inline]
    pub fn usable_size(size: uint, align: uint) -> uint {
        unsafe { je_nallocx(size as size_t, mallocx_align(align)) as uint }
    }

    pub fn stats_print() {
        unsafe {
            je_malloc_stats_print(None, mut_null(), null())
        }
    }
}

#[cfg(not(jemalloc), unix)]
mod imp {
    use core::mem;
    use core::ptr;
    use libc;
    use libc_heap;

    extern {
        fn posix_memalign(memptr: *mut *mut libc::c_void,
                          align: libc::size_t,
                          size: libc::size_t) -> libc::c_int;
    }

    #[inline]
    pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        // The posix_memalign manpage states
        //
        //      alignment [...] must be a power of and a multiple of
        //      sizeof(void *)
        //
        // The `align` parameter to this function is the *minimum* alignment for
        // a block of memory, so we special case everything under `*uint` to
        // just pass it to malloc, which is guaranteed to align to at least the
        // size of `*uint`.
        if align < mem::size_of::<uint>() {
            libc_heap::malloc_raw(size)
        } else {
            let mut out = 0 as *mut libc::c_void;
            let ret = posix_memalign(&mut out,
                                     align as libc::size_t,
                                     size as libc::size_t);
            if ret != 0 {
                ::oom();
            }
            out as *mut u8
        }
    }

    #[inline]
    pub unsafe fn reallocate(ptr: *mut u8, size: uint, align: uint,
                             old_size: uint) -> *mut u8 {
        let new_ptr = allocate(size, align);
        ptr::copy_memory(new_ptr, ptr as *const u8, old_size);
        deallocate(ptr, old_size, align);
        return new_ptr;
    }

    #[inline]
    pub unsafe fn reallocate_inplace(_ptr: *mut u8, _size: uint, _align: uint,
                                     _old_size: uint) -> bool {
        false
    }

    #[inline]
    pub unsafe fn deallocate(ptr: *mut u8, _size: uint, _align: uint) {
        libc::free(ptr as *mut libc::c_void)
    }

    #[inline]
    pub fn usable_size(size: uint, _align: uint) -> uint {
        size
    }

    pub fn stats_print() {
    }
}

#[cfg(not(jemalloc), windows)]
mod imp {
    use libc::{c_void, size_t};
    use core::ptr::RawPtr;

    extern {
        fn _aligned_malloc(size: size_t, align: size_t) -> *mut c_void;
        fn _aligned_realloc(block: *mut c_void, size: size_t,
                            align: size_t) -> *mut c_void;
        fn _aligned_free(ptr: *mut c_void);
    }

    #[inline]
    pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        let ptr = _aligned_malloc(size as size_t, align as size_t);
        if ptr.is_null() {
            ::oom();
        }
        ptr as *mut u8
    }

    #[inline]
    pub unsafe fn reallocate(ptr: *mut u8, size: uint, align: uint,
                             _old_size: uint) -> *mut u8 {
        let ptr = _aligned_realloc(ptr as *mut c_void, size as size_t,
                                   align as size_t);
        if ptr.is_null() {
            ::oom();
        }
        ptr as *mut u8
    }

    #[inline]
    pub unsafe fn reallocate_inplace(_ptr: *mut u8, _size: uint, _align: uint,
                                     _old_size: uint) -> bool {
        false
    }

    #[inline]
    pub unsafe fn deallocate(ptr: *mut u8, _size: uint, _align: uint) {
        _aligned_free(ptr as *mut c_void)
    }

    #[inline]
    pub fn usable_size(size: uint, _align: uint) -> uint {
        size
    }

    pub fn stats_print() {}
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;

    #[bench]
    fn alloc_owned_small(b: &mut Bencher) {
        b.iter(|| {
            box 10i
        })
    }
}
