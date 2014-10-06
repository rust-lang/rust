// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: #13996: mark the `allocate` and `reallocate` return value as `noalias`

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
pub unsafe fn reallocate(ptr: *mut u8, old_size: uint, size: uint, align: uint) -> *mut u8 {
    imp::reallocate(ptr, old_size, size, align)
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
pub unsafe fn reallocate_inplace(ptr: *mut u8, old_size: uint, size: uint, align: uint) -> bool {
    imp::reallocate_inplace(ptr, old_size, size, align)
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

/// An arbitrary non-null address to represent zero-size allocations.
///
/// This preserves the non-null invariant for types like `Box<T>`. The address may overlap with
/// non-zero-size memory allocations.
pub const EMPTY: *mut () = 0x1 as *mut ();

/// The allocator for unique pointers.
#[cfg(not(test))]
#[lang="exchange_malloc"]
#[inline]
unsafe fn exchange_malloc(size: uint, align: uint) -> *mut u8 {
    if size == 0 {
        EMPTY as *mut u8
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

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values. In practice, the alignment is a
// constant at the call site and the branch will be optimized out.
#[cfg(any(target_arch = "arm",
          target_arch = "mips",
          target_arch = "mipsel"))]
static MIN_ALIGN: uint = 8;
#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
static MIN_ALIGN: uint = 16;

#[cfg(jemalloc)]
mod imp {
    use core::option::{None, Option};
    use core::ptr::{RawPtr, null_mut, null};
    use core::num::Int;
    use libc::{c_char, c_int, c_void, size_t};
    use super::MIN_ALIGN;

    #[link(name = "jemalloc", kind = "static")]
    #[cfg(not(test))]
    extern {}

    extern {
        fn je_mallocx(size: size_t, flags: c_int) -> *mut c_void;
        fn je_rallocx(ptr: *mut c_void, size: size_t,
                      flags: c_int) -> *mut c_void;
        fn je_xallocx(ptr: *mut c_void, size: size_t, extra: size_t,
                      flags: c_int) -> size_t;
        fn je_sdallocx(ptr: *mut c_void, size: size_t, flags: c_int);
        fn je_nallocx(size: size_t, flags: c_int) -> size_t;
        fn je_malloc_stats_print(write_cb: Option<extern "C" fn(cbopaque: *mut c_void,
                                                                *const c_char)>,
                                 cbopaque: *mut c_void,
                                 opts: *const c_char);
    }

    // -lpthread needs to occur after -ljemalloc, the earlier argument isn't enough
    #[cfg(all(not(windows), not(target_os = "android")))]
    #[link(name = "pthread")]
    extern {}

    // MALLOCX_ALIGN(a) macro
    #[inline(always)]
    fn mallocx_align(a: uint) -> c_int { a.trailing_zeros() as c_int }

    #[inline(always)]
    fn align_to_flags(align: uint) -> c_int {
        if align <= MIN_ALIGN { 0 } else { mallocx_align(align) }
    }

    #[inline]
    pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        let flags = align_to_flags(align);
        let ptr = je_mallocx(size as size_t, flags) as *mut u8;
        if ptr.is_null() {
            ::oom()
        }
        ptr
    }

    #[inline]
    pub unsafe fn reallocate(ptr: *mut u8, _old_size: uint, size: uint, align: uint) -> *mut u8 {
        let flags = align_to_flags(align);
        let ptr = je_rallocx(ptr as *mut c_void, size as size_t, flags) as *mut u8;
        if ptr.is_null() {
            ::oom()
        }
        ptr
    }

    #[inline]
    pub unsafe fn reallocate_inplace(ptr: *mut u8, old_size: uint, size: uint,
                                     align: uint) -> bool {
        let flags = align_to_flags(align);
        let new_size = je_xallocx(ptr as *mut c_void, size as size_t, 0, flags) as uint;
        // checking for failure to shrink is tricky
        if size < old_size {
            usable_size(size, align) == new_size as uint
        } else {
            new_size >= size
        }
    }

    #[inline]
    pub unsafe fn deallocate(ptr: *mut u8, size: uint, align: uint) {
        let flags = align_to_flags(align);
        je_sdallocx(ptr as *mut c_void, size as size_t, flags)
    }

    #[inline]
    pub fn usable_size(size: uint, align: uint) -> uint {
        let flags = align_to_flags(align);
        unsafe { je_nallocx(size as size_t, flags) as uint }
    }

    pub fn stats_print() {
        unsafe {
            je_malloc_stats_print(None, null_mut(), null())
        }
    }
}

#[cfg(all(not(jemalloc), unix))]
mod imp {
    use core::cmp;
    use core::ptr;
    use libc;
    use libc_heap;
    use super::MIN_ALIGN;

    extern {
        fn posix_memalign(memptr: *mut *mut libc::c_void,
                          align: libc::size_t,
                          size: libc::size_t) -> libc::c_int;
    }

    #[inline]
    pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        if align <= MIN_ALIGN {
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
    pub unsafe fn reallocate(ptr: *mut u8, old_size: uint, size: uint, align: uint) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc_heap::realloc_raw(ptr, size)
        } else {
            let new_ptr = allocate(size, align);
            ptr::copy_memory(new_ptr, ptr as *const u8, cmp::min(size, old_size));
            deallocate(ptr, old_size, align);
            new_ptr
        }
    }

    #[inline]
    pub unsafe fn reallocate_inplace(_ptr: *mut u8, old_size: uint, size: uint,
                                     _align: uint) -> bool {
        size == old_size
    }

    #[inline]
    pub unsafe fn deallocate(ptr: *mut u8, _size: uint, _align: uint) {
        libc::free(ptr as *mut libc::c_void)
    }

    #[inline]
    pub fn usable_size(size: uint, _align: uint) -> uint {
        size
    }

    pub fn stats_print() {}
}

#[cfg(all(not(jemalloc), windows))]
mod imp {
    use libc::{c_void, size_t};
    use libc;
    use libc_heap;
    use core::ptr::RawPtr;
    use super::MIN_ALIGN;

    extern {
        fn _aligned_malloc(size: size_t, align: size_t) -> *mut c_void;
        fn _aligned_realloc(block: *mut c_void, size: size_t,
                            align: size_t) -> *mut c_void;
        fn _aligned_free(ptr: *mut c_void);
    }

    #[inline]
    pub unsafe fn allocate(size: uint, align: uint) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc_heap::malloc_raw(size)
        } else {
            let ptr = _aligned_malloc(size as size_t, align as size_t);
            if ptr.is_null() {
                ::oom();
            }
            ptr as *mut u8
        }
    }

    #[inline]
    pub unsafe fn reallocate(ptr: *mut u8, _old_size: uint, size: uint, align: uint) -> *mut u8 {
        if align <= MIN_ALIGN {
            libc_heap::realloc_raw(ptr, size)
        } else {
            let ptr = _aligned_realloc(ptr as *mut c_void, size as size_t,
                                       align as size_t);
            if ptr.is_null() {
                ::oom();
            }
            ptr as *mut u8
        }
    }

    #[inline]
    pub unsafe fn reallocate_inplace(_ptr: *mut u8, old_size: uint, size: uint,
                                     _align: uint) -> bool {
        size == old_size
    }

    #[inline]
    pub unsafe fn deallocate(ptr: *mut u8, _size: uint, align: uint) {
        if align <= MIN_ALIGN {
            libc::free(ptr as *mut libc::c_void)
        } else {
            _aligned_free(ptr as *mut c_void)
        }
    }

    #[inline]
    pub fn usable_size(size: uint, _align: uint) -> uint {
        size
    }

    pub fn stats_print() {}
}

#[cfg(test)]
mod test {
    extern crate test;
    use self::test::Bencher;
    use heap;

    #[test]
    fn basic_reallocate_inplace_noop() {
        unsafe {
            let size = 4000;
            let ptr = heap::allocate(size, 8);
            let ret = heap::reallocate_inplace(ptr, size, size, 8);
            heap::deallocate(ptr, size, 8);
            assert!(ret);
        }
    }

    #[bench]
    fn alloc_owned_small(b: &mut Bencher) {
        b.iter(|| {
            box 10i
        })
    }
}
