// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "alloc_jemalloc"]
#![crate_type = "rlib"]
#![no_std]
#![allocator]
#![unstable(feature = "alloc_jemalloc",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "27783")]
#![deny(warnings)]
#![feature(allocator)]
#![feature(libc)]
#![feature(staged_api)]

extern crate libc;

pub use imp::*;

// See comments in build.rs for why we sometimes build a crate that does nothing
#[cfg(not(dummy_jemalloc))]
mod imp {
    use libc::{c_int, c_void, size_t};

    // Note that the symbols here are prefixed by default on OSX and Windows (we
    // don't explicitly request it), and on Android and DragonFly we explicitly
    // request it as unprefixing cause segfaults (mismatches in allocators).
    extern "C" {
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows"),
                   link_name = "je_mallocx")]
        fn mallocx(size: size_t, flags: c_int) -> *mut c_void;
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows"),
                   link_name = "je_rallocx")]
        fn rallocx(ptr: *mut c_void, size: size_t, flags: c_int) -> *mut c_void;
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows"),
                   link_name = "je_xallocx")]
        fn xallocx(ptr: *mut c_void, size: size_t, extra: size_t, flags: c_int) -> size_t;
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows"),
                   link_name = "je_sdallocx")]
        fn sdallocx(ptr: *mut c_void, size: size_t, flags: c_int);
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows"),
                   link_name = "je_nallocx")]
        fn nallocx(size: size_t, flags: c_int) -> size_t;
    }

    // The minimum alignment guaranteed by the architecture. This value is used to
    // add fast paths for low alignment values. In practice, the alignment is a
    // constant at the call site and the branch will be optimized out.
    #[cfg(all(any(target_arch = "arm",
                  target_arch = "mips",
                  target_arch = "powerpc")))]
    const MIN_ALIGN: usize = 8;
    #[cfg(all(any(target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "aarch64",
                  target_arch = "powerpc64",
                  target_arch = "mips64",
                  target_arch = "s390x",
                  target_arch = "sparc64")))]
    const MIN_ALIGN: usize = 16;

    // MALLOCX_ALIGN(a) macro
    fn mallocx_align(a: usize) -> c_int {
        a.trailing_zeros() as c_int
    }

    fn align_to_flags(align: usize) -> c_int {
        if align <= MIN_ALIGN {
            0
        } else {
            mallocx_align(align)
        }
    }

    #[no_mangle]
    pub extern "C" fn __rust_allocate(size: usize, align: usize) -> *mut u8 {
        let flags = align_to_flags(align);
        unsafe { mallocx(size as size_t, flags) as *mut u8 }
    }

    #[no_mangle]
    pub extern "C" fn __rust_reallocate(ptr: *mut u8,
                                        _old_size: usize,
                                        size: usize,
                                        align: usize)
                                        -> *mut u8 {
        let flags = align_to_flags(align);
        unsafe { rallocx(ptr as *mut c_void, size as size_t, flags) as *mut u8 }
    }

    #[no_mangle]
    pub extern "C" fn __rust_reallocate_inplace(ptr: *mut u8,
                                                _old_size: usize,
                                                size: usize,
                                                align: usize)
                                                -> usize {
        let flags = align_to_flags(align);
        unsafe { xallocx(ptr as *mut c_void, size as size_t, 0, flags) as usize }
    }

    #[no_mangle]
    pub extern "C" fn __rust_deallocate(ptr: *mut u8, old_size: usize, align: usize) {
        let flags = align_to_flags(align);
        unsafe { sdallocx(ptr as *mut c_void, old_size as size_t, flags) }
    }

    #[no_mangle]
    pub extern "C" fn __rust_usable_size(size: usize, align: usize) -> usize {
        let flags = align_to_flags(align);
        unsafe { nallocx(size as size_t, flags) as usize }
    }
}

#[cfg(dummy_jemalloc)]
mod imp {
    fn bogus() -> ! {
        panic!("jemalloc is not implemented for this platform");
    }

    #[no_mangle]
    pub extern "C" fn __rust_allocate(_size: usize, _align: usize) -> *mut u8 {
        bogus()
    }

    #[no_mangle]
    pub extern "C" fn __rust_reallocate(_ptr: *mut u8,
                                        _old_size: usize,
                                        _size: usize,
                                        _align: usize)
                                        -> *mut u8 {
        bogus()
    }

    #[no_mangle]
    pub extern "C" fn __rust_reallocate_inplace(_ptr: *mut u8,
                                                _old_size: usize,
                                                _size: usize,
                                                _align: usize)
                                                -> usize {
        bogus()
    }

    #[no_mangle]
    pub extern "C" fn __rust_deallocate(_ptr: *mut u8, _old_size: usize, _align: usize) {
        bogus()
    }

    #[no_mangle]
    pub extern "C" fn __rust_usable_size(_size: usize, _align: usize) -> usize {
        bogus()
    }
}
