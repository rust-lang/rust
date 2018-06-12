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
#![unstable(feature = "alloc_jemalloc",
            reason = "implementation detail of std, does not provide any public API",
            issue = "0")]
#![feature(core_intrinsics)]
#![feature(libc)]
#![feature(linkage)]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![cfg_attr(dummy_jemalloc, allow(dead_code, unused_extern_crates))]
#![cfg_attr(not(dummy_jemalloc), feature(allocator_api))]
#![rustc_alloc_kind = "exe"]

extern crate libc;

#[cfg(not(dummy_jemalloc))]
pub use contents::*;
#[cfg(not(dummy_jemalloc))]
mod contents {
    use libc::{c_int, c_void, size_t};

    // Note that the symbols here are prefixed by default on macOS and Windows (we
    // don't explicitly request it), and on Android and DragonFly we explicitly
    // request it as unprefixing cause segfaults (mismatches in allocators).
    extern "C" {
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows", target_env = "musl"),
                   link_name = "je_mallocx")]
        fn mallocx(size: size_t, flags: c_int) -> *mut c_void;
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows", target_env = "musl"),
                   link_name = "je_calloc")]
        fn calloc(size: size_t, flags: c_int) -> *mut c_void;
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows", target_env = "musl"),
                   link_name = "je_rallocx")]
        fn rallocx(ptr: *mut c_void, size: size_t, flags: c_int) -> *mut c_void;
        #[cfg_attr(any(target_os = "macos", target_os = "android", target_os = "ios",
                       target_os = "dragonfly", target_os = "windows", target_env = "musl"),
                   link_name = "je_sdallocx")]
        fn sdallocx(ptr: *mut c_void, size: size_t, flags: c_int);
    }

    const MALLOCX_ZERO: c_int = 0x40;

    // The minimum alignment guaranteed by the architecture. This value is used to
    // add fast paths for low alignment values.
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

    fn align_to_flags(align: usize, size: usize) -> c_int {
        if align <= MIN_ALIGN && align <= size {
            0
        } else {
            mallocx_align(align)
        }
    }

    // for symbol names src/librustc/middle/allocator.rs
    // for signatures src/librustc_allocator/lib.rs

    // linkage directives are provided as part of the current compiler allocator
    // ABI

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rde_alloc(size: usize, align: usize) -> *mut u8 {
        let flags = align_to_flags(align, size);
        let ptr = mallocx(size as size_t, flags) as *mut u8;
        ptr
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rde_dealloc(ptr: *mut u8,
                                       size: usize,
                                       align: usize) {
        let flags = align_to_flags(align, size);
        sdallocx(ptr as *mut c_void, size, flags);
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rde_realloc(ptr: *mut u8,
                                       _old_size: usize,
                                       align: usize,
                                       new_size: usize) -> *mut u8 {
        let flags = align_to_flags(align, new_size);
        let ptr = rallocx(ptr as *mut c_void, new_size, flags) as *mut u8;
        ptr
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rde_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
        let ptr = if align <= MIN_ALIGN && align <= size {
            calloc(size as size_t, 1) as *mut u8
        } else {
            let flags = align_to_flags(align, size) | MALLOCX_ZERO;
            mallocx(size as size_t, flags) as *mut u8
        };
        ptr
    }
}
