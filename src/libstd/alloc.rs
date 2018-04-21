// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! dox

#![unstable(issue = "32838", feature = "allocator_api")]

#[doc(inline)] #[allow(deprecated)] pub use alloc_crate::alloc::Heap;
#[doc(inline)] pub use alloc_crate::alloc::{Global, oom};
#[doc(inline)] pub use alloc_system::System;
#[doc(inline)] pub use core::alloc::*;

#[cfg(not(stage0))]
#[cfg(not(test))]
#[doc(hidden)]
#[lang = "oom"]
pub extern fn rust_oom() -> ! {
    rtabort!("memory allocation failed");
}

#[cfg(not(test))]
#[doc(hidden)]
#[allow(unused_attributes)]
pub mod __default_lib_allocator {
    use super::{System, Layout, GlobalAlloc, Opaque};
    // for symbol names src/librustc/middle/allocator.rs
    // for signatures src/librustc_allocator/lib.rs

    // linkage directives are provided as part of the current compiler allocator
    // ABI

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc(size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        System.alloc(layout) as *mut u8
    }

    #[cfg(stage0)]
    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_oom() -> ! {
        super::oom()
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_dealloc(ptr: *mut u8,
                                       size: usize,
                                       align: usize) {
        System.dealloc(ptr as *mut Opaque, Layout::from_size_align_unchecked(size, align))
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_realloc(ptr: *mut u8,
                                       old_size: usize,
                                       align: usize,
                                       new_size: usize) -> *mut u8 {
        let old_layout = Layout::from_size_align_unchecked(old_size, align);
        System.realloc(ptr as *mut Opaque, old_layout, new_size) as *mut u8
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        System.alloc_zeroed(layout) as *mut u8
    }

    #[cfg(stage0)]
    pub mod stage0 {
        #[no_mangle]
        #[rustc_std_internal_symbol]
        pub unsafe extern fn __rdl_usable_size(_layout: *const u8,
                                               _min: *mut usize,
                                               _max: *mut usize) {
            unimplemented!()
        }

        #[no_mangle]
        #[rustc_std_internal_symbol]
        pub unsafe extern fn __rdl_alloc_excess(_size: usize,
                                                _align: usize,
                                                _excess: *mut usize,
                                                _err: *mut u8) -> *mut u8 {
            unimplemented!()
        }

        #[no_mangle]
        #[rustc_std_internal_symbol]
        pub unsafe extern fn __rdl_realloc_excess(_ptr: *mut u8,
                                                  _old_size: usize,
                                                  _old_align: usize,
                                                  _new_size: usize,
                                                  _new_align: usize,
                                                  _excess: *mut usize,
                                                  _err: *mut u8) -> *mut u8 {
            unimplemented!()
        }

        #[no_mangle]
        #[rustc_std_internal_symbol]
        pub unsafe extern fn __rdl_grow_in_place(_ptr: *mut u8,
                                                 _old_size: usize,
                                                 _old_align: usize,
                                                 _new_size: usize,
                                                 _new_align: usize) -> u8 {
            unimplemented!()
        }

        #[no_mangle]
        #[rustc_std_internal_symbol]
        pub unsafe extern fn __rdl_shrink_in_place(_ptr: *mut u8,
                                                   _old_size: usize,
                                                   _old_align: usize,
                                                   _new_size: usize,
                                                   _new_align: usize) -> u8 {
            unimplemented!()
        }

    }
}
