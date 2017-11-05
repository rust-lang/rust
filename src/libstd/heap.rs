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

pub use alloc::heap::{Heap, Alloc, Layout, Excess, CannotReallocInPlace, AllocErr};
pub use alloc_system::System;

#[cfg(not(test))]
#[doc(hidden)]
#[allow(unused_attributes)]
pub mod __default_lib_allocator {
    use super::{System, Layout, Alloc, AllocErr};
    use ptr;

    // for symbol names src/librustc/middle/allocator.rs
    // for signatures src/librustc_allocator/lib.rs

    // linkage directives are provided as part of the current compiler allocator
    // ABI

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc(size: usize,
                                     align: usize,
                                     err: *mut u8) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        match System.alloc(layout) {
            Ok(p) => p,
            Err(e) => {
                ptr::write(err as *mut AllocErr, e);
                0 as *mut u8
            }
        }
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_oom(err: *const u8) -> ! {
        System.oom((*(err as *const AllocErr)).clone())
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_dealloc(ptr: *mut u8,
                                       size: usize,
                                       align: usize) {
        System.dealloc(ptr, Layout::from_size_align_unchecked(size, align))
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_usable_size(layout: *const u8,
                                           min: *mut usize,
                                           max: *mut usize) {
        let pair = System.usable_size(&*(layout as *const Layout));
        *min = pair.0;
        *max = pair.1;
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_realloc(ptr: *mut u8,
                                       old_size: usize,
                                       old_align: usize,
                                       new_size: usize,
                                       new_align: usize,
                                       err: *mut u8) -> *mut u8 {
        let old_layout = Layout::from_size_align_unchecked(old_size, old_align);
        let new_layout = Layout::from_size_align_unchecked(new_size, new_align);
        match System.realloc(ptr, old_layout, new_layout) {
            Ok(p) => p,
            Err(e) => {
                ptr::write(err as *mut AllocErr, e);
                0 as *mut u8
            }
        }
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc_zeroed(size: usize,
                                            align: usize,
                                            err: *mut u8) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        match System.alloc_zeroed(layout) {
            Ok(p) => p,
            Err(e) => {
                ptr::write(err as *mut AllocErr, e);
                0 as *mut u8
            }
        }
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc_excess(size: usize,
                                            align: usize,
                                            excess: *mut usize,
                                            err: *mut u8) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        match System.alloc_excess(layout) {
            Ok(p) => {
                *excess = p.1;
                p.0
            }
            Err(e) => {
                ptr::write(err as *mut AllocErr, e);
                0 as *mut u8
            }
        }
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_realloc_excess(ptr: *mut u8,
                                              old_size: usize,
                                              old_align: usize,
                                              new_size: usize,
                                              new_align: usize,
                                              excess: *mut usize,
                                              err: *mut u8) -> *mut u8 {
        let old_layout = Layout::from_size_align_unchecked(old_size, old_align);
        let new_layout = Layout::from_size_align_unchecked(new_size, new_align);
        match System.realloc_excess(ptr, old_layout, new_layout) {
            Ok(p) => {
                *excess = p.1;
                p.0
            }
            Err(e) => {
                ptr::write(err as *mut AllocErr, e);
                0 as *mut u8
            }
        }
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_grow_in_place(ptr: *mut u8,
                                             old_size: usize,
                                             old_align: usize,
                                             new_size: usize,
                                             new_align: usize) -> u8 {
        let old_layout = Layout::from_size_align_unchecked(old_size, old_align);
        let new_layout = Layout::from_size_align_unchecked(new_size, new_align);
        match System.grow_in_place(ptr, old_layout, new_layout) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_shrink_in_place(ptr: *mut u8,
                                               old_size: usize,
                                               old_align: usize,
                                               new_size: usize,
                                               new_align: usize) -> u8 {
        let old_layout = Layout::from_size_align_unchecked(old_size, old_align);
        let new_layout = Layout::from_size_align_unchecked(new_size, new_align);
        match System.shrink_in_place(ptr, old_layout, new_layout) {
            Ok(()) => 1,
            Err(_) => 0,
        }
    }
}
