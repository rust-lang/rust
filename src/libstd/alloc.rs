// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Memory allocation APIs
//!
//! In a given program, the standard library has one “global” memory allocator
//! that is used for example by `Box<T>` and `Vec<T>`.
//!
//! Currently the default global allocator is unspecified.
//! The compiler may link to a version of [jemalloc] on some platforms,
//! but this is not guaranteed.
//! Libraries, however, like `cdylib`s and `staticlib`s are guaranteed
//! to use the [`System`] by default.
//!
//! [jemalloc]: https://github.com/jemalloc/jemalloc
//! [`System`]: struct.System.html
//!
//! # The `#[global_allocator]` attribute
//!
//! This attribute allows configuring the choice of global allocator.
//! You can use this to implement a completely custom global allocator
//! to route all default allocation requests to a custom object.
//!
//! ```rust
//! use std::alloc::{GlobalAlloc, System, Layout};
//!
//! struct MyAllocator;
//!
//! unsafe impl GlobalAlloc for MyAllocator {
//!     unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
//!         System.alloc(layout)
//!     }
//!
//!     unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
//!         System.dealloc(ptr, layout)
//!     }
//! }
//!
//! #[global_allocator]
//! static GLOBAL: MyAllocator = MyAllocator;
//!
//! fn main() {
//!     // This `Vec` will allocate memory through `GLOBAL` above
//!     let mut v = Vec::new();
//!     v.push(1);
//! }
//! ```
//!
//! The attribute is used on a `static` item whose type implements the
//! [`GlobalAlloc`] trait. This type can be provided by an external library:
//!
//! [`GlobalAlloc`]: ../../core/alloc/trait.GlobalAlloc.html
//!
//! ```rust,ignore (demonstrates crates.io usage)
//! extern crate jemallocator;
//!
//! use jemallacator::Jemalloc;
//!
//! #[global_allocator]
//! static GLOBAL: Jemalloc = Jemalloc;
//!
//! fn main() {}
//! ```
//!
//! The `#[global_allocator]` can only be used once in a crate
//! or its recursive dependencies.

#![stable(feature = "alloc_module", since = "1.28.0")]

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use alloc_crate::alloc::*;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
#[doc(inline)]
pub use alloc_system::System;

#[cfg(not(test))]
#[doc(hidden)]
#[allow(unused_attributes)]
#[unstable(feature = "alloc_internals", issue = "0")]
pub mod __default_lib_allocator {
    use super::{System, Layout, GlobalAlloc};
    use sys_common::util::dumb_print;

    // for symbol names src/librustc/middle/allocator.rs
    // for signatures src/librustc_allocator/lib.rs

    // linkage directives are provided as part of the current compiler allocator
    // ABI

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc(size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        System.alloc(layout)
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
    pub unsafe extern fn __rdl_realloc(ptr: *mut u8,
                                       old_size: usize,
                                       align: usize,
                                       new_size: usize) -> *mut u8 {
        let old_layout = Layout::from_size_align_unchecked(old_size, align);
        System.realloc(ptr, old_layout, new_size)
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rdl_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        System.alloc_zeroed(layout)
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub extern fn __rust_default_alloc_error_hook(size: usize, _align: usize) {
        dumb_print(format_args!("memory allocation of {} bytes failed", size));
    }

    #[no_mangle]
    #[rustc_std_internal_symbol]
    pub unsafe extern fn __rust_abort_internal() {
        ::sys::abort_internal()
    }
}
