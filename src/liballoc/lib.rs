// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # The Rust core allocation library
//!
//! This is the lowest level library through which allocation in Rust can be
//! performed.
//!
//! This library, like libcore, is not intended for general usage, but rather as
//! a building block of other libraries. The types and interfaces in this
//! library are reexported through the [standard library](../std/index.html),
//! and should not be used through this library.
//!
//! Currently, there are four major definitions in this library.
//!
//! ## Boxed values
//!
//! The [`Box`](boxed/index.html) type is a smart pointer type. There can
//! only be one owner of a `Box`, and the owner can decide to mutate the
//! contents, which live on the heap.
//!
//! This type can be sent among threads efficiently as the size of a `Box` value
//! is the same as that of a pointer. Tree-like data structures are often built
//! with boxes because each node often has only one owner, the parent.
//!
//! ## Reference counted pointers
//!
//! The [`Rc`](rc/index.html) type is a non-threadsafe reference-counted pointer
//! type intended for sharing memory within a thread. An `Rc` pointer wraps a
//! type, `T`, and only allows access to `&T`, a shared reference.
//!
//! This type is useful when inherited mutability (such as using `Box`) is too
//! constraining for an application, and is often paired with the `Cell` or
//! `RefCell` types in order to allow mutation.
//!
//! ## Atomically reference counted pointers
//!
//! The [`Arc`](arc/index.html) type is the threadsafe equivalent of the `Rc`
//! type. It provides all the same functionality of `Rc`, except it requires
//! that the contained type `T` is shareable. Additionally, `Arc<T>` is itself
//! sendable while `Rc<T>` is not.
//!
//! This types allows for shared access to the contained data, and is often
//! paired with synchronization primitives such as mutexes to allow mutation of
//! shared resources.
//!
//! ## Heap interfaces
//!
//! The [`heap`](heap/index.html) module defines the low-level interface to the
//! default global allocator. It is not compatible with the libc allocator API.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "alloc"]
#![crate_type = "rlib"]
#![staged_api]
#![allow(unused_attributes)]
#![unstable(feature = "alloc",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "27783")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(allow(unused_variables), deny(warnings))))]
#![no_std]
#![cfg_attr(not(stage0), needs_allocator)]

#![cfg_attr(stage0, feature(rustc_attrs))]
#![cfg_attr(stage0, allow(unused_attributes))]
#![feature(allocator)]
#![feature(box_syntax)]
#![feature(coerce_unsized)]
#![feature(core)]
#![feature(core_intrinsics)]
#![feature(core_slice_ext)]
#![feature(custom_attribute)]
#![feature(fundamental)]
#![feature(lang_items)]
#![feature(no_std)]
#![feature(nonzero)]
#![feature(num_bits_bytes)]
#![feature(optin_builtin_traits)]
#![feature(placement_in_syntax)]
#![feature(placement_new_protocol)]
#![feature(raw)]
#![feature(shared)]
#![feature(staged_api)]
#![feature(unboxed_closures)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag, filling_drop)]
// SNAP 1af31d4
#![allow(unused_features)]
// SNAP 1af31d4
#![allow(unused_attributes)]
#![feature(dropck_parametricity)]
#![feature(unsize)]
#![feature(core_slice_ext)]
#![feature(core_str_ext)]
#![feature(drop_in_place)]

#![cfg_attr(stage0, feature(alloc_system))]
#![cfg_attr(not(stage0), feature(needs_allocator))]

#![cfg_attr(test, feature(test, rustc_private, box_heap))]

#[cfg(stage0)]
extern crate alloc_system;

// Allow testing this library

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
#[macro_use]
extern crate log;

// Heaps provided for low-level allocation strategies

pub mod heap;

// Primitive types using the heaps above

// Need to conditionally define the mod from `boxed.rs` to avoid
// duplicating the lang-items when building in test cfg; but also need
// to allow code to have `use boxed::HEAP;`
// and `use boxed::Box;` declarations.
#[cfg(not(test))]
pub mod boxed;
#[cfg(test)]
mod boxed {
    pub use std::boxed::{Box, HEAP};
}
#[cfg(test)]
mod boxed_test;
pub mod arc;
pub mod rc;
pub mod raw_vec;

/// Common out-of-memory routine
#[cold]
#[inline(never)]
#[unstable(feature = "oom", reason = "not a scrutinized interface",
           issue = "27700")]
pub fn oom() -> ! {
    // FIXME(#14674): This really needs to do something other than just abort
    //                here, but any printing done must be *guaranteed* to not
    //                allocate.
    unsafe { core::intrinsics::abort() }
}
