// Copyright 2014-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # The Rust core allocation and collections library
//!
//! This library provides smart pointers and collections for managing
//! heap-allocated values.
//!
//! This library, like libcore, normally doesn’t need to be used directly
//! since its contents are re-exported in the [`std` crate](../std/index.html).
//! Crates that use the `#![no_std]` attribute however will typically
//! not depend on `std`, so they’d use this crate instead.
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
//! The [`Arc`](sync/index.html) type is the threadsafe equivalent of the `Rc`
//! type. It provides all the same functionality of `Rc`, except it requires
//! that the contained type `T` is shareable. Additionally, `Arc<T>` is itself
//! sendable while `Rc<T>` is not.
//!
//! This type allows for shared access to the contained data, and is often
//! paired with synchronization primitives such as mutexes to allow mutation of
//! shared resources.
//!
//! ## Collections
//!
//! Implementations of the most common general purpose data structures are
//! defined in this library. They are re-exported through the
//! [standard collections library](../std/collections/index.html).
//!
//! ## Heap interfaces
//!
//! The [`alloc`](alloc/index.html) module defines the low-level interface to the
//! default global allocator. It is not compatible with the libc allocator API.

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
#![needs_allocator]
#![deny(missing_debug_implementations)]

#![cfg_attr(not(test), feature(fn_traits))]
#![cfg_attr(not(test), feature(generator_trait))]
#![cfg_attr(test, feature(test))]

#![feature(allocator_api)]
#![feature(allow_internal_unstable)]
#![feature(arbitrary_self_types)]
#![feature(box_into_raw_non_null)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(cfg_target_has_atomic)]
#![feature(coerce_unsized)]
#![feature(dispatch_from_dyn)]
#![feature(core_intrinsics)]
#![feature(custom_attribute)]
#![feature(dropck_eyepatch)]
#![feature(exact_size_is_empty)]
#![feature(fmt_internals)]
#![feature(fundamental)]
#![feature(futures_api)]
#![feature(lang_items)]
#![feature(libc)]
#![feature(needs_allocator)]
#![feature(nll)]
#![feature(optin_builtin_traits)]
#![feature(pattern)]
#![feature(pin)]
#![feature(ptr_internals)]
#![feature(ptr_offset_from)]
#![feature(rustc_attrs)]
#![feature(receiver_trait)]
#![feature(specialization)]
#![feature(split_ascii_whitespace)]
#![feature(staged_api)]
#![feature(str_internals)]
#![feature(trusted_len)]
#![feature(try_reserve)]
#![feature(unboxed_closures)]
#![feature(unicode_internals)]
#![feature(unsize)]
#![feature(allocator_internals)]
#![feature(on_unimplemented)]
#![feature(rustc_const_unstable)]
#![feature(const_vec_new)]
#![feature(slice_partition_dedup)]
#![feature(maybe_uninit)]
#![feature(alloc_layout_extra)]

// Allow testing this library

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate test;
#[cfg(test)]
extern crate rand;

// Module with internal macros used by other modules (needs to be included before other modules).
#[macro_use]
mod macros;

// Heaps provided for low-level allocation strategies

pub mod alloc;

#[unstable(feature = "futures_api",
           reason = "futures in libcore are unstable",
           issue = "50547")]
pub mod task;
// Primitive types using the heaps above

// Need to conditionally define the mod from `boxed.rs` to avoid
// duplicating the lang-items when building in test cfg; but also need
// to allow code to have `use boxed::Box;` declarations.
#[cfg(not(test))]
pub mod boxed;
#[cfg(test)]
mod boxed {
    pub use std::boxed::Box;
}
#[cfg(test)]
mod boxed_test;
pub mod collections;
#[cfg(all(target_has_atomic = "ptr", target_has_atomic = "cas"))]
pub mod sync;
pub mod rc;
pub mod raw_vec;
pub mod prelude;
pub mod borrow;
pub mod fmt;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;

#[cfg(not(test))]
mod std {
    pub use core::ops;      // RangeFull
}
