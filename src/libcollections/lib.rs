// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Collection types.
//!
//! See [std::collections](../std/collections) for a detailed discussion of
//! collections in Rust.

#![crate_name = "collections"]
#![crate_type = "rlib"]
#![unstable(feature = "collections",
            reason = "library is unlikely to be stabilized with the current \
                      layout and name, use std::collections instead",
            issue = "27783")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(allow(unused_variables), deny(warnings))))]

#![allow(trivial_casts)]
#![cfg_attr(test, allow(deprecated))] // rand

#![feature(alloc)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(clone_from_slice)]
#![feature(core_intrinsics)]
#![feature(decode_utf16)]
#![feature(drop_in_place)]
#![feature(dropck_parametricity)]
#![feature(fmt_internals)]
#![feature(fmt_radix)]
#![feature(heap_api)]
#![feature(iter_arith)]
#![feature(iter_arith)]
#![feature(lang_items)]
#![feature(num_bits_bytes)]
#![feature(oom)]
#![feature(pattern)]
#![feature(shared)]
#![feature(slice_bytes)]
#![feature(slice_patterns)]
#![feature(staged_api)]
#![feature(step_by)]
#![feature(str_char)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag, filling_drop)]
#![cfg_attr(test, feature(clone_from_slice, rand, test))]

#![no_std]

extern crate rustc_unicode;
extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate test;

pub use binary_heap::BinaryHeap;
pub use btree_map::BTreeMap;
pub use btree_set::BTreeSet;
pub use linked_list::LinkedList;
pub use enum_set::EnumSet;
pub use vec_deque::VecDeque;
pub use string::String;
pub use vec::Vec;

// Needed for the vec! macro
pub use alloc::boxed;

#[macro_use]
mod macros;

pub mod binary_heap;
mod btree;
pub mod borrow;
pub mod enum_set;
pub mod fmt;
pub mod linked_list;
pub mod range;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod vec_deque;

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_map {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use btree::map::*;
}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_set {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use btree::set::*;
}

#[cfg(not(test))]
mod std {
    pub use core::ops;      // RangeFull
}

/// An endpoint of a range of keys.
#[unstable(feature = "collections_bound", issue = "27787")]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bound<T> {
    /// An inclusive bound.
    Included(T),
    /// An exclusive bound.
    Excluded(T),
    /// An infinite endpoint. Indicates that there is no bound in this direction.
    Unbounded,
}
