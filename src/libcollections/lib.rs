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
//! See [std::collections](../std/collections) for a detailed discussion of collections in Rust.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "collections"]
#![unstable(feature = "collections")]
#![staged_api]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]
#![doc(test(no_crate_inject))]

#![allow(trivial_casts)]
#![feature(alloc)]
#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(core)]
#![feature(lang_items)]
#![feature(staged_api)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unsafe_destructor)]
#![feature(unique)]
#![feature(unsafe_no_drop_flag, filling_drop)]
#![feature(step_by)]
#![feature(str_char)]
#![feature(slice_patterns)]
#![feature(debug_builders)]
#![cfg_attr(test, feature(rand, rustc_private, test, hash, collections))]
#![cfg_attr(test, allow(deprecated))] // rand

#![feature(no_std)]
#![no_std]

#[macro_use]
extern crate core;

extern crate unicode;
extern crate alloc;

#[cfg(test)] #[macro_use] extern crate std;
#[cfg(test)] extern crate test;

pub use binary_heap::BinaryHeap;
pub use bit_vec::BitVec;
pub use bit_set::BitSet;
pub use btree_map::BTreeMap;
pub use btree_set::BTreeSet;
pub use linked_list::LinkedList;
pub use enum_set::EnumSet;
pub use vec_deque::VecDeque;
pub use string::String;
pub use vec::Vec;
pub use vec_map::VecMap;

// Needed for the vec! macro
pub use alloc::boxed;

#[macro_use]
mod macros;

pub mod binary_heap;
mod bit;
mod btree;
pub mod borrow;
pub mod enum_set;
pub mod fmt;
pub mod linked_list;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod vec_deque;
pub mod vec_map;

#[unstable(feature = "collections",
           reason = "RFC 509")]
pub mod bit_vec {
    pub use bit::{BitVec, Iter};
}

#[unstable(feature = "collections",
           reason = "RFC 509")]
pub mod bit_set {
    pub use bit::{BitSet, Union, Intersection, Difference, SymmetricDifference};
    pub use bit::SetIter as Iter;
}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_map {
    pub use btree::map::*;
}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_set {
    pub use btree::set::*;
}


// FIXME(#14344) this shouldn't be necessary
#[doc(hidden)]
pub fn fixme_14344_be_sure_to_link_to_collections() {}

#[cfg(not(test))]
mod std {
    pub use core::ops;      // RangeFull
}

/// An endpoint of a range of keys.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Bound<T> {
    /// An inclusive bound.
    Included(T),
    /// An exclusive bound.
    Excluded(T),
    /// An infinite endpoint. Indicates that there is no bound in this direction.
    Unbounded,
}
