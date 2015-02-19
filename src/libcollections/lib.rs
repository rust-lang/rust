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


#![crate_name = "collections"]
#![unstable(feature = "collections")]
#![staged_api]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(alloc)]
#![feature(box_syntax)]
#![feature(box_patterns)]
#![feature(core)]
#![feature(staged_api)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unsafe_destructor)]
#![feature(unsafe_no_drop_flag)]
#![cfg_attr(test, feature(rand, rustc_private, test))]
#![cfg_attr(test, allow(deprecated))] // rand

#![feature(no_std)]
#![no_std]

#[macro_use]
extern crate core;

extern crate unicode;
extern crate alloc;

#[cfg(test)] extern crate test;
#[cfg(test)] #[macro_use] extern crate std;
#[cfg(test)] #[macro_use] extern crate log;

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

#[deprecated(since = "1.0.0", reason = "renamed to vec_deque")]
#[unstable(feature = "collections")]
pub use vec_deque as ring_buf;

#[deprecated(since = "1.0.0", reason = "renamed to linked_list")]
#[unstable(feature = "collections")]
pub use linked_list as dlist;

#[deprecated(since = "1.0.0", reason = "renamed to bit_vec")]
#[unstable(feature = "collections")]
pub use bit_vec as bitv;

#[deprecated(since = "1.0.0", reason = "renamed to bit_set")]
#[unstable(feature = "collections")]
pub use bit_set as bitv_set;

// Needed for the vec! macro
pub use alloc::boxed;

#[macro_use]
mod macros;

#[cfg(test)] #[macro_use] mod bench;

pub mod binary_heap;
mod bit;
mod btree;
pub mod linked_list;
pub mod enum_set;
pub mod fmt;
pub mod vec_deque;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod vec_map;

#[cfg(stage0)]
#[path = "borrow_stage0.rs"]
pub mod borrow;

#[cfg(not(stage0))]
pub mod borrow;

#[unstable(feature = "collections",
           reason = "RFC 509")]
pub mod bit_vec {
    pub use bit::{BitVec, Iter};

    #[deprecated(since = "1.0.0", reason = "renamed to BitVec")]
    #[unstable(feature = "collections")]
    pub use bit::BitVec as Bitv;
}

#[unstable(feature = "collections",
           reason = "RFC 509")]
pub mod bit_set {
    pub use bit::{BitSet, Union, Intersection, Difference, SymmetricDifference};
    pub use bit::SetIter as Iter;

    #[deprecated(since = "1.0.0", reason = "renamed to BitSet")]
    #[unstable(feature = "collections")]
    pub use bit::BitSet as BitvSet;
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

#[cfg(test)]
mod prelude {
    // from core.
    pub use core::clone::Clone;
    pub use core::cmp::{PartialEq, Eq, PartialOrd, Ord};
    pub use core::cmp::Ordering::{Less, Equal, Greater};
    pub use core::iter::range;
    pub use core::iter::{FromIterator, Extend, IteratorExt};
    pub use core::iter::{Iterator, DoubleEndedIterator, RandomAccessIterator};
    pub use core::iter::{ExactSizeIterator};
    pub use core::marker::{Copy, Send, Sized, Sync};
    pub use core::mem::drop;
    pub use core::ops::{Drop, Fn, FnMut, FnOnce};
    pub use core::option::Option;
    pub use core::option::Option::{Some, None};
    pub use core::ptr::PtrExt;
    pub use core::result::Result;
    pub use core::result::Result::{Ok, Err};

    // in core and collections (may differ).
    pub use slice::{AsSlice, SliceExt};
    pub use str::{Str, StrExt};

    // from other crates.
    pub use alloc::boxed::Box;
    pub use unicode::char::CharExt;

    // from collections.
    pub use borrow::IntoCow;
    pub use slice::SliceConcatExt;
    pub use string::{String, ToString};
    pub use vec::Vec;
}

/// An endpoint of a range of keys.
pub enum Bound<T> {
    /// An inclusive bound.
    Included(T),
    /// An exclusive bound.
    Excluded(T),
    /// An infinite endpoint. Indicates that there is no bound in this direction.
    Unbounded,
}
