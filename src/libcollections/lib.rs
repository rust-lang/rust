// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "collections"]
#![crate_type = "rlib"]
#![allow(unused_attributes)]
#![unstable(feature = "collections",
            reason = "this library is unlikely to be stabilized in its current \
                      form or name",
            issue = "27783")]
#![rustc_deprecated(since = "1.20.0",
                    reason = "collections moved to `alloc`")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/",
       test(no_crate_inject, attr(allow(unused_variables), deny(warnings))))]
#![no_std]
#![needs_allocator]
#![deny(warnings)]

#![feature(alloc)]
#![feature(collections_range)]
#![feature(macro_reexport)]
#![feature(needs_allocator)]
#![feature(staged_api)]

//! Collection types
//!
//! See [`std::collections`](../std/collections/index.html) for a detailed
//! discussion of collections in Rust.

#[macro_reexport(vec, format)]
extern crate alloc;

pub use alloc::Bound;

pub use alloc::binary_heap;
pub use alloc::borrow;
pub use alloc::fmt;
pub use alloc::linked_list;
pub use alloc::range;
pub use alloc::slice;
pub use alloc::str;
pub use alloc::string;
pub use alloc::vec;
pub use alloc::vec_deque;

pub use alloc::btree_map;
pub use alloc::btree_set;

#[doc(no_inline)]
pub use alloc::binary_heap::BinaryHeap;
#[doc(no_inline)]
pub use alloc::btree_map::BTreeMap;
#[doc(no_inline)]
pub use alloc::btree_set::BTreeSet;
#[doc(no_inline)]
pub use alloc::linked_list::LinkedList;
#[doc(no_inline)]
pub use alloc::vec_deque::VecDeque;
#[doc(no_inline)]
pub use alloc::string::String;
#[doc(no_inline)]
pub use alloc::vec::Vec;
