// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Collection types.
 */

#[crate_id = "collections#0.10-pre"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[license = "MIT/ASL2"];

#[feature(macro_rules, managed_boxes)];

extern crate serialize;
#[cfg(test)] extern crate extra; // benchmark tests need this

pub use bitv::Bitv;
pub use btree::BTree;
pub use deque::Deque;
pub use dlist::DList;
pub use list::List;
pub use lru_cache::LruCache;
pub use priority_queue::PriorityQueue;
pub use ringbuf::RingBuf;
pub use smallintmap::SmallIntMap;
pub use treemap::{TreeMap, TreeSet};

pub mod bitv;
pub mod btree;
pub mod deque;
pub mod dlist;
pub mod list;
pub mod lru_cache;
pub mod priority_queue;
pub mod ringbuf;
pub mod smallintmap;
pub mod treemap;
