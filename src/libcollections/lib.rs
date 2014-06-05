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

#![crate_id = "collections#0.11.0-pre"]
#![crate_type = "rlib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/")]

#![feature(macro_rules, managed_boxes, default_type_params, phase, globs)]
#![no_std]

#[phase(syntax, link)] extern crate core;
extern crate alloc;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;
#[cfg(test)] extern crate debug;
#[cfg(test)] #[phase(syntax, link)] extern crate std;
#[cfg(test)] #[phase(syntax, link)] extern crate log;

pub use bitv::{Bitv, BitvSet};
pub use btree::BTree;
pub use deque::Deque;
pub use dlist::DList;
pub use enum_set::EnumSet;
pub use priority_queue::PriorityQueue;
pub use ringbuf::RingBuf;
pub use smallintmap::SmallIntMap;
pub use treemap::{TreeMap, TreeSet};
pub use trie::{TrieMap, TrieSet};

mod macros;

pub mod bitv;
pub mod btree;
pub mod deque;
pub mod dlist;
pub mod enum_set;
pub mod priority_queue;
pub mod ringbuf;
pub mod smallintmap;
pub mod treemap;
pub mod trie;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod hash;

// Internal unicode fiddly bits for the str module
mod unicode;

// FIXME(#14008) should this actually exist, or should a method be added?
fn expect<T>(a: core::option::Option<T>, b: &str) -> T {
    match a {
        core::option::Some(a) => a,
        core::option::None => fail!("{}", b),
    }
}

#[cfg(not(test))]
mod std {
    pub use core::fmt;      // necessary for fail!()
    pub use core::option;   // necessary for fail!()
    pub use core::clone;    // deriving(Clone)
    pub use core::cmp;      // deriving(Eq, Ord, etc.)
    pub use hash;           // deriving(Hash)
}
