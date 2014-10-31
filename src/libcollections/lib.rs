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
#![experimental]
#![crate_type = "rlib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![allow(unknown_features)]
#![feature(macro_rules, default_type_params, phase, globs)]
#![feature(unsafe_destructor, import_shadowing, slicing_syntax)]
#![no_std]

#[phase(plugin, link)] extern crate core;
extern crate unicode;
extern crate alloc;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;

#[cfg(test)] #[phase(plugin, link)] extern crate std;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

pub use bitv::{Bitv, BitvSet};
pub use btree::{BTreeMap, BTreeSet};
pub use dlist::DList;
pub use enum_set::EnumSet;
pub use priority_queue::PriorityQueue;
pub use ringbuf::RingBuf;
pub use smallintmap::SmallIntMap;
pub use string::String;
pub use treemap::{TreeMap, TreeSet};
pub use trie::{TrieMap, TrieSet};
pub use vec::Vec;

mod macros;

pub mod bitv;
pub mod btree;
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

#[cfg(test)] mod bench;

// FIXME(#14344) this shouldn't be necessary
#[doc(hidden)]
pub fn fixme_14344_be_sure_to_link_to_collections() {}

#[cfg(not(test))]
mod std {
    pub use core::fmt;      // necessary for panic!()
    pub use core::option;   // necessary for panic!()
    pub use core::clone;    // deriving(Clone)
    pub use core::cmp;      // deriving(Eq, Ord, etc.)
    pub use hash;           // deriving(Hash)
}
