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
       html_root_url = "http://doc.rust-lang.org/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(macro_rules, managed_boxes, default_type_params, phase, globs)]
#![no_std]

#[phase(plugin, link)] extern crate core;
extern crate alloc;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;
#[cfg(test)] extern crate debug;

#[cfg(test)] #[phase(plugin, link)] extern crate std;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

use core::prelude::*;

pub use core::collections::Collection;
pub use bitv::{Bitv, BitvSet};
pub use btree::BTree;
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

// Internal unicode fiddly bits for the str module
mod unicode;

mod deque;

/// A trait to represent mutable containers
pub trait Mutable: Collection {
    /// Clear the container, removing all values.
    fn clear(&mut self);
}

/// A map is a key-value store where values may be looked up by their keys. This
/// trait provides basic operations to operate on these stores.
pub trait Map<K, V>: Collection {
    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &K) -> Option<&'a V>;

    /// Return true if the map contains a value for the specified key
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }
}

/// This trait provides basic operations to modify the contents of a map.
pub trait MutableMap<K, V>: Map<K, V> + Mutable {
    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    #[inline]
    fn insert(&mut self, key: K, value: V) -> bool {
        self.swap(key, value).is_none()
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    #[inline]
    fn remove(&mut self, key: &K) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, k: K, v: V) -> Option<V>;

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, k: &K) -> Option<V>;

    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>;
}

/// A set is a group of objects which are each distinct from one another. This
/// trait represents actions which can be performed on sets to iterate over
/// them.
pub trait Set<T>: Collection {
    /// Return true if the set contains a value
    fn contains(&self, value: &T) -> bool;

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    fn is_disjoint(&self, other: &Self) -> bool;

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &Self) -> bool;

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    // FIXME #8154: Add difference, sym. difference, intersection and union iterators
}

/// This trait represents actions which can be performed on sets to mutate
/// them.
pub trait MutableSet<T>: Set<T> + Mutable {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: T) -> bool;

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &T) -> bool;
}

/// A double-ended sequence that allows querying, insertion and deletion at both
/// ends.
pub trait Deque<T> : Mutable {
    /// Provide a reference to the front element, or None if the sequence is
    /// empty
    fn front<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the front element, or None if the
    /// sequence is empty
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Provide a reference to the back element, or None if the sequence is
    /// empty
    fn back<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the back element, or None if the sequence
    /// is empty
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Insert an element first in the sequence
    fn push_front(&mut self, elt: T);

    /// Insert an element last in the sequence
    fn push_back(&mut self, elt: T);

    /// Remove the last element and return it, or None if the sequence is empty
    fn pop_back(&mut self) -> Option<T>;

    /// Remove the first element and return it, or None if the sequence is empty
    fn pop_front(&mut self) -> Option<T>;
}

// FIXME(#14344) this shouldn't be necessary
#[doc(hidden)]
pub fn fixme_14344_be_sure_to_link_to_collections() {}

#[cfg(not(test))]
mod std {
    pub use core::fmt;      // necessary for fail!()
    pub use core::option;   // necessary for fail!()
    pub use core::clone;    // deriving(Clone)
    pub use core::cmp;      // deriving(Eq, Ord, etc.)
    pub use hash;           // deriving(Hash)
}
