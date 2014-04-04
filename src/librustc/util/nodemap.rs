// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An efficient hash map for node IDs.
//!
//! The hashmap in libcollections by default uses SipHash which isn't quite as speedy as we want.
//! In the compiler we're not really worried about DOS attempts, so we just default to a
//! non-cryptographic hash.

use collections::{HashMap, HashSet};
use std::hash::fnv::FnvHasher;
use syntax::ast;

pub type FnvHashMap<K, V> = HashMap<K, V, FnvHasher>;
pub type FnvHashSet<V> = HashSet<V, FnvHasher>;

pub type NodeMap<T> = FnvHashMap<ast::NodeId, T>;
pub type DefIdMap<T> = FnvHashMap<ast::DefId, T>;

pub type NodeSet = FnvHashSet<ast::NodeId>;
pub type DefIdSet = FnvHashSet<ast::DefId>;

// Hacks to get good names
pub mod FnvHashMap {
    use std::hash::Hash;
    use std::hash::fnv::{FnvHasher, FnvState};
    use collections::HashMap;
    pub fn new<K: Hash<FnvState> + TotalEq, V>() -> super::FnvHashMap<K, V> {
        HashMap::with_hasher(FnvHasher)
    }
}
pub mod FnvHashSet {
    use std::hash::Hash;
    use std::hash::fnv::{FnvHasher, FnvState};
    use collections::HashSet;
    pub fn new<V: Hash<FnvState> + TotalEq>() -> super::FnvHashSet<V> {
        HashSet::with_hasher(FnvHasher)
    }
}
pub mod NodeMap {
    pub fn new<T>() -> super::NodeMap<T> {
        super::FnvHashMap::new()
    }
}
pub mod DefIdMap {
    pub fn new<T>() -> super::DefIdMap<T> {
        super::FnvHashMap::new()
    }
}
pub mod NodeSet {
    use std::hash::fnv::FnvHasher;
    use collections::HashSet;
    pub fn new() -> super::NodeSet {
        HashSet::with_hasher(FnvHasher)
    }
}
pub mod DefIdSet {
    use std::hash::fnv::FnvHasher;
    use collections::HashSet;
    pub fn new() -> super::DefIdSet {
        HashSet::with_hasher(FnvHasher)
    }
}
