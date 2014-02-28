// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An efficient hash map for node IDs

use collections::hashmap;
use collections::{HashMap, HashSet};
use std::hash::{Hasher, Hash};
use std::iter;
use ast;

pub type NodeMap<T> = HashMap<ast::NodeId, T, NodeHasher>;
pub type DefIdMap<T> = HashMap<ast::DefId, T, NodeHasher>;
pub type NodeSet = HashSet<ast::NodeId, NodeHasher>;
pub type DefIdSet = HashSet<ast::DefId, NodeHasher>;

#[deriving(Clone)]
struct NodeHasher;

impl Hasher<u64> for NodeHasher {
    fn hash<T: Hash<u64>>(&self, t: &T) -> u64 {
        let mut last = 0;
        t.hash(&mut last);
        return last
    }
}

impl Hash<u64> for ast::NodeId {
    fn hash(&self, state: &mut u64) {
        *state = self.get() as u64;
    }
}

impl Hash<u64> for ast::DefId {
    fn hash(&self, state: &mut u64) {
        let ast::DefId { krate, node } = *self;
        // assert that these two types are each 32 bits
        let krate: u32 = krate;
        let node: u32 = node;
        *state = (krate << 32) as u64 | (node as u64);
    }
}

// Hacks to get good names
pub mod NodeMap {
    use collections::HashMap;
    pub fn new<T>() -> super::NodeMap<T> {
        HashMap::with_hasher(super::NodeHasher)
    }
}
pub mod NodeSet {
    use collections::HashSet;
    pub fn new() -> super::NodeSet {
        HashSet::with_hasher(super::NodeHasher)
    }
}
pub mod DefIdMap {
    use collections::HashMap;
    pub fn new<T>() -> super::DefIdMap<T> {
        HashMap::with_hasher(super::NodeHasher)
    }
}
pub mod DefIdSet {
    use collections::HashSet;
    pub fn new() -> super::DefIdSet {
        HashSet::with_hasher(super::NodeHasher)
    }
}
