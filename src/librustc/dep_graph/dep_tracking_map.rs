// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::fnv::FnvHashMap;
use std::cell::RefCell;
use std::ops::Index;
use std::hash::Hash;
use std::marker::PhantomData;

use super::{DepNode, DepGraph};

/// A DepTrackingMap offers a subset of the `Map` API and ensures that
/// we make calls to `read` and `write` as appropriate. We key the
/// maps with a unique type for brevity.
pub struct DepTrackingMap<M: DepTrackingMapId> {
    phantom: PhantomData<M>,
    graph: DepGraph,
    map: FnvHashMap<M::Key, M::Value>,
}

pub trait DepTrackingMapId {
    type Key: Eq + Hash + Clone;
    type Value: Clone;
    fn to_dep_node(key: &Self::Key) -> DepNode;
}

impl<M: DepTrackingMapId> DepTrackingMap<M> {
    pub fn new(graph: DepGraph) -> DepTrackingMap<M> {
        DepTrackingMap {
            phantom: PhantomData,
            graph: graph,
            map: FnvHashMap()
        }
    }

    /// Registers a (synthetic) read from the key `k`. Usually this
    /// is invoked automatically by `get`.
    fn read(&self, k: &M::Key) {
        let dep_node = M::to_dep_node(k);
        self.graph.read(dep_node);
    }

    /// Registers a (synthetic) write to the key `k`. Usually this is
    /// invoked automatically by `insert`.
    fn write(&self, k: &M::Key) {
        let dep_node = M::to_dep_node(k);
        self.graph.write(dep_node);
    }

    pub fn get(&self, k: &M::Key) -> Option<&M::Value> {
        self.read(k);
        self.map.get(k)
    }

    pub fn insert(&mut self, k: M::Key, v: M::Value) -> Option<M::Value> {
        self.write(&k);
        self.map.insert(k, v)
    }

    pub fn contains_key(&self, k: &M::Key) -> bool {
        self.read(k);
        self.map.contains_key(k)
    }
}

impl<'k, M: DepTrackingMapId> Index<&'k M::Key> for DepTrackingMap<M> {
    type Output = M::Value;

    #[inline]
    fn index(&self, k: &'k M::Key) -> &M::Value {
        self.get(k).unwrap()
    }
}

