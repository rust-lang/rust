//! Cache for candidate selection.

use std::hash::Hash;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;

use crate::dep_graph::{DepContext, DepNodeIndex};

pub struct WithDepNodeCache<Key, Value> {
    hashmap: Lock<FxHashMap<Key, WithDepNode<Value>>>,
}

impl<Key: Clone, Value: Clone> Clone for WithDepNodeCache<Key, Value> {
    fn clone(&self) -> Self {
        Self { hashmap: Lock::new(self.hashmap.borrow().clone()) }
    }
}

impl<Key, Value> Default for WithDepNodeCache<Key, Value> {
    fn default() -> Self {
        Self { hashmap: Default::default() }
    }
}

impl<Key: Eq + Hash, Value: Clone> WithDepNodeCache<Key, Value> {
    pub fn get<Tcx: DepContext>(&self, key: &Key, tcx: Tcx) -> Option<Value> {
        Some(self.hashmap.borrow().get(key)?.get(tcx))
    }

    pub fn insert(&self, key: Key, dep_node: DepNodeIndex, value: Value) {
        self.hashmap.borrow_mut().insert(key, WithDepNode::new(dep_node, value));
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WithDepNode<T> {
    dep_node: DepNodeIndex,
    cached_value: T,
}

impl<T: Clone> WithDepNode<T> {
    pub fn new(dep_node: DepNodeIndex, cached_value: T) -> Self {
        WithDepNode { dep_node, cached_value }
    }

    pub fn get<Tcx: DepContext>(&self, tcx: Tcx) -> T {
        tcx.dep_graph().read_index(self.dep_node);
        self.cached_value.clone()
    }
}
