//! Cache for candidate selection.

use crate::dep_graph::DepNodeIndex;
use crate::query::QueryContext;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::HashMapExt;
use rustc_data_structures::sync::Lock;

use std::hash::Hash;

#[derive(Clone)]
pub struct Cache<Key, Value> {
    hashmap: Lock<FxHashMap<Key, WithDepNode<Value>>>,
}

impl<Key, Value> Default for Cache<Key, Value> {
    fn default() -> Self {
        Self { hashmap: Default::default() }
    }
}

impl<Key, Value> Cache<Key, Value> {
    /// Actually frees the underlying memory in contrast to what stdlib containers do on `clear`
    pub fn clear(&self) {
        *self.hashmap.borrow_mut() = Default::default();
    }
}

impl<Key: Eq + Hash, Value: Clone> Cache<Key, Value> {
    pub fn get<CTX: QueryContext>(&self, key: &Key, tcx: CTX) -> Option<Value> {
        Some(self.hashmap.borrow().get(key)?.get(tcx))
    }

    pub fn insert(&self, key: Key, dep_node: DepNodeIndex, value: Value) {
        self.hashmap.borrow_mut().insert(key, WithDepNode::new(dep_node, value));
    }

    pub fn insert_same(&self, key: Key, dep_node: DepNodeIndex, value: Value)
    where
        Value: Eq,
    {
        self.hashmap.borrow_mut().insert_same(key, WithDepNode::new(dep_node, value));
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct WithDepNode<T> {
    dep_node: DepNodeIndex,
    cached_value: T,
}

impl<T: Clone> WithDepNode<T> {
    pub fn new(dep_node: DepNodeIndex, cached_value: T) -> Self {
        WithDepNode { dep_node, cached_value }
    }

    pub fn get<CTX: QueryContext>(&self, tcx: CTX) -> T {
        tcx.dep_graph().read_index(self.dep_node);
        self.cached_value.clone()
    }
}
