//! Cache for candidate selection.

use crate::dep_graph::{DepContext, DepNodeIndex};

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;

use std::fmt::Debug;
use std::hash::Hash;

pub struct Cache<Key, Value> {
    hashmap: Lock<FxHashMap<Key, WithDepNode<Value>>>,
}

impl<Key: Clone, Value: Clone> Clone for Cache<Key, Value> {
    fn clone(&self) -> Self {
        Self { hashmap: Lock::new(self.hashmap.borrow().clone()) }
    }
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

impl<Key: Eq + Hash, Value: Eq + Clone + Debug> Cache<Key, Value> {
    pub fn get<Tcx: DepContext>(&self, key: &Key, tcx: Tcx) -> Option<Value> {
        Some(self.hashmap.borrow().get(key)?.get(tcx))
    }

    pub fn insert(&self, key: Key, dep_node: DepNodeIndex, value: Value) {
        // FIXME(#50507): For some reason we're getting different `DepNodeIndex`es
        // for the same evaluation in the parallel compiler. Once that's fixed
        // we can just use `HashMap::insert_same` here instead of doing all this.
        self.hashmap
            .borrow_mut()
            .entry(key)
            .and_modify(|old_value| {
                assert_eq!(old_value.cached_value, value, "unstable cached value")
            })
            .or_insert(WithDepNode::new(dep_node, value));
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

    pub fn get<Tcx: DepContext>(&self, tcx: Tcx) -> T {
        tcx.dep_graph().read_index(self.dep_node);
        self.cached_value.clone()
    }
}
