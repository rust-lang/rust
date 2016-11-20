// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use rustc_data_structures::fx::FxHashMap;
use std::cell::RefCell;
use std::ops::Index;
use std::hash::Hash;
use std::marker::PhantomData;
use util::common::MemoizationMap;

use super::{DepNode, DepGraph};

/// A DepTrackingMap offers a subset of the `Map` API and ensures that
/// we make calls to `read` and `write` as appropriate. We key the
/// maps with a unique type for brevity.
pub struct DepTrackingMap<M: DepTrackingMapConfig> {
    phantom: PhantomData<M>,
    graph: DepGraph,
    map: FxHashMap<M::Key, M::Value>,
}

pub trait DepTrackingMapConfig {
    type Key: Eq + Hash + Clone;
    type Value: Clone;
    fn to_dep_node(key: &Self::Key) -> DepNode<DefId>;
}

impl<M: DepTrackingMapConfig> DepTrackingMap<M> {
    pub fn new(graph: DepGraph) -> DepTrackingMap<M> {
        DepTrackingMap {
            phantom: PhantomData,
            graph: graph,
            map: FxHashMap()
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

    pub fn get_mut(&mut self, k: &M::Key) -> Option<&mut M::Value> {
        self.read(k);
        self.write(k);
        self.map.get_mut(k)
    }

    pub fn insert(&mut self, k: M::Key, v: M::Value) -> Option<M::Value> {
        self.write(&k);
        self.map.insert(k, v)
    }

    pub fn contains_key(&self, k: &M::Key) -> bool {
        self.read(k);
        self.map.contains_key(k)
    }

    pub fn keys(&self) -> Vec<M::Key> {
        self.map.keys().cloned().collect()
    }

    /// Append `elem` to the vector stored for `k`, creating a new vector if needed.
    /// This is considered a write to `k`.
    pub fn push<E: Clone>(&mut self, k: M::Key, elem: E)
        where M: DepTrackingMapConfig<Value=Vec<E>>
    {
        self.write(&k);
        self.map.entry(k)
                .or_insert(Vec::new())
                .push(elem);
    }
}

impl<M: DepTrackingMapConfig> MemoizationMap for RefCell<DepTrackingMap<M>> {
    type Key = M::Key;
    type Value = M::Value;

    /// Memoizes an entry in the dep-tracking-map. If the entry is not
    /// already present, then `op` will be executed to compute its value.
    /// The resulting dependency graph looks like this:
    ///
    ///     [op] -> Map(key) -> CurrentTask
    ///
    /// Here, `[op]` represents whatever nodes `op` reads in the
    /// course of execution; `Map(key)` represents the node for this
    /// map; and `CurrentTask` represents the current task when
    /// `memoize` is invoked.
    ///
    /// **Important:* when `op` is invoked, the current task will be
    /// switched to `Map(key)`. Therefore, if `op` makes use of any
    /// HIR nodes or shared state accessed through its closure
    /// environment, it must explicitly register a read of that
    /// state. As an example, see `type_of_item` in `collect`,
    /// which looks something like this:
    ///
    /// ```
    /// fn type_of_item(..., item: &hir::Item) -> Ty<'tcx> {
    ///     let item_def_id = ccx.tcx.map.local_def_id(it.id);
    ///     ccx.tcx.item_types.memoized(item_def_id, || {
    ///         ccx.tcx.dep_graph.read(DepNode::Hir(item_def_id)); // (*)
    ///         compute_type_of_item(ccx, item)
    ///     });
    /// }
    /// ```
    ///
    /// The key is the line marked `(*)`: the closure implicitly
    /// accesses the body of the item `item`, so we register a read
    /// from `Hir(item_def_id)`.
    fn memoize<OP>(&self, key: M::Key, op: OP) -> M::Value
        where OP: FnOnce() -> M::Value
    {
        let graph;
        {
            let this = self.borrow();
            if let Some(result) = this.map.get(&key) {
                this.read(&key);
                return result.clone();
            }
            graph = this.graph.clone();
        }

        let _task = graph.in_task(M::to_dep_node(&key));
        let result = op();
        self.borrow_mut().map.insert(key, result.clone());
        result
    }
}

impl<'k, M: DepTrackingMapConfig> Index<&'k M::Key> for DepTrackingMap<M> {
    type Output = M::Value;

    #[inline]
    fn index(&self, k: &'k M::Key) -> &M::Value {
        self.get(k).unwrap()
    }
}

