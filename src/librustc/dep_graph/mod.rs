// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::DefId;
use middle::ty;
use rustc_data_structures::fnv::FnvHashMap;
use rustc_data_structures::dependency;
use rustc_front::hir;
use rustc_front::intravisit::Visitor;
use std::ops::Index;
use std::hash::Hash;
use std::marker::PhantomData;
use std::rc::Rc;
use util::common;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepNode {
    // Represents the `Krate` as a whole (the `hir::Krate` value) (as
    // distinct from the krate module). This is basically a hash of
    // the entire krate, so if you read from `Krate` (e.g., by calling
    // `tcx.map.krate()`), we will have to assume that any change
    // means that you need to be recompiled. This is because the
    // `Krate` value gives you access to all other items. To avoid
    // this fate, do not call `tcx.map.krate()`; instead, prefer
    // wrappers like `tcx.visit_all_items_in_krate()`.  If there is no
    // suitable wrapper, you can use `tcx.dep_graph.ignore()` to gain
    // access to the krate, but you must remember to add suitable
    // edges yourself for the individual items that you read.
    Krate,

    // Represents the HIR node with the given node-id
    Hir(DefId),

    // Represents different phases in the compiler.
    CollectItem(DefId),
    TypeScheme(DefId),
    Coherence,
    CoherenceOverlapCheck(DefId),
    CoherenceOverlapCheckSpecial(DefId),
    CoherenceOrphanCheck(DefId),
    Variance,
    WfCheck(DefId),
    TypeckItemType(DefId),
    TypeckItemBody(DefId),
    Dropck(DefId),
    CheckConst(DefId),
    Privacy,
    IntrinsicCheck(DefId),
    MatchCheck(DefId),
    MirMapConstruction(DefId),
    BorrowCheck(DefId),
    RvalueCheck(DefId),
    Reachability,
    DeadCheck,
    StabilityCheck,
    LateLintCheck,
    IntrinsicUseCheck,
    TransLinkMeta,
    TransCrateItem(DefId),
    TransInlinedItem(DefId),
    TransWriteMetadata,

    // Nodes representing bits of computed IR in the tcx. Each of
    // these corresponds to a particular table in the tcx.
    ImplOrTraitItems(DefId),
    TraitItemDefIds(DefId),
    ImplTraitRef(DefId),
    Tcache(DefId),
    TraitDefs(DefId),
    AdtDefs(DefId),
    Predicates(DefId),
    ItemVarianceMap(DefId),
}

impl dependency::DepNodeId for DepNode { }

pub type DepGraph = dependency::DepGraph<DepNode>;

/// A DepTrackingMap offers a subset of the `Map` API and ensures that
/// we make calls to `read` and `write` as appropriate. We key the
/// maps with a unique type for brevity.
pub struct DepTrackingMap<M: DepTrackingMapId> {
    phantom: PhantomData<M>,
    graph: Rc<DepGraph>,
    map: FnvHashMap<M::Key, M::Value>,
}

pub trait DepTrackingMapId {
    type Key: Eq + Hash + Clone;
    type Value: Clone;
    fn to_dep_node(key: &Self::Key) -> DepNode;
}

impl<M: DepTrackingMapId> DepTrackingMap<M> {
    pub fn new(graph: Rc<DepGraph>) -> DepTrackingMap<M> {
        DepTrackingMap {
            phantom: PhantomData,
            graph: graph,
            map: FnvHashMap()
        }
    }

    /// Registers a (synthetic) read from the key `k`. Usually this
    /// is invoked automatically by `get`.
    pub fn read(&self, k: &M::Key) {
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

impl<M: DepTrackingMapId> common::MemoizationMap for DepTrackingMap<M> {
    type Key = M::Key;
    type Value = M::Value;
    fn get(&self, key: &M::Key) -> Option<&M::Value> {
        self.get(key)
    }
    fn insert(&mut self, key: M::Key, value: M::Value) -> Option<M::Value> {
        self.insert(key, value)
    }
}

impl<'k, M: DepTrackingMapId> Index<&'k M::Key> for DepTrackingMap<M> {
    type Output = M::Value;

    #[inline]
    fn index(&self, k: &'k M::Key) -> &M::Value {
        self.get(k).unwrap()
    }
}

/// Visit all the items in the krate in some order. When visiting a
/// particular item, first create a dep-node by calling `dep_node_fn`
/// and push that onto the dep-graph stack of tasks, and also create a
/// read edge from the corresponding AST node. This is used in
/// compiler passes to automatically record the item that they are
/// working on.
pub fn visit_all_items_in_krate<'tcx,V,F>(tcx: &ty::ctxt<'tcx>,
                                          mut dep_node_fn: F,
                                          visitor: &mut V)
    where F: FnMut(DefId) -> DepNode, V: Visitor<'tcx>
{
    struct TrackingVisitor<'visit, 'tcx: 'visit, F: 'visit, V: 'visit> {
        tcx: &'visit ty::ctxt<'tcx>,
        dep_node_fn: &'visit mut F,
        visitor: &'visit mut V
    }

    impl<'visit, 'tcx, F, V> Visitor<'tcx> for TrackingVisitor<'visit, 'tcx, F, V>
        where F: FnMut(DefId) -> DepNode, V: Visitor<'tcx>
    {
        fn visit_item(&mut self, i: &'tcx hir::Item) {
            let item_def_id = self.tcx.map.local_def_id(i.id);
            let task_id = (self.dep_node_fn)(item_def_id);
            debug!("About to start task {:?}", task_id);
            let _task = self.tcx.dep_graph.in_task(task_id);
            self.tcx.dep_graph.read(DepNode::Hir(item_def_id));
            self.visitor.visit_item(i)
        }
    }

    let krate = tcx.dep_graph.with_ignore(|| tcx.map.krate());
    let mut tracking_visitor = TrackingVisitor {
        tcx: tcx,
        dep_node_fn: &mut dep_node_fn,
        visitor: visitor
    };
    krate.visit_all_items(&mut tracking_visitor)
}
