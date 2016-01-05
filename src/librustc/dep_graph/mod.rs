// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::thread::{DepGraphThreadData, DepMessage};
use middle::def_id::DefId;
use middle::ty;
use middle::ty::fast_reject::SimplifiedType;
use rustc_front::hir;
use rustc_front::intravisit::Visitor;
use std::rc::Rc;

mod dep_tracking_map;
mod edges;
mod query;
mod raii;
mod thread;

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
    Coherence,
    CoherenceCheckImpl(DefId),
    CoherenceOverlapCheck(DefId),
    CoherenceOverlapCheckSpecial(DefId),
    CoherenceOrphanCheck(DefId),
    Variance,
    WfCheck(DefId),
    TypeckItemType(DefId),
    TypeckItemBody(DefId),
    Dropck,
    DropckImpl(DefId),
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
    TransCrate,
    TransCrateItem(DefId),
    TransInlinedItem(DefId),
    TransWriteMetadata,

    // Nodes representing bits of computed IR in the tcx. Each shared
    // table in the tcx (or elsewhere) maps to one of these
    // nodes. Often we map multiple tables to the same node if there
    // is no point in distinguishing them (e.g., both the type and
    // predicates for an item wind up in `ItemSignature`). Other
    // times, such as `ImplItems` vs `TraitItemDefIds`, tables which
    // might be mergable are kept distinct because the sets of def-ids
    // to which they apply are disjoint, and hence we might as well
    // have distinct labels for easier debugging.
    ImplOrTraitItems(DefId),
    ItemSignature(DefId),
    FieldTy(DefId),
    TraitItemDefIds(DefId),
    InherentImpls(DefId),
    ImplItems(DefId),

    // The set of impls for a given trait. Ultimately, it would be
    // nice to get more fine-grained here (e.g., to include a
    // simplified type), but we can't do that until we restructure the
    // HIR to distinguish the *header* of an impl from its body.  This
    // is because changes to the header may change the self-type of
    // the impl and hence would require us to be more conservative
    // than changes in the impl body.
    TraitImpls(DefId),

    // Nodes representing caches. To properly handle a true cache, we
    // don't use a DepTrackingMap, but rather we push a task node.
    // Otherwise the write into the map would be incorrectly
    // attributed to the first task that happened to fill the cache,
    // which would yield an overly conservative dep-graph.
    TraitItems(DefId),
    ReprHints(DefId),
    TraitSelect(DefId, Option<SimplifiedType>),
}

#[derive(Clone)]
pub struct DepGraph {
    data: Rc<DepGraphThreadData>
}

impl DepGraph {
    pub fn new(enabled: bool) -> DepGraph {
        DepGraph {
            data: Rc::new(DepGraphThreadData::new(enabled))
        }
    }

    pub fn query(&self) -> DepGraphQuery {
        self.data.query()
    }

    pub fn in_ignore<'graph>(&'graph self) -> raii::IgnoreTask<'graph> {
        raii::IgnoreTask::new(&self.data)
    }

    pub fn in_task<'graph>(&'graph self, key: DepNode) -> raii::DepTask<'graph> {
        raii::DepTask::new(&self.data, key)
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_ignore();
        op()
    }

    pub fn with_task<OP,R>(&self, key: DepNode, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_task(key);
        op()
    }

    pub fn read(&self, v: DepNode) {
        self.data.enqueue(DepMessage::Read(v));
    }

    pub fn write(&self, v: DepNode) {
        self.data.enqueue(DepMessage::Write(v));
    }
}

pub use self::dep_tracking_map::{DepTrackingMap, DepTrackingMapId};

pub use self::query::DepGraphQuery;

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
