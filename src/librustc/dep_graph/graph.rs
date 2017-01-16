// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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
use session::config::OutputType;
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use super::dep_node::{DepNode, WorkProductId};
use super::query::DepGraphQuery;
use super::raii;
use super::thread::{DepGraphThreadData, DepMessage};

#[derive(Clone)]
pub struct DepGraph {
    data: Rc<DepGraphData>
}

struct DepGraphData {
    /// We send messages to the thread to let it build up the dep-graph
    /// from the current run.
    thread: DepGraphThreadData,

    /// When we load, there may be `.o` files, cached mir, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: RefCell<FxHashMap<Arc<WorkProductId>, WorkProduct>>,

    /// Work-products that we generate in this run.
    work_products: RefCell<FxHashMap<Arc<WorkProductId>, WorkProduct>>,
}

impl DepGraph {
    pub fn new(enabled: bool) -> DepGraph {
        DepGraph {
            data: Rc::new(DepGraphData {
                thread: DepGraphThreadData::new(enabled),
                previous_work_products: RefCell::new(FxHashMap()),
                work_products: RefCell::new(FxHashMap()),
            })
        }
    }

    /// True if we are actually building the full dep-graph.
    #[inline]
    pub fn is_fully_enabled(&self) -> bool {
        self.data.thread.is_fully_enabled()
    }

    pub fn query(&self) -> DepGraphQuery<DefId> {
        self.data.thread.query()
    }

    pub fn in_ignore<'graph>(&'graph self) -> Option<raii::IgnoreTask<'graph>> {
        raii::IgnoreTask::new(&self.data.thread)
    }

    pub fn in_task<'graph>(&'graph self, key: DepNode<DefId>) -> Option<raii::DepTask<'graph>> {
        raii::DepTask::new(&self.data.thread, key)
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_ignore();
        op()
    }

    pub fn with_task<OP,R>(&self, key: DepNode<DefId>, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_task(key);
        op()
    }

    pub fn read(&self, v: DepNode<DefId>) {
        if self.data.thread.is_enqueue_enabled() {
            self.data.thread.enqueue(DepMessage::Read(v));
        }
    }

    pub fn write(&self, v: DepNode<DefId>) {
        if self.data.thread.is_enqueue_enabled() {
            self.data.thread.enqueue(DepMessage::Write(v));
        }
    }

    /// Indicates that a previous work product exists for `v`. This is
    /// invoked during initial start-up based on what nodes are clean
    /// (and what files exist in the incr. directory).
    pub fn insert_previous_work_product(&self, v: &Arc<WorkProductId>, data: WorkProduct) {
        debug!("insert_previous_work_product({:?}, {:?})", v, data);
        self.data.previous_work_products.borrow_mut()
                                        .insert(v.clone(), data);
    }

    /// Indicates that we created the given work-product in this run
    /// for `v`. This record will be preserved and loaded in the next
    /// run.
    pub fn insert_work_product(&self, v: &Arc<WorkProductId>, data: WorkProduct) {
        debug!("insert_work_product({:?}, {:?})", v, data);
        self.data.work_products.borrow_mut()
                               .insert(v.clone(), data);
    }

    /// Check whether a previous work product exists for `v` and, if
    /// so, return the path that leads to it. Used to skip doing work.
    pub fn previous_work_product(&self, v: &Arc<WorkProductId>) -> Option<WorkProduct> {
        self.data.previous_work_products.borrow()
                                        .get(v)
                                        .cloned()
    }

    /// Access the map of work-products created during this run. Only
    /// used during saving of the dep-graph.
    pub fn work_products(&self) -> Ref<FxHashMap<Arc<WorkProductId>, WorkProduct>> {
        self.data.work_products.borrow()
    }

    /// Access the map of work-products created during the cached run. Only
    /// used during saving of the dep-graph.
    pub fn previous_work_products(&self) -> Ref<FxHashMap<Arc<WorkProductId>, WorkProduct>> {
        self.data.previous_work_products.borrow()
    }
}

/// A "work product" is an intermediate result that we save into the
/// incremental directory for later re-use. The primary example are
/// the object files that we save for each partition at code
/// generation time.
///
/// Each work product is associated with a dep-node, representing the
/// process that produced the work-product. If that dep-node is found
/// to be dirty when we load up, then we will delete the work-product
/// at load time. If the work-product is found to be clean, then we
/// will keep a record in the `previous_work_products` list.
///
/// In addition, work products have an associated hash. This hash is
/// an extra hash that can be used to decide if the work-product from
/// a previous compilation can be re-used (in addition to the dirty
/// edges check).
///
/// As the primary example, consider the object files we generate for
/// each partition. In the first run, we create partitions based on
/// the symbols that need to be compiled. For each partition P, we
/// hash the symbols in P and create a `WorkProduct` record associated
/// with `DepNode::TransPartition(P)`; the hash is the set of symbols
/// in P.
///
/// The next time we compile, if the `DepNode::TransPartition(P)` is
/// judged to be clean (which means none of the things we read to
/// generate the partition were found to be dirty), it will be loaded
/// into previous work products. We will then regenerate the set of
/// symbols in the partition P and hash them (note that new symbols
/// may be added -- for example, new monomorphizations -- even if
/// nothing in P changed!). We will compare that hash against the
/// previous hash. If it matches up, we can reuse the object file.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct WorkProduct {
    /// Extra hash used to decide if work-product is still suitable;
    /// note that this is *not* a hash of the work-product itself.
    /// See documentation on `WorkProduct` type for an example.
    pub input_hash: u64,

    /// Saved files associated with this CGU
    pub saved_files: Vec<(OutputType, String)>,
}
