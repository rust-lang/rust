// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHashingContextProvider};
use session::config::OutputType;
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use util::common::{ProfileQueriesMsg, profq_msg};

use ich::Fingerprint;

use super::dep_node::{DepNode, DepKind, WorkProductId};
use super::query::DepGraphQuery;
use super::raii;
use super::safe::DepGraphSafe;
use super::edges::{DepGraphEdges, DepNodeIndex};

#[derive(Clone)]
pub struct DepGraph {
    data: Option<Rc<DepGraphData>>
}

struct DepGraphData {
    /// The actual graph data.
    edges: RefCell<DepGraphEdges>,

    /// When we load, there may be `.o` files, cached mir, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: RefCell<FxHashMap<WorkProductId, WorkProduct>>,

    /// Work-products that we generate in this run.
    work_products: RefCell<FxHashMap<WorkProductId, WorkProduct>>,

    dep_node_debug: RefCell<FxHashMap<DepNode, String>>,
}

impl DepGraph {
    pub fn new(enabled: bool) -> DepGraph {
        DepGraph {
            data: if enabled {
                Some(Rc::new(DepGraphData {
                    previous_work_products: RefCell::new(FxHashMap()),
                    work_products: RefCell::new(FxHashMap()),
                    edges: RefCell::new(DepGraphEdges::new()),
                    dep_node_debug: RefCell::new(FxHashMap()),
                }))
            } else {
                None
            }
        }
    }

    /// True if we are actually building the full dep-graph.
    #[inline]
    pub fn is_fully_enabled(&self) -> bool {
        self.data.is_some()
    }

    pub fn query(&self) -> DepGraphQuery {
        self.data.as_ref().unwrap().edges.borrow().query()
    }

    pub fn in_ignore<'graph>(&'graph self) -> Option<raii::IgnoreTask<'graph>> {
        self.data.as_ref().map(|data| raii::IgnoreTask::new(&data.edges))
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        let _task = self.in_ignore();
        op()
    }

    /// Starts a new dep-graph task. Dep-graph tasks are specified
    /// using a free function (`task`) and **not** a closure -- this
    /// is intentional because we want to exercise tight control over
    /// what state they have access to. In particular, we want to
    /// prevent implicit 'leaks' of tracked state into the task (which
    /// could then be read without generating correct edges in the
    /// dep-graph -- see the [README] for more details on the
    /// dep-graph). To this end, the task function gets exactly two
    /// pieces of state: the context `cx` and an argument `arg`. Both
    /// of these bits of state must be of some type that implements
    /// `DepGraphSafe` and hence does not leak.
    ///
    /// The choice of two arguments is not fundamental. One argument
    /// would work just as well, since multiple values can be
    /// collected using tuples. However, using two arguments works out
    /// to be quite convenient, since it is common to need a context
    /// (`cx`) and some argument (e.g., a `DefId` identifying what
    /// item to process).
    ///
    /// For cases where you need some other number of arguments:
    ///
    /// - If you only need one argument, just use `()` for the `arg`
    ///   parameter.
    /// - If you need 3+ arguments, use a tuple for the
    ///   `arg` parameter.
    ///
    /// [README]: README.md
    pub fn with_task<C, A, R, HCX>(&self,
                                   key: DepNode,
                                   cx: C,
                                   arg: A,
                                   task: fn(C, A) -> R)
                                   -> (R, DepNodeIndex)
        where C: DepGraphSafe + StableHashingContextProvider<ContextType=HCX>,
              R: HashStable<HCX>,
    {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().push_task(key);
            if cfg!(debug_assertions) {
                profq_msg(ProfileQueriesMsg::TaskBegin(key.clone()))
            };

            // In incremental mode, hash the result of the task. We don't
            // do anything with the hash yet, but we are computing it
            // anyway so that
            //  - we make sure that the infrastructure works and
            //  - we can get an idea of the runtime cost.
            let mut hcx = cx.create_stable_hashing_context();

            let result = task(cx, arg);
            if cfg!(debug_assertions) {
                profq_msg(ProfileQueriesMsg::TaskEnd)
            };
            let dep_node_index = data.edges.borrow_mut().pop_task(key);

            let mut stable_hasher = StableHasher::new();
            result.hash_stable(&mut hcx, &mut stable_hasher);
            let _: Fingerprint = stable_hasher.finish();

            (result, dep_node_index)
        } else {
            (task(cx, arg), DepNodeIndex::INVALID)
        }
    }

    /// Execute something within an "anonymous" task, that is, a task the
    /// DepNode of which is determined by the list of inputs it read from.
    pub fn with_anon_task<OP,R>(&self, dep_kind: DepKind, op: OP) -> (R, DepNodeIndex)
        where OP: FnOnce() -> R
    {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().push_anon_task();
            let result = op();
            let dep_node = data.edges.borrow_mut().pop_anon_task(dep_kind);
            (result, dep_node)
        } else {
            (op(), DepNodeIndex::INVALID)
        }
    }

    #[inline]
    pub fn read(&self, v: DepNode) {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().read(v);
        }
    }

    #[inline]
    pub fn read_index(&self, v: DepNodeIndex) {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().read_index(v);
        }
    }

    /// Only to be used during graph loading
    #[inline]
    pub fn add_edge_directly(&self, source: DepNode, target: DepNode) {
        self.data.as_ref().unwrap().edges.borrow_mut().add_edge(source, target);
    }

    /// Only to be used during graph loading
    pub fn add_node_directly(&self, node: DepNode) {
        self.data.as_ref().unwrap().edges.borrow_mut().add_node(node);
    }

    pub fn alloc_input_node(&self, node: DepNode) -> DepNodeIndex {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().add_node(node)
        } else {
            DepNodeIndex::INVALID
        }
    }

    /// Indicates that a previous work product exists for `v`. This is
    /// invoked during initial start-up based on what nodes are clean
    /// (and what files exist in the incr. directory).
    pub fn insert_previous_work_product(&self, v: &WorkProductId, data: WorkProduct) {
        debug!("insert_previous_work_product({:?}, {:?})", v, data);
        self.data
            .as_ref()
            .unwrap()
            .previous_work_products
            .borrow_mut()
            .insert(v.clone(), data);
    }

    /// Indicates that we created the given work-product in this run
    /// for `v`. This record will be preserved and loaded in the next
    /// run.
    pub fn insert_work_product(&self, v: &WorkProductId, data: WorkProduct) {
        debug!("insert_work_product({:?}, {:?})", v, data);
        self.data
            .as_ref()
            .unwrap()
            .work_products
            .borrow_mut()
            .insert(v.clone(), data);
    }

    /// Check whether a previous work product exists for `v` and, if
    /// so, return the path that leads to it. Used to skip doing work.
    pub fn previous_work_product(&self, v: &WorkProductId) -> Option<WorkProduct> {
        self.data
            .as_ref()
            .and_then(|data| {
                data.previous_work_products.borrow().get(v).cloned()
            })
    }

    /// Access the map of work-products created during this run. Only
    /// used during saving of the dep-graph.
    pub fn work_products(&self) -> Ref<FxHashMap<WorkProductId, WorkProduct>> {
        self.data.as_ref().unwrap().work_products.borrow()
    }

    /// Access the map of work-products created during the cached run. Only
    /// used during saving of the dep-graph.
    pub fn previous_work_products(&self) -> Ref<FxHashMap<WorkProductId, WorkProduct>> {
        self.data.as_ref().unwrap().previous_work_products.borrow()
    }

    #[inline(always)]
    pub fn register_dep_node_debug_str<F>(&self,
                                          dep_node: DepNode,
                                          debug_str_gen: F)
        where F: FnOnce() -> String
    {
        let dep_node_debug = &self.data.as_ref().unwrap().dep_node_debug;

        if dep_node_debug.borrow().contains_key(&dep_node) {
            return
        }
        let debug_str = debug_str_gen();
        dep_node_debug.borrow_mut().insert(dep_node, debug_str);
    }

    pub(super) fn dep_node_debug_str(&self, dep_node: DepNode) -> Option<String> {
        self.data.as_ref().and_then(|t| t.dep_node_debug.borrow().get(&dep_node).cloned())
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
    pub cgu_name: String,
    /// Extra hash used to decide if work-product is still suitable;
    /// note that this is *not* a hash of the work-product itself.
    /// See documentation on `WorkProduct` type for an example.
    pub input_hash: u64,

    /// Saved files associated with this CGU
    pub saved_files: Vec<(OutputType, String)>,
}
