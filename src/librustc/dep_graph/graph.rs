// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::stable_hasher::{HashStable, StableHasher,
                                           StableHashingContextProvider};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use session::config::OutputType;
use std::cell::{Ref, RefCell};
use std::hash::Hash;
use std::rc::Rc;
use util::common::{ProfileQueriesMsg, profq_msg};

use ich::Fingerprint;

use super::dep_node::{DepNode, DepKind, WorkProductId};
use super::query::DepGraphQuery;
use super::raii;
use super::safe::DepGraphSafe;
use super::edges::{self, DepGraphEdges};
use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};
use super::prev::PreviousDepGraph;

#[derive(Clone)]
pub struct DepGraph {
    data: Option<Rc<DepGraphData>>,

    // At the moment we are using DepNode as key here. In the future it might
    // be possible to use an IndexVec<DepNodeIndex, _> here. At the moment there
    // are a few problems with that:
    // - Some fingerprints are needed even if incr. comp. is disabled -- yet
    //   we need to have a dep-graph to generate DepNodeIndices.
    // - The architecture is still in flux and it's not clear what how to best
    //   implement things.
    fingerprints: Rc<RefCell<FxHashMap<DepNode, Fingerprint>>>
}

/// As a temporary measure, while transitioning to the new DepGraph
/// implementation, we maintain the old and the new dep-graph encoding in
/// parallel, so a DepNodeIndex actually contains two indices, one for each
/// version.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DepNodeIndex {
    legacy: edges::DepNodeIndex,
    new: DepNodeIndexNew,
}

impl DepNodeIndex {
    pub const INVALID: DepNodeIndex = DepNodeIndex {
        legacy: edges::DepNodeIndex::INVALID,
        new: DepNodeIndexNew::INVALID,
    };
}

struct DepGraphData {
    /// The old, initial encoding of the dependency graph. This will soon go
    /// away.
    edges: RefCell<DepGraphEdges>,

    /// The new encoding of the dependency graph, optimized for red/green
    /// tracking. The `current` field is the dependency graph of only the
    /// current compilation session: We don't merge the previous dep-graph into
    /// current one anymore.
    current: RefCell<CurrentDepGraph>,

    /// The dep-graph from the previous compilation session. It contains all
    /// nodes and edges as well as all fingerprints of nodes that have them.
    previous: PreviousDepGraph,

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

    pub fn new(prev_graph: PreviousDepGraph) -> DepGraph {
        DepGraph {
            data: Some(Rc::new(DepGraphData {
                previous_work_products: RefCell::new(FxHashMap()),
                work_products: RefCell::new(FxHashMap()),
                edges: RefCell::new(DepGraphEdges::new()),
                dep_node_debug: RefCell::new(FxHashMap()),
                current: RefCell::new(CurrentDepGraph::new()),
                previous: prev_graph,
            })),
            fingerprints: Rc::new(RefCell::new(FxHashMap())),
        }
    }

    pub fn new_disabled() -> DepGraph {
        DepGraph {
            data: None,
            fingerprints: Rc::new(RefCell::new(FxHashMap())),
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
        self.data.as_ref().map(|data| raii::IgnoreTask::new(&data.edges,
                                                            &data.current))
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
            data.current.borrow_mut().push_task(key);
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

            let dep_node_index_legacy = data.edges.borrow_mut().pop_task(key);
            let dep_node_index_new = data.current.borrow_mut().pop_task(key);

            let mut stable_hasher = StableHasher::new();
            result.hash_stable(&mut hcx, &mut stable_hasher);

            assert!(self.fingerprints
                        .borrow_mut()
                        .insert(key, stable_hasher.finish())
                        .is_none());

            (result, DepNodeIndex {
                legacy: dep_node_index_legacy,
                new: dep_node_index_new,
            })
        } else {
            if key.kind.fingerprint_needed_for_crate_hash() {
                let mut hcx = cx.create_stable_hashing_context();
                let result = task(cx, arg);
                let mut stable_hasher = StableHasher::new();
                result.hash_stable(&mut hcx, &mut stable_hasher);
                assert!(self.fingerprints
                            .borrow_mut()
                            .insert(key, stable_hasher.finish())
                            .is_none());
                (result, DepNodeIndex::INVALID)
            } else {
                (task(cx, arg), DepNodeIndex::INVALID)
            }
        }
    }

    /// Execute something within an "anonymous" task, that is, a task the
    /// DepNode of which is determined by the list of inputs it read from.
    pub fn with_anon_task<OP,R>(&self, dep_kind: DepKind, op: OP) -> (R, DepNodeIndex)
        where OP: FnOnce() -> R
    {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().push_anon_task();
            data.current.borrow_mut().push_anon_task();
            let result = op();
            let dep_node_index_legacy = data.edges.borrow_mut().pop_anon_task(dep_kind);
            let dep_node_index_new = data.current.borrow_mut().pop_anon_task(dep_kind);
            (result, DepNodeIndex {
                legacy: dep_node_index_legacy,
                new: dep_node_index_new,
            })
        } else {
            (op(), DepNodeIndex::INVALID)
        }
    }

    #[inline]
    pub fn read(&self, v: DepNode) {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().read(v);

            let mut current = data.current.borrow_mut();
            if let Some(&dep_node_index_new) = current.node_to_node_index.get(&v) {
                current.read_index(dep_node_index_new);
            } else {
                bug!("DepKind {:?} should be pre-allocated but isn't.", v.kind)
            }
        }
    }

    #[inline]
    pub fn read_index(&self, v: DepNodeIndex) {
        if let Some(ref data) = self.data {
            data.edges.borrow_mut().read_index(v.legacy);
            data.current.borrow_mut().read_index(v.new);
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

    pub fn fingerprint_of(&self, dep_node: &DepNode) -> Fingerprint {
        self.fingerprints.borrow()[dep_node]
    }

    pub fn prev_fingerprint_of(&self, dep_node: &DepNode) -> Fingerprint {
        self.data.as_ref().unwrap().previous.fingerprint_of(dep_node)
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

    pub fn serialize(&self) -> SerializedDepGraph {
        let fingerprints = self.fingerprints.borrow();
        let current_dep_graph = self.data.as_ref().unwrap().current.borrow();

        let nodes: IndexVec<_, _> = current_dep_graph.nodes.iter().map(|dep_node| {
            let fingerprint = fingerprints.get(dep_node)
                                          .cloned()
                                          .unwrap_or(Fingerprint::zero());
            (*dep_node, fingerprint)
        }).collect();

        let total_edge_count: usize = current_dep_graph.edges.iter()
                                                             .map(|v| v.len())
                                                             .sum();

        let mut edge_list_indices = IndexVec::with_capacity(nodes.len());
        let mut edge_list_data = Vec::with_capacity(total_edge_count);

        for (current_dep_node_index, edges) in current_dep_graph.edges.iter_enumerated() {
            let start = edge_list_data.len() as u32;
            // This should really just be a memcpy :/
            edge_list_data.extend(edges.iter().map(|i| SerializedDepNodeIndex(i.index)));
            let end = edge_list_data.len() as u32;

            debug_assert_eq!(current_dep_node_index.index(), edge_list_indices.len());
            edge_list_indices.push((start, end));
        }

        debug_assert!(edge_list_data.len() <= ::std::u32::MAX as usize);
        debug_assert_eq!(edge_list_data.len(), total_edge_count);

        SerializedDepGraph {
            nodes,
            edge_list_indices,
            edge_list_data,
        }
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

pub(super) struct CurrentDepGraph {
    nodes: IndexVec<DepNodeIndexNew, DepNode>,
    edges: IndexVec<DepNodeIndexNew, Vec<DepNodeIndexNew>>,
    node_to_node_index: FxHashMap<DepNode, DepNodeIndexNew>,

    task_stack: Vec<OpenTask>,
}

impl CurrentDepGraph {
    fn new() -> CurrentDepGraph {
        CurrentDepGraph {
            nodes: IndexVec::new(),
            edges: IndexVec::new(),
            node_to_node_index: FxHashMap(),
            task_stack: Vec::new(),
        }
    }

    pub(super) fn push_ignore(&mut self) {
        self.task_stack.push(OpenTask::Ignore);
    }

    pub(super) fn pop_ignore(&mut self) {
        let popped_node = self.task_stack.pop().unwrap();
        debug_assert_eq!(popped_node, OpenTask::Ignore);
    }

    pub(super) fn push_task(&mut self, key: DepNode) {
        self.task_stack.push(OpenTask::Regular {
            node: key,
            reads: Vec::new(),
            read_set: FxHashSet(),
        });
    }

    pub(super) fn pop_task(&mut self, key: DepNode) -> DepNodeIndexNew {
        let popped_node = self.task_stack.pop().unwrap();

        if let OpenTask::Regular {
            node,
            read_set: _,
            reads
        } = popped_node {
            debug_assert_eq!(node, key);
            self.alloc_node(node, reads)
        } else {
            bug!("pop_task() - Expected regular task to be popped")
        }
    }

    fn push_anon_task(&mut self) {
        self.task_stack.push(OpenTask::Anon {
            reads: Vec::new(),
            read_set: FxHashSet(),
        });
    }

    fn pop_anon_task(&mut self, kind: DepKind) -> DepNodeIndexNew {
        let popped_node = self.task_stack.pop().unwrap();

        if let OpenTask::Anon {
            read_set: _,
            reads
        } = popped_node {
            let mut fingerprint = Fingerprint::zero();
            let mut hasher = StableHasher::new();

            for &read in reads.iter() {
                let read_dep_node = self.nodes[read];

                ::std::mem::discriminant(&read_dep_node.kind).hash(&mut hasher);

                // Fingerprint::combine() is faster than sending Fingerprint
                // through the StableHasher (at least as long as StableHasher
                // is so slow).
                fingerprint = fingerprint.combine(read_dep_node.hash);
            }

            fingerprint = fingerprint.combine(hasher.finish());

            let target_dep_node = DepNode {
                kind,
                hash: fingerprint,
            };

            if let Some(&index) = self.node_to_node_index.get(&target_dep_node) {
                return index;
            }

            self.alloc_node(target_dep_node, reads)
        } else {
            bug!("pop_anon_task() - Expected anonymous task to be popped")
        }
    }

    fn read_index(&mut self, source: DepNodeIndexNew) {
        match self.task_stack.last_mut() {
            Some(&mut OpenTask::Regular {
                ref mut reads,
                ref mut read_set,
                node: _,
            }) => {
                if read_set.insert(source) {
                    reads.push(source);
                }
            }
            Some(&mut OpenTask::Anon {
                ref mut reads,
                ref mut read_set,
            }) => {
                if read_set.insert(source) {
                    reads.push(source);
                }
            }
            Some(&mut OpenTask::Ignore) | None => {
                // ignore
            }
        }
    }

    fn alloc_node(&mut self,
                  dep_node: DepNode,
                  edges: Vec<DepNodeIndexNew>)
                  -> DepNodeIndexNew {
        debug_assert_eq!(self.edges.len(), self.nodes.len());
        debug_assert_eq!(self.node_to_node_index.len(), self.nodes.len());
        debug_assert!(!self.node_to_node_index.contains_key(&dep_node));
        let dep_node_index = DepNodeIndexNew::new(self.nodes.len());
        self.nodes.push(dep_node);
        self.node_to_node_index.insert(dep_node, dep_node_index);
        self.edges.push(edges);
        dep_node_index
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct DepNodeIndexNew {
    index: u32,
}

impl Idx for DepNodeIndexNew {
    fn new(idx: usize) -> Self {
        DepNodeIndexNew::new(idx)
    }
    fn index(self) -> usize {
        self.index()
    }
}

impl DepNodeIndexNew {

    const INVALID: DepNodeIndexNew = DepNodeIndexNew {
        index: ::std::u32::MAX,
    };

    fn new(v: usize) -> DepNodeIndexNew {
        assert!((v & 0xFFFF_FFFF) == v);
        DepNodeIndexNew { index: v as u32 }
    }

    fn index(self) -> usize {
        self.index as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
enum OpenTask {
    Regular {
        node: DepNode,
        reads: Vec<DepNodeIndexNew>,
        read_set: FxHashSet<DepNodeIndexNew>,
    },
    Anon {
        reads: Vec<DepNodeIndexNew>,
        read_set: FxHashSet<DepNodeIndexNew>,
    },
    Ignore,
}
