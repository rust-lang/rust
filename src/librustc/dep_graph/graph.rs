use errors::DiagnosticBuilder;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use smallvec::SmallVec;
use rustc_data_structures::sync::{Lrc, Lock};
use std::env;
use std::hash::Hash;
use ty::{self, TyCtxt};
use util::common::{ProfileQueriesMsg, profq_msg};

use ich::{StableHashingContext, StableHashingContextProvider, Fingerprint};

use super::debug::EdgeFilter;
use super::dep_node::{DepNode, DepKind, WorkProductId};
use super::query::DepGraphQuery;
use super::safe::DepGraphSafe;
use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};
use super::prev::PreviousDepGraph;

#[derive(Clone)]
pub struct DepGraph {
    data: Option<Lrc<DepGraphData>>,

    // A vector mapping depnodes from the current graph to their associated
    // result value fingerprints. Do not rely on the length of this vector
    // being the same as the number of nodes in the graph. The vector can
    // contain an arbitrary number of zero-entries at the end.
    fingerprints: Lrc<Lock<IndexVec<DepNodeIndex, Fingerprint>>>
}

newtype_index! {
    pub struct DepNodeIndex { .. }
}

impl DepNodeIndex {
    const INVALID: DepNodeIndex = DepNodeIndex::MAX;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepNodeColor {
    Red,
    Green(DepNodeIndex)
}

impl DepNodeColor {
    pub fn is_green(self) -> bool {
        match self {
            DepNodeColor::Red => false,
            DepNodeColor::Green(_) => true,
        }
    }
}

struct DepGraphData {
    /// The new encoding of the dependency graph, optimized for red/green
    /// tracking. The `current` field is the dependency graph of only the
    /// current compilation session: We don't merge the previous dep-graph into
    /// current one anymore.
    current: Lock<CurrentDepGraph>,

    /// The dep-graph from the previous compilation session. It contains all
    /// nodes and edges as well as all fingerprints of nodes that have them.
    previous: PreviousDepGraph,

    colors: Lock<DepNodeColorMap>,

    /// When we load, there may be `.o` files, cached mir, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: FxHashMap<WorkProductId, WorkProduct>,

    dep_node_debug: Lock<FxHashMap<DepNode, String>>,

    // Used for testing, only populated when -Zquery-dep-graph is specified.
    loaded_from_cache: Lock<FxHashMap<DepNodeIndex, bool>>,
}

impl DepGraph {

    pub fn new(prev_graph: PreviousDepGraph,
               prev_work_products: FxHashMap<WorkProductId, WorkProduct>) -> DepGraph {
        // Pre-allocate the fingerprints array. We over-allocate a little so
        // that we hopefully don't have to re-allocate during this compilation
        // session.
        let prev_graph_node_count = prev_graph.node_count();

        let fingerprints = IndexVec::from_elem_n(Fingerprint::ZERO,
                                                 (prev_graph_node_count * 115) / 100);
        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: prev_work_products,
                dep_node_debug: Default::default(),
                current: Lock::new(CurrentDepGraph::new()),
                previous: prev_graph,
                colors: Lock::new(DepNodeColorMap::new(prev_graph_node_count)),
                loaded_from_cache: Default::default(),
            })),
            fingerprints: Lrc::new(Lock::new(fingerprints)),
        }
    }

    pub fn new_disabled() -> DepGraph {
        DepGraph {
            data: None,
            fingerprints: Lrc::new(Lock::new(IndexVec::new())),
        }
    }

    /// True if we are actually building the full dep-graph.
    #[inline]
    pub fn is_fully_enabled(&self) -> bool {
        self.data.is_some()
    }

    pub fn query(&self) -> DepGraphQuery {
        let current_dep_graph = self.data.as_ref().unwrap().current.borrow();
        let nodes: Vec<_> = current_dep_graph.nodes.iter().cloned().collect();
        let mut edges = Vec::new();
        for (index, edge_targets) in current_dep_graph.edges.iter_enumerated() {
            let from = current_dep_graph.nodes[index];
            for &edge_target in edge_targets.iter() {
                let to = current_dep_graph.nodes[edge_target];
                edges.push((from, to));
            }
        }

        DepGraphQuery::new(&nodes[..], &edges[..])
    }

    pub fn assert_ignored(&self)
    {
        if let Some(..) = self.data {
            ty::tls::with_context_opt(|icx| {
                let icx = if let Some(icx) = icx { icx } else { return };
                match *icx.task {
                    OpenTask::Ignore => {
                        // ignored
                    }
                    _ => panic!("expected an ignore context")
                }
            })
        }
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        ty::tls::with_context(|icx| {
            let icx = ty::tls::ImplicitCtxt {
                task: &OpenTask::Ignore,
                ..icx.clone()
            };

            ty::tls::enter_context(&icx, |_| {
                op()
            })
        })
    }

    /// Starts a new dep-graph task. Dep-graph tasks are specified
    /// using a free function (`task`) and **not** a closure -- this
    /// is intentional because we want to exercise tight control over
    /// what state they have access to. In particular, we want to
    /// prevent implicit 'leaks' of tracked state into the task (which
    /// could then be read without generating correct edges in the
    /// dep-graph -- see the [rustc guide] for more details on
    /// the dep-graph). To this end, the task function gets exactly two
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
    /// [rustc guide]: https://rust-lang.github.io/rustc-guide/incremental-compilation.html
    pub fn with_task<'gcx, C, A, R>(&self,
                                   key: DepNode,
                                   cx: C,
                                   arg: A,
                                   task: fn(C, A) -> R)
                                   -> (R, DepNodeIndex)
        where C: DepGraphSafe + StableHashingContextProvider<'gcx>,
              R: HashStable<StableHashingContext<'gcx>>,
    {
        self.with_task_impl(key, cx, arg, false, task,
            |key| OpenTask::Regular(Lock::new(RegularOpenTask {
                node: key,
                reads: SmallVec::new(),
                read_set: Default::default(),
            })),
            |data, key, task| data.borrow_mut().complete_task(key, task))
    }

    /// Creates a new dep-graph input with value `input`
    pub fn input_task<'gcx, C, R>(&self,
                                   key: DepNode,
                                   cx: C,
                                   input: R)
                                   -> (R, DepNodeIndex)
        where C: DepGraphSafe + StableHashingContextProvider<'gcx>,
              R: HashStable<StableHashingContext<'gcx>>,
    {
        fn identity_fn<C, A>(_: C, arg: A) -> A {
            arg
        }

        self.with_task_impl(key, cx, input, true, identity_fn,
            |_| OpenTask::Ignore,
            |data, key, _| data.borrow_mut().alloc_node(key, SmallVec::new()))
    }

    fn with_task_impl<'gcx, C, A, R>(
        &self,
        key: DepNode,
        cx: C,
        arg: A,
        no_tcx: bool,
        task: fn(C, A) -> R,
        create_task: fn(DepNode) -> OpenTask,
        finish_task_and_alloc_depnode: fn(&Lock<CurrentDepGraph>,
                                          DepNode,
                                          OpenTask) -> DepNodeIndex
    ) -> (R, DepNodeIndex)
    where
        C: DepGraphSafe + StableHashingContextProvider<'gcx>,
        R: HashStable<StableHashingContext<'gcx>>,
    {
        if let Some(ref data) = self.data {
            let open_task = create_task(key);

            // In incremental mode, hash the result of the task. We don't
            // do anything with the hash yet, but we are computing it
            // anyway so that
            //  - we make sure that the infrastructure works and
            //  - we can get an idea of the runtime cost.
            let mut hcx = cx.get_stable_hashing_context();

            if cfg!(debug_assertions) {
                profq_msg(hcx.sess(), ProfileQueriesMsg::TaskBegin(key.clone()))
            };

            let result = if no_tcx {
                task(cx, arg)
            } else {
                ty::tls::with_context(|icx| {
                    let icx = ty::tls::ImplicitCtxt {
                        task: &open_task,
                        ..icx.clone()
                    };

                    ty::tls::enter_context(&icx, |_| {
                        task(cx, arg)
                    })
                })
            };

            if cfg!(debug_assertions) {
                profq_msg(hcx.sess(), ProfileQueriesMsg::TaskEnd)
            };

            let dep_node_index = finish_task_and_alloc_depnode(&data.current, key, open_task);

            let mut stable_hasher = StableHasher::new();
            result.hash_stable(&mut hcx, &mut stable_hasher);

            let current_fingerprint = stable_hasher.finish();

            // Store the current fingerprint
            {
                let mut fingerprints = self.fingerprints.borrow_mut();

                if dep_node_index.index() >= fingerprints.len() {
                    fingerprints.resize(dep_node_index.index() + 1, Fingerprint::ZERO);
                }

                debug_assert!(fingerprints[dep_node_index] == Fingerprint::ZERO,
                              "DepGraph::with_task() - Duplicate fingerprint \
                               insertion for {:?}", key);
                fingerprints[dep_node_index] = current_fingerprint;
            }

            // Determine the color of the new DepNode.
            if let Some(prev_index) = data.previous.node_to_index_opt(&key) {
                let prev_fingerprint = data.previous.fingerprint_by_index(prev_index);

                let color = if current_fingerprint == prev_fingerprint {
                    DepNodeColor::Green(dep_node_index)
                } else {
                    DepNodeColor::Red
                };

                let mut colors = data.colors.borrow_mut();
                debug_assert!(colors.get(prev_index).is_none(),
                              "DepGraph::with_task() - Duplicate DepNodeColor \
                               insertion for {:?}", key);

                colors.insert(prev_index, color);
            }

            (result, dep_node_index)
        } else {
            if key.kind.fingerprint_needed_for_crate_hash() {
                let mut hcx = cx.get_stable_hashing_context();
                let result = task(cx, arg);
                let mut stable_hasher = StableHasher::new();
                result.hash_stable(&mut hcx, &mut stable_hasher);
                let fingerprint = stable_hasher.finish();

                let mut fingerprints = self.fingerprints.borrow_mut();
                let dep_node_index = DepNodeIndex::new(fingerprints.len());
                fingerprints.push(fingerprint);

                debug_assert!(fingerprints[dep_node_index] == fingerprint,
                              "DepGraph::with_task() - Assigned fingerprint to \
                               unexpected index for {:?}", key);

                (result, dep_node_index)
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
            let (result, open_task) = ty::tls::with_context(|icx| {
                let task = OpenTask::Anon(Lock::new(AnonOpenTask {
                    reads: SmallVec::new(),
                    read_set: Default::default(),
                }));

                let r = {
                    let icx = ty::tls::ImplicitCtxt {
                        task: &task,
                        ..icx.clone()
                    };

                    ty::tls::enter_context(&icx, |_| {
                        op()
                    })
                };

                (r, task)
            });
            let dep_node_index = data.current
                                     .borrow_mut()
                                     .pop_anon_task(dep_kind, open_task);
            (result, dep_node_index)
        } else {
            (op(), DepNodeIndex::INVALID)
        }
    }

    /// Execute something within an "eval-always" task which is a task
    // that runs whenever anything changes.
    pub fn with_eval_always_task<'gcx, C, A, R>(&self,
                                   key: DepNode,
                                   cx: C,
                                   arg: A,
                                   task: fn(C, A) -> R)
                                   -> (R, DepNodeIndex)
        where C: DepGraphSafe + StableHashingContextProvider<'gcx>,
              R: HashStable<StableHashingContext<'gcx>>,
    {
        self.with_task_impl(key, cx, arg, false, task,
            |key| OpenTask::EvalAlways { node: key },
            |data, key, task| data.borrow_mut().complete_eval_always_task(key, task))
    }

    #[inline]
    pub fn read(&self, v: DepNode) {
        if let Some(ref data) = self.data {
            let mut current = data.current.borrow_mut();
            if let Some(&dep_node_index) = current.node_to_node_index.get(&v) {
                current.read_index(dep_node_index);
            } else {
                bug!("DepKind {:?} should be pre-allocated but isn't.", v.kind)
            }
        }
    }

    #[inline]
    pub fn read_index(&self, dep_node_index: DepNodeIndex) {
        if let Some(ref data) = self.data {
            data.current.borrow_mut().read_index(dep_node_index);
        }
    }

    #[inline]
    pub fn dep_node_index_of(&self, dep_node: &DepNode) -> DepNodeIndex {
        self.data
            .as_ref()
            .unwrap()
            .current
            .borrow_mut()
            .node_to_node_index
            .get(dep_node)
            .cloned()
            .unwrap()
    }

    #[inline]
    pub fn dep_node_exists(&self, dep_node: &DepNode) -> bool {
        if let Some(ref data) = self.data {
            data.current.borrow_mut().node_to_node_index.contains_key(dep_node)
        } else {
            false
        }
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        match self.fingerprints.borrow().get(dep_node_index) {
            Some(&fingerprint) => fingerprint,
            None => {
                if let Some(ref data) = self.data {
                    let dep_node = data.current.borrow().nodes[dep_node_index];
                    bug!("Could not find current fingerprint for {:?}", dep_node)
                } else {
                    bug!("Could not find current fingerprint for {:?}", dep_node_index)
                }
            }
        }
    }

    pub fn prev_fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.data.as_ref().unwrap().previous.fingerprint_of(dep_node)
    }

    #[inline]
    pub fn prev_dep_node_index_of(&self, dep_node: &DepNode) -> SerializedDepNodeIndex {
        self.data.as_ref().unwrap().previous.node_to_index(dep_node)
    }

    /// Check whether a previous work product exists for `v` and, if
    /// so, return the path that leads to it. Used to skip doing work.
    pub fn previous_work_product(&self, v: &WorkProductId) -> Option<WorkProduct> {
        self.data
            .as_ref()
            .and_then(|data| {
                data.previous_work_products.get(v).cloned()
            })
    }

    /// Access the map of work-products created during the cached run. Only
    /// used during saving of the dep-graph.
    pub fn previous_work_products(&self) -> &FxHashMap<WorkProductId, WorkProduct> {
        &self.data.as_ref().unwrap().previous_work_products
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
        self.data
            .as_ref()?
            .dep_node_debug
            .borrow()
            .get(&dep_node)
            .cloned()
    }

    pub fn edge_deduplication_data(&self) -> (u64, u64) {
        let current_dep_graph = self.data.as_ref().unwrap().current.borrow();

        (current_dep_graph.total_read_count, current_dep_graph.total_duplicate_read_count)
    }

    pub fn serialize(&self) -> SerializedDepGraph {
        let current_dep_graph = self.data.as_ref().unwrap().current.borrow();

        let fingerprints = self.fingerprints.borrow().clone().convert_index_type();
        let nodes = current_dep_graph.nodes.clone().convert_index_type();

        let total_edge_count: usize = current_dep_graph.edges.iter()
                                                             .map(|v| v.len())
                                                             .sum();

        let mut edge_list_indices = IndexVec::with_capacity(nodes.len());
        let mut edge_list_data = Vec::with_capacity(total_edge_count);

        for (current_dep_node_index, edges) in current_dep_graph.edges.iter_enumerated() {
            let start = edge_list_data.len() as u32;
            // This should really just be a memcpy :/
            edge_list_data.extend(edges.iter().map(|i| SerializedDepNodeIndex::new(i.index())));
            let end = edge_list_data.len() as u32;

            debug_assert_eq!(current_dep_node_index.index(), edge_list_indices.len());
            edge_list_indices.push((start, end));
        }

        debug_assert!(edge_list_data.len() <= ::std::u32::MAX as usize);
        debug_assert_eq!(edge_list_data.len(), total_edge_count);

        SerializedDepGraph {
            nodes,
            fingerprints,
            edge_list_indices,
            edge_list_data,
        }
    }

    pub fn node_color(&self, dep_node: &DepNode) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            if let Some(prev_index) = data.previous.node_to_index_opt(dep_node) {
                return data.colors.borrow().get(prev_index)
            } else {
                // This is a node that did not exist in the previous compilation
                // session, so we consider it to be red.
                return Some(DepNodeColor::Red)
            }
        }

        None
    }

    pub fn try_mark_green<'tcx>(&self,
                                tcx: TyCtxt<'_, 'tcx, 'tcx>,
                                dep_node: &DepNode)
                                -> Option<DepNodeIndex> {
        debug!("try_mark_green({:?}) - BEGIN", dep_node);
        let data = self.data.as_ref().unwrap();

        #[cfg(not(parallel_queries))]
        debug_assert!(!data.current.borrow().node_to_node_index.contains_key(dep_node));

        if dep_node.kind.is_input() {
            // We should only hit try_mark_green() for inputs that do not exist
            // anymore in the current compilation session. Existing inputs are
            // eagerly marked as either red/green before any queries are
            // executed.
            debug_assert!(dep_node.extract_def_id(tcx).is_none());
            debug!("try_mark_green({:?}) - END - DepNode is deleted input", dep_node);
            return None;
        }

        let (prev_deps, prev_dep_node_index) = match data.previous.edges_from(dep_node) {
            Some(prev) => {
                // This DepNode and the corresponding query invocation existed
                // in the previous compilation session too, so we can try to
                // mark it as green by recursively marking all of its
                // dependencies green.
                prev
            }
            None => {
                // This DepNode did not exist in the previous compilation session,
                // so we cannot mark it as green.
                debug!("try_mark_green({:?}) - END - DepNode does not exist in \
                        current compilation session anymore", dep_node);
                return None
            }
        };

        debug_assert!(data.colors.borrow().get(prev_dep_node_index).is_none());

        let mut current_deps = SmallVec::new();

        for &dep_dep_node_index in prev_deps {
            let dep_dep_node_color = data.colors.borrow().get(dep_dep_node_index);

            match dep_dep_node_color {
                Some(DepNodeColor::Green(node_index)) => {
                    // This dependency has been marked as green before, we are
                    // still fine and can continue with checking the other
                    // dependencies.
                    debug!("try_mark_green({:?}) --- found dependency {:?} to \
                            be immediately green",
                            dep_node,
                            data.previous.index_to_node(dep_dep_node_index));
                    current_deps.push(node_index);
                }
                Some(DepNodeColor::Red) => {
                    // We found a dependency the value of which has changed
                    // compared to the previous compilation session. We cannot
                    // mark the DepNode as green and also don't need to bother
                    // with checking any of the other dependencies.
                    debug!("try_mark_green({:?}) - END - dependency {:?} was \
                            immediately red",
                            dep_node,
                            data.previous.index_to_node(dep_dep_node_index));
                    return None
                }
                None => {
                    let dep_dep_node = &data.previous.index_to_node(dep_dep_node_index);

                    // We don't know the state of this dependency. If it isn't
                    // an input node, let's try to mark it green recursively.
                    if !dep_dep_node.kind.is_input() {
                         debug!("try_mark_green({:?}) --- state of dependency {:?} \
                                 is unknown, trying to mark it green", dep_node,
                                 dep_dep_node);

                        if let Some(node_index) = self.try_mark_green(tcx, dep_dep_node) {
                            debug!("try_mark_green({:?}) --- managed to MARK \
                                    dependency {:?} as green", dep_node, dep_dep_node);
                            current_deps.push(node_index);
                            continue;
                        }
                    } else {
                        match dep_dep_node.kind {
                            DepKind::Hir |
                            DepKind::HirBody |
                            DepKind::CrateMetadata => {
                                if dep_node.extract_def_id(tcx).is_none() {
                                    // If the node does not exist anymore, we
                                    // just fail to mark green.
                                    return None
                                } else {
                                    // If the node does exist, it should have
                                    // been pre-allocated.
                                    bug!("DepNode {:?} should have been \
                                          pre-allocated but wasn't.",
                                          dep_dep_node)
                                }
                            }
                            _ => {
                                // For other kinds of inputs it's OK to be
                                // forced.
                            }
                        }
                    }

                    // We failed to mark it green, so we try to force the query.
                    debug!("try_mark_green({:?}) --- trying to force \
                            dependency {:?}", dep_node, dep_dep_node);
                    if ::ty::query::force_from_dep_node(tcx, dep_dep_node) {
                        let dep_dep_node_color = data.colors.borrow().get(dep_dep_node_index);

                        match dep_dep_node_color {
                            Some(DepNodeColor::Green(node_index)) => {
                                debug!("try_mark_green({:?}) --- managed to \
                                        FORCE dependency {:?} to green",
                                        dep_node, dep_dep_node);
                                current_deps.push(node_index);
                            }
                            Some(DepNodeColor::Red) => {
                                debug!("try_mark_green({:?}) - END - \
                                        dependency {:?} was red after forcing",
                                       dep_node,
                                       dep_dep_node);
                                return None
                            }
                            None => {
                                if !tcx.sess.has_errors() {
                                    bug!("try_mark_green() - Forcing the DepNode \
                                          should have set its color")
                                } else {
                                    // If the query we just forced has resulted
                                    // in some kind of compilation error, we
                                    // don't expect that the corresponding
                                    // dep-node color has been updated.
                                }
                            }
                        }
                    } else {
                        // The DepNode could not be forced.
                        debug!("try_mark_green({:?}) - END - dependency {:?} \
                                could not be forced", dep_node, dep_dep_node);
                        return None
                    }
                }
            }
        }

        // If we got here without hitting a `return` that means that all
        // dependencies of this DepNode could be marked as green. Therefore we
        // can also mark this DepNode as green.

        // There may be multiple threads trying to mark the same dep node green concurrently

        let (dep_node_index, did_allocation) = {
            let mut current = data.current.borrow_mut();

            if let Some(&dep_node_index) = current.node_to_node_index.get(&dep_node) {
                // Someone else allocated it before us
                (dep_node_index, false)
            } else {
                // We allocating an entry for the node in the current dependency graph and
                // adding all the appropriate edges imported from the previous graph
                (current.alloc_node(*dep_node, current_deps), true)
            }
        };

        // ... copying the fingerprint from the previous graph too, so we don't
        // have to recompute it ...
        {
            let fingerprint = data.previous.fingerprint_by_index(prev_dep_node_index);
            let mut fingerprints = self.fingerprints.borrow_mut();

            if dep_node_index.index() >= fingerprints.len() {
                fingerprints.resize(dep_node_index.index() + 1, Fingerprint::ZERO);
            }

            // Multiple threads can all write the same fingerprint here
            #[cfg(not(parallel_queries))]
            debug_assert!(fingerprints[dep_node_index] == Fingerprint::ZERO,
                "DepGraph::try_mark_green() - Duplicate fingerprint \
                insertion for {:?}", dep_node);

            fingerprints[dep_node_index] = fingerprint;
        }

        // ... emitting any stored diagnostic ...
        if did_allocation {
            // Only the thread which did the allocation emits the error messages

            // FIXME: Ensure that these are printed before returning for all threads.
            // Currently threads where did_allocation = false can continue on
            // and emit other diagnostics before these diagnostics are emitted.
            // Such diagnostics should be emitted after these.
            // See https://github.com/rust-lang/rust/issues/48685
            let diagnostics = tcx.queries.on_disk_cache
                                 .load_diagnostics(tcx, prev_dep_node_index);

            if diagnostics.len() > 0 {
                let handle = tcx.sess.diagnostic();

                // Promote the previous diagnostics to the current session.
                tcx.queries.on_disk_cache
                   .store_diagnostics(dep_node_index, diagnostics.clone());

                for diagnostic in diagnostics {
                    DiagnosticBuilder::new_diagnostic(handle, diagnostic).emit();
                }
            }
        }

        // ... and finally storing a "Green" entry in the color map.
        let mut colors = data.colors.borrow_mut();
        // Multiple threads can all write the same color here
        #[cfg(not(parallel_queries))]
        debug_assert!(colors.get(prev_dep_node_index).is_none(),
                      "DepGraph::try_mark_green() - Duplicate DepNodeColor \
                      insertion for {:?}", dep_node);

        colors.insert(prev_dep_node_index, DepNodeColor::Green(dep_node_index));

        debug!("try_mark_green({:?}) - END - successfully marked as green", dep_node);
        Some(dep_node_index)
    }

    // Returns true if the given node has been marked as green during the
    // current compilation session. Used in various assertions
    pub fn is_green(&self, dep_node: &DepNode) -> bool {
        self.node_color(dep_node).map(|c| c.is_green()).unwrap_or(false)
    }

    // This method loads all on-disk cacheable query results into memory, so
    // they can be written out to the new cache file again. Most query results
    // will already be in memory but in the case where we marked something as
    // green but then did not need the value, that value will never have been
    // loaded from disk.
    //
    // This method will only load queries that will end up in the disk cache.
    // Other queries will not be executed.
    pub fn exec_cache_promotions<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) {
        let green_nodes: Vec<DepNode> = {
            let data = self.data.as_ref().unwrap();
            let colors = data.colors.borrow();
            colors.values.indices().filter_map(|prev_index| {
                match colors.get(prev_index) {
                    Some(DepNodeColor::Green(_)) => {
                        let dep_node = data.previous.index_to_node(prev_index);
                        if dep_node.cache_on_disk(tcx) {
                            Some(dep_node)
                        } else {
                            None
                        }
                    }
                    None |
                    Some(DepNodeColor::Red) => {
                        // We can skip red nodes because a node can only be marked
                        // as red if the query result was recomputed and thus is
                        // already in memory.
                        None
                    }
                }
            }).collect()
        };

        for dep_node in green_nodes {
            dep_node.load_from_on_disk_cache(tcx);
        }
    }

    pub fn mark_loaded_from_cache(&self, dep_node_index: DepNodeIndex, state: bool) {
        debug!("mark_loaded_from_cache({:?}, {})",
               self.data.as_ref().unwrap().current.borrow().nodes[dep_node_index],
               state);

        self.data
            .as_ref()
            .unwrap()
            .loaded_from_cache
            .borrow_mut()
            .insert(dep_node_index, state);
    }

    pub fn was_loaded_from_cache(&self, dep_node: &DepNode) -> Option<bool> {
        let data = self.data.as_ref().unwrap();
        let dep_node_index = data.current.borrow().node_to_node_index[dep_node];
        data.loaded_from_cache.borrow().get(&dep_node_index).cloned()
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
/// with `DepNode::CodegenUnit(P)`; the hash is the set of symbols
/// in P.
///
/// The next time we compile, if the `DepNode::CodegenUnit(P)` is
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
    /// Saved files associated with this CGU
    pub saved_files: Vec<(WorkProductFileKind, String)>,
}

#[derive(Clone, Copy, Debug, RustcEncodable, RustcDecodable, PartialEq)]
pub enum WorkProductFileKind {
    Object,
    Bytecode,
    BytecodeCompressed,
}

pub(super) struct CurrentDepGraph {
    nodes: IndexVec<DepNodeIndex, DepNode>,
    edges: IndexVec<DepNodeIndex, SmallVec<[DepNodeIndex; 8]>>,
    node_to_node_index: FxHashMap<DepNode, DepNodeIndex>,
    forbidden_edge: Option<EdgeFilter>,

    // Anonymous DepNodes are nodes the ID of which we compute from the list of
    // their edges. This has the beneficial side-effect that multiple anonymous
    // nodes can be coalesced into one without changing the semantics of the
    // dependency graph. However, the merging of nodes can lead to a subtle
    // problem during red-green marking: The color of an anonymous node from
    // the current session might "shadow" the color of the node with the same
    // ID from the previous session. In order to side-step this problem, we make
    // sure that anon-node IDs allocated in different sessions don't overlap.
    // This is implemented by mixing a session-key into the ID fingerprint of
    // each anon node. The session-key is just a random number generated when
    // the DepGraph is created.
    anon_id_seed: Fingerprint,

    total_read_count: u64,
    total_duplicate_read_count: u64,
}

impl CurrentDepGraph {
    fn new() -> CurrentDepGraph {
        use std::time::{SystemTime, UNIX_EPOCH};

        let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let nanos = duration.as_secs() * 1_000_000_000 +
                    duration.subsec_nanos() as u64;
        let mut stable_hasher = StableHasher::new();
        nanos.hash(&mut stable_hasher);

        let forbidden_edge = if cfg!(debug_assertions) {
            match env::var("RUST_FORBID_DEP_GRAPH_EDGE") {
                Ok(s) => {
                    match EdgeFilter::new(&s) {
                        Ok(f) => Some(f),
                        Err(err) => bug!("RUST_FORBID_DEP_GRAPH_EDGE invalid: {}", err),
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        CurrentDepGraph {
            nodes: IndexVec::new(),
            edges: IndexVec::new(),
            node_to_node_index: Default::default(),
            anon_id_seed: stable_hasher.finish(),
            forbidden_edge,
            total_read_count: 0,
            total_duplicate_read_count: 0,
        }
    }

    fn complete_task(&mut self, key: DepNode, task: OpenTask) -> DepNodeIndex {
        if let OpenTask::Regular(task) = task {
            let RegularOpenTask {
                node,
                read_set: _,
                reads
            } = task.into_inner();
            assert_eq!(node, key);

            // If this is an input node, we expect that it either has no
            // dependencies, or that it just depends on DepKind::CrateMetadata
            // or DepKind::Krate. This happens for some "thin wrapper queries"
            // like `crate_disambiguator` which sometimes have zero deps (for
            // when called for LOCAL_CRATE) or they depend on a CrateMetadata
            // node.
            if cfg!(debug_assertions) {
                if node.kind.is_input() && reads.len() > 0 &&
                   // FIXME(mw): Special case for DefSpan until Spans are handled
                   //            better in general.
                   node.kind != DepKind::DefSpan &&
                    reads.iter().any(|&i| {
                        !(self.nodes[i].kind == DepKind::CrateMetadata ||
                          self.nodes[i].kind == DepKind::Krate)
                    })
                {
                    bug!("Input node {:?} with unexpected reads: {:?}",
                        node,
                        reads.iter().map(|&i| self.nodes[i]).collect::<Vec<_>>())
                }
            }

            self.alloc_node(node, reads)
        } else {
            bug!("complete_task() - Expected regular task to be popped")
        }
    }

    fn pop_anon_task(&mut self, kind: DepKind, task: OpenTask) -> DepNodeIndex {
        if let OpenTask::Anon(task) = task {
            let AnonOpenTask {
                read_set: _,
                reads
            } = task.into_inner();
            debug_assert!(!kind.is_input());

            let mut fingerprint = self.anon_id_seed;
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
                index
            } else {
                self.alloc_node(target_dep_node, reads)
            }
        } else {
            bug!("pop_anon_task() - Expected anonymous task to be popped")
        }
    }

    fn complete_eval_always_task(&mut self, key: DepNode, task: OpenTask) -> DepNodeIndex {
        if let OpenTask::EvalAlways {
            node,
        } = task {
            debug_assert_eq!(node, key);
            let krate_idx = self.node_to_node_index[&DepNode::new_no_params(DepKind::Krate)];
            self.alloc_node(node, smallvec![krate_idx])
        } else {
            bug!("complete_eval_always_task() - Expected eval always task to be popped");
        }
    }

    fn read_index(&mut self, source: DepNodeIndex) {
        ty::tls::with_context_opt(|icx| {
            let icx = if let Some(icx) = icx { icx } else { return };
            match *icx.task {
                OpenTask::Regular(ref task) => {
                    let mut task = task.lock();
                    self.total_read_count += 1;
                    if task.read_set.insert(source) {
                        task.reads.push(source);

                        if cfg!(debug_assertions) {
                            if let Some(ref forbidden_edge) = self.forbidden_edge {
                                let target = &task.node;
                                let source = self.nodes[source];
                                if forbidden_edge.test(&source, &target) {
                                    bug!("forbidden edge {:?} -> {:?} created",
                                        source,
                                        target)
                                }
                            }
                        }
                    } else {
                        self.total_duplicate_read_count += 1;
                    }
                }
                OpenTask::Anon(ref task) => {
                    let mut task = task.lock();
                    if task.read_set.insert(source) {
                        task.reads.push(source);
                    }
                }
                OpenTask::Ignore | OpenTask::EvalAlways { .. } => {
                    // ignore
                }
            }
        })
    }

    fn alloc_node(&mut self,
                  dep_node: DepNode,
                  edges: SmallVec<[DepNodeIndex; 8]>)
                  -> DepNodeIndex {
        debug_assert_eq!(self.edges.len(), self.nodes.len());
        debug_assert_eq!(self.node_to_node_index.len(), self.nodes.len());
        debug_assert!(!self.node_to_node_index.contains_key(&dep_node));
        let dep_node_index = DepNodeIndex::new(self.nodes.len());
        self.nodes.push(dep_node);
        self.node_to_node_index.insert(dep_node, dep_node_index);
        self.edges.push(edges);
        dep_node_index
    }
}

pub struct RegularOpenTask {
    node: DepNode,
    reads: SmallVec<[DepNodeIndex; 8]>,
    read_set: FxHashSet<DepNodeIndex>,
}

pub struct AnonOpenTask {
    reads: SmallVec<[DepNodeIndex; 8]>,
    read_set: FxHashSet<DepNodeIndex>,
}

pub enum OpenTask {
    Regular(Lock<RegularOpenTask>),
    Anon(Lock<AnonOpenTask>),
    Ignore,
    EvalAlways {
        node: DepNode,
    },
}

// A data structure that stores Option<DepNodeColor> values as a contiguous
// array, using one u32 per entry.
struct DepNodeColorMap {
    values: IndexVec<SerializedDepNodeIndex, u32>,
}

const COMPRESSED_NONE: u32 = 0;
const COMPRESSED_RED: u32 = 1;
const COMPRESSED_FIRST_GREEN: u32 = 2;

impl DepNodeColorMap {
    fn new(size: usize) -> DepNodeColorMap {
        DepNodeColorMap {
            values: IndexVec::from_elem_n(COMPRESSED_NONE, size)
        }
    }

    fn get(&self, index: SerializedDepNodeIndex) -> Option<DepNodeColor> {
        match self.values[index] {
            COMPRESSED_NONE => None,
            COMPRESSED_RED => Some(DepNodeColor::Red),
            value => Some(DepNodeColor::Green(DepNodeIndex::from_u32(
                value - COMPRESSED_FIRST_GREEN
            )))
        }
    }

    fn insert(&mut self, index: SerializedDepNodeIndex, color: DepNodeColor) {
        self.values[index] = match color {
            DepNodeColor::Red => COMPRESSED_RED,
            DepNodeColor::Green(index) => index.as_u32() + COMPRESSED_FIRST_GREEN,
        }
    }
}
