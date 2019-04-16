use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::profiling::QueryInvocationId;
use rustc_data_structures::sharded::{self, Sharded};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{AtomicCell, AtomicU32, AtomicU64, Lock, Lrc};
use rustc_data_structures::unlikely;
use rustc_errors::Diagnostic;
use rustc_index::vec::{Idx, IndexVec};

use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::env;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::sync::atomic::Ordering::Relaxed;

use super::debug::EdgeFilter;
use super::prev::PreviousDepGraph;
use super::query::DepGraphQuery;
use super::serialized::SerializedDepGraph;
use super::{DepContext, DepKind, DepNode, WorkProductId};

#[derive(Clone)]
pub struct DepGraph<K: DepKind> {
    data: Option<Lrc<DepGraphData<K>>>,

    /// This field is used for assigning DepNodeIndices when running in
    /// non-incremental mode. Even in non-incremental mode we make sure that
    /// each task has a `DepNodeIndex` that uniquely identifies it. This unique
    /// ID is used for self-profiling.
    virtual_dep_node_index: Lrc<AtomicU32>,
}

rustc_index::newtype_index! {
    pub struct DepNodeIndex { .. }
}

impl DepNodeIndex {
    pub const INVALID: DepNodeIndex = DepNodeIndex::MAX;
}

impl std::convert::From<DepNodeIndex> for QueryInvocationId {
    #[inline]
    fn from(dep_node_index: DepNodeIndex) -> Self {
        QueryInvocationId(dep_node_index.as_u32())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepNodeColor {
    Red,
    Green,
}

impl DepNodeColor {
    pub fn is_green(self) -> bool {
        match self {
            DepNodeColor::Red => false,
            DepNodeColor::Green => true,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DepNodeState {
    /// The dep node index is invalid and does not refer to any dep node.
    Invalid,

    /// The node is from the previous session, but its state is unknown
    Unknown,

    Red,

    Green,
}

impl DepNodeState {
    pub fn color(self) -> Option<DepNodeColor> {
        match self {
            DepNodeState::Invalid => panic!(),
            DepNodeState::Unknown => None,
            DepNodeState::Red => Some(DepNodeColor::Red),
            DepNodeState::Green => Some(DepNodeColor::Green),
        }
    }
}

struct DepGraphData<K: DepKind> {
    /// The new encoding of the dependency graph, optimized for red/green
    /// tracking. The `current` field is the dependency graph of only the
    /// current compilation session: We don't merge the previous dep-graph into
    /// current one anymore.
    current: CurrentDepGraph<K>,

    /// The dep-graph from the previous compilation session. It contains all
    /// nodes and edges as well as all fingerprints of nodes that have them.
    previous: PreviousDepGraph<K>,

    colors: DepNodeColorMap,

    /// A set of loaded diagnostics that is in the progress of being emitted.
    emitting_diagnostics: Mutex<FxHashSet<DepNodeIndex>>,

    /// Used to wait for diagnostics to be emitted.
    emitting_diagnostics_cond_var: Condvar,

    /// When we load, there may be `.o` files, cached MIR, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: FxHashMap<WorkProductId, WorkProduct>,

    dep_node_debug: Lock<FxHashMap<DepNode<K>, String>>,
}

pub fn hash_result<HashCtxt, R>(hcx: &mut HashCtxt, result: &R) -> Option<Fingerprint>
where
    R: HashStable<HashCtxt>,
{
    let mut stable_hasher = StableHasher::new();
    result.hash_stable(hcx, &mut stable_hasher);

    Some(stable_hasher.finish())
}

pub struct DepGraphArgs<K: DepKind> {
    pub prev_graph: PreviousDepGraph<K>,
    pub prev_work_products: FxHashMap<WorkProductId, WorkProduct>,
    pub state: IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>,
    pub current: CurrentDepGraph<K>,
}

impl<K: DepKind> Default for DepGraphArgs<K> {
    fn default() -> Self {
        let prev_graph = Default::default();
        Self {
            current: CurrentDepGraph::new(&prev_graph),
            prev_work_products: Default::default(),
            state: Default::default(),
            prev_graph,
        }
    }
}

impl<K: DepKind> DepGraph<K> {
    pub fn new(args: DepGraphArgs<K>) -> Self {
        let colors = DepNodeColorMap { values: args.state };
        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: args.prev_work_products,
                dep_node_debug: Default::default(),
                current: args.current,
                emitting_diagnostics: Default::default(),
                emitting_diagnostics_cond_var: Condvar::new(),
                colors,
                previous: args.prev_graph,
            })),
            virtual_dep_node_index: Lrc::new(AtomicU32::new(0)),
        }
    }

    pub fn new_disabled() -> DepGraph<K> {
        DepGraph { data: None, virtual_dep_node_index: Lrc::new(AtomicU32::new(0)) }
    }

    /// Returns `true` if we are actually building the full dep-graph, and `false` otherwise.
    #[inline]
    pub fn is_fully_enabled(&self) -> bool {
        self.data.is_some()
    }

    pub fn query(&self) -> DepGraphQuery<K> {
        let data = self.data.as_ref().unwrap();
        let node_data = data.current.data.lock();

        // Recreate the dep graph, but ignore nodes from the previous dep graph
        let current_nodes =
            || node_data.iter_enumerated().filter(|&(i, _)| data.current(i)).map(|(_, n)| n);
        let nodes: Vec<_> = current_nodes().map(|n| n.node).collect();
        let mut edges = Vec::new();
        for (from, edge_targets) in current_nodes().map(|d| (d.node, &d.edges)) {
            for &edge_target in edge_targets.iter() {
                assert!(data.current(edge_target));
                let to = node_data[edge_target].node;
                edges.push((from, to));
            }
        }

        DepGraphQuery::new(&nodes[..], &edges[..])
    }

    pub fn assert_ignored(&self) {
        if let Some(..) = self.data {
            K::read_deps(|task_deps| {
                assert!(task_deps.is_none(), "expected no task dependency tracking");
            })
        }
    }

    pub fn with_ignore<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R,
    {
        K::with_deps(None, op)
    }

    /// Starts a new dep-graph task. Dep-graph tasks are specified
    /// using a free function (`task`) and **not** a closure -- this
    /// is intentional because we want to exercise tight control over
    /// what state they have access to. In particular, we want to
    /// prevent implicit 'leaks' of tracked state into the task (which
    /// could then be read without generating correct edges in the
    /// dep-graph -- see the [rustc dev guide] for more details on
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
    /// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/incremental-compilation.html
    pub fn with_task<Ctxt: DepContext<DepKind = K>, A, R>(
        &self,
        key: DepNode<K>,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        hash_result: impl FnOnce(&mut Ctxt::StableHashingContext, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex) {
        self.with_task_impl(
            key,
            cx,
            arg,
            false,
            task,
            |_key| {
                Some(TaskDeps {
                    #[cfg(debug_assertions)]
                    node: Some(_key),
                    reads: SmallVec::new(),
                    read_set: Default::default(),
                    phantom_data: PhantomData,
                })
            },
            hash_result,
        )
    }

    fn with_task_impl<Ctxt: DepContext<DepKind = K>, A, R>(
        &self,
        key: DepNode<K>,
        cx: Ctxt,
        arg: A,
        no_tcx: bool,
        task: fn(Ctxt, A) -> R,
        create_task: fn(DepNode<K>) -> Option<TaskDeps<K>>,
        hash_result: impl FnOnce(&mut Ctxt::StableHashingContext, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex) {
        if let Some(ref data) = self.data {
            let task_deps = create_task(key).map(Lock::new);

            // In incremental mode, hash the result of the task. We don't
            // do anything with the hash yet, but we are computing it
            // anyway so that
            //  - we make sure that the infrastructure works and
            //  - we can get an idea of the runtime cost.
            let mut hcx = cx.create_stable_hashing_context();

            let result = if no_tcx {
                task(cx, arg)
            } else {
                K::with_deps(task_deps.as_ref(), || task(cx, arg))
            };

            let current_fingerprint = hash_result(&mut hcx, &result);

            let deps = task_deps.map(|lock| lock.into_inner().reads).unwrap_or(SmallVec::new());
            let fingerprint = current_fingerprint.unwrap_or(Fingerprint::ZERO);

            let print_status = cfg!(debug_assertions) && cx.debug_dep_tasks();

            let dep_node_index = if let Some(prev_index) = data.previous.node_to_index_opt(&key) {
                let prev_fingerprint = data.previous.fingerprint_by_index(prev_index);

                // Determine the color of the DepNode.
                let color = if let Some(current_fingerprint) = current_fingerprint {
                    if current_fingerprint == prev_fingerprint {
                        if print_status {
                            eprintln!("[task::green] {:?}", key);
                        }
                        DepNodeState::Green
                    } else {
                        if print_status {
                            eprintln!("[task::red] {:?}", key);
                        }
                        DepNodeState::Red
                    }
                } else {
                    if print_status {
                        eprintln!("[task::unknown] {:?}", key);
                    }
                    // Mark the node as Red if we can't hash the result
                    DepNodeState::Red
                };

                debug_assert_eq!(
                    data.colors.get(prev_index).color(),
                    None,
                    "DepGraph::with_task() - Duplicate DepNodeState insertion for {:?}",
                    key
                );

                data.colors.insert(prev_index, color);

                data.current.update_node(prev_index, key, deps, fingerprint);

                prev_index
            } else {
                if print_status {
                    eprintln!("[task::new] {:?}", key);
                }

                data.current.new_node(key, deps, fingerprint, &data.previous)
            };

            (result, dep_node_index)
        } else {
            (task(cx, arg), self.next_virtual_depnode_index())
        }
    }

    /// Executes something within an "anonymous" task, that is, a task the
    /// `DepNode` of which is determined by the list of inputs it read from.
    pub fn with_anon_task<OP, R>(&self, dep_kind: K, op: OP) -> (R, DepNodeIndex)
    where
        OP: FnOnce() -> R,
    {
        if let Some(ref data) = self.data {
            let task_deps = Lock::new(TaskDeps::default());

            let result = K::with_deps(Some(&task_deps), op);
            let task_deps = task_deps.into_inner();

            let dep_node_index = data.current.complete_anon_task(dep_kind, task_deps);
            (result, dep_node_index)
        } else {
            (op(), self.next_virtual_depnode_index())
        }
    }

    /// Executes something within an "eval-always" task which is a task
    /// that runs whenever anything changes.
    pub fn with_eval_always_task<Ctxt: DepContext<DepKind = K>, A, R>(
        &self,
        key: DepNode<K>,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        hash_result: impl FnOnce(&mut Ctxt::StableHashingContext, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex) {
        self.with_task_impl(key, cx, arg, false, task, |_| None, hash_result)
    }

    #[inline]
    pub fn read(&self, v: DepNode<K>) {
        if let Some(ref data) = self.data {
            let map = data.current.node_to_node_index.get_shard_by_value(&v).lock();
            if let Some(dep_node_index) = map.get(&v).copied() {
                std::mem::drop(map);
                data.read_index(dep_node_index);
            } else {
                panic!("DepKind {:?} should be pre-allocated but isn't.", v.kind)
            }
        }
    }

    #[inline]
    pub fn read_index(&self, dep_node_index: DepNodeIndex) {
        if let Some(ref data) = self.data {
            data.read_index(dep_node_index);
        }
    }

    #[inline]
    pub fn dep_node_index_of(&self, dep_node: &DepNode<K>) -> DepNodeIndex {
        self.data
            .as_ref()
            .unwrap()
            .current
            .node_to_node_index
            .get_shard_by_value(dep_node)
            .lock()
            .get(dep_node)
            .cloned()
            .unwrap()
    }

    #[inline]
    pub fn dep_node_exists(&self, dep_node: &DepNode<K>) -> bool {
        if let Some(ref data) = self.data {
            data.current
                .node_to_node_index
                .get_shard_by_value(&dep_node)
                .lock()
                .get(dep_node)
                .map_or(false, |index| data.current(*index))
        } else {
            false
        }
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        let data = self.data.as_ref().expect("dep graph enabled");
        assert!(data.current(dep_node_index));
        data.current.data.lock()[dep_node_index].fingerprint
    }

    pub fn prev_fingerprint_of(&self, dep_node: &DepNode<K>) -> Option<Fingerprint> {
        self.data.as_ref().unwrap().previous.fingerprint_of(dep_node)
    }

    /// Checks whether a previous work product exists for `v` and, if
    /// so, return the path that leads to it. Used to skip doing work.
    pub fn previous_work_product(&self, v: &WorkProductId) -> Option<WorkProduct> {
        self.data.as_ref().and_then(|data| data.previous_work_products.get(v).cloned())
    }

    /// Access the map of work-products created during the cached run. Only
    /// used during saving of the dep-graph.
    pub fn previous_work_products(&self) -> &FxHashMap<WorkProductId, WorkProduct> {
        &self.data.as_ref().unwrap().previous_work_products
    }

    #[inline(always)]
    pub fn register_dep_node_debug_str<F>(&self, dep_node: DepNode<K>, debug_str_gen: F)
    where
        F: FnOnce() -> String,
    {
        let dep_node_debug = &self.data.as_ref().unwrap().dep_node_debug;

        if dep_node_debug.borrow().contains_key(&dep_node) {
            return;
        }
        let debug_str = debug_str_gen();
        dep_node_debug.borrow_mut().insert(dep_node, debug_str);
    }

    pub fn dep_node_debug_str(&self, dep_node: DepNode<K>) -> Option<String> {
        self.data.as_ref()?.dep_node_debug.borrow().get(&dep_node).cloned()
    }

    pub fn edge_deduplication_data(&self) -> Option<(u64, u64)> {
        if cfg!(debug_assertions) {
            let current_dep_graph = &self.data.as_ref().unwrap().current;

            Some((
                current_dep_graph.total_read_count.load(Relaxed),
                current_dep_graph.total_duplicate_read_count.load(Relaxed),
            ))
        } else {
            None
        }
    }

    pub fn serialize(&self) -> SerializedDepGraph<K> {
        let data = self.data.as_ref().unwrap();
        let colors = &data.colors;
        let data = data.current.data.lock();

        let fingerprints: IndexVec<DepNodeIndex, _> = data.iter().map(|d| d.fingerprint).collect();
        let mut nodes: IndexVec<DepNodeIndex, _> = data.iter().map(|d| d.node).collect();

        // Invalidate dep nodes with unknown state as these cannot safely
        // be marked green in the next session. One of the dependencies of the
        // unknown node may have changed in this session (and is currently marked red),
        // but might be green again in the next session, which may cause the unknown node
        // to incorrectly be marked green in the next session, even though one of its dependencies
        // did actually change.
        for index in colors.values.indices() {
            match colors.get(index) {
                // In order to this invalidation to be safe, none of the valid nodes can
                // point to unknown nodes.
                DepNodeState::Unknown => {
                    // Change the node kind to Null so we know this node is invalidated when
                    // we load the dep graph
                    nodes[index] = DepNode::new_no_params(DepKind::NULL);
                    // FIXME: Also clear edges
                }

                // For green nodes, we either executed the query (which always uses valid nodes)
                // or we marked it as green because all its dependencies are green and valid.
                DepNodeState::Green |
                // Red nodes were always exexuted.
                DepNodeState::Red |
                // We don't need to invalidate already invalid nodes
                DepNodeState::Invalid => {},
            }
        }

        let total_edge_count: usize = data.iter().map(|d| d.edges.len()).sum();

        let mut edge_list_indices = IndexVec::with_capacity(nodes.len());
        let mut edge_list_data = Vec::with_capacity(total_edge_count);

        for (current_dep_node_index, edges) in data.iter_enumerated().map(|(i, d)| (i, &d.edges)) {
            let start = edge_list_data.len() as u32;
            // This should really just be a memcpy :/
            edge_list_data.extend(edges.iter().map(|i| DepNodeIndex::new(i.index())));
            let end = edge_list_data.len() as u32;

            debug_assert_eq!(current_dep_node_index.index(), edge_list_indices.len());
            edge_list_indices.push((start, end));
        }

        debug_assert!(edge_list_data.len() <= u32::MAX as usize);
        debug_assert_eq!(edge_list_data.len(), total_edge_count);

        SerializedDepGraph { nodes, fingerprints, edge_list_indices, edge_list_data }
    }

    pub fn node_color(&self, dep_node: &DepNode<K>) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            if let Some(prev_index) = data.previous.node_to_index_opt(dep_node) {
                return data.colors.get(prev_index).color();
            } else {
                // This is a node that did not exist in the previous compilation
                // session, so we consider it to be red.
                return Some(DepNodeColor::Red);
            }
        }

        None
    }

    /// Try to read a node index for the node dep_node.
    /// A node will have an index, when it's already been marked green, or when we can mark it
    /// green. This function will mark the current task as a reader of the specified node, when
    /// a node index can be found for that node.
    pub fn try_mark_green_and_read<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        dep_node: &DepNode<K>,
    ) -> Option<DepNodeIndex> {
        self.try_mark_green(tcx, dep_node).map(|prev_index| {
            debug_assert!(self.is_green(&dep_node));
            self.read_index(prev_index);
            prev_index
        })
    }

    pub fn try_mark_green<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        dep_node: &DepNode<K>,
    ) -> Option<DepNodeIndex> {
        debug_assert!(!dep_node.kind.is_eval_always());

        // Return None if the dep graph is disabled
        let data = self.data.as_ref()?;

        // Return None if the dep node didn't exist in the previous session
        let prev_index = data.previous.node_to_index_opt(dep_node)?;

        match data.colors.get(prev_index) {
            DepNodeState::Invalid => panic!(),
            DepNodeState::Green => Some(prev_index),
            DepNodeState::Red => None,
            DepNodeState::Unknown => {
                // This DepNode and the corresponding query invocation existed
                // in the previous compilation session too, so we can try to
                // mark it as green by recursively marking all of its
                // dependencies green.
                if self.try_mark_previous_green(tcx, data, prev_index, &dep_node) {
                    Some(prev_index)
                } else {
                    None
                }
            }
        }
    }

    /// Try to force a dep node to execute and see if it's green
    fn try_force_previous_green<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        dep_node_index: DepNodeIndex,
    ) -> bool {
        let dep_node = &data.previous.index_to_node(dep_node_index);

        debug!("try_force_previous_green({:?}) --- trying to force", dep_node);
        if tcx.try_force_from_dep_node(dep_node) {
            match data.colors.get(dep_node_index) {
                DepNodeState::Green => {
                    debug!(
                        "try_force_previous_green({:?}) --- managed to \
                            FORCE to green",
                        dep_node
                    );
                    true
                }
                DepNodeState::Red => {
                    debug!(
                        "try_force_previous_green({:?}) - END - was red after forcing",
                        dep_node
                    );
                    false
                }
                DepNodeState::Invalid => panic!(),
                DepNodeState::Unknown => {
                    if !tcx.has_errors_or_delayed_span_bugs() {
                        panic!(
                            "try_force_previous_green() - Forcing the DepNode \
                            should have set its color"
                        )
                    } else {
                        // If the query we just forced has resulted in
                        // some kind of compilation error, we cannot rely on
                        // the dep-node color having been properly updated.
                        // This means that the query system has reached an
                        // invalid state. We let the compiler continue (by
                        // returning `None`) so it can emit error messages
                        // and wind down, but rely on the fact that this
                        // invalid state will not be persisted to the
                        // incremental compilation cache because of
                        // compilation errors being present.
                        debug!(
                            "try_force_previous_green({:?}) resulted in compilation error",
                            dep_node
                        );
                        false
                    }
                }
            }
        } else {
            // The DepNode could not be forced.
            debug!("try_force_previous_green({:?}) - END - could not be forced", dep_node);
            false
        }
    }

    /// Try to mark a dep-node which existed in the previous compilation session as green.
    fn try_mark_previous_green<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        dep_node_index: DepNodeIndex,
        // FIXME: Remove this, only used in debug statements
        dep_node: &DepNode<K>,
    ) -> bool {
        debug!("try_mark_previous_green({:?}) - BEGIN", dep_node);

        #[cfg(not(parallel_compiler))]
        debug_assert!(data.colors.get(dep_node_index).color().is_none());

        // We never try to mark eval_always nodes as green
        debug_assert!(!dep_node.kind.is_eval_always());

        debug_assert_eq!(data.previous.index_to_node(dep_node_index), *dep_node);

        let prev_deps = data.previous.edge_targets_from(dep_node_index);

        for &dep_dep_node_index in prev_deps {
            let dep_dep_node_color = data.colors.get(dep_dep_node_index);

            match dep_dep_node_color {
                DepNodeState::Invalid => panic!(),
                DepNodeState::Green => {
                    // This dependency has been marked as green before, we are
                    // still fine and can continue with checking the other
                    // dependencies.
                    debug!(
                        "try_mark_previous_green({:?}) --- found dependency {:?} to \
                            be immediately green",
                        dep_node,
                        data.previous.index_to_node(dep_dep_node_index)
                    );
                }
                DepNodeState::Red => {
                    // We found a dependency the value of which has changed
                    // compared to the previous compilation session. We cannot
                    // mark the DepNode as green and also don't need to bother
                    // with checking any of the other dependencies.
                    debug!(
                        "try_mark_previous_green({:?}) - END - dependency {:?} was \
                            immediately red",
                        dep_node,
                        data.previous.index_to_node(dep_dep_node_index)
                    );
                    return false;
                }
                DepNodeState::Unknown => {
                    let dep_dep_node = &data.previous.index_to_node(dep_dep_node_index);

                    // We don't know the state of this dependency. If it isn't
                    // an eval_always node, let's try to mark it green recursively.
                    if !dep_dep_node.kind.is_eval_always() {
                        // We don't know the state of this dependency.
                        // We known it is not an eval_always node, since those get marked as `Invalid`.
                        // Let's try to mark it green recursively.
                        if self.try_mark_previous_green(tcx, data, dep_dep_node_index, dep_dep_node)
                        {
                            debug!(
                                "try_mark_previous_green({:?}) --- managed to MARK \
                                dependency {:?} as green",
                                dep_node, dep_dep_node
                            );
                            continue;
                        }
                    }

                    // We failed to mark it green, so we try to force the query.
                    if !self.try_force_previous_green(tcx, data, dep_dep_node_index) {
                        return false;
                    }
                }
            }
        }

        // If we got here without hitting a `return` that means that all
        // dependencies of this DepNode could be marked as green. Therefore we
        // can also mark this DepNode as green.

        // There may be multiple threads trying to mark the same dep node green concurrently

        #[cfg(not(parallel_compiler))]
        debug_assert_eq!(
            data.colors.get(dep_node_index).color(),
            None,
            "DepGraph::try_mark_previous_green() - Duplicate DepNodeState \
                      insertion for {:?}",
            dep_node
        );

        // ... emitting any stored diagnostic ...

        // FIXME: Store the fact that a node has diagnostics in a bit in the dep graph somewhere
        // Maybe store a list on disk and encode this fact in the DepNodeState
        let diagnostics = tcx.load_diagnostics(dep_node_index);

        #[cfg(not(parallel_compiler))]
        debug_assert_eq!(
            data.colors.get(dep_node_index).color(),
            None,
            "DepGraph::try_mark_previous_green() - Duplicate DepNodeColor \
                        insertion for {:?}",
            dep_node
        );

        if unlikely!(!diagnostics.is_empty()) {
            self.emit_diagnostics(tcx, data, dep_node_index, diagnostics);
        }

        // ... and finally storing a "Green" entry in the color map.
        // Multiple threads can all write the same color here

        data.colors.insert(dep_node_index, DepNodeState::Green);

        debug!("try_mark_previous_green({:?}) - END - successfully marked as green", dep_node);

        true
    }

    /// Atomically emits some loaded diagnostics.
    /// This may be called concurrently on multiple threads for the same dep node.
    #[cold]
    #[inline(never)]
    fn emit_diagnostics<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        dep_node_index: DepNodeIndex,
        diagnostics: Vec<Diagnostic>,
    ) {
        let mut emitting = data.emitting_diagnostics.lock();

        if data.colors.get(dep_node_index) == DepNodeState::Green {
            // The node is already green so diagnostics must have been emitted already
            return;
        }

        if emitting.insert(dep_node_index) {
            // We were the first to insert the node in the set so this thread
            // must emit the diagnostics and signal other potentially waiting
            // threads after.
            mem::drop(emitting);

            // Promote the previous diagnostics to the current session.
            tcx.store_diagnostics(dep_node_index, diagnostics.clone().into());

            let handle = tcx.diagnostic();

            for diagnostic in diagnostics {
                handle.emit_diagnostic(&diagnostic);
            }

            // Mark the node as green now that diagnostics are emitted
            data.colors.insert(dep_node_index, DepNodeState::Green);

            // Remove the node from the set
            data.emitting_diagnostics.lock().remove(&dep_node_index);

            // Wake up waiters
            data.emitting_diagnostics_cond_var.notify_all();
        } else {
            // We must wait for the other thread to finish emitting the diagnostic

            loop {
                data.emitting_diagnostics_cond_var.wait(&mut emitting);
                if data.colors.get(dep_node_index) == DepNodeState::Green {
                    break;
                }
            }
        }
    }

    // Returns true if the given node has been marked as green during the
    // current compilation session. Used in various assertions
    pub fn is_green(&self, dep_node: &DepNode<K>) -> bool {
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
    pub fn exec_cache_promotions<Ctxt: DepContext<DepKind = K>>(&self, tcx: Ctxt) {
        let _prof_timer = tcx.profiler().generic_activity("incr_comp_query_cache_promotion");

        let data = self.data.as_ref().unwrap();
        for prev_index in data.colors.values.indices() {
            match data.colors.get(prev_index) {
                DepNodeState::Green => {
                    let dep_node = data.previous.index_to_node(prev_index);
                    tcx.try_load_from_on_disk_cache(&dep_node);
                }

                // There cannot be results stored for invalid indices.
                DepNodeState::Invalid |

                // Unknown nodes are unused, so we don't want to promote these and we would
                // not to mark their colors in order to do so anyway.
                DepNodeState::Unknown |

                DepNodeState::Red => {
                    // We can skip red nodes because a node can only be marked
                    // as red if the query result was recomputed and thus is
                    // already in memory.
                }
            }
        }
    }

    fn next_virtual_depnode_index(&self) -> DepNodeIndex {
        let index = self.virtual_dep_node_index.fetch_add(1, Relaxed);
        DepNodeIndex::from_u32(index)
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
    /// Saved files associated with this CGU.
    pub saved_files: Vec<(WorkProductFileKind, String)>,
}

#[derive(Clone, Copy, Debug, RustcEncodable, RustcDecodable, PartialEq)]
pub enum WorkProductFileKind {
    Object,
    Bytecode,
    BytecodeCompressed,
}

#[derive(Clone)]
struct DepNodeData<K> {
    node: DepNode<K>,
    edges: EdgesVec,
    fingerprint: Fingerprint,
}

/// `CurrentDepGraph` stores the dependency graph for the current session.
/// It will be populated as we run queries or tasks.
///
/// The nodes in it are identified by an index (`DepNodeIndex`).
/// The data for each node is stored in its `DepNodeData`, found in the `data` field.
///
/// We never remove nodes from the graph: they are only added.
///
/// This struct uses two locks internally. The `data` and `node_to_node_index` fields are
/// locked separately. Operations that take a `DepNodeIndex` typically just access
/// the data field.
///
/// The only operation that must manipulate both locks is adding new nodes, in which case
/// we first acquire the `node_to_node_index` lock and then, once a new node is to be inserted,
/// acquire the lock on `data.`
pub struct CurrentDepGraph<K> {
    data: Lock<IndexVec<DepNodeIndex, DepNodeData<K>>>,
    node_to_node_index: Sharded<FxHashMap<DepNode<K>, DepNodeIndex>>,

    /// Used to trap when a specific edge is added to the graph.
    /// This is used for debug purposes and is only active with `debug_assertions`.
    #[allow(dead_code)]
    forbidden_edge: Option<EdgeFilter>,

    /// Anonymous `DepNode`s are nodes whose IDs we compute from the list of
    /// their edges. This has the beneficial side-effect that multiple anonymous
    /// nodes can be coalesced into one without changing the semantics of the
    /// dependency graph. However, the merging of nodes can lead to a subtle
    /// problem during red-green marking: The color of an anonymous node from
    /// the current session might "shadow" the color of the node with the same
    /// ID from the previous session. In order to side-step this problem, we make
    /// sure that anonymous `NodeId`s allocated in different sessions don't overlap.
    /// This is implemented by mixing a session-key into the ID fingerprint of
    /// each anon node. The session-key is just a random number generated when
    /// the `DepGraph` is created.
    anon_id_seed: Fingerprint,

    /// These are simple counters that are for profiling and
    /// debugging and only active with `debug_assertions`.
    total_read_count: AtomicU64,
    total_duplicate_read_count: AtomicU64,
}

impl<K: DepKind> CurrentDepGraph<K> {
    pub fn new(prev_graph: &PreviousDepGraph<K>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let nanos = duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64;
        let mut stable_hasher = StableHasher::new();
        nanos.hash(&mut stable_hasher);

        let forbidden_edge = if cfg!(debug_assertions) {
            match env::var("RUST_FORBID_DEP_GRAPH_EDGE") {
                Ok(s) => match EdgeFilter::new(&s) {
                    Ok(f) => Some(f),
                    Err(err) => panic!("RUST_FORBID_DEP_GRAPH_EDGE invalid: {}", err),
                },
                Err(_) => None,
            }
        } else {
            None
        };

        // Pre-allocate the dep node structures. We over-allocate a little so
        // that we hopefully don't have to re-allocate during this compilation
        // session. The over-allocation is 2% plus a small constant to account
        // for the fact that in very small crates 2% might not be enough.
        let new_node_count_estimate = (prev_graph.node_count() * 102) / 100 + 200;

        let mut current = CurrentDepGraph {
            data: Lock::new(IndexVec::with_capacity(new_node_count_estimate)),
            node_to_node_index: Sharded::new(|_| {
                FxHashMap::with_capacity_and_hasher(
                    new_node_count_estimate / sharded::SHARDS,
                    Default::default(),
                )
            }),
            anon_id_seed: stable_hasher.finish(),
            forbidden_edge,
            total_read_count: AtomicU64::new(0),
            total_duplicate_read_count: AtomicU64::new(0),
        };

        // Fill in with the nodes from the previous session
        for (idx, &node) in prev_graph.data.nodes.iter_enumerated() {
            current.data.get_mut().push(DepNodeData {
                node,
                fingerprint: prev_graph.fingerprint_by_index(idx),
                edges: prev_graph.edge_targets_from(idx).iter().copied().collect(),
            });
        }

        // Fill in with the node to index map from the previous session
        for (node, idx) in prev_graph.index.iter() {
            current.node_to_node_index.get_shard_by_value_mut(node).insert(*node, *idx);
        }

        current
    }

    fn complete_anon_task(&self, kind: K, task_deps: TaskDeps<K>) -> DepNodeIndex {
        debug_assert!(!kind.is_eval_always());

        let mut hasher = StableHasher::new();

        // The dep node indices are hashed here instead of hashing the dep nodes of the
        // dependencies. These indices may refer to different nodes per session, but this isn't
        // a problem here because we that ensure the final dep node hash is per session only by
        // combining it with the per session random number `anon_id_seed`. This hash only need
        // to map the dependencies to a single value on a per session basis.
        task_deps.reads.hash(&mut hasher);

        let target_dep_node = DepNode {
            kind,

            // Fingerprint::combine() is faster than sending Fingerprint
            // through the StableHasher (at least as long as StableHasher
            // is so slow).
            hash: self.anon_id_seed.combine(hasher.finish()),
        };

        self.intern_node(target_dep_node, task_deps.reads, Fingerprint::ZERO)
    }

    fn new_node(
        &self,
        dep_node: DepNode<K>,
        edges: EdgesVec,
        fingerprint: Fingerprint,
        previous: &PreviousDepGraph<K>,
    ) -> DepNodeIndex {
        debug_assert!(previous.node_to_index_opt(&dep_node).is_none());
        debug_assert!(
            !self.node_to_node_index.get_shard_by_value(&dep_node).lock().contains_key(&dep_node)
        );
        self.intern_node(dep_node, edges, fingerprint)
    }

    fn update_node(
        &self,
        index: DepNodeIndex,
        dep_node: DepNode<K>,
        edges: SmallVec<[DepNodeIndex; 8]>,
        fingerprint: Fingerprint,
    ) {
        // Update the edges and fingerprint for this dep node
        self.data.lock()[index] = DepNodeData { node: dep_node, edges, fingerprint };
    }

    fn intern_node(
        &self,
        dep_node: DepNode<K>,
        edges: EdgesVec,
        fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        match self.node_to_node_index.get_shard_by_value(&dep_node).lock().entry(dep_node) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let mut data = self.data.lock();
                let dep_node_index = DepNodeIndex::new(data.len());
                data.push(DepNodeData { node: dep_node, edges, fingerprint });
                entry.insert(dep_node_index);
                dep_node_index
            }
        }
    }
}

impl<K: DepKind> DepGraphData<K> {
    fn current(&self, dep_node_index: DepNodeIndex) -> bool {
        if dep_node_index.as_usize() < self.previous.node_count() {
            // From this session if it's colored
            self.colors.get(dep_node_index).color().is_some()
        } else {
            // From this session
            true
        }
    }

    #[inline(never)]
    fn read_index(&self, source: DepNodeIndex) {
        K::read_deps(|task_deps| {
            if let Some(task_deps) = task_deps {
                let mut task_deps = task_deps.lock();
                let task_deps = &mut *task_deps;
                if cfg!(debug_assertions) {
                    self.current.total_read_count.fetch_add(1, Relaxed);
                }

                // As long as we only have a low number of reads we can avoid doing a hash
                // insert and potentially allocating/reallocating the hashmap
                let new_read = if task_deps.reads.len() < TASK_DEPS_READS_CAP {
                    task_deps.reads.iter().all(|other| *other != source)
                } else {
                    task_deps.read_set.insert(source)
                };
                if new_read {
                    task_deps.reads.push(source);
                    if task_deps.reads.len() == TASK_DEPS_READS_CAP {
                        // Fill `read_set` with what we have so far so we can use the hashset next
                        // time
                        task_deps.read_set.extend(task_deps.reads.iter().copied());
                    }

                    #[cfg(debug_assertions)]
                    {
                        if let Some(target) = task_deps.node {
                            let data = self.current.data.lock();
                            if let Some(ref forbidden_edge) = self.current.forbidden_edge {
                                let source = data[source].node;
                                if forbidden_edge.test(&source, &target) {
                                    panic!("forbidden edge {:?} -> {:?} created", source, target)
                                }
                            }
                        }
                    }
                } else if cfg!(debug_assertions) {
                    self.current.total_duplicate_read_count.fetch_add(1, Relaxed);
                }
            }
        })
    }
}

/// The capacity of the `reads` field `SmallVec`
const TASK_DEPS_READS_CAP: usize = 8;
type EdgesVec = SmallVec<[DepNodeIndex; TASK_DEPS_READS_CAP]>;

pub struct TaskDeps<K> {
    #[cfg(debug_assertions)]
    node: Option<DepNode<K>>,
    reads: EdgesVec,
    read_set: FxHashSet<DepNodeIndex>,
    phantom_data: PhantomData<DepNode<K>>,
}

impl<K> Default for TaskDeps<K> {
    fn default() -> Self {
        Self {
            #[cfg(debug_assertions)]
            node: None,
            reads: EdgesVec::new(),
            read_set: FxHashSet::default(),
            phantom_data: PhantomData,
        }
    }
}

struct DepNodeColorMap {
    values: IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>,
}

impl DepNodeColorMap {
    #[inline]
    fn get(&self, index: DepNodeIndex) -> DepNodeState {
        self.values[index].load()
    }

    #[inline]
    fn insert(&self, index: DepNodeIndex, state: DepNodeState) {
        self.values[index].store(state)
    }
}
