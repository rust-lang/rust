use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::profiling::QueryInvocationId;
use rustc_data_structures::sharded::{self, Sharded};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{AtomicU32, AtomicU64, Lock, LockGuard, Lrc, Ordering};
use rustc_data_structures::unlikely;
use rustc_errors::Diagnostic;
use rustc_index::vec::{Idx, IndexVec};

use parking_lot::{Condvar, Mutex};
use smallvec::{smallvec, SmallVec};
use std::collections::hash_map::Entry;
use std::env;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::atomic::Ordering::Relaxed;

use super::debug::EdgeFilter;
use super::prev::PreviousDepGraph;
use super::query::DepGraphQuery;
use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};
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

#[derive(PartialEq)]
pub enum DepNodeColor {
    Red,
    Green(DepNodeIndex),
}

impl DepNodeColor {
    pub fn is_green(self) -> bool {
        match self {
            DepNodeColor::Red => false,
            DepNodeColor::Green(_) => true,
        }
    }
}

struct DepGraphData<K: DepKind> {
    /// The new encoding of the dependency graph, optimized for red/green
    /// tracking. The `current` field is the dependency graph of only the
    /// current compilation session: We don't merge the previous dep-graph into
    /// current one anymore, but we do reference shared data to save space.
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

impl<K: DepKind> DepGraph<K> {
    pub fn new(
        prev_graph: PreviousDepGraph<K>,
        prev_work_products: FxHashMap<WorkProductId, WorkProduct>,
    ) -> DepGraph<K> {
        let prev_graph_node_count = prev_graph.node_count();

        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: prev_work_products,
                dep_node_debug: Default::default(),
                current: CurrentDepGraph::new(prev_graph_node_count),
                emitting_diagnostics: Default::default(),
                emitting_diagnostics_cond_var: Condvar::new(),
                previous: prev_graph,
                colors: DepNodeColorMap::new(prev_graph_node_count),
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
        let previous = &data.previous;

        // Note locking order: `prev_index_to_index`, then `data`.
        let prev_index_to_index = data.current.prev_index_to_index.lock();
        let data = data.current.data.lock();
        let node_count = data.hybrid_indices.len();
        let edge_count = self.edge_count(&data);

        let mut nodes = Vec::with_capacity(node_count);
        let mut edge_list_indices = Vec::with_capacity(node_count);
        let mut edge_list_data = Vec::with_capacity(edge_count);

        // See `serialize` for notes on the approach used here.

        edge_list_data.extend(data.unshared_edges.iter().map(|i| i.index()));

        for &hybrid_index in data.hybrid_indices.iter() {
            match hybrid_index.into() {
                HybridIndex::New(new_index) => {
                    nodes.push(data.new.nodes[new_index]);
                    let edges = &data.new.edges[new_index];
                    edge_list_indices.push((edges.start.index(), edges.end.index()));
                }
                HybridIndex::Red(red_index) => {
                    nodes.push(previous.index_to_node(data.red.node_indices[red_index]));
                    let edges = &data.red.edges[red_index];
                    edge_list_indices.push((edges.start.index(), edges.end.index()));
                }
                HybridIndex::LightGreen(lg_index) => {
                    nodes.push(previous.index_to_node(data.light_green.node_indices[lg_index]));
                    let edges = &data.light_green.edges[lg_index];
                    edge_list_indices.push((edges.start.index(), edges.end.index()));
                }
                HybridIndex::DarkGreen(prev_index) => {
                    nodes.push(previous.index_to_node(prev_index));

                    let edges_iter = previous
                        .edge_targets_from(prev_index)
                        .iter()
                        .map(|&dst| prev_index_to_index[dst].unwrap().index());

                    let start = edge_list_data.len();
                    edge_list_data.extend(edges_iter);
                    let end = edge_list_data.len();
                    edge_list_indices.push((start, end));
                }
            }
        }

        debug_assert_eq!(nodes.len(), node_count);
        debug_assert_eq!(edge_list_indices.len(), node_count);
        debug_assert_eq!(edge_list_data.len(), edge_count);

        DepGraphQuery::new(&nodes[..], &edge_list_indices[..], &edge_list_data[..])
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
        task: fn(Ctxt, A) -> R,
        create_task: fn(DepNode<K>) -> Option<TaskDeps<K>>,
        hash_result: impl FnOnce(&mut Ctxt::StableHashingContext, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex) {
        if let Some(ref data) = self.data {
            let task_deps = create_task(key).map(Lock::new);
            let result = K::with_deps(task_deps.as_ref(), || task(cx, arg));
            let edges = task_deps.map_or_else(|| smallvec![], |lock| lock.into_inner().reads);

            let mut hcx = cx.create_stable_hashing_context();
            let current_fingerprint = hash_result(&mut hcx, &result);

            let print_status = cfg!(debug_assertions) && cx.debug_dep_tasks();

            // Intern the new `DepNode`.
            let dep_node_index = if let Some(prev_index) = data.previous.node_to_index_opt(&key) {
                // Determine the color and index of the new `DepNode`.
                let (color, dep_node_index) = if let Some(current_fingerprint) = current_fingerprint
                {
                    if current_fingerprint == data.previous.fingerprint_by_index(prev_index) {
                        if print_status {
                            eprintln!("[task::green] {:?}", key);
                        }

                        // This is a light green node: it existed in the previous compilation,
                        // its query was re-executed, and it has the same result as before.
                        let dep_node_index =
                            data.current.intern_light_green_node(&data.previous, prev_index, edges);

                        (DepNodeColor::Green(dep_node_index), dep_node_index)
                    } else {
                        if print_status {
                            eprintln!("[task::red] {:?}", key);
                        }

                        // This is a red node: it existed in the previous compilation, its query
                        // was re-executed, but it has a different result from before.
                        let dep_node_index = data.current.intern_red_node(
                            &data.previous,
                            prev_index,
                            edges,
                            current_fingerprint,
                        );

                        (DepNodeColor::Red, dep_node_index)
                    }
                } else {
                    if print_status {
                        eprintln!("[task::unknown] {:?}", key);
                    }

                    // This is a red node, effectively: it existed in the previous compilation
                    // session, its query was re-executed, but it doesn't compute a result hash
                    // (i.e. it represents a `no_hash` query), so we have no way of determining
                    // whether or not the result was the same as before.
                    let dep_node_index = data.current.intern_red_node(
                        &data.previous,
                        prev_index,
                        edges,
                        Fingerprint::ZERO,
                    );

                    (DepNodeColor::Red, dep_node_index)
                };

                debug_assert!(
                    data.colors.get(prev_index).is_none(),
                    "DepGraph::with_task() - Duplicate DepNodeColor \
                            insertion for {:?}",
                    key
                );

                data.colors.insert(prev_index, color);
                dep_node_index
            } else {
                if print_status {
                    eprintln!("[task::new] {:?}", key);
                }

                // This is a new node: it didn't exist in the previous compilation session.
                data.current.intern_new_node(
                    &data.previous,
                    key,
                    edges,
                    current_fingerprint.unwrap_or(Fingerprint::ZERO),
                )
            };

            (result, dep_node_index)
        } else {
            // Incremental compilation is turned off. We just execute the task
            // without tracking. We still provide a dep-node index that uniquely
            // identifies the task so that we have a cheap way of referring to
            // the query for self-profiling.
            (task(cx, arg), self.next_virtual_depnode_index())
        }
    }

    /// Executes something within an "anonymous" task, that is, a task the
    /// `DepNode` of which is determined by the list of inputs it read from.
    pub fn with_anon_task<OP, R>(&self, dep_kind: K, op: OP) -> (R, DepNodeIndex)
    where
        OP: FnOnce() -> R,
    {
        debug_assert!(!dep_kind.is_eval_always());

        if let Some(ref data) = self.data {
            let task_deps = Lock::new(TaskDeps::default());
            let result = K::with_deps(Some(&task_deps), op);
            let task_deps = task_deps.into_inner();

            // The dep node indices are hashed here instead of hashing the dep nodes of the
            // dependencies. These indices may refer to different nodes per session, but this isn't
            // a problem here because we that ensure the final dep node hash is per session only by
            // combining it with the per session random number `anon_id_seed`. This hash only need
            // to map the dependencies to a single value on a per session basis.
            let mut hasher = StableHasher::new();
            task_deps.reads.hash(&mut hasher);

            let target_dep_node = DepNode {
                kind: dep_kind,
                // Fingerprint::combine() is faster than sending Fingerprint
                // through the StableHasher (at least as long as StableHasher
                // is so slow).
                hash: data.current.anon_id_seed.combine(hasher.finish()).into(),
            };

            let dep_node_index = data.current.intern_new_node(
                &data.previous,
                target_dep_node,
                task_deps.reads,
                Fingerprint::ZERO,
            );

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
        self.with_task_impl(key, cx, arg, task, |_| None, hash_result)
    }

    #[inline]
    pub fn read_index(&self, dep_node_index: DepNodeIndex) {
        if let Some(ref data) = self.data {
            K::read_deps(|task_deps| {
                if let Some(task_deps) = task_deps {
                    let mut task_deps = task_deps.lock();
                    let task_deps = &mut *task_deps;
                    if cfg!(debug_assertions) {
                        data.current.total_read_count.fetch_add(1, Relaxed);
                    }

                    // As long as we only have a low number of reads we can avoid doing a hash
                    // insert and potentially allocating/reallocating the hashmap
                    let new_read = if task_deps.reads.len() < TASK_DEPS_READS_CAP {
                        task_deps.reads.iter().all(|other| *other != dep_node_index)
                    } else {
                        task_deps.read_set.insert(dep_node_index)
                    };
                    if new_read {
                        task_deps.reads.push(dep_node_index);
                        if task_deps.reads.len() == TASK_DEPS_READS_CAP {
                            // Fill `read_set` with what we have so far so we can use the hashset
                            // next time
                            task_deps.read_set.extend(task_deps.reads.iter().copied());
                        }

                        #[cfg(debug_assertions)]
                        {
                            if let Some(target) = task_deps.node {
                                if let Some(ref forbidden_edge) = data.current.forbidden_edge {
                                    let src = self.dep_node_of(dep_node_index);
                                    if forbidden_edge.test(&src, &target) {
                                        panic!("forbidden edge {:?} -> {:?} created", src, target)
                                    }
                                }
                            }
                        }
                    } else if cfg!(debug_assertions) {
                        data.current.total_duplicate_read_count.fetch_add(1, Relaxed);
                    }
                }
            })
        }
    }

    #[inline]
    pub fn dep_node_index_of(&self, dep_node: &DepNode<K>) -> DepNodeIndex {
        self.dep_node_index_of_opt(dep_node).unwrap()
    }

    #[inline]
    pub fn dep_node_index_of_opt(&self, dep_node: &DepNode<K>) -> Option<DepNodeIndex> {
        let data = self.data.as_ref().unwrap();
        let current = &data.current;

        if let Some(prev_index) = data.previous.node_to_index_opt(dep_node) {
            current.prev_index_to_index.lock()[prev_index]
        } else {
            current.new_node_to_index.get_shard_by_value(dep_node).lock().get(dep_node).copied()
        }
    }

    #[inline]
    pub fn dep_node_exists(&self, dep_node: &DepNode<K>) -> bool {
        self.data.is_some() && self.dep_node_index_of_opt(dep_node).is_some()
    }

    #[inline]
    pub fn dep_node_of(&self, dep_node_index: DepNodeIndex) -> DepNode<K> {
        let data = self.data.as_ref().unwrap();
        let previous = &data.previous;
        let data = data.current.data.lock();

        match data.hybrid_indices[dep_node_index].into() {
            HybridIndex::New(new_index) => data.new.nodes[new_index],
            HybridIndex::Red(red_index) => previous.index_to_node(data.red.node_indices[red_index]),
            HybridIndex::LightGreen(light_green_index) => {
                previous.index_to_node(data.light_green.node_indices[light_green_index])
            }
            HybridIndex::DarkGreen(prev_index) => previous.index_to_node(prev_index),
        }
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        let data = self.data.as_ref().unwrap();
        let previous = &data.previous;
        let data = data.current.data.lock();

        match data.hybrid_indices[dep_node_index].into() {
            HybridIndex::New(new_index) => data.new.fingerprints[new_index],
            HybridIndex::Red(red_index) => data.red.fingerprints[red_index],
            HybridIndex::LightGreen(light_green_index) => {
                previous.fingerprint_by_index(data.light_green.node_indices[light_green_index])
            }
            HybridIndex::DarkGreen(prev_index) => previous.fingerprint_by_index(prev_index),
        }
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

    fn edge_count(&self, node_data: &LockGuard<'_, DepNodeData<K>>) -> usize {
        let data = self.data.as_ref().unwrap();
        let previous = &data.previous;

        let mut edge_count = node_data.unshared_edges.len();

        for &hybrid_index in node_data.hybrid_indices.iter() {
            if let HybridIndex::DarkGreen(prev_index) = hybrid_index.into() {
                edge_count += previous.edge_targets_from(prev_index).len()
            }
        }

        edge_count
    }

    pub fn serialize(&self) -> SerializedDepGraph<K> {
        type SDNI = SerializedDepNodeIndex;

        let data = self.data.as_ref().unwrap();
        let previous = &data.previous;

        // Note locking order: `prev_index_to_index`, then `data`.
        let prev_index_to_index = data.current.prev_index_to_index.lock();
        let data = data.current.data.lock();
        let node_count = data.hybrid_indices.len();
        let edge_count = self.edge_count(&data);

        let mut nodes = IndexVec::with_capacity(node_count);
        let mut fingerprints = IndexVec::with_capacity(node_count);
        let mut edge_list_indices = IndexVec::with_capacity(node_count);
        let mut edge_list_data = Vec::with_capacity(edge_count);

        // `rustc_middle::ty::query::OnDiskCache` expects nodes to be in
        // `DepNodeIndex` order. The edges in `edge_list_data`, on the other
        // hand, don't need to be in a particular order, as long as each node
        // can reference its edges as a contiguous range within it. This is why
        // we're able to copy `unshared_edges` directly into `edge_list_data`.
        // It meets the above requirements, and each non-dark-green node already
        // knows the range of edges to reference within it, which they'll push
        // onto `edge_list_indices`. Dark green nodes, however, don't have their
        // edges in `unshared_edges`, so need to add them to `edge_list_data`.

        edge_list_data.extend(data.unshared_edges.iter().map(|i| SDNI::new(i.index())));

        for &hybrid_index in data.hybrid_indices.iter() {
            match hybrid_index.into() {
                HybridIndex::New(i) => {
                    let new = &data.new;
                    nodes.push(new.nodes[i]);
                    fingerprints.push(new.fingerprints[i]);
                    let edges = &new.edges[i];
                    edge_list_indices.push((edges.start.as_u32(), edges.end.as_u32()));
                }
                HybridIndex::Red(i) => {
                    let red = &data.red;
                    nodes.push(previous.index_to_node(red.node_indices[i]));
                    fingerprints.push(red.fingerprints[i]);
                    let edges = &red.edges[i];
                    edge_list_indices.push((edges.start.as_u32(), edges.end.as_u32()));
                }
                HybridIndex::LightGreen(i) => {
                    let lg = &data.light_green;
                    nodes.push(previous.index_to_node(lg.node_indices[i]));
                    fingerprints.push(previous.fingerprint_by_index(lg.node_indices[i]));
                    let edges = &lg.edges[i];
                    edge_list_indices.push((edges.start.as_u32(), edges.end.as_u32()));
                }
                HybridIndex::DarkGreen(prev_index) => {
                    nodes.push(previous.index_to_node(prev_index));
                    fingerprints.push(previous.fingerprint_by_index(prev_index));

                    let edges_iter = previous
                        .edge_targets_from(prev_index)
                        .iter()
                        .map(|&dst| prev_index_to_index[dst].as_ref().unwrap());

                    let start = edge_list_data.len() as u32;
                    edge_list_data.extend(edges_iter.map(|i| SDNI::new(i.index())));
                    let end = edge_list_data.len() as u32;
                    edge_list_indices.push((start, end));
                }
            }
        }

        debug_assert_eq!(nodes.len(), node_count);
        debug_assert_eq!(fingerprints.len(), node_count);
        debug_assert_eq!(edge_list_indices.len(), node_count);
        debug_assert_eq!(edge_list_data.len(), edge_count);
        debug_assert!(edge_list_data.len() <= u32::MAX as usize);

        SerializedDepGraph { nodes, fingerprints, edge_list_indices, edge_list_data }
    }

    pub fn node_color(&self, dep_node: &DepNode<K>) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            if let Some(prev_index) = data.previous.node_to_index_opt(dep_node) {
                return data.colors.get(prev_index);
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
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        self.try_mark_green(tcx, dep_node).map(|(prev_index, dep_node_index)| {
            debug_assert!(self.is_green(&dep_node));
            self.read_index(dep_node_index);
            (prev_index, dep_node_index)
        })
    }

    pub fn try_mark_green<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        dep_node: &DepNode<K>,
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        debug_assert!(!dep_node.kind.is_eval_always());

        // Return None if the dep graph is disabled
        let data = self.data.as_ref()?;

        // Return None if the dep node didn't exist in the previous session
        let prev_index = data.previous.node_to_index_opt(dep_node)?;

        match data.colors.get(prev_index) {
            Some(DepNodeColor::Green(dep_node_index)) => Some((prev_index, dep_node_index)),
            Some(DepNodeColor::Red) => None,
            None => {
                // This DepNode and the corresponding query invocation existed
                // in the previous compilation session too, so we can try to
                // mark it as green by recursively marking all of its
                // dependencies green.
                self.try_mark_previous_green(tcx, data, prev_index, &dep_node)
                    .map(|dep_node_index| (prev_index, dep_node_index))
            }
        }
    }

    /// Try to mark a dep-node which existed in the previous compilation session as green.
    fn try_mark_previous_green<Ctxt: DepContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        prev_dep_node_index: SerializedDepNodeIndex,
        dep_node: &DepNode<K>,
    ) -> Option<DepNodeIndex> {
        debug!("try_mark_previous_green({:?}) - BEGIN", dep_node);

        #[cfg(not(parallel_compiler))]
        {
            debug_assert!(!self.dep_node_exists(dep_node));
            debug_assert!(data.colors.get(prev_dep_node_index).is_none());
        }

        // We never try to mark eval_always nodes as green
        debug_assert!(!dep_node.kind.is_eval_always());

        debug_assert_eq!(data.previous.index_to_node(prev_dep_node_index), *dep_node);

        let prev_deps = data.previous.edge_targets_from(prev_dep_node_index);

        for &dep_dep_node_index in prev_deps {
            let dep_dep_node_color = data.colors.get(dep_dep_node_index);

            match dep_dep_node_color {
                Some(DepNodeColor::Green(_)) => {
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
                Some(DepNodeColor::Red) => {
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
                    return None;
                }
                None => {
                    let dep_dep_node = &data.previous.index_to_node(dep_dep_node_index);

                    // We don't know the state of this dependency. If it isn't
                    // an eval_always node, let's try to mark it green recursively.
                    if !dep_dep_node.kind.is_eval_always() {
                        debug!(
                            "try_mark_previous_green({:?}) --- state of dependency {:?} ({}) \
                                 is unknown, trying to mark it green",
                            dep_node, dep_dep_node, dep_dep_node.hash,
                        );

                        let node_index = self.try_mark_previous_green(
                            tcx,
                            data,
                            dep_dep_node_index,
                            dep_dep_node,
                        );
                        if node_index.is_some() {
                            debug!(
                                "try_mark_previous_green({:?}) --- managed to MARK \
                                    dependency {:?} as green",
                                dep_node, dep_dep_node
                            );
                            continue;
                        }
                    }

                    // We failed to mark it green, so we try to force the query.
                    debug!(
                        "try_mark_previous_green({:?}) --- trying to force \
                            dependency {:?}",
                        dep_node, dep_dep_node
                    );
                    if tcx.try_force_from_dep_node(dep_dep_node) {
                        let dep_dep_node_color = data.colors.get(dep_dep_node_index);

                        match dep_dep_node_color {
                            Some(DepNodeColor::Green(_)) => {
                                debug!(
                                    "try_mark_previous_green({:?}) --- managed to \
                                        FORCE dependency {:?} to green",
                                    dep_node, dep_dep_node
                                );
                            }
                            Some(DepNodeColor::Red) => {
                                debug!(
                                    "try_mark_previous_green({:?}) - END - \
                                        dependency {:?} was red after forcing",
                                    dep_node, dep_dep_node
                                );
                                return None;
                            }
                            None => {
                                if !tcx.has_errors_or_delayed_span_bugs() {
                                    panic!(
                                        "try_mark_previous_green() - Forcing the DepNode \
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
                                        "try_mark_previous_green({:?}) - END - \
                                            dependency {:?} resulted in compilation error",
                                        dep_node, dep_dep_node
                                    );
                                    return None;
                                }
                            }
                        }
                    } else {
                        // The DepNode could not be forced.
                        debug!(
                            "try_mark_previous_green({:?}) - END - dependency {:?} \
                                could not be forced",
                            dep_node, dep_dep_node
                        );
                        return None;
                    }
                }
            }
        }

        // If we got here without hitting a `return` that means that all
        // dependencies of this DepNode could be marked as green. Therefore we
        // can also mark this DepNode as green.

        // There may be multiple threads trying to mark the same dep node green concurrently

        let dep_node_index = {
            // We allocating an entry for the node in the current dependency graph and
            // adding all the appropriate edges imported from the previous graph
            data.current.intern_dark_green_node(&data.previous, prev_dep_node_index)
        };

        // ... emitting any stored diagnostic ...

        // FIXME: Store the fact that a node has diagnostics in a bit in the dep graph somewhere
        // Maybe store a list on disk and encode this fact in the DepNodeState
        let diagnostics = tcx.load_diagnostics(prev_dep_node_index);

        #[cfg(not(parallel_compiler))]
        debug_assert!(
            data.colors.get(prev_dep_node_index).is_none(),
            "DepGraph::try_mark_previous_green() - Duplicate DepNodeColor \
                      insertion for {:?}",
            dep_node
        );

        if unlikely!(!diagnostics.is_empty()) {
            self.emit_diagnostics(tcx, data, dep_node_index, prev_dep_node_index, diagnostics);
        }

        // ... and finally storing a "Green" entry in the color map.
        // Multiple threads can all write the same color here
        data.colors.insert(prev_dep_node_index, DepNodeColor::Green(dep_node_index));

        debug!("try_mark_previous_green({:?}) - END - successfully marked as green", dep_node);
        Some(dep_node_index)
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
        prev_dep_node_index: SerializedDepNodeIndex,
        diagnostics: Vec<Diagnostic>,
    ) {
        let mut emitting = data.emitting_diagnostics.lock();

        if data.colors.get(prev_dep_node_index) == Some(DepNodeColor::Green(dep_node_index)) {
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
            data.colors.insert(prev_dep_node_index, DepNodeColor::Green(dep_node_index));

            // Remove the node from the set
            data.emitting_diagnostics.lock().remove(&dep_node_index);

            // Wake up waiters
            data.emitting_diagnostics_cond_var.notify_all();
        } else {
            // We must wait for the other thread to finish emitting the diagnostic

            loop {
                data.emitting_diagnostics_cond_var.wait(&mut emitting);
                if data.colors.get(prev_dep_node_index) == Some(DepNodeColor::Green(dep_node_index))
                {
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
                Some(DepNodeColor::Green(_)) => {
                    let dep_node = data.previous.index_to_node(prev_index);
                    tcx.try_load_from_on_disk_cache(&dep_node);
                }
                None | Some(DepNodeColor::Red) => {
                    // We can skip red nodes because a node can only be marked
                    // as red if the query result was recomputed and thus is
                    // already in memory.
                }
            }
        }
    }

    // Register reused dep nodes (i.e. nodes we've marked red or green) with the context.
    pub fn register_reused_dep_nodes<Ctxt: DepContext<DepKind = K>>(&self, tcx: Ctxt) {
        let data = self.data.as_ref().unwrap();
        for prev_index in data.colors.values.indices() {
            match data.colors.get(prev_index) {
                Some(DepNodeColor::Red) | Some(DepNodeColor::Green(_)) => {
                    let dep_node = data.previous.index_to_node(prev_index);
                    tcx.register_reused_dep_node(&dep_node);
                }
                None => {}
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
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct WorkProduct {
    pub cgu_name: String,
    /// Saved file associated with this CGU.
    pub saved_file: Option<String>,
}

// The maximum value of the follow index types leaves the upper two bits unused
// so that we can store multiple index types in `CompressedHybridIndex`, and use
// those bits to encode which index type it contains.

// Index type for `NewDepNodeData`.
rustc_index::newtype_index! {
    struct NewDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

// Index type for `RedDepNodeData`.
rustc_index::newtype_index! {
    struct RedDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

// Index type for `LightGreenDepNodeData`.
rustc_index::newtype_index! {
    struct LightGreenDepNodeIndex {
        MAX = 0x7FFF_FFFF
    }
}

/// Compressed representation of `HybridIndex` enum. Bits unused by the
/// contained index types are used to encode which index type it contains.
#[derive(Copy, Clone)]
struct CompressedHybridIndex(u32);

impl CompressedHybridIndex {
    const NEW_TAG: u32 = 0b0000_0000_0000_0000_0000_0000_0000_0000;
    const RED_TAG: u32 = 0b0100_0000_0000_0000_0000_0000_0000_0000;
    const LIGHT_GREEN_TAG: u32 = 0b1000_0000_0000_0000_0000_0000_0000_0000;
    const DARK_GREEN_TAG: u32 = 0b1100_0000_0000_0000_0000_0000_0000_0000;

    const TAG_MASK: u32 = 0b1100_0000_0000_0000_0000_0000_0000_0000;
    const INDEX_MASK: u32 = !Self::TAG_MASK;
}

impl From<NewDepNodeIndex> for CompressedHybridIndex {
    #[inline]
    fn from(index: NewDepNodeIndex) -> Self {
        CompressedHybridIndex(Self::NEW_TAG | index.as_u32())
    }
}

impl From<RedDepNodeIndex> for CompressedHybridIndex {
    #[inline]
    fn from(index: RedDepNodeIndex) -> Self {
        CompressedHybridIndex(Self::RED_TAG | index.as_u32())
    }
}

impl From<LightGreenDepNodeIndex> for CompressedHybridIndex {
    #[inline]
    fn from(index: LightGreenDepNodeIndex) -> Self {
        CompressedHybridIndex(Self::LIGHT_GREEN_TAG | index.as_u32())
    }
}

impl From<SerializedDepNodeIndex> for CompressedHybridIndex {
    #[inline]
    fn from(index: SerializedDepNodeIndex) -> Self {
        CompressedHybridIndex(Self::DARK_GREEN_TAG | index.as_u32())
    }
}

/// Contains an index into one of several node data collections. Elsewhere, we
/// store `CompressedHyridIndex` instead of this to save space, but convert to
/// this type during processing to take advantage of the enum match ergonomics.
enum HybridIndex {
    New(NewDepNodeIndex),
    Red(RedDepNodeIndex),
    LightGreen(LightGreenDepNodeIndex),
    DarkGreen(SerializedDepNodeIndex),
}

impl From<CompressedHybridIndex> for HybridIndex {
    #[inline]
    fn from(hybrid_index: CompressedHybridIndex) -> Self {
        let index = hybrid_index.0 & CompressedHybridIndex::INDEX_MASK;

        match hybrid_index.0 & CompressedHybridIndex::TAG_MASK {
            CompressedHybridIndex::NEW_TAG => HybridIndex::New(NewDepNodeIndex::from_u32(index)),
            CompressedHybridIndex::RED_TAG => HybridIndex::Red(RedDepNodeIndex::from_u32(index)),
            CompressedHybridIndex::LIGHT_GREEN_TAG => {
                HybridIndex::LightGreen(LightGreenDepNodeIndex::from_u32(index))
            }
            CompressedHybridIndex::DARK_GREEN_TAG => {
                HybridIndex::DarkGreen(SerializedDepNodeIndex::from_u32(index))
            }
            _ => unreachable!(),
        }
    }
}

// Index type for `DepNodeData`'s edges.
rustc_index::newtype_index! {
    struct EdgeIndex { .. }
}

/// Data for nodes in the current graph, divided into different collections
/// based on their presence in the previous graph, and if present, their color.
/// We divide nodes this way because different types of nodes are able to share
/// more or less data with the previous graph.
///
/// To enable more sharing, we distinguish between two kinds of green nodes.
/// Light green nodes are nodes in the previous graph that have been marked
/// green because we re-executed their queries and the results were the same as
/// in the previous session. Dark green nodes are nodes in the previous graph
/// that have been marked green because we were able to mark all of their
/// dependencies green.
///
/// Both light and dark green nodes can share the dep node and fingerprint with
/// the previous graph, but for light green nodes, we can't be sure that the
/// edges may be shared without comparing them against the previous edges, so we
/// store them directly (an approach in which we compare edges with the previous
/// edges to see if they can be shared was evaluated, but was not found to be
/// very profitable).
///
/// For dark green nodes, we can share everything with the previous graph, which
/// is why the `HybridIndex::DarkGreen` enum variant contains the index of the
/// node in the previous graph, and why we don't have a separate collection for
/// dark green node data--the collection is the `PreviousDepGraph` itself.
///
/// (Note that for dark green nodes, the edges in the previous graph
/// (`SerializedDepNodeIndex`s) must be converted to edges in the current graph
/// (`DepNodeIndex`s). `CurrentDepGraph` contains `prev_index_to_index`, which
/// can perform this conversion. It should always be possible, as by definition,
/// a dark green node is one whose dependencies from the previous session have
/// all been marked green--which means `prev_index_to_index` contains them.)
///
/// Node data is stored in parallel vectors to eliminate the padding between
/// elements that would be needed to satisfy alignment requirements of the
/// structure that would contain all of a node's data. We could group tightly
/// packing subsets of node data together and use fewer vectors, but for
/// consistency's sake, we use separate vectors for each piece of data.
struct DepNodeData<K> {
    /// Data for nodes not in previous graph.
    new: NewDepNodeData<K>,

    /// Data for nodes in previous graph that have been marked red.
    red: RedDepNodeData,

    /// Data for nodes in previous graph that have been marked light green.
    light_green: LightGreenDepNodeData,

    // Edges for all nodes other than dark-green ones. Edges for each node
    // occupy a contiguous region of this collection, which a node can reference
    // using two indices. Storing edges this way rather than using an `EdgesVec`
    // for each node reduces memory consumption by a not insignificant amount
    // when compiling large crates. The downside is that we have to copy into
    // this collection the edges from the `EdgesVec`s that are built up during
    // query execution. But this is mostly balanced out by the more efficient
    // implementation of `DepGraph::serialize` enabled by this representation.
    unshared_edges: IndexVec<EdgeIndex, DepNodeIndex>,

    /// Mapping from `DepNodeIndex` to an index into a collection above.
    /// Indicates which of the above collections contains a node's data.
    ///
    /// This collection is wasteful in time and space during incr-full builds,
    /// because for those, all nodes are new. However, the waste is relatively
    /// small, and the maintenance cost of avoiding using this for incr-full
    /// builds is somewhat high and prone to bugginess. It does not seem worth
    /// it at the time of this writing, but we may want to revisit the idea.
    hybrid_indices: IndexVec<DepNodeIndex, CompressedHybridIndex>,
}

/// Data for nodes not in previous graph. Since we cannot share any data with
/// the previous graph, so we must store all of such a node's data here.
struct NewDepNodeData<K> {
    nodes: IndexVec<NewDepNodeIndex, DepNode<K>>,
    edges: IndexVec<NewDepNodeIndex, Range<EdgeIndex>>,
    fingerprints: IndexVec<NewDepNodeIndex, Fingerprint>,
}

/// Data for nodes in previous graph that have been marked red. We can share the
/// dep node with the previous graph, but the edges may be different, and the
/// fingerprint is known to be different, so we store the latter two directly.
struct RedDepNodeData {
    node_indices: IndexVec<RedDepNodeIndex, SerializedDepNodeIndex>,
    edges: IndexVec<RedDepNodeIndex, Range<EdgeIndex>>,
    fingerprints: IndexVec<RedDepNodeIndex, Fingerprint>,
}

/// Data for nodes in previous graph that have been marked green because we
/// re-executed their queries and the results were the same as in the previous
/// session. We can share the dep node and the fingerprint with the previous
/// graph, but the edges may be different, so we store them directly.
struct LightGreenDepNodeData {
    node_indices: IndexVec<LightGreenDepNodeIndex, SerializedDepNodeIndex>,
    edges: IndexVec<LightGreenDepNodeIndex, Range<EdgeIndex>>,
}

/// `CurrentDepGraph` stores the dependency graph for the current session. It
/// will be populated as we run queries or tasks. We never remove nodes from the
/// graph: they are only added.
///
/// The nodes in it are identified by a `DepNodeIndex`. Internally, this maps to
/// a `HybridIndex`, which identifies which collection in the `data` field
/// contains a node's data. Which collection is used for a node depends on
/// whether the node was present in the `PreviousDepGraph`, and if so, the color
/// of the node. Each type of node can share more or less data with the previous
/// graph. When possible, we can store just the index of the node in the
/// previous graph, rather than duplicating its data in our own collections.
/// This is important, because these graph structures are some of the largest in
/// the compiler.
///
/// For the same reason, we also avoid storing `DepNode`s more than once as map
/// keys. The `new_node_to_index` map only contains nodes not in the previous
/// graph, and we map nodes in the previous graph to indices via a two-step
/// mapping. `PreviousDepGraph` maps from `DepNode` to `SerializedDepNodeIndex`,
/// and the `prev_index_to_index` vector (which is more compact and faster than
/// using a map) maps from `SerializedDepNodeIndex` to `DepNodeIndex`.
///
/// This struct uses three locks internally. The `data`, `new_node_to_index`,
/// and `prev_index_to_index` fields are locked separately. Operations that take
/// a `DepNodeIndex` typically just access the `data` field.
///
/// We only need to manipulate at most two locks simultaneously:
/// `new_node_to_index` and `data`, or `prev_index_to_index` and `data`. When
/// manipulating both, we acquire `new_node_to_index` or `prev_index_to_index`
/// first, and `data` second.
pub(super) struct CurrentDepGraph<K> {
    data: Lock<DepNodeData<K>>,
    new_node_to_index: Sharded<FxHashMap<DepNode<K>, DepNodeIndex>>,
    prev_index_to_index: Lock<IndexVec<SerializedDepNodeIndex, Option<DepNodeIndex>>>,

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
    fn new(prev_graph_node_count: usize) -> CurrentDepGraph<K> {
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
        // session. The over-allocation for new nodes is 2% plus a small
        // constant to account for the fact that in very small crates 2% might
        // not be enough. The allocation for red and green node data doesn't
        // include a constant, as we don't want to allocate anything for these
        // structures during full incremental builds, where they aren't used.
        //
        // These estimates are based on the distribution of node and edge counts
        // seen in rustc-perf benchmarks, adjusted somewhat to account for the
        // fact that these benchmarks aren't perfectly representative.
        //
        // FIXME Use a collection type that doesn't copy node and edge data and
        // grow multiplicatively on reallocation. Without such a collection or
        // solution having the same effect, there is a performance hazard here
        // in both time and space, as growing these collections means copying a
        // large amount of data and doubling already large buffer capacities. A
        // solution for this will also mean that it's less important to get
        // these estimates right.
        let new_node_count_estimate = (prev_graph_node_count * 2) / 100 + 200;
        let red_node_count_estimate = (prev_graph_node_count * 3) / 100;
        let light_green_node_count_estimate = (prev_graph_node_count * 25) / 100;
        let total_node_count_estimate = prev_graph_node_count + new_node_count_estimate;

        let average_edges_per_node_estimate = 6;
        let unshared_edge_count_estimate = average_edges_per_node_estimate
            * (new_node_count_estimate + red_node_count_estimate + light_green_node_count_estimate);

        // We store a large collection of these in `prev_index_to_index` during
        // non-full incremental builds, and want to ensure that the element size
        // doesn't inadvertently increase.
        static_assert_size!(Option<DepNodeIndex>, 4);

        CurrentDepGraph {
            data: Lock::new(DepNodeData {
                new: NewDepNodeData {
                    nodes: IndexVec::with_capacity(new_node_count_estimate),
                    edges: IndexVec::with_capacity(new_node_count_estimate),
                    fingerprints: IndexVec::with_capacity(new_node_count_estimate),
                },
                red: RedDepNodeData {
                    node_indices: IndexVec::with_capacity(red_node_count_estimate),
                    edges: IndexVec::with_capacity(red_node_count_estimate),
                    fingerprints: IndexVec::with_capacity(red_node_count_estimate),
                },
                light_green: LightGreenDepNodeData {
                    node_indices: IndexVec::with_capacity(light_green_node_count_estimate),
                    edges: IndexVec::with_capacity(light_green_node_count_estimate),
                },
                unshared_edges: IndexVec::with_capacity(unshared_edge_count_estimate),
                hybrid_indices: IndexVec::with_capacity(total_node_count_estimate),
            }),
            new_node_to_index: Sharded::new(|| {
                FxHashMap::with_capacity_and_hasher(
                    new_node_count_estimate / sharded::SHARDS,
                    Default::default(),
                )
            }),
            prev_index_to_index: Lock::new(IndexVec::from_elem_n(None, prev_graph_node_count)),
            anon_id_seed: stable_hasher.finish(),
            forbidden_edge,
            total_read_count: AtomicU64::new(0),
            total_duplicate_read_count: AtomicU64::new(0),
        }
    }

    fn intern_new_node(
        &self,
        prev_graph: &PreviousDepGraph<K>,
        dep_node: DepNode<K>,
        edges: EdgesVec,
        fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        debug_assert!(
            prev_graph.node_to_index_opt(&dep_node).is_none(),
            "node in previous graph should be interned using one \
            of `intern_red_node`, `intern_light_green_node`, etc."
        );

        match self.new_node_to_index.get_shard_by_value(&dep_node).lock().entry(dep_node) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let data = &mut *self.data.lock();
                let new_index = data.new.nodes.push(dep_node);
                add_edges(&mut data.unshared_edges, &mut data.new.edges, edges);
                data.new.fingerprints.push(fingerprint);
                let dep_node_index = data.hybrid_indices.push(new_index.into());
                entry.insert(dep_node_index);
                dep_node_index
            }
        }
    }

    fn intern_red_node(
        &self,
        prev_graph: &PreviousDepGraph<K>,
        prev_index: SerializedDepNodeIndex,
        edges: EdgesVec,
        fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        self.debug_assert_not_in_new_nodes(prev_graph, prev_index);

        let mut prev_index_to_index = self.prev_index_to_index.lock();

        match prev_index_to_index[prev_index] {
            Some(dep_node_index) => dep_node_index,
            None => {
                let data = &mut *self.data.lock();
                let red_index = data.red.node_indices.push(prev_index);
                add_edges(&mut data.unshared_edges, &mut data.red.edges, edges);
                data.red.fingerprints.push(fingerprint);
                let dep_node_index = data.hybrid_indices.push(red_index.into());
                prev_index_to_index[prev_index] = Some(dep_node_index);
                dep_node_index
            }
        }
    }

    fn intern_light_green_node(
        &self,
        prev_graph: &PreviousDepGraph<K>,
        prev_index: SerializedDepNodeIndex,
        edges: EdgesVec,
    ) -> DepNodeIndex {
        self.debug_assert_not_in_new_nodes(prev_graph, prev_index);

        let mut prev_index_to_index = self.prev_index_to_index.lock();

        match prev_index_to_index[prev_index] {
            Some(dep_node_index) => dep_node_index,
            None => {
                let data = &mut *self.data.lock();
                let light_green_index = data.light_green.node_indices.push(prev_index);
                add_edges(&mut data.unshared_edges, &mut data.light_green.edges, edges);
                let dep_node_index = data.hybrid_indices.push(light_green_index.into());
                prev_index_to_index[prev_index] = Some(dep_node_index);
                dep_node_index
            }
        }
    }

    fn intern_dark_green_node(
        &self,
        prev_graph: &PreviousDepGraph<K>,
        prev_index: SerializedDepNodeIndex,
    ) -> DepNodeIndex {
        self.debug_assert_not_in_new_nodes(prev_graph, prev_index);

        let mut prev_index_to_index = self.prev_index_to_index.lock();

        match prev_index_to_index[prev_index] {
            Some(dep_node_index) => dep_node_index,
            None => {
                let mut data = self.data.lock();
                let dep_node_index = data.hybrid_indices.push(prev_index.into());
                prev_index_to_index[prev_index] = Some(dep_node_index);
                dep_node_index
            }
        }
    }

    #[inline]
    fn debug_assert_not_in_new_nodes(
        &self,
        prev_graph: &PreviousDepGraph<K>,
        prev_index: SerializedDepNodeIndex,
    ) {
        let node = &prev_graph.index_to_node(prev_index);
        debug_assert!(
            !self.new_node_to_index.get_shard_by_value(node).lock().contains_key(node),
            "node from previous graph present in new node collection"
        );
    }
}

#[inline]
fn add_edges<I: Idx>(
    edges: &mut IndexVec<EdgeIndex, DepNodeIndex>,
    edge_indices: &mut IndexVec<I, Range<EdgeIndex>>,
    new_edges: EdgesVec,
) {
    let start = edges.next_index();
    edges.extend(new_edges);
    let end = edges.next_index();
    edge_indices.push(start..end);
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

// A data structure that stores Option<DepNodeColor> values as a contiguous
// array, using one u32 per entry.
struct DepNodeColorMap {
    values: IndexVec<SerializedDepNodeIndex, AtomicU32>,
}

const COMPRESSED_NONE: u32 = 0;
const COMPRESSED_RED: u32 = 1;
const COMPRESSED_FIRST_GREEN: u32 = 2;

impl DepNodeColorMap {
    fn new(size: usize) -> DepNodeColorMap {
        DepNodeColorMap { values: (0..size).map(|_| AtomicU32::new(COMPRESSED_NONE)).collect() }
    }

    #[inline]
    fn get(&self, index: SerializedDepNodeIndex) -> Option<DepNodeColor> {
        match self.values[index].load(Ordering::Acquire) {
            COMPRESSED_NONE => None,
            COMPRESSED_RED => Some(DepNodeColor::Red),
            value => {
                Some(DepNodeColor::Green(DepNodeIndex::from_u32(value - COMPRESSED_FIRST_GREEN)))
            }
        }
    }

    fn insert(&self, index: SerializedDepNodeIndex, color: DepNodeColor) {
        self.values[index].store(
            match color {
                DepNodeColor::Red => COMPRESSED_RED,
                DepNodeColor::Green(index) => index.as_u32() + COMPRESSED_FIRST_GREEN,
            },
            Ordering::Release,
        )
    }
}
