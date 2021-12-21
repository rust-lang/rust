use parking_lot::Mutex;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::profiling::{EventId, QueryInvocationId, SelfProfilerRef};
use rustc_data_structures::sharded::{self, Sharded};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::{AtomicU32, AtomicU64, Lock, Lrc, Ordering};
use rustc_index::vec::IndexVec;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use smallvec::{smallvec, SmallVec};
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::Ordering::Relaxed;

use super::query::DepGraphQuery;
use super::serialized::{GraphEncoder, SerializedDepGraph, SerializedDepNodeIndex};
use super::{DepContext, DepKind, DepNode, HasDepContext, WorkProductId};
use crate::ich::StableHashingContext;
use crate::query::{QueryContext, QuerySideEffects};

#[cfg(debug_assertions)]
use {super::debug::EdgeFilter, std::env};

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
    pub const SINGLETON_DEPENDENCYLESS_ANON_NODE: DepNodeIndex = DepNodeIndex::from_u32(0);
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
    previous: SerializedDepGraph<K>,

    colors: DepNodeColorMap,

    processed_side_effects: Mutex<FxHashSet<DepNodeIndex>>,

    /// When we load, there may be `.o` files, cached MIR, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: FxHashMap<WorkProductId, WorkProduct>,

    dep_node_debug: Lock<FxHashMap<DepNode<K>, String>>,

    /// Used by incremental compilation tests to assert that
    /// a particular query result was decoded from disk
    /// (not just marked green)
    debug_loaded_from_disk: Lock<FxHashSet<DepNode<K>>>,
}

pub fn hash_result<R>(hcx: &mut StableHashingContext<'_>, result: &R) -> Fingerprint
where
    R: for<'a> HashStable<StableHashingContext<'a>>,
{
    let mut stable_hasher = StableHasher::new();
    result.hash_stable(hcx, &mut stable_hasher);
    stable_hasher.finish()
}

impl<K: DepKind> DepGraph<K> {
    pub fn new(
        profiler: &SelfProfilerRef,
        prev_graph: SerializedDepGraph<K>,
        prev_work_products: FxHashMap<WorkProductId, WorkProduct>,
        encoder: FileEncoder,
        record_graph: bool,
        record_stats: bool,
    ) -> DepGraph<K> {
        let prev_graph_node_count = prev_graph.node_count();

        let current = CurrentDepGraph::new(
            profiler,
            prev_graph_node_count,
            encoder,
            record_graph,
            record_stats,
        );

        // Instantiate a dependy-less node only once for anonymous queries.
        let _green_node_index = current.intern_new_node(
            profiler,
            DepNode { kind: DepKind::NULL, hash: current.anon_id_seed.into() },
            smallvec![],
            Fingerprint::ZERO,
        );
        debug_assert_eq!(_green_node_index, DepNodeIndex::SINGLETON_DEPENDENCYLESS_ANON_NODE);

        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: prev_work_products,
                dep_node_debug: Default::default(),
                current,
                processed_side_effects: Default::default(),
                previous: prev_graph,
                colors: DepNodeColorMap::new(prev_graph_node_count),
                debug_loaded_from_disk: Default::default(),
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

    pub fn with_query(&self, f: impl Fn(&DepGraphQuery<K>)) {
        if let Some(data) = &self.data {
            data.current.encoder.borrow().with_query(f)
        }
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
    pub fn with_task<Ctxt: HasDepContext<DepKind = K>, A: Debug, R>(
        &self,
        key: DepNode<K>,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        hash_result: Option<fn(&mut StableHashingContext<'_>, &R) -> Fingerprint>,
    ) -> (R, DepNodeIndex) {
        if self.is_fully_enabled() {
            self.with_task_impl(key, cx, arg, task, hash_result)
        } else {
            // Incremental compilation is turned off. We just execute the task
            // without tracking. We still provide a dep-node index that uniquely
            // identifies the task so that we have a cheap way of referring to
            // the query for self-profiling.
            (task(cx, arg), self.next_virtual_depnode_index())
        }
    }

    fn with_task_impl<Ctxt: HasDepContext<DepKind = K>, A: Debug, R>(
        &self,
        key: DepNode<K>,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        hash_result: Option<fn(&mut StableHashingContext<'_>, &R) -> Fingerprint>,
    ) -> (R, DepNodeIndex) {
        // This function is only called when the graph is enabled.
        let data = self.data.as_ref().unwrap();

        // If the following assertion triggers, it can have two reasons:
        // 1. Something is wrong with DepNode creation, either here or
        //    in `DepGraph::try_mark_green()`.
        // 2. Two distinct query keys get mapped to the same `DepNode`
        //    (see for example #48923).
        assert!(
            !self.dep_node_exists(&key),
            "forcing query with already existing `DepNode`\n\
                 - query-key: {:?}\n\
                 - dep-node: {:?}",
            arg,
            key
        );

        let task_deps = if cx.dep_context().is_eval_always(key.kind) {
            None
        } else {
            Some(Lock::new(TaskDeps {
                #[cfg(debug_assertions)]
                node: Some(key),
                reads: SmallVec::new(),
                read_set: Default::default(),
                phantom_data: PhantomData,
            }))
        };
        let result = K::with_deps(task_deps.as_ref(), || task(cx, arg));
        let edges = task_deps.map_or_else(|| smallvec![], |lock| lock.into_inner().reads);

        let dcx = cx.dep_context();
        let hashing_timer = dcx.profiler().incr_result_hashing();
        let current_fingerprint = hash_result.map(|f| {
            let mut hcx = dcx.create_stable_hashing_context();
            f(&mut hcx, &result)
        });

        let print_status = cfg!(debug_assertions) && dcx.sess().opts.debugging_opts.dep_tasks;

        // Intern the new `DepNode`.
        let (dep_node_index, prev_and_color) = data.current.intern_node(
            dcx.profiler(),
            &data.previous,
            key,
            edges,
            current_fingerprint,
            print_status,
        );

        hashing_timer.finish_with_query_invocation_id(dep_node_index.into());

        if let Some((prev_index, color)) = prev_and_color {
            debug_assert!(
                data.colors.get(prev_index).is_none(),
                "DepGraph::with_task() - Duplicate DepNodeColor \
                            insertion for {:?}",
                key
            );

            data.colors.insert(prev_index, color);
        }

        (result, dep_node_index)
    }

    /// Executes something within an "anonymous" task, that is, a task the
    /// `DepNode` of which is determined by the list of inputs it read from.
    pub fn with_anon_task<Ctxt: DepContext<DepKind = K>, OP, R>(
        &self,
        cx: Ctxt,
        dep_kind: K,
        op: OP,
    ) -> (R, DepNodeIndex)
    where
        OP: FnOnce() -> R,
    {
        debug_assert!(!cx.is_eval_always(dep_kind));

        if let Some(ref data) = self.data {
            let task_deps = Lock::new(TaskDeps::default());
            let result = K::with_deps(Some(&task_deps), op);
            let task_deps = task_deps.into_inner();
            let task_deps = task_deps.reads;

            let dep_node_index = match task_deps.len() {
                0 => {
                    // Because the dep-node id of anon nodes is computed from the sets of its
                    // dependencies we already know what the ID of this dependency-less node is
                    // going to be (i.e. equal to the precomputed
                    // `SINGLETON_DEPENDENCYLESS_ANON_NODE`). As a consequence we can skip creating
                    // a `StableHasher` and sending the node through interning.
                    DepNodeIndex::SINGLETON_DEPENDENCYLESS_ANON_NODE
                }
                1 => {
                    // When there is only one dependency, don't bother creating a node.
                    task_deps[0]
                }
                _ => {
                    // The dep node indices are hashed here instead of hashing the dep nodes of the
                    // dependencies. These indices may refer to different nodes per session, but this isn't
                    // a problem here because we that ensure the final dep node hash is per session only by
                    // combining it with the per session random number `anon_id_seed`. This hash only need
                    // to map the dependencies to a single value on a per session basis.
                    let mut hasher = StableHasher::new();
                    task_deps.hash(&mut hasher);

                    let target_dep_node = DepNode {
                        kind: dep_kind,
                        // Fingerprint::combine() is faster than sending Fingerprint
                        // through the StableHasher (at least as long as StableHasher
                        // is so slow).
                        hash: data.current.anon_id_seed.combine(hasher.finish()).into(),
                    };

                    data.current.intern_new_node(
                        cx.profiler(),
                        target_dep_node,
                        task_deps,
                        Fingerprint::ZERO,
                    )
                }
            };

            (result, dep_node_index)
        } else {
            (op(), self.next_virtual_depnode_index())
        }
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
                                    let src = forbidden_edge.index_to_node.lock()[&dep_node_index];
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

    pub fn mark_debug_loaded_from_disk(&self, dep_node: DepNode<K>) {
        self.data.as_ref().unwrap().debug_loaded_from_disk.lock().insert(dep_node);
    }

    pub fn debug_was_loaded_from_disk(&self, dep_node: DepNode<K>) -> bool {
        self.data.as_ref().unwrap().debug_loaded_from_disk.lock().contains(&dep_node)
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

    fn node_color(&self, dep_node: &DepNode<K>) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            if let Some(prev_index) = data.previous.node_to_index_opt(dep_node) {
                return data.colors.get(prev_index);
            } else {
                // This is a node that did not exist in the previous compilation session.
                return None;
            }
        }

        None
    }

    /// Try to mark a node index for the node dep_node.
    ///
    /// A node will have an index, when it's already been marked green, or when we can mark it
    /// green. This function will mark the current task as a reader of the specified node, when
    /// a node index can be found for that node.
    pub fn try_mark_green<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        dep_node: &DepNode<K>,
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        debug_assert!(!tcx.dep_context().is_eval_always(dep_node.kind));

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

    fn try_mark_parent_green<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        parent_dep_node_index: SerializedDepNodeIndex,
        dep_node: &DepNode<K>,
    ) -> Option<()> {
        let dep_dep_node_color = data.colors.get(parent_dep_node_index);
        let dep_dep_node = &data.previous.index_to_node(parent_dep_node_index);

        match dep_dep_node_color {
            Some(DepNodeColor::Green(_)) => {
                // This dependency has been marked as green before, we are
                // still fine and can continue with checking the other
                // dependencies.
                debug!(
                    "try_mark_previous_green({:?}) --- found dependency {:?} to \
                            be immediately green",
                    dep_node, dep_dep_node,
                );
                return Some(());
            }
            Some(DepNodeColor::Red) => {
                // We found a dependency the value of which has changed
                // compared to the previous compilation session. We cannot
                // mark the DepNode as green and also don't need to bother
                // with checking any of the other dependencies.
                debug!(
                    "try_mark_previous_green({:?}) - END - dependency {:?} was immediately red",
                    dep_node, dep_dep_node,
                );
                return None;
            }
            None => {}
        }

        // We don't know the state of this dependency. If it isn't
        // an eval_always node, let's try to mark it green recursively.
        if !tcx.dep_context().is_eval_always(dep_dep_node.kind) {
            debug!(
                "try_mark_previous_green({:?}) --- state of dependency {:?} ({}) \
                                 is unknown, trying to mark it green",
                dep_node, dep_dep_node, dep_dep_node.hash,
            );

            let node_index =
                self.try_mark_previous_green(tcx, data, parent_dep_node_index, dep_dep_node);
            if node_index.is_some() {
                debug!(
                    "try_mark_previous_green({:?}) --- managed to MARK dependency {:?} as green",
                    dep_node, dep_dep_node
                );
                return Some(());
            }
        }

        // We failed to mark it green, so we try to force the query.
        debug!(
            "try_mark_previous_green({:?}) --- trying to force dependency {:?}",
            dep_node, dep_dep_node
        );
        if !tcx.dep_context().try_force_from_dep_node(*dep_dep_node) {
            // The DepNode could not be forced.
            debug!(
                "try_mark_previous_green({:?}) - END - dependency {:?} could not be forced",
                dep_node, dep_dep_node
            );
            return None;
        }

        let dep_dep_node_color = data.colors.get(parent_dep_node_index);

        match dep_dep_node_color {
            Some(DepNodeColor::Green(_)) => {
                debug!(
                    "try_mark_previous_green({:?}) --- managed to FORCE dependency {:?} to green",
                    dep_node, dep_dep_node
                );
                return Some(());
            }
            Some(DepNodeColor::Red) => {
                debug!(
                    "try_mark_previous_green({:?}) - END - dependency {:?} was red after forcing",
                    dep_node, dep_dep_node
                );
                return None;
            }
            None => {}
        }

        if !tcx.dep_context().sess().has_errors_or_delayed_span_bugs() {
            panic!("try_mark_previous_green() - Forcing the DepNode should have set its color")
        }

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
            "try_mark_previous_green({:?}) - END - dependency {:?} resulted in compilation error",
            dep_node, dep_dep_node
        );
        return None;
    }

    /// Try to mark a dep-node which existed in the previous compilation session as green.
    fn try_mark_previous_green<Ctxt: QueryContext<DepKind = K>>(
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
        debug_assert!(!tcx.dep_context().is_eval_always(dep_node.kind));

        debug_assert_eq!(data.previous.index_to_node(prev_dep_node_index), *dep_node);

        let prev_deps = data.previous.edge_targets_from(prev_dep_node_index);

        for &dep_dep_node_index in prev_deps {
            self.try_mark_parent_green(tcx, data, dep_dep_node_index, dep_node)?
        }

        // If we got here without hitting a `return` that means that all
        // dependencies of this DepNode could be marked as green. Therefore we
        // can also mark this DepNode as green.

        // There may be multiple threads trying to mark the same dep node green concurrently

        // We allocating an entry for the node in the current dependency graph and
        // adding all the appropriate edges imported from the previous graph
        let dep_node_index = data.current.promote_node_and_deps_to_current(
            tcx.dep_context().profiler(),
            &data.previous,
            prev_dep_node_index,
        );

        // ... emitting any stored diagnostic ...

        // FIXME: Store the fact that a node has diagnostics in a bit in the dep graph somewhere
        // Maybe store a list on disk and encode this fact in the DepNodeState
        let side_effects = tcx.load_side_effects(prev_dep_node_index);

        #[cfg(not(parallel_compiler))]
        debug_assert!(
            data.colors.get(prev_dep_node_index).is_none(),
            "DepGraph::try_mark_previous_green() - Duplicate DepNodeColor \
                      insertion for {:?}",
            dep_node
        );

        if unlikely!(!side_effects.is_empty()) {
            self.emit_side_effects(tcx, data, dep_node_index, side_effects);
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
    fn emit_side_effects<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    ) {
        let mut processed = data.processed_side_effects.lock();

        if processed.insert(dep_node_index) {
            // We were the first to insert the node in the set so this thread
            // must process side effects

            // Promote the previous diagnostics to the current session.
            tcx.store_side_effects(dep_node_index, side_effects.clone());

            let handle = tcx.dep_context().sess().diagnostic();

            for diagnostic in side_effects.diagnostics {
                handle.emit_diagnostic(&diagnostic);
            }
        }
    }

    // Returns true if the given node has been marked as red during the
    // current compilation session. Used in various assertions
    pub fn is_red(&self, dep_node: &DepNode<K>) -> bool {
        self.node_color(dep_node) == Some(DepNodeColor::Red)
    }

    // Returns true if the given node has been marked as green during the
    // current compilation session. Used in various assertions
    pub fn is_green(&self, dep_node: &DepNode<K>) -> bool {
        self.node_color(dep_node).map_or(false, |c| c.is_green())
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
                    tcx.try_load_from_on_disk_cache(dep_node);
                }
                None | Some(DepNodeColor::Red) => {
                    // We can skip red nodes because a node can only be marked
                    // as red if the query result was recomputed and thus is
                    // already in memory.
                }
            }
        }
    }

    pub fn print_incremental_info(&self) {
        if let Some(data) = &self.data {
            data.current.encoder.borrow().print_incremental_info(
                data.current.total_read_count.load(Relaxed),
                data.current.total_duplicate_read_count.load(Relaxed),
            )
        }
    }

    pub fn encode(&self, profiler: &SelfProfilerRef) -> FileEncodeResult {
        if let Some(data) = &self.data {
            data.current.encoder.steal().finish(profiler)
        } else {
            Ok(())
        }
    }

    pub(crate) fn next_virtual_depnode_index(&self) -> DepNodeIndex {
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

// Index type for `DepNodeData`'s edges.
rustc_index::newtype_index! {
    struct EdgeIndex { .. }
}

/// `CurrentDepGraph` stores the dependency graph for the current session. It
/// will be populated as we run queries or tasks. We never remove nodes from the
/// graph: they are only added.
///
/// The nodes in it are identified by a `DepNodeIndex`. We avoid keeping the nodes
/// in memory.  This is important, because these graph structures are some of the
/// largest in the compiler.
///
/// For this reason, we avoid storing `DepNode`s more than once as map
/// keys. The `new_node_to_index` map only contains nodes not in the previous
/// graph, and we map nodes in the previous graph to indices via a two-step
/// mapping. `SerializedDepGraph` maps from `DepNode` to `SerializedDepNodeIndex`,
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
pub(super) struct CurrentDepGraph<K: DepKind> {
    encoder: Steal<GraphEncoder<K>>,
    new_node_to_index: Sharded<FxHashMap<DepNode<K>, DepNodeIndex>>,
    prev_index_to_index: Lock<IndexVec<SerializedDepNodeIndex, Option<DepNodeIndex>>>,

    /// Used to trap when a specific edge is added to the graph.
    /// This is used for debug purposes and is only active with `debug_assertions`.
    #[cfg(debug_assertions)]
    forbidden_edge: Option<EdgeFilter<K>>,

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

    /// The cached event id for profiling node interning. This saves us
    /// from having to look up the event id every time we intern a node
    /// which may incur too much overhead.
    /// This will be None if self-profiling is disabled.
    node_intern_event_id: Option<EventId>,
}

impl<K: DepKind> CurrentDepGraph<K> {
    fn new(
        profiler: &SelfProfilerRef,
        prev_graph_node_count: usize,
        encoder: FileEncoder,
        record_graph: bool,
        record_stats: bool,
    ) -> CurrentDepGraph<K> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let nanos = duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64;
        let mut stable_hasher = StableHasher::new();
        nanos.hash(&mut stable_hasher);

        #[cfg(debug_assertions)]
        let forbidden_edge = match env::var("RUST_FORBID_DEP_GRAPH_EDGE") {
            Ok(s) => match EdgeFilter::new(&s) {
                Ok(f) => Some(f),
                Err(err) => panic!("RUST_FORBID_DEP_GRAPH_EDGE invalid: {}", err),
            },
            Err(_) => None,
        };

        // We store a large collection of these in `prev_index_to_index` during
        // non-full incremental builds, and want to ensure that the element size
        // doesn't inadvertently increase.
        static_assert_size!(Option<DepNodeIndex>, 4);

        let new_node_count_estimate = 102 * prev_graph_node_count / 100 + 200;

        let node_intern_event_id = profiler
            .get_or_alloc_cached_string("incr_comp_intern_dep_graph_node")
            .map(EventId::from_label);

        CurrentDepGraph {
            encoder: Steal::new(GraphEncoder::new(
                encoder,
                prev_graph_node_count,
                record_graph,
                record_stats,
            )),
            new_node_to_index: Sharded::new(|| {
                FxHashMap::with_capacity_and_hasher(
                    new_node_count_estimate / sharded::SHARDS,
                    Default::default(),
                )
            }),
            prev_index_to_index: Lock::new(IndexVec::from_elem_n(None, prev_graph_node_count)),
            anon_id_seed: stable_hasher.finish(),
            #[cfg(debug_assertions)]
            forbidden_edge,
            total_read_count: AtomicU64::new(0),
            total_duplicate_read_count: AtomicU64::new(0),
            node_intern_event_id,
        }
    }

    #[cfg(debug_assertions)]
    fn record_edge(&self, dep_node_index: DepNodeIndex, key: DepNode<K>) {
        if let Some(forbidden_edge) = &self.forbidden_edge {
            forbidden_edge.index_to_node.lock().insert(dep_node_index, key);
        }
    }

    /// Writes the node to the current dep-graph and allocates a `DepNodeIndex` for it.
    /// Assumes that this is a node that has no equivalent in the previous dep-graph.
    fn intern_new_node(
        &self,
        profiler: &SelfProfilerRef,
        key: DepNode<K>,
        edges: EdgesVec,
        current_fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        match self.new_node_to_index.get_shard_by_value(&key).lock().entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let dep_node_index =
                    self.encoder.borrow().send(profiler, key, current_fingerprint, edges);
                entry.insert(dep_node_index);
                #[cfg(debug_assertions)]
                self.record_edge(dep_node_index, key);
                dep_node_index
            }
        }
    }

    fn intern_node(
        &self,
        profiler: &SelfProfilerRef,
        prev_graph: &SerializedDepGraph<K>,
        key: DepNode<K>,
        edges: EdgesVec,
        fingerprint: Option<Fingerprint>,
        print_status: bool,
    ) -> (DepNodeIndex, Option<(SerializedDepNodeIndex, DepNodeColor)>) {
        let print_status = cfg!(debug_assertions) && print_status;

        // Get timer for profiling `DepNode` interning
        let _node_intern_timer =
            self.node_intern_event_id.map(|eid| profiler.generic_activity_with_event_id(eid));

        if let Some(prev_index) = prev_graph.node_to_index_opt(&key) {
            // Determine the color and index of the new `DepNode`.
            if let Some(fingerprint) = fingerprint {
                if fingerprint == prev_graph.fingerprint_by_index(prev_index) {
                    if print_status {
                        eprintln!("[task::green] {:?}", key);
                    }

                    // This is a green node: it existed in the previous compilation,
                    // its query was re-executed, and it has the same result as before.
                    let mut prev_index_to_index = self.prev_index_to_index.lock();

                    let dep_node_index = match prev_index_to_index[prev_index] {
                        Some(dep_node_index) => dep_node_index,
                        None => {
                            let dep_node_index =
                                self.encoder.borrow().send(profiler, key, fingerprint, edges);
                            prev_index_to_index[prev_index] = Some(dep_node_index);
                            dep_node_index
                        }
                    };

                    #[cfg(debug_assertions)]
                    self.record_edge(dep_node_index, key);
                    (dep_node_index, Some((prev_index, DepNodeColor::Green(dep_node_index))))
                } else {
                    if print_status {
                        eprintln!("[task::red] {:?}", key);
                    }

                    // This is a red node: it existed in the previous compilation, its query
                    // was re-executed, but it has a different result from before.
                    let mut prev_index_to_index = self.prev_index_to_index.lock();

                    let dep_node_index = match prev_index_to_index[prev_index] {
                        Some(dep_node_index) => dep_node_index,
                        None => {
                            let dep_node_index =
                                self.encoder.borrow().send(profiler, key, fingerprint, edges);
                            prev_index_to_index[prev_index] = Some(dep_node_index);
                            dep_node_index
                        }
                    };

                    #[cfg(debug_assertions)]
                    self.record_edge(dep_node_index, key);
                    (dep_node_index, Some((prev_index, DepNodeColor::Red)))
                }
            } else {
                if print_status {
                    eprintln!("[task::unknown] {:?}", key);
                }

                // This is a red node, effectively: it existed in the previous compilation
                // session, its query was re-executed, but it doesn't compute a result hash
                // (i.e. it represents a `no_hash` query), so we have no way of determining
                // whether or not the result was the same as before.
                let mut prev_index_to_index = self.prev_index_to_index.lock();

                let dep_node_index = match prev_index_to_index[prev_index] {
                    Some(dep_node_index) => dep_node_index,
                    None => {
                        let dep_node_index =
                            self.encoder.borrow().send(profiler, key, Fingerprint::ZERO, edges);
                        prev_index_to_index[prev_index] = Some(dep_node_index);
                        dep_node_index
                    }
                };

                #[cfg(debug_assertions)]
                self.record_edge(dep_node_index, key);
                (dep_node_index, Some((prev_index, DepNodeColor::Red)))
            }
        } else {
            if print_status {
                eprintln!("[task::new] {:?}", key);
            }

            let fingerprint = fingerprint.unwrap_or(Fingerprint::ZERO);

            // This is a new node: it didn't exist in the previous compilation session.
            let dep_node_index = self.intern_new_node(profiler, key, edges, fingerprint);

            (dep_node_index, None)
        }
    }

    fn promote_node_and_deps_to_current(
        &self,
        profiler: &SelfProfilerRef,
        prev_graph: &SerializedDepGraph<K>,
        prev_index: SerializedDepNodeIndex,
    ) -> DepNodeIndex {
        self.debug_assert_not_in_new_nodes(prev_graph, prev_index);

        let mut prev_index_to_index = self.prev_index_to_index.lock();

        match prev_index_to_index[prev_index] {
            Some(dep_node_index) => dep_node_index,
            None => {
                let key = prev_graph.index_to_node(prev_index);
                let dep_node_index = self.encoder.borrow().send(
                    profiler,
                    key,
                    prev_graph.fingerprint_by_index(prev_index),
                    prev_graph
                        .edge_targets_from(prev_index)
                        .iter()
                        .map(|i| prev_index_to_index[*i].unwrap())
                        .collect(),
                );
                prev_index_to_index[prev_index] = Some(dep_node_index);
                #[cfg(debug_assertions)]
                self.record_edge(dep_node_index, key);
                dep_node_index
            }
        }
    }

    #[inline]
    fn debug_assert_not_in_new_nodes(
        &self,
        prev_graph: &SerializedDepGraph<K>,
        prev_index: SerializedDepNodeIndex,
    ) {
        let node = &prev_graph.index_to_node(prev_index);
        debug_assert!(
            !self.new_node_to_index.get_shard_by_value(node).lock().contains_key(node),
            "node from previous graph present in new node collection"
        );
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
