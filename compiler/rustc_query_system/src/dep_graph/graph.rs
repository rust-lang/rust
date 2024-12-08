use std::assert_matches::assert_matches;
use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::profiling::{QueryInvocationId, SelfProfilerRef};
use rustc_data_structures::sharded::{self, Sharded};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{AtomicU32, AtomicU64, Lock, Lrc};
use rustc_data_structures::unord::UnordMap;
use rustc_index::IndexVec;
use rustc_macros::{Decodable, Encodable};
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use tracing::{debug, instrument};
#[cfg(debug_assertions)]
use {super::debug::EdgeFilter, std::env};

use super::query::DepGraphQuery;
use super::serialized::{GraphEncoder, SerializedDepGraph, SerializedDepNodeIndex};
use super::{DepContext, DepKind, DepNode, Deps, HasDepContext, WorkProductId};
use crate::dep_graph::edges::EdgesVec;
use crate::ich::StableHashingContext;
use crate::query::{QueryContext, QuerySideEffects};

#[derive(Clone)]
pub struct DepGraph<D: Deps> {
    data: Option<Lrc<DepGraphData<D>>>,

    /// This field is used for assigning DepNodeIndices when running in
    /// non-incremental mode. Even in non-incremental mode we make sure that
    /// each task has a `DepNodeIndex` that uniquely identifies it. This unique
    /// ID is used for self-profiling.
    virtual_dep_node_index: Lrc<AtomicU32>,
}

rustc_index::newtype_index! {
    pub struct DepNodeIndex {}
}

// We store a large collection of these in `prev_index_to_index` during
// non-full incremental builds, and want to ensure that the element size
// doesn't inadvertently increase.
rustc_data_structures::static_assert_size!(Option<DepNodeIndex>, 4);

impl DepNodeIndex {
    const SINGLETON_DEPENDENCYLESS_ANON_NODE: DepNodeIndex = DepNodeIndex::ZERO;
    pub const FOREVER_RED_NODE: DepNodeIndex = DepNodeIndex::from_u32(1);
}

impl From<DepNodeIndex> for QueryInvocationId {
    #[inline(always)]
    fn from(dep_node_index: DepNodeIndex) -> Self {
        QueryInvocationId(dep_node_index.as_u32())
    }
}

pub struct MarkFrame<'a> {
    index: SerializedDepNodeIndex,
    parent: Option<&'a MarkFrame<'a>>,
}

enum DepNodeColor {
    Red,
    Green(DepNodeIndex),
}

impl DepNodeColor {
    #[inline]
    fn is_green(self) -> bool {
        match self {
            DepNodeColor::Red => false,
            DepNodeColor::Green(_) => true,
        }
    }
}

pub(crate) struct DepGraphData<D: Deps> {
    /// The new encoding of the dependency graph, optimized for red/green
    /// tracking. The `current` field is the dependency graph of only the
    /// current compilation session: We don't merge the previous dep-graph into
    /// current one anymore, but we do reference shared data to save space.
    current: CurrentDepGraph<D>,

    /// The dep-graph from the previous compilation session. It contains all
    /// nodes and edges as well as all fingerprints of nodes that have them.
    previous: Arc<SerializedDepGraph>,

    colors: DepNodeColorMap,

    processed_side_effects: Lock<FxHashSet<DepNodeIndex>>,

    /// When we load, there may be `.o` files, cached MIR, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: WorkProductMap,

    dep_node_debug: Lock<FxHashMap<DepNode, String>>,

    /// Used by incremental compilation tests to assert that
    /// a particular query result was decoded from disk
    /// (not just marked green)
    debug_loaded_from_disk: Lock<FxHashSet<DepNode>>,
}

pub fn hash_result<R>(hcx: &mut StableHashingContext<'_>, result: &R) -> Fingerprint
where
    R: for<'a> HashStable<StableHashingContext<'a>>,
{
    let mut stable_hasher = StableHasher::new();
    result.hash_stable(hcx, &mut stable_hasher);
    stable_hasher.finish()
}

impl<D: Deps> DepGraph<D> {
    pub fn new(
        profiler: &SelfProfilerRef,
        prev_graph: Arc<SerializedDepGraph>,
        prev_work_products: WorkProductMap,
        encoder: FileEncoder,
        record_graph: bool,
        record_stats: bool,
    ) -> DepGraph<D> {
        let prev_graph_node_count = prev_graph.node_count();

        let current = CurrentDepGraph::new(
            profiler,
            prev_graph_node_count,
            encoder,
            record_graph,
            record_stats,
            Arc::clone(&prev_graph),
        );

        let colors = DepNodeColorMap::new(prev_graph_node_count);

        // Instantiate a dependy-less node only once for anonymous queries.
        let _green_node_index = current.intern_new_node(
            DepNode { kind: D::DEP_KIND_NULL, hash: current.anon_id_seed.into() },
            EdgesVec::new(),
            Fingerprint::ZERO,
        );
        assert_eq!(_green_node_index, DepNodeIndex::SINGLETON_DEPENDENCYLESS_ANON_NODE);

        // Instantiate a dependy-less red node only once for anonymous queries.
        let (red_node_index, red_node_prev_index_and_color) = current.intern_node(
            &prev_graph,
            DepNode { kind: D::DEP_KIND_RED, hash: Fingerprint::ZERO.into() },
            EdgesVec::new(),
            None,
        );
        assert_eq!(red_node_index, DepNodeIndex::FOREVER_RED_NODE);
        match red_node_prev_index_and_color {
            None => {
                // This is expected when we have no previous compilation session.
                assert!(prev_graph_node_count == 0);
            }
            Some((prev_red_node_index, DepNodeColor::Red)) => {
                assert_eq!(prev_red_node_index.as_usize(), red_node_index.as_usize());
                colors.insert(prev_red_node_index, DepNodeColor::Red);
            }
            Some((_, DepNodeColor::Green(_))) => {
                // There must be a logic error somewhere if we hit this branch.
                panic!("DepNodeIndex::FOREVER_RED_NODE evaluated to DepNodeColor::Green")
            }
        }

        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: prev_work_products,
                dep_node_debug: Default::default(),
                current,
                processed_side_effects: Default::default(),
                previous: prev_graph,
                colors,
                debug_loaded_from_disk: Default::default(),
            })),
            virtual_dep_node_index: Lrc::new(AtomicU32::new(0)),
        }
    }

    pub fn new_disabled() -> DepGraph<D> {
        DepGraph { data: None, virtual_dep_node_index: Lrc::new(AtomicU32::new(0)) }
    }

    #[inline]
    pub(crate) fn data(&self) -> Option<&DepGraphData<D>> {
        self.data.as_deref()
    }

    /// Returns `true` if we are actually building the full dep-graph, and `false` otherwise.
    #[inline]
    pub fn is_fully_enabled(&self) -> bool {
        self.data.is_some()
    }

    pub fn with_query(&self, f: impl Fn(&DepGraphQuery)) {
        if let Some(data) = &self.data {
            data.current.encoder.with_query(f)
        }
    }

    pub fn assert_ignored(&self) {
        if let Some(..) = self.data {
            D::read_deps(|task_deps| {
                assert_matches!(
                    task_deps,
                    TaskDepsRef::Ignore,
                    "expected no task dependency tracking"
                );
            })
        }
    }

    pub fn with_ignore<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R,
    {
        D::with_deps(TaskDepsRef::Ignore, op)
    }

    /// Used to wrap the deserialization of a query result from disk,
    /// This method enforces that no new `DepNodes` are created during
    /// query result deserialization.
    ///
    /// Enforcing this makes the query dep graph simpler - all nodes
    /// must be created during the query execution, and should be
    /// created from inside the 'body' of a query (the implementation
    /// provided by a particular compiler crate).
    ///
    /// Consider the case of three queries `A`, `B`, and `C`, where
    /// `A` invokes `B` and `B` invokes `C`:
    ///
    /// `A -> B -> C`
    ///
    /// Suppose that decoding the result of query `B` required re-computing
    /// the query `C`. If we did not create a fresh `TaskDeps` when
    /// decoding `B`, we would still be using the `TaskDeps` for query `A`
    /// (if we needed to re-execute `A`). This would cause us to create
    /// a new edge `A -> C`. If this edge did not previously
    /// exist in the `DepGraph`, then we could end up with a different
    /// `DepGraph` at the end of compilation, even if there were no
    /// meaningful changes to the overall program (e.g. a newline was added).
    /// In addition, this edge might cause a subsequent compilation run
    /// to try to force `C` before marking other necessary nodes green. If
    /// `C` did not exist in the new compilation session, then we could
    /// get an ICE. Normally, we would have tried (and failed) to mark
    /// some other query green (e.g. `item_children`) which was used
    /// to obtain `C`, which would prevent us from ever trying to force
    /// a nonexistent `D`.
    ///
    /// It might be possible to enforce that all `DepNode`s read during
    /// deserialization already exist in the previous `DepGraph`. In
    /// the above example, we would invoke `D` during the deserialization
    /// of `B`. Since we correctly create a new `TaskDeps` from the decoding
    /// of `B`, this would result in an edge `B -> D`. If that edge already
    /// existed (with the same `DepPathHash`es), then it should be correct
    /// to allow the invocation of the query to proceed during deserialization
    /// of a query result. We would merely assert that the dep-graph fragment
    /// that would have been added by invoking `C` while decoding `B`
    /// is equivalent to the dep-graph fragment that we already instantiated for B
    /// (at the point where we successfully marked B as green).
    ///
    /// However, this would require additional complexity
    /// in the query infrastructure, and is not currently needed by the
    /// decoding of any query results. Should the need arise in the future,
    /// we should consider extending the query system with this functionality.
    pub fn with_query_deserialization<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R,
    {
        D::with_deps(TaskDepsRef::Forbid, op)
    }

    #[inline(always)]
    pub fn with_task<Ctxt: HasDepContext<Deps = D>, A: Debug, R>(
        &self,
        key: DepNode,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        hash_result: Option<fn(&mut StableHashingContext<'_>, &R) -> Fingerprint>,
    ) -> (R, DepNodeIndex) {
        match self.data() {
            Some(data) => data.with_task(key, cx, arg, task, hash_result),
            None => (task(cx, arg), self.next_virtual_depnode_index()),
        }
    }

    pub fn with_anon_task<Tcx: DepContext<Deps = D>, OP, R>(
        &self,
        cx: Tcx,
        dep_kind: DepKind,
        op: OP,
    ) -> (R, DepNodeIndex)
    where
        OP: FnOnce() -> R,
    {
        match self.data() {
            Some(data) => {
                let (result, index) = data.with_anon_task_inner(cx, dep_kind, op);
                self.read_index(index);
                (result, index)
            }
            None => (op(), self.next_virtual_depnode_index()),
        }
    }
}

impl<D: Deps> DepGraphData<D> {
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
    /// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/queries/incremental-compilation.html
    #[inline(always)]
    pub(crate) fn with_task<Ctxt: HasDepContext<Deps = D>, A: Debug, R>(
        &self,
        key: DepNode,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        hash_result: Option<fn(&mut StableHashingContext<'_>, &R) -> Fingerprint>,
    ) -> (R, DepNodeIndex) {
        // If the following assertion triggers, it can have two reasons:
        // 1. Something is wrong with DepNode creation, either here or
        //    in `DepGraph::try_mark_green()`.
        // 2. Two distinct query keys get mapped to the same `DepNode`
        //    (see for example #48923).
        assert!(
            !self.dep_node_exists(&key),
            "forcing query with already existing `DepNode`\n\
                 - query-key: {arg:?}\n\
                 - dep-node: {key:?}"
        );

        let with_deps = |task_deps| D::with_deps(task_deps, || task(cx, arg));
        let (result, edges) = if cx.dep_context().is_eval_always(key.kind) {
            (with_deps(TaskDepsRef::EvalAlways), EdgesVec::new())
        } else {
            let task_deps = Lock::new(TaskDeps {
                #[cfg(debug_assertions)]
                node: Some(key),
                reads: EdgesVec::new(),
                read_set: Default::default(),
                phantom_data: PhantomData,
            });
            (with_deps(TaskDepsRef::Allow(&task_deps)), task_deps.into_inner().reads)
        };

        let dcx = cx.dep_context();
        let hashing_timer = dcx.profiler().incr_result_hashing();
        let current_fingerprint =
            hash_result.map(|f| dcx.with_stable_hashing_context(|mut hcx| f(&mut hcx, &result)));

        // Intern the new `DepNode`.
        let (dep_node_index, prev_and_color) =
            self.current.intern_node(&self.previous, key, edges, current_fingerprint);

        hashing_timer.finish_with_query_invocation_id(dep_node_index.into());

        if let Some((prev_index, color)) = prev_and_color {
            debug_assert!(
                self.colors.get(prev_index).is_none(),
                "DepGraph::with_task() - Duplicate DepNodeColor \
                            insertion for {key:?}"
            );

            self.colors.insert(prev_index, color);
        }

        (result, dep_node_index)
    }

    /// Executes something within an "anonymous" task, that is, a task the
    /// `DepNode` of which is determined by the list of inputs it read from.
    ///
    /// NOTE: this does not actually count as a read of the DepNode here.
    /// Using the result of this task without reading the DepNode will result
    /// in untracked dependencies which may lead to ICEs as nodes are
    /// incorrectly marked green.
    ///
    /// FIXME: This could perhaps return a `WithDepNode` to ensure that the
    /// user of this function actually performs the read; we'll have to see
    /// how to make that work with `anon` in `execute_job_incr`, though.
    pub(crate) fn with_anon_task_inner<Tcx: DepContext<Deps = D>, OP, R>(
        &self,
        cx: Tcx,
        dep_kind: DepKind,
        op: OP,
    ) -> (R, DepNodeIndex)
    where
        OP: FnOnce() -> R,
    {
        debug_assert!(!cx.is_eval_always(dep_kind));

        let task_deps = Lock::new(TaskDeps::default());
        let result = D::with_deps(TaskDepsRef::Allow(&task_deps), op);
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
                    hash: self.current.anon_id_seed.combine(hasher.finish()).into(),
                };

                self.current.intern_new_node(target_dep_node, task_deps, Fingerprint::ZERO)
            }
        };

        (result, dep_node_index)
    }
}

impl<D: Deps> DepGraph<D> {
    #[inline]
    pub fn read_index(&self, dep_node_index: DepNodeIndex) {
        if let Some(ref data) = self.data {
            D::read_deps(|task_deps| {
                let mut task_deps = match task_deps {
                    TaskDepsRef::Allow(deps) => deps.lock(),
                    TaskDepsRef::EvalAlways => {
                        // We don't need to record dependencies of eval_always
                        // queries. They are re-evaluated unconditionally anyway.
                        return;
                    }
                    TaskDepsRef::Ignore => return,
                    TaskDepsRef::Forbid => {
                        // Reading is forbidden in this context. ICE with a useful error message.
                        panic_on_forbidden_read(data, dep_node_index)
                    }
                };
                let task_deps = &mut *task_deps;

                if cfg!(debug_assertions) {
                    data.current.total_read_count.fetch_add(1, Ordering::Relaxed);
                }

                // As long as we only have a low number of reads we can avoid doing a hash
                // insert and potentially allocating/reallocating the hashmap
                let new_read = if task_deps.reads.len() < EdgesVec::INLINE_CAPACITY {
                    task_deps.reads.iter().all(|other| *other != dep_node_index)
                } else {
                    task_deps.read_set.insert(dep_node_index)
                };
                if new_read {
                    task_deps.reads.push(dep_node_index);
                    if task_deps.reads.len() == EdgesVec::INLINE_CAPACITY {
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
                    data.current.total_duplicate_read_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        }
    }

    /// Create a node when we force-feed a value into the query cache.
    /// This is used to remove cycles during type-checking const generic parameters.
    ///
    /// As usual in the query system, we consider the current state of the calling query
    /// only depends on the list of dependencies up to now. As a consequence, the value
    /// that this query gives us can only depend on those dependencies too. Therefore,
    /// it is sound to use the current dependency set for the created node.
    ///
    /// During replay, the order of the nodes is relevant in the dependency graph.
    /// So the unchanged replay will mark the caller query before trying to mark this one.
    /// If there is a change to report, the caller query will be re-executed before this one.
    ///
    /// FIXME: If the code is changed enough for this node to be marked before requiring the
    /// caller's node, we suppose that those changes will be enough to mark this node red and
    /// force a recomputation using the "normal" way.
    pub fn with_feed_task<Ctxt: DepContext<Deps = D>, A: Debug, R: Debug>(
        &self,
        node: DepNode,
        cx: Ctxt,
        key: A,
        result: &R,
        hash_result: Option<fn(&mut StableHashingContext<'_>, &R) -> Fingerprint>,
    ) -> DepNodeIndex {
        if let Some(data) = self.data.as_ref() {
            // The caller query has more dependencies than the node we are creating. We may
            // encounter a case where this created node is marked as green, but the caller query is
            // subsequently marked as red or recomputed. In this case, we will end up feeding a
            // value to an existing node.
            //
            // For sanity, we still check that the loaded stable hash and the new one match.
            if let Some(prev_index) = data.previous.node_to_index_opt(&node) {
                let dep_node_index = data.current.prev_index_to_index.lock()[prev_index];
                if let Some(dep_node_index) = dep_node_index {
                    crate::query::incremental_verify_ich(
                        cx,
                        data,
                        result,
                        prev_index,
                        hash_result,
                        |value| format!("{value:?}"),
                    );

                    #[cfg(debug_assertions)]
                    if hash_result.is_some() {
                        data.current.record_edge(
                            dep_node_index,
                            node,
                            data.prev_fingerprint_of(prev_index),
                        );
                    }

                    return dep_node_index;
                }
            }

            let mut edges = EdgesVec::new();
            D::read_deps(|task_deps| match task_deps {
                TaskDepsRef::Allow(deps) => edges.extend(deps.lock().reads.iter().copied()),
                TaskDepsRef::EvalAlways => {
                    edges.push(DepNodeIndex::FOREVER_RED_NODE);
                }
                TaskDepsRef::Ignore => {}
                TaskDepsRef::Forbid => {
                    panic!("Cannot summarize when dependencies are not recorded.")
                }
            });

            let hashing_timer = cx.profiler().incr_result_hashing();
            let current_fingerprint = hash_result.map(|hash_result| {
                cx.with_stable_hashing_context(|mut hcx| hash_result(&mut hcx, result))
            });

            // Intern the new `DepNode` with the dependencies up-to-now.
            let (dep_node_index, prev_and_color) =
                data.current.intern_node(&data.previous, node, edges, current_fingerprint);

            hashing_timer.finish_with_query_invocation_id(dep_node_index.into());

            if let Some((prev_index, color)) = prev_and_color {
                debug_assert!(
                    data.colors.get(prev_index).is_none(),
                    "DepGraph::with_task() - Duplicate DepNodeColor insertion for {key:?}",
                );

                data.colors.insert(prev_index, color);
            }

            dep_node_index
        } else {
            // Incremental compilation is turned off. We just execute the task
            // without tracking. We still provide a dep-node index that uniquely
            // identifies the task so that we have a cheap way of referring to
            // the query for self-profiling.
            self.next_virtual_depnode_index()
        }
    }
}

impl<D: Deps> DepGraphData<D> {
    #[inline]
    fn dep_node_index_of_opt(&self, dep_node: &DepNode) -> Option<DepNodeIndex> {
        if let Some(prev_index) = self.previous.node_to_index_opt(dep_node) {
            self.current.prev_index_to_index.lock()[prev_index]
        } else {
            self.current.new_node_to_index.lock_shard_by_value(dep_node).get(dep_node).copied()
        }
    }

    #[inline]
    fn dep_node_exists(&self, dep_node: &DepNode) -> bool {
        self.dep_node_index_of_opt(dep_node).is_some()
    }

    fn node_color(&self, dep_node: &DepNode) -> Option<DepNodeColor> {
        if let Some(prev_index) = self.previous.node_to_index_opt(dep_node) {
            self.colors.get(prev_index)
        } else {
            // This is a node that did not exist in the previous compilation session.
            None
        }
    }

    /// Returns true if the given node has been marked as green during the
    /// current compilation session. Used in various assertions
    #[inline]
    pub(crate) fn is_index_green(&self, prev_index: SerializedDepNodeIndex) -> bool {
        self.colors.get(prev_index).is_some_and(|c| c.is_green())
    }

    #[inline]
    pub(crate) fn prev_fingerprint_of(&self, prev_index: SerializedDepNodeIndex) -> Fingerprint {
        self.previous.fingerprint_by_index(prev_index)
    }

    #[inline]
    pub(crate) fn prev_node_of(&self, prev_index: SerializedDepNodeIndex) -> DepNode {
        self.previous.index_to_node(prev_index)
    }

    pub(crate) fn mark_debug_loaded_from_disk(&self, dep_node: DepNode) {
        self.debug_loaded_from_disk.lock().insert(dep_node);
    }
}

impl<D: Deps> DepGraph<D> {
    #[inline]
    pub fn dep_node_exists(&self, dep_node: &DepNode) -> bool {
        self.data.as_ref().is_some_and(|data| data.dep_node_exists(dep_node))
    }

    /// Checks whether a previous work product exists for `v` and, if
    /// so, return the path that leads to it. Used to skip doing work.
    pub fn previous_work_product(&self, v: &WorkProductId) -> Option<WorkProduct> {
        self.data.as_ref().and_then(|data| data.previous_work_products.get(v).cloned())
    }

    /// Access the map of work-products created during the cached run. Only
    /// used during saving of the dep-graph.
    pub fn previous_work_products(&self) -> &WorkProductMap {
        &self.data.as_ref().unwrap().previous_work_products
    }

    pub fn debug_was_loaded_from_disk(&self, dep_node: DepNode) -> bool {
        self.data.as_ref().unwrap().debug_loaded_from_disk.lock().contains(&dep_node)
    }

    #[cfg(debug_assertions)]
    #[inline(always)]
    pub(crate) fn register_dep_node_debug_str<F>(&self, dep_node: DepNode, debug_str_gen: F)
    where
        F: FnOnce() -> String,
    {
        let dep_node_debug = &self.data.as_ref().unwrap().dep_node_debug;

        if dep_node_debug.borrow().contains_key(&dep_node) {
            return;
        }
        let debug_str = self.with_ignore(debug_str_gen);
        dep_node_debug.borrow_mut().insert(dep_node, debug_str);
    }

    pub fn dep_node_debug_str(&self, dep_node: DepNode) -> Option<String> {
        self.data.as_ref()?.dep_node_debug.borrow().get(&dep_node).cloned()
    }

    fn node_color(&self, dep_node: &DepNode) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            return data.node_color(dep_node);
        }

        None
    }

    pub fn try_mark_green<Qcx: QueryContext<Deps = D>>(
        &self,
        qcx: Qcx,
        dep_node: &DepNode,
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        self.data().and_then(|data| data.try_mark_green(qcx, dep_node))
    }
}

impl<D: Deps> DepGraphData<D> {
    /// Try to mark a node index for the node dep_node.
    ///
    /// A node will have an index, when it's already been marked green, or when we can mark it
    /// green. This function will mark the current task as a reader of the specified node, when
    /// a node index can be found for that node.
    pub(crate) fn try_mark_green<Qcx: QueryContext<Deps = D>>(
        &self,
        qcx: Qcx,
        dep_node: &DepNode,
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        debug_assert!(!qcx.dep_context().is_eval_always(dep_node.kind));

        // Return None if the dep node didn't exist in the previous session
        let prev_index = self.previous.node_to_index_opt(dep_node)?;

        match self.colors.get(prev_index) {
            Some(DepNodeColor::Green(dep_node_index)) => Some((prev_index, dep_node_index)),
            Some(DepNodeColor::Red) => None,
            None => {
                // This DepNode and the corresponding query invocation existed
                // in the previous compilation session too, so we can try to
                // mark it as green by recursively marking all of its
                // dependencies green.
                self.try_mark_previous_green(qcx, prev_index, dep_node, None)
                    .map(|dep_node_index| (prev_index, dep_node_index))
            }
        }
    }

    #[instrument(skip(self, qcx, parent_dep_node_index, frame), level = "debug")]
    fn try_mark_parent_green<Qcx: QueryContext<Deps = D>>(
        &self,
        qcx: Qcx,
        parent_dep_node_index: SerializedDepNodeIndex,
        frame: Option<&MarkFrame<'_>>,
    ) -> Option<()> {
        let dep_dep_node_color = self.colors.get(parent_dep_node_index);
        let dep_dep_node = &self.previous.index_to_node(parent_dep_node_index);

        match dep_dep_node_color {
            Some(DepNodeColor::Green(_)) => {
                // This dependency has been marked as green before, we are
                // still fine and can continue with checking the other
                // dependencies.
                debug!("dependency {dep_dep_node:?} was immediately green");
                return Some(());
            }
            Some(DepNodeColor::Red) => {
                // We found a dependency the value of which has changed
                // compared to the previous compilation session. We cannot
                // mark the DepNode as green and also don't need to bother
                // with checking any of the other dependencies.
                debug!("dependency {dep_dep_node:?} was immediately red");
                return None;
            }
            None => {}
        }

        // We don't know the state of this dependency. If it isn't
        // an eval_always node, let's try to mark it green recursively.
        if !qcx.dep_context().is_eval_always(dep_dep_node.kind) {
            debug!(
                "state of dependency {:?} ({}) is unknown, trying to mark it green",
                dep_dep_node, dep_dep_node.hash,
            );

            let node_index =
                self.try_mark_previous_green(qcx, parent_dep_node_index, dep_dep_node, frame);

            if node_index.is_some() {
                debug!("managed to MARK dependency {dep_dep_node:?} as green",);
                return Some(());
            }
        }

        // We failed to mark it green, so we try to force the query.
        debug!("trying to force dependency {dep_dep_node:?}");
        if !qcx.dep_context().try_force_from_dep_node(*dep_dep_node, frame) {
            // The DepNode could not be forced.
            debug!("dependency {dep_dep_node:?} could not be forced");
            return None;
        }

        let dep_dep_node_color = self.colors.get(parent_dep_node_index);

        match dep_dep_node_color {
            Some(DepNodeColor::Green(_)) => {
                debug!("managed to FORCE dependency {dep_dep_node:?} to green");
                return Some(());
            }
            Some(DepNodeColor::Red) => {
                debug!("dependency {dep_dep_node:?} was red after forcing",);
                return None;
            }
            None => {}
        }

        if let None = qcx.dep_context().sess().dcx().has_errors_or_delayed_bugs() {
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
        debug!("dependency {dep_dep_node:?} resulted in compilation error",);
        return None;
    }

    /// Try to mark a dep-node which existed in the previous compilation session as green.
    #[instrument(skip(self, qcx, prev_dep_node_index, frame), level = "debug")]
    fn try_mark_previous_green<Qcx: QueryContext<Deps = D>>(
        &self,
        qcx: Qcx,
        prev_dep_node_index: SerializedDepNodeIndex,
        dep_node: &DepNode,
        frame: Option<&MarkFrame<'_>>,
    ) -> Option<DepNodeIndex> {
        let frame = MarkFrame { index: prev_dep_node_index, parent: frame };

        // We never try to mark eval_always nodes as green
        debug_assert!(!qcx.dep_context().is_eval_always(dep_node.kind));

        debug_assert_eq!(self.previous.index_to_node(prev_dep_node_index), *dep_node);

        let prev_deps = self.previous.edge_targets_from(prev_dep_node_index);

        for dep_dep_node_index in prev_deps {
            self.try_mark_parent_green(qcx, dep_dep_node_index, Some(&frame))?;
        }

        // If we got here without hitting a `return` that means that all
        // dependencies of this DepNode could be marked as green. Therefore we
        // can also mark this DepNode as green.

        // There may be multiple threads trying to mark the same dep node green concurrently

        // We allocating an entry for the node in the current dependency graph and
        // adding all the appropriate edges imported from the previous graph
        let dep_node_index =
            self.current.promote_node_and_deps_to_current(&self.previous, prev_dep_node_index);

        // ... emitting any stored diagnostic ...

        // FIXME: Store the fact that a node has diagnostics in a bit in the dep graph somewhere
        // Maybe store a list on disk and encode this fact in the DepNodeState
        let side_effects = qcx.load_side_effects(prev_dep_node_index);

        if side_effects.maybe_any() {
            qcx.dep_context().dep_graph().with_query_deserialization(|| {
                self.emit_side_effects(qcx, dep_node_index, side_effects)
            });
        }

        // ... and finally storing a "Green" entry in the color map.
        // Multiple threads can all write the same color here
        self.colors.insert(prev_dep_node_index, DepNodeColor::Green(dep_node_index));

        debug!("successfully marked {dep_node:?} as green");
        Some(dep_node_index)
    }

    /// Atomically emits some loaded diagnostics.
    /// This may be called concurrently on multiple threads for the same dep node.
    #[cold]
    #[inline(never)]
    fn emit_side_effects<Qcx: QueryContext<Deps = D>>(
        &self,
        qcx: Qcx,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    ) {
        let mut processed = self.processed_side_effects.lock();

        if processed.insert(dep_node_index) {
            // We were the first to insert the node in the set so this thread
            // must process side effects

            // Promote the previous diagnostics to the current session.
            qcx.store_side_effects(dep_node_index, side_effects.clone());

            let dcx = qcx.dep_context().sess().dcx();

            for diagnostic in side_effects.diagnostics {
                dcx.emit_diagnostic(diagnostic);
            }
        }
    }
}

impl<D: Deps> DepGraph<D> {
    /// Returns true if the given node has been marked as red during the
    /// current compilation session. Used in various assertions
    pub fn is_red(&self, dep_node: &DepNode) -> bool {
        matches!(self.node_color(dep_node), Some(DepNodeColor::Red))
    }

    /// Returns true if the given node has been marked as green during the
    /// current compilation session. Used in various assertions
    pub fn is_green(&self, dep_node: &DepNode) -> bool {
        self.node_color(dep_node).is_some_and(|c| c.is_green())
    }

    /// This method loads all on-disk cacheable query results into memory, so
    /// they can be written out to the new cache file again. Most query results
    /// will already be in memory but in the case where we marked something as
    /// green but then did not need the value, that value will never have been
    /// loaded from disk.
    ///
    /// This method will only load queries that will end up in the disk cache.
    /// Other queries will not be executed.
    pub fn exec_cache_promotions<Tcx: DepContext>(&self, tcx: Tcx) {
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
            data.current.encoder.print_incremental_info(
                data.current.total_read_count.load(Ordering::Relaxed),
                data.current.total_duplicate_read_count.load(Ordering::Relaxed),
            )
        }
    }

    pub fn finish_encoding(&self) -> FileEncodeResult {
        if let Some(data) = &self.data { data.current.encoder.finish() } else { Ok(0) }
    }

    pub(crate) fn next_virtual_depnode_index(&self) -> DepNodeIndex {
        debug_assert!(self.data.is_none());
        let index = self.virtual_dep_node_index.fetch_add(1, Ordering::Relaxed);
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
    /// Saved files associated with this CGU. In each key/value pair, the value is the path to the
    /// saved file and the key is some identifier for the type of file being saved.
    ///
    /// By convention, file extensions are currently used as identifiers, i.e. the key "o" maps to
    /// the object file's path, and "dwo" to the dwarf object file's path.
    pub saved_files: UnordMap<String, String>,
}

pub type WorkProductMap = UnordMap<WorkProductId, WorkProduct>;

// Index type for `DepNodeData`'s edges.
rustc_index::newtype_index! {
    struct EdgeIndex {}
}

/// `CurrentDepGraph` stores the dependency graph for the current session. It
/// will be populated as we run queries or tasks. We never remove nodes from the
/// graph: they are only added.
///
/// The nodes in it are identified by a `DepNodeIndex`. We avoid keeping the nodes
/// in memory. This is important, because these graph structures are some of the
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
pub(super) struct CurrentDepGraph<D: Deps> {
    encoder: GraphEncoder<D>,
    new_node_to_index: Sharded<FxHashMap<DepNode, DepNodeIndex>>,
    prev_index_to_index: Lock<IndexVec<SerializedDepNodeIndex, Option<DepNodeIndex>>>,

    /// This is used to verify that fingerprints do not change between the creation of a node
    /// and its recomputation.
    #[cfg(debug_assertions)]
    fingerprints: Lock<IndexVec<DepNodeIndex, Option<Fingerprint>>>,

    /// Used to trap when a specific edge is added to the graph.
    /// This is used for debug purposes and is only active with `debug_assertions`.
    #[cfg(debug_assertions)]
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

impl<D: Deps> CurrentDepGraph<D> {
    fn new(
        profiler: &SelfProfilerRef,
        prev_graph_node_count: usize,
        encoder: FileEncoder,
        record_graph: bool,
        record_stats: bool,
        previous: Arc<SerializedDepGraph>,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let nanos = duration.as_nanos();
        let mut stable_hasher = StableHasher::new();
        nanos.hash(&mut stable_hasher);
        let anon_id_seed = stable_hasher.finish();

        #[cfg(debug_assertions)]
        let forbidden_edge = match env::var("RUST_FORBID_DEP_GRAPH_EDGE") {
            Ok(s) => match EdgeFilter::new(&s) {
                Ok(f) => Some(f),
                Err(err) => panic!("RUST_FORBID_DEP_GRAPH_EDGE invalid: {}", err),
            },
            Err(_) => None,
        };

        let new_node_count_estimate = 102 * prev_graph_node_count / 100 + 200;

        CurrentDepGraph {
            encoder: GraphEncoder::new(
                encoder,
                prev_graph_node_count,
                record_graph,
                record_stats,
                profiler,
                previous,
            ),
            new_node_to_index: Sharded::new(|| {
                FxHashMap::with_capacity_and_hasher(
                    new_node_count_estimate / sharded::shards(),
                    Default::default(),
                )
            }),
            prev_index_to_index: Lock::new(IndexVec::from_elem_n(None, prev_graph_node_count)),
            anon_id_seed,
            #[cfg(debug_assertions)]
            forbidden_edge,
            #[cfg(debug_assertions)]
            fingerprints: Lock::new(IndexVec::from_elem_n(None, new_node_count_estimate)),
            total_read_count: AtomicU64::new(0),
            total_duplicate_read_count: AtomicU64::new(0),
        }
    }

    #[cfg(debug_assertions)]
    fn record_edge(&self, dep_node_index: DepNodeIndex, key: DepNode, fingerprint: Fingerprint) {
        if let Some(forbidden_edge) = &self.forbidden_edge {
            forbidden_edge.index_to_node.lock().insert(dep_node_index, key);
        }
        let previous = *self.fingerprints.lock().get_or_insert_with(dep_node_index, || fingerprint);
        assert_eq!(previous, fingerprint, "Unstable fingerprints for {:?}", key);
    }

    /// Writes the node to the current dep-graph and allocates a `DepNodeIndex` for it.
    /// Assumes that this is a node that has no equivalent in the previous dep-graph.
    #[inline(always)]
    fn intern_new_node(
        &self,
        key: DepNode,
        edges: EdgesVec,
        current_fingerprint: Fingerprint,
    ) -> DepNodeIndex {
        let dep_node_index = match self.new_node_to_index.lock_shard_by_value(&key).entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let dep_node_index = self.encoder.send(key, current_fingerprint, edges);
                entry.insert(dep_node_index);
                dep_node_index
            }
        };

        #[cfg(debug_assertions)]
        self.record_edge(dep_node_index, key, current_fingerprint);

        dep_node_index
    }

    fn intern_node(
        &self,
        prev_graph: &SerializedDepGraph,
        key: DepNode,
        edges: EdgesVec,
        fingerprint: Option<Fingerprint>,
    ) -> (DepNodeIndex, Option<(SerializedDepNodeIndex, DepNodeColor)>) {
        if let Some(prev_index) = prev_graph.node_to_index_opt(&key) {
            let get_dep_node_index = |fingerprint| {
                let mut prev_index_to_index = self.prev_index_to_index.lock();

                let dep_node_index = match prev_index_to_index[prev_index] {
                    Some(dep_node_index) => dep_node_index,
                    None => {
                        let dep_node_index = self.encoder.send(key, fingerprint, edges);
                        prev_index_to_index[prev_index] = Some(dep_node_index);
                        dep_node_index
                    }
                };

                #[cfg(debug_assertions)]
                self.record_edge(dep_node_index, key, fingerprint);

                dep_node_index
            };

            // Determine the color and index of the new `DepNode`.
            if let Some(fingerprint) = fingerprint {
                if fingerprint == prev_graph.fingerprint_by_index(prev_index) {
                    // This is a green node: it existed in the previous compilation,
                    // its query was re-executed, and it has the same result as before.
                    let dep_node_index = get_dep_node_index(fingerprint);
                    (dep_node_index, Some((prev_index, DepNodeColor::Green(dep_node_index))))
                } else {
                    // This is a red node: it existed in the previous compilation, its query
                    // was re-executed, but it has a different result from before.
                    let dep_node_index = get_dep_node_index(fingerprint);
                    (dep_node_index, Some((prev_index, DepNodeColor::Red)))
                }
            } else {
                // This is a red node, effectively: it existed in the previous compilation
                // session, its query was re-executed, but it doesn't compute a result hash
                // (i.e. it represents a `no_hash` query), so we have no way of determining
                // whether or not the result was the same as before.
                let dep_node_index = get_dep_node_index(Fingerprint::ZERO);
                (dep_node_index, Some((prev_index, DepNodeColor::Red)))
            }
        } else {
            let fingerprint = fingerprint.unwrap_or(Fingerprint::ZERO);

            // This is a new node: it didn't exist in the previous compilation session.
            let dep_node_index = self.intern_new_node(key, edges, fingerprint);

            (dep_node_index, None)
        }
    }

    fn promote_node_and_deps_to_current(
        &self,
        prev_graph: &SerializedDepGraph,
        prev_index: SerializedDepNodeIndex,
    ) -> DepNodeIndex {
        self.debug_assert_not_in_new_nodes(prev_graph, prev_index);

        let mut prev_index_to_index = self.prev_index_to_index.lock();

        match prev_index_to_index[prev_index] {
            Some(dep_node_index) => dep_node_index,
            None => {
                let dep_node_index = self.encoder.send_promoted(prev_index, &*prev_index_to_index);
                prev_index_to_index[prev_index] = Some(dep_node_index);
                #[cfg(debug_assertions)]
                self.record_edge(
                    dep_node_index,
                    prev_graph.index_to_node(prev_index),
                    prev_graph.fingerprint_by_index(prev_index),
                );
                dep_node_index
            }
        }
    }

    #[inline]
    fn debug_assert_not_in_new_nodes(
        &self,
        prev_graph: &SerializedDepGraph,
        prev_index: SerializedDepNodeIndex,
    ) {
        let node = &prev_graph.index_to_node(prev_index);
        debug_assert!(
            !self.new_node_to_index.lock_shard_by_value(node).contains_key(node),
            "node from previous graph present in new node collection"
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TaskDepsRef<'a> {
    /// New dependencies can be added to the
    /// `TaskDeps`. This is used when executing a 'normal' query
    /// (no `eval_always` modifier)
    Allow(&'a Lock<TaskDeps>),
    /// This is used when executing an `eval_always` query. We don't
    /// need to track dependencies for a query that's always
    /// re-executed -- but we need to know that this is an `eval_always`
    /// query in order to emit dependencies to `DepNodeIndex::FOREVER_RED_NODE`
    /// when directly feeding other queries.
    EvalAlways,
    /// New dependencies are ignored. This is also used for `dep_graph.with_ignore`.
    Ignore,
    /// Any attempt to add new dependencies will cause a panic.
    /// This is used when decoding a query result from disk,
    /// to ensure that the decoding process doesn't itself
    /// require the execution of any queries.
    Forbid,
}

#[derive(Debug)]
pub struct TaskDeps {
    #[cfg(debug_assertions)]
    node: Option<DepNode>,
    reads: EdgesVec,
    read_set: FxHashSet<DepNodeIndex>,
    phantom_data: PhantomData<DepNode>,
}

impl Default for TaskDeps {
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

    #[inline]
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

#[inline(never)]
#[cold]
pub(crate) fn print_markframe_trace<D: Deps>(graph: &DepGraph<D>, frame: Option<&MarkFrame<'_>>) {
    let data = graph.data.as_ref().unwrap();

    eprintln!("there was a panic while trying to force a dep node");
    eprintln!("try_mark_green dep node stack:");

    let mut i = 0;
    let mut current = frame;
    while let Some(frame) = current {
        let node = data.previous.index_to_node(frame.index);
        eprintln!("#{i} {node:?}");
        current = frame.parent;
        i += 1;
    }

    eprintln!("end of try_mark_green dep node stack");
}

#[cold]
#[inline(never)]
fn panic_on_forbidden_read<D: Deps>(data: &DepGraphData<D>, dep_node_index: DepNodeIndex) -> ! {
    // We have to do an expensive reverse-lookup of the DepNode that
    // corresponds to `dep_node_index`, but that's OK since we are about
    // to ICE anyway.
    let mut dep_node = None;

    // First try to find the dep node among those that already existed in the
    // previous session
    for (prev_index, index) in data.current.prev_index_to_index.lock().iter_enumerated() {
        if index == &Some(dep_node_index) {
            dep_node = Some(data.previous.index_to_node(prev_index));
            break;
        }
    }

    if dep_node.is_none() {
        // Try to find it among the new nodes
        for shard in data.current.new_node_to_index.lock_shards() {
            if let Some((node, _)) = shard.iter().find(|(_, index)| **index == dep_node_index) {
                dep_node = Some(*node);
                break;
            }
        }
    }

    let dep_node = dep_node.map_or_else(
        || format!("with index {:?}", dep_node_index),
        |dep_node| format!("`{:?}`", dep_node),
    );

    panic!(
        "Error: trying to record dependency on DepNode {dep_node} in a \
         context that does not allow it (e.g. during query deserialization). \
         The most common case of recording a dependency on a DepNode `foo` is \
         when the corresponding query `foo` is invoked. Invoking queries is not \
         allowed as part of loading something from the incremental on-disk cache. \
         See <https://github.com/rust-lang/rust/pull/91919>."
    )
}
