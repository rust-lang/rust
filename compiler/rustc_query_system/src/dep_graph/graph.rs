use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::profiling::QueryInvocationId;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{AtomicU32, AtomicU64, Lock, Lrc, RwLock};
use rustc_data_structures::unlikely;
use rustc_errors::Diagnostic;
use rustc_index::vec::IndexVec;
use rustc_serialize::{Encodable, Encoder};

use smallvec::{smallvec, SmallVec};
use std::env;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::Ordering::Relaxed;

use super::debug::EdgeFilter;
use super::query::DepGraphQuery;
use super::serialized::{
    CurrentDepGraph, DepNodeColor, DepNodeIndex, SerializedDepGraph, SerializedDepNodeIndex,
};
use super::{DepContext, DepKind, DepNode, HasDepContext, WorkProductId};
use crate::query::QueryContext;

#[derive(Clone)]
pub struct DepGraph<K: DepKind> {
    data: Option<Lrc<DepGraphData<K>>>,

    /// This field is used for assigning DepNodeIndices when running in
    /// non-incremental mode. Even in non-incremental mode we make sure that
    /// each task has a `DepNodeIndex` that uniquely identifies it. This unique
    /// ID is used for self-profiling.
    virtual_dep_node_index: Lrc<AtomicU32>,
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

struct DepGraphData<K: DepKind> {
    /// The dep-graph from the previous compilation session. It contains all
    /// nodes and edges as well as all fingerprints of nodes that have them.
    previous: RwLock<CurrentDepGraph<K>>,

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

    /// A set of loaded diagnostics that is in the progress of being emitted.
    emitting_diagnostics: Lock<FxHashSet<DepNodeIndex>>,

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
        prev_graph: SerializedDepGraph<K>,
        prev_work_products: FxHashMap<WorkProductId, WorkProduct>,
    ) -> DepGraph<K> {
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

        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: prev_work_products,
                dep_node_debug: Default::default(),
                anon_id_seed: stable_hasher.finish(),
                forbidden_edge,
                total_read_count: AtomicU64::new(0),
                total_duplicate_read_count: AtomicU64::new(0),
                emitting_diagnostics: Default::default(),
                previous: RwLock::new(CurrentDepGraph::new(prev_graph)),
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
        data.previous.read().query()
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
    pub fn with_task<Ctxt: HasDepContext<DepKind = K>, A, R>(
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

    fn with_task_impl<Ctxt: HasDepContext<DepKind = K>, A, R>(
        &self,
        key: DepNode<K>,
        cx: Ctxt,
        arg: A,
        task: fn(Ctxt, A) -> R,
        create_task: fn(DepNode<K>) -> Option<TaskDeps<K>>,
        hash_result: impl FnOnce(&mut Ctxt::StableHashingContext, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex) {
        if let Some(ref data) = self.data {
            let dcx = cx.dep_context();
            let task_deps = create_task(key).map(Lock::new);
            let result = K::with_deps(task_deps.as_ref(), || task(cx, arg));
            let edges = task_deps.map_or_else(|| smallvec![], |lock| lock.into_inner().reads);

            let mut hcx = dcx.create_stable_hashing_context();
            let current_fingerprint = hash_result(&mut hcx, &result);

            // Intern the new `DepNode`.
            let dep_node_index = data.previous.write().intern_task_node(
                key,
                &edges[..],
                current_fingerprint,
                dcx.sess().opts.debugging_opts.dep_tasks,
            );

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
                hash: data.anon_id_seed.combine(hasher.finish()).into(),
            };

            let mut previous = data.previous.write();
            let dep_node_index = previous.intern_anon_node(target_dep_node, &task_deps.reads[..]);

            (result, dep_node_index)
        } else {
            (op(), self.next_virtual_depnode_index())
        }
    }

    /// Executes something within an "eval-always" task which is a task
    /// that runs whenever anything changes.
    pub fn with_eval_always_task<Ctxt: HasDepContext<DepKind = K>, A, R>(
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
                        data.total_read_count.fetch_add(1, Relaxed);
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
                                if let Some(ref forbidden_edge) = data.forbidden_edge {
                                    let src = self.dep_node_of(dep_node_index);
                                    if forbidden_edge.test(&src, &target) {
                                        panic!("forbidden edge {:?} -> {:?} created", src, target)
                                    }
                                }
                            }
                        }
                    } else if cfg!(debug_assertions) {
                        data.total_duplicate_read_count.fetch_add(1, Relaxed);
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
        data.previous.read().dep_node_index_of_opt(dep_node)
    }

    #[inline]
    pub fn dep_node_exists(&self, dep_node: &DepNode<K>) -> bool {
        self.data.is_some() && self.dep_node_index_of_opt(dep_node).is_some()
    }

    #[inline]
    pub fn dep_node_of(&self, dep_node_index: DepNodeIndex) -> DepNode<K> {
        let data = self.data.as_ref().unwrap();
        data.previous.read().dep_node_of(dep_node_index)
    }

    #[inline]
    pub fn fingerprint_of(&self, dep_node_index: DepNodeIndex) -> Fingerprint {
        let data = self.data.as_ref().unwrap();
        data.previous.read().fingerprint_of(dep_node_index)
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

    pub fn node_color(&self, dep_node: &DepNode<K>) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            let previous = data.previous.read();
            if let Some(prev_index) = previous.node_to_index_opt(dep_node) {
                return previous.color(prev_index);
            } else {
                // This is a node that did not exist in the previous compilation
                // session, so we consider it to be red.
                return Some(DepNodeColor::New);
            }
        }

        None
    }

    /// Try to read a node index for the node dep_node.
    /// A node will have an index, when it's already been marked green, or when we can mark it
    /// green. This function will mark the current task as a reader of the specified node, when
    /// a node index can be found for that node.
    pub fn try_mark_green_and_read<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        dep_node: &DepNode<K>,
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        self.try_mark_green(tcx, dep_node).map(|(prev_index, dep_node_index)| {
            debug_assert!(self.node_color(&dep_node) == Some(DepNodeColor::Green));
            self.read_index(dep_node_index);
            (prev_index, dep_node_index)
        })
    }

    pub fn try_mark_green<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        dep_node: &DepNode<K>,
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        debug_assert!(!dep_node.kind.is_eval_always());

        // Return None if the dep graph is disabled
        let data = self.data.as_ref()?;

        // Return None if the dep node didn't exist in the previous session
        let prev_index = data.previous.read().node_to_index_opt(dep_node)?;
        let prev_deps = data.previous.read().color_or_edges(prev_index);
        let prev_deps = match prev_deps {
            Err(prev_deps) => prev_deps,
            Ok(DepNodeColor::Green) => return Some((prev_index, prev_index.rejuvenate())),
            Ok(DepNodeColor::Red) | Ok(DepNodeColor::New) => return None,
        };

        // This DepNode and the corresponding query invocation existed
        // in the previous compilation session too, so we can try to
        // mark it as green by recursively marking all of its
        // dependencies green.
        let dep_node_index =
            self.try_mark_previous_green(tcx, data, prev_index, prev_deps, &dep_node)?;
        Some((prev_index, dep_node_index))
    }

    fn try_mark_parent_green<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        parent_dep_node_index: SerializedDepNodeIndex,
        dep_node: &DepNode<K>,
    ) -> Option<()> {
        let dep_dep_node_color = data.previous.read().color_or_edges(parent_dep_node_index);
        let prev_deps = match dep_dep_node_color {
            Ok(DepNodeColor::Green) => {
                // This dependency has been marked as green before, we are
                // still fine and can continue with checking the other
                // dependencies.
                debug!(
                    "try_mark_parent_green({:?}) --- found dependency {:?} to be immediately green",
                    dep_node,
                    data.previous.read().index_to_node(parent_dep_node_index)
                );
                return Some(());
            }
            Ok(DepNodeColor::Red) | Ok(DepNodeColor::New) => {
                // We found a dependency the value of which has changed
                // compared to the previous compilation session. We cannot
                // mark the DepNode as green and also don't need to bother
                // with checking any of the other dependencies.
                debug!(
                    "try_mark_parent_green({:?}) - END - dependency {:?} was immediately red",
                    dep_node,
                    data.previous.read().index_to_node(parent_dep_node_index)
                );
                return None;
            }
            Err(prev_deps) => prev_deps,
        };

        // We don't know the state of this dependency. If it isn't
        // an eval_always node, let's try to mark it green recursively.
        debug!(
            "try_mark_parent_green({:?}) --- state of dependency {:?} \
                                 is unknown, trying to mark it green",
            dep_node,
            {
                let dep_dep_node = data.previous.read().index_to_node(parent_dep_node_index);
                (dep_dep_node, dep_dep_node.hash)
            }
        );

        let dep_dep_node = &data.previous.read().index_to_node(parent_dep_node_index);
        let node_index =
            self.try_mark_previous_green(tcx, data, parent_dep_node_index, prev_deps, dep_dep_node);
        if node_index.is_some() {
            debug!(
                "try_mark_parent_green({:?}) --- managed to MARK dependency {:?} as green",
                dep_node, dep_dep_node
            );
            return Some(());
        }

        // We failed to mark it green, so we try to force the query.
        debug!(
            "try_mark_parent_green({:?}) --- trying to force dependency {:?}",
            dep_node, dep_dep_node
        );
        if !tcx.try_force_from_dep_node(dep_dep_node) {
            // The DepNode could not be forced.
            debug!(
                "try_mark_parent_green({:?}) - END - dependency {:?} could not be forced",
                dep_node, dep_dep_node
            );
            return None;
        }

        let dep_dep_node_color = data.previous.read().color(parent_dep_node_index);

        match dep_dep_node_color {
            Some(DepNodeColor::Green) => {
                debug!(
                    "try_mark_parent_green({:?}) --- managed to FORCE dependency {:?} to green",
                    dep_node, dep_dep_node
                );
                return Some(());
            }
            Some(DepNodeColor::Red) | Some(DepNodeColor::New) => {
                debug!(
                    "try_mark_parent_green({:?}) - END - dependency {:?} was red after forcing",
                    dep_node, dep_dep_node
                );
                return None;
            }
            None => {}
        }

        if !tcx.dep_context().sess().has_errors_or_delayed_span_bugs() {
            panic!("try_mark_parent_green() - Forcing the DepNode should have set its color")
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
            "try_mark_parent_green({:?}) - END - dependency {:?} resulted in compilation error",
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
        prev_deps: &[SerializedDepNodeIndex],
        dep_node: &DepNode<K>,
    ) -> Option<DepNodeIndex> {
        // We never try to mark eval_always nodes as green
        if dep_node.kind.is_eval_always() {
            return None;
        }

        debug!("try_mark_previous_green({:?}) - BEGIN", dep_node);

        // We never try to mark eval_always nodes as green
        debug_assert!(!dep_node.kind.is_eval_always());
        debug_assert_eq!(data.previous.read().index_to_node(prev_dep_node_index), *dep_node);

        for &dep_dep_node_index in prev_deps {
            self.try_mark_parent_green(tcx, data, dep_dep_node_index, dep_node)?
        }

        #[cfg(not(parallel_compiler))]
        debug_assert_eq!(
            data.previous.read().color(prev_dep_node_index),
            None,
            "DepGraph::try_mark_previous_green() - Duplicate DepNodeColor \
                      insertion for {:?}",
            dep_node
        );

        // If we got here without hitting a `return` that means that all
        // dependencies of this DepNode could be marked as green. Therefore we
        // can also mark this DepNode as green.

        // There may be multiple threads trying to mark the same dep node green concurrently

        let dep_node_index = {
            // We allocating an entry for the node in the current dependency graph and
            // adding all the appropriate edges imported from the previous graph
            data.previous.write().intern_dark_green_node(prev_dep_node_index)
        };

        // ... and emitting any stored diagnostic.

        // FIXME: Store the fact that a node has diagnostics in a bit in the dep graph somewhere
        // Maybe store a list on disk and encode this fact in the DepNodeState
        let diagnostics = tcx.load_diagnostics(prev_dep_node_index);

        if unlikely!(!diagnostics.is_empty()) {
            self.emit_diagnostics(tcx, data, dep_node_index, diagnostics);
        }

        debug!("try_mark_previous_green({:?}) - END - successfully marked as green", dep_node);
        Some(dep_node_index)
    }

    /// Atomically emits some loaded diagnostics.
    /// This may be called concurrently on multiple threads for the same dep node.
    #[cold]
    #[inline(never)]
    fn emit_diagnostics<Ctxt: QueryContext<DepKind = K>>(
        &self,
        tcx: Ctxt,
        data: &DepGraphData<K>,
        dep_node_index: DepNodeIndex,
        diagnostics: Vec<Diagnostic>,
    ) {
        let should_emit = data.emitting_diagnostics.lock().insert(dep_node_index);
        if !should_emit {
            return;
        }

        // Promote the previous diagnostics to the current session.
        tcx.store_diagnostics(dep_node_index, diagnostics.clone().into());

        let handle = tcx.dep_context().sess().diagnostic();

        for diagnostic in diagnostics {
            handle.emit_diagnostic(&diagnostic);
        }
    }

    // This method loads all on-disk cacheable query results into memory, so
    // they can be written out to the new cache file again. Most query results
    // will already be in memory but in the case where we marked something as
    // green but then did not need the value, that value will never have been
    // loaded from disk.
    //
    // This method will only load queries that will end up in the disk cache.
    // Other queries will not be executed.
    pub fn exec_cache_promotions<Ctxt: QueryContext<DepKind = K>>(&self, qcx: Ctxt) {
        let tcx = qcx.dep_context();
        let _prof_timer = tcx.profiler().generic_activity("incr_comp_query_cache_promotion");

        let data = self.data.as_ref().unwrap();
        let previous = data.previous.read();
        for prev_index in previous.serialized_indices() {
            match previous.color(prev_index) {
                Some(DepNodeColor::Green) => {
                    let dep_node = data.previous.read().index_to_node(prev_index);
                    debug!("PROMOTE {:?} {:?}", prev_index, dep_node);
                    qcx.try_load_from_on_disk_cache(&dep_node);
                }
                None | Some(DepNodeColor::Red) | Some(DepNodeColor::New) => {
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
        let previous = data.previous.read();
        for prev_index in previous.serialized_indices() {
            match previous.color(prev_index) {
                Some(_) => {
                    let dep_node = data.previous.read().index_to_node(prev_index);
                    tcx.register_reused_dep_node(&dep_node);
                }
                None => {}
            }
        }
    }

    pub fn print_incremental_info(&self) {
        #[derive(Clone)]
        struct Stat<Kind: DepKind> {
            kind: Kind,
            node_counter: u64,
            edge_counter: u64,
        }

        let data = self.data.as_ref().unwrap();
        let prev = &data.previous.read();

        let mut stats: FxHashMap<_, Stat<K>> = FxHashMap::with_hasher(Default::default());

        for index in prev.live_indices() {
            let kind = prev.dep_node_of(index).kind;
            let edge_count = prev.edge_targets_from(index).len();

            let stat = stats.entry(kind).or_insert(Stat { kind, node_counter: 0, edge_counter: 0 });
            stat.node_counter += 1;
            stat.edge_counter += edge_count as u64;
        }

        let total_node_count = prev.node_count();
        let total_edge_count = prev.edge_count();

        // Drop the lock guard.
        std::mem::drop(prev);

        let mut stats: Vec<_> = stats.values().cloned().collect();
        stats.sort_by_key(|s| -(s.node_counter as i64));

        const SEPARATOR: &str = "[incremental] --------------------------------\
                                 ----------------------------------------------\
                                 ------------";

        eprintln!("[incremental]");
        eprintln!("[incremental] DepGraph Statistics");
        eprintln!("{}", SEPARATOR);
        eprintln!("[incremental]");
        eprintln!("[incremental] Total Node Count: {}", total_node_count);
        eprintln!("[incremental] Total Edge Count: {}", total_edge_count);

        if cfg!(debug_assertions) {
            let total_edge_reads = data.total_read_count.load(Relaxed);
            let total_duplicate_edge_reads = data.total_duplicate_read_count.load(Relaxed);

            eprintln!("[incremental] Total Edge Reads: {}", total_edge_reads);
            eprintln!("[incremental] Total Duplicate Edge Reads: {}", total_duplicate_edge_reads);
        }

        eprintln!("[incremental]");

        eprintln!(
            "[incremental]  {:<36}| {:<17}| {:<12}| {:<17}|",
            "Node Kind", "Node Frequency", "Node Count", "Avg. Edge Count"
        );

        eprintln!(
            "[incremental] -------------------------------------\
                  |------------------\
                  |-------------\
                  |------------------|"
        );

        for stat in stats {
            let node_kind_ratio = (100.0 * (stat.node_counter as f64)) / (total_node_count as f64);
            let node_kind_avg_edges = (stat.edge_counter as f64) / (stat.node_counter as f64);

            eprintln!(
                "[incremental]  {:<36}|{:>16.1}% |{:>12} |{:>17.1} |",
                format!("{:?}", stat.kind),
                node_kind_ratio,
                stat.node_counter,
                node_kind_avg_edges,
            );
        }

        eprintln!("{}", SEPARATOR);
        eprintln!("[incremental]");
    }

    fn next_virtual_depnode_index(&self) -> DepNodeIndex {
        let index = self.virtual_dep_node_index.fetch_add(1, Relaxed);
        DepNodeIndex::from_u32(index)
    }

    pub fn compression_map(&self) -> IndexVec<DepNodeIndex, Option<SerializedDepNodeIndex>> {
        self.data.as_ref().unwrap().previous.read().compression_map()
    }

    pub fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error>
    where
        K: Encodable<E>,
    {
        if let Some(data) = &self.data { data.previous.read().encode(encoder) } else { Ok(()) }
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
