use errors::{Diagnostic, DiagnosticBuilder};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use smallvec::SmallVec;
use rustc_data_structures::sync::{Lrc, Lock, AtomicCell, AtomicU64};
use std::sync::atomic::Ordering::{Acquire, SeqCst};
use std::env;
use std::fs::File;
use std::hash::Hash;
use std::collections::hash_map::Entry;
use std::mem;
use crate::ty::{self, TyCtxt};
use crate::util::common::{ProfileQueriesMsg, profq_msg};
use parking_lot::{Mutex, Condvar};

use crate::ich::{StableHashingContext, StableHashingContextProvider, Fingerprint};

use super::debug::EdgeFilter;
use super::dep_node::{DepNode, DepKind, WorkProductId};
use super::query::DepGraphQuery;
use super::safe::DepGraphSafe;
use super::serialized::Serializer;
use super::prev::PreviousDepGraph;

#[derive(Clone)]
pub struct DepGraph {
    data: Option<Lrc<DepGraphData>>,
}

newtype_index! {
    pub struct DepNodeIndex { .. }
}

impl DepNodeIndex {
    pub(super) const INVALID: DepNodeIndex = DepNodeIndex::MAX;
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
    /// The node is invalid since its result is older than the previous session.
    Invalid,

    /// The node is from the previous session, but its state is unknown
    Unknown,

    /// The node will eventually be green (it was previously Unknown),
    /// but its side effects (like error messages) have not yet happened.
    /// This only exist so we can know which thread
    /// marked a node as green in `try_mark_previous_green`.
    WasUnknownWillBeGreen,

    Red,

    Green,
}

impl DepNodeState {
    pub fn color(self) -> Option<DepNodeColor> {
        match self {
            DepNodeState::Invalid |
            DepNodeState::Unknown |
            DepNodeState::WasUnknownWillBeGreen => None,
            DepNodeState::Red => Some(DepNodeColor::Red),
            DepNodeState::Green => Some(DepNodeColor::Green),
        }
    }
}

struct DepGraphData {
    /// The new encoding of the dependency graph, optimized for red/green
    /// tracking. The `current` field is the dependency graph of only the
    /// current compilation session: We don't merge the previous dep-graph into
    /// current one anymore.
    current: CurrentDepGraph,

    /// The dep-graph from the previous compilation session. It contains all
    /// nodes and edges as well as all fingerprints of nodes that have them.
    previous: Lrc<PreviousDepGraph>,

    colors: DepNodeColorMap,

    /// A set of loaded diagnostics that have been emitted.
    emitted_diagnostics: Mutex<FxHashSet<DepNodeIndex>>,

    /// Used to wait for diagnostics to be emitted.
    emitted_diagnostics_cond_var: Condvar,

    /// When we load, there may be `.o` files, cached MIR, or other such
    /// things available to us. If we find that they are not dirty, we
    /// load the path to the file storing those work-products here into
    /// this map. We can later look for and extract that data.
    previous_work_products: FxHashMap<WorkProductId, WorkProduct>,

    dep_node_debug: Lock<FxHashMap<DepNode, String>>,

    // Used for testing, only populated when -Zquery-dep-graph is specified.
    loaded_from_cache: Lock<FxHashMap<DepNode, bool>>,
}

pub fn hash_result<R>(hcx: &mut StableHashingContext<'_>, result: &R) -> Option<Fingerprint>
where
    R: for<'a> HashStable<StableHashingContext<'a>>,
{
    let mut stable_hasher = StableHasher::new();
    result.hash_stable(hcx, &mut stable_hasher);

    Some(stable_hasher.finish())
}

pub struct DepGraphArgs {
    pub prev_graph: PreviousDepGraph,
    pub prev_work_products: FxHashMap<WorkProductId, WorkProduct>,
    pub file: File,
    pub state: IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>,
}

impl DepGraph {
    pub fn new(args: DepGraphArgs) -> DepGraph {
        let colors = DepNodeColorMap {
            values: args.state,
        };
        let prev_graph = Lrc::new(args.prev_graph);

        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: args.prev_work_products,
                dep_node_debug: Default::default(),
                current: CurrentDepGraph::new(prev_graph.clone(), args.file),
                emitted_diagnostics: Default::default(),
                emitted_diagnostics_cond_var: Condvar::new(),
                colors,
                previous: prev_graph,
                loaded_from_cache: Default::default(),
            })),
        }
    }

    pub fn new_disabled() -> DepGraph {
        DepGraph {
            data: None,
        }
    }

    /// Returns `true` if we are actually building the full dep-graph, and `false` otherwise.
    #[inline]
    pub fn is_fully_enabled(&self) -> bool {
        self.data.is_some()
    }

    pub fn query(&self) -> DepGraphQuery {
        // FIXME
        panic!()/*
        let current_dep_graph = self.data.as_ref().unwrap().current.borrow();
        let nodes: Vec<_> = current_dep_graph.data.iter().map(|n| n.node).collect();
        let mut edges = Vec::new();
        for (from, edge_targets) in current_dep_graph.data.iter()
                                                           .map(|d| (d.node, &d.edges)) {
            for &edge_target in edge_targets.iter() {
                let to = current_dep_graph.data[edge_target].node;
                edges.push((from, to));
            }
        }

        DepGraphQuery::new(&nodes[..], &edges[..])*/
    }

    pub fn assert_ignored(&self)
    {
        if let Some(..) = self.data {
            ty::tls::with_context_opt(|icx| {
                let icx = if let Some(icx) = icx { icx } else { return };
                assert!(icx.task_deps.is_none(), "expected no task dependency tracking");
            })
        }
    }

    pub fn with_ignore<OP,R>(&self, op: OP) -> R
        where OP: FnOnce() -> R
    {
        ty::tls::with_context(|icx| {
            let icx = ty::tls::ImplicitCtxt {
                task_deps: None,
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
    pub fn with_task<'a, C, A, R>(
        &self,
        key: DepNode,
        cx: C,
        arg: A,
        task: fn(C, A) -> R,
        hash_result: impl FnOnce(&mut StableHashingContext<'_>, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex)
    where
        C: DepGraphSafe + StableHashingContextProvider<'a>,
    {
        self.with_task_impl(key, cx, arg, false, task,
            |_key| Some(TaskDeps {
                #[cfg(debug_assertions)]
                node: Some(_key),
                reads: SmallVec::new(),
                read_set: Default::default(),
            }),
            hash_result)
    }

    /// Creates a new dep-graph input with value `input`.
    /// Dep nodes created by this function can be used by the `read` method.
    pub fn input_task<'a, C, R>(&self,
                                   key: DepNode,
                                   cx: C,
                                   input: R)
                                   -> (R, DepNodeIndex)
        where C: DepGraphSafe + StableHashingContextProvider<'a>,
              R: for<'b> HashStable<StableHashingContext<'b>>,
    {
        fn identity_fn<C, A>(_: C, arg: A) -> A {
            arg
        }

        let (r, index) = self.with_task_impl(key, cx, input, true, identity_fn,
            |_| None,
            hash_result::<R>);

        self.data.as_ref().map(|data| {
            assert!(data.current.input_node_to_node_index.lock().insert(key, index).is_none());
        });

        (r, index)
    }

    fn with_task_impl<'a, C, A, R>(
        &self,
        key: DepNode,
        cx: C,
        arg: A,
        no_tcx: bool,
        task: fn(C, A) -> R,
        create_task: fn(DepNode) -> Option<TaskDeps>,
        hash_result: impl FnOnce(&mut StableHashingContext<'_>, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex)
    where
        C: DepGraphSafe + StableHashingContextProvider<'a>,
    {
        if let Some(ref data) = self.data {
            let task_deps = create_task(key).map(|deps| Lock::new(deps));

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
                        task_deps: task_deps.as_ref(),
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

            let current_fingerprint = hash_result(&mut hcx, &result);

            let deps = task_deps.map(|lock| lock.into_inner().reads).unwrap_or(SmallVec::new());
            let fingerprint = current_fingerprint.unwrap_or(Fingerprint::ZERO);

            let print_status = cfg!(debug_assertions) && hcx.sess().opts.debugging_opts.dep_tasks;

            let dep_node_index = if let Some(prev_index) = data.previous.node_to_index_opt(&key) {
                let prev_fingerprint = data.previous.fingerprint_by_index(prev_index);

                // Determine the color of the previous DepNode.
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
                    data.colors.get(prev_index),
                    DepNodeState::Unknown,
                    "DepGraph::with_task() - Duplicate DepNodeState insertion for {:?}",
                    key
                );

                data.colors.insert(prev_index, color);

                data.current.update_node(
                    prev_index,
                    key,
                    deps,
                    fingerprint
                );

                prev_index
            } else {
                if print_status {
                    eprintln!("[task::new] {:?}", key);
                }

                data.current.new_node(
                    key,
                    deps,
                    fingerprint,
                    &data.previous,
                )
            };

            (result, dep_node_index)
        } else {
            (task(cx, arg), DepNodeIndex::INVALID)
        }
    }

    /// Executes something within an "anonymous" task, that is, a task the
    /// `DepNode` of which is determined by the list of inputs it read from.
    pub fn with_anon_task<OP,R>(&self, dep_kind: DepKind, op: OP) -> (R, DepNodeIndex)
        where OP: FnOnce() -> R
    {
        if let Some(ref data) = self.data {
            let (result, task_deps) = ty::tls::with_context(|icx| {
                let task_deps = Lock::new(TaskDeps {
                    #[cfg(debug_assertions)]
                    node: None,
                    reads: SmallVec::new(),
                    read_set: Default::default(),
                });

                let r = {
                    let icx = ty::tls::ImplicitCtxt {
                        task_deps: Some(&task_deps),
                        ..icx.clone()
                    };

                    ty::tls::enter_context(&icx, |_| {
                        op()
                    })
                };

                (r, task_deps.into_inner())
            });
            let dep_node_index = data.current
                                     .complete_anon_task(dep_kind, task_deps, &data.previous);
            (result, dep_node_index)
        } else {
            (op(), DepNodeIndex::INVALID)
        }
    }

    /// Executes something within an "eval-always" task which is a task
    /// that runs whenever anything changes.
    pub fn with_eval_always_task<'a, C, A, R>(
        &self,
        key: DepNode,
        cx: C,
        arg: A,
        task: fn(C, A) -> R,
        hash_result: impl FnOnce(&mut StableHashingContext<'_>, &R) -> Option<Fingerprint>,
    ) -> (R, DepNodeIndex)
    where
        C: DepGraphSafe + StableHashingContextProvider<'a>,
    {
        self.with_task_impl(key, cx, arg, false, task,
            |_| None,
            hash_result)
    }

    #[inline]
    pub fn read(&self, v: DepNode) {
        if let Some(ref data) = self.data {
            let map = data.current.input_node_to_node_index.lock();
            if let Some(dep_node_index) = map.get(&v).cloned() {
                std::mem::drop(map);
                data.read_index(dep_node_index);
            } else {
                bug!("DepKind {:?} should be pre-allocated but isn't.", v.kind)
            }
        }
    }

    #[inline]
    pub fn read_index(&self, dep_node_index: DepNodeIndex) {
        if let Some(ref data) = self.data {
            data.read_index(dep_node_index);
        }
    }
/*
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
*/
    #[inline]
    pub fn fingerprint_of(&self, _dep_node_index: DepNodeIndex) -> Fingerprint {
        panic!()/*
        let current = self.data.as_ref().expect("dep graph enabled").current.borrow_mut();
        current.data[dep_node_index].fingerprint*/
    }

    pub fn prev_fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.data.as_ref().unwrap().previous.fingerprint_of(dep_node)
    }

    /// Checks whether a previous work product exists for `v` and, if
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

    pub fn edge_deduplication_data(&self) -> Option<(u64, u64)> {
        if cfg!(debug_assertions) {
            let current_dep_graph = &self.data.as_ref().unwrap().current;

            Some((current_dep_graph.total_read_count.load(Acquire),
                  current_dep_graph.total_duplicate_read_count.load(Acquire)))
        } else {
            None
        }
    }

    pub fn serialize(&self) -> IndexVec<DepNodeIndex, Fingerprint> {
        let data = self.data.as_ref().unwrap();
        // Invalidate dep nodes with unknown state as these cannot safely
        // be marked green in the next session.
        let invalidate = data.colors.values.indices().filter_map(|prev_index| {
            match data.colors.get(prev_index) {
                DepNodeState::Unknown => Some(prev_index),
                _ => None,
            }
        }).collect();
        // FIXME: Can this deadlock?
        data.current.serializer.lock().complete(invalidate)
    }

    pub fn node_color(&self, dep_node: &DepNode) -> Option<DepNodeColor> {
        if let Some(ref data) = self.data {
            if let Some(prev_index) = data.previous.node_to_index_opt(dep_node) {
                return data.colors.get(prev_index).color()
            } else {
                // This is a node that did not exist in the previous compilation
                // session, so we consider it to be red.
                return Some(DepNodeColor::Red)
            }
        }

        None
    }

    /// Try to read a node index for the node dep_node.
    /// A node will have an index, when it's already been marked green, or when we can mark it
    /// green. This function will mark the current task as a reader of the specified node, when
    /// a node index can be found for that node.
    pub fn try_mark_green_and_read(
        &self,
        tcx: TyCtxt<'_>,
        dep_node: &DepNode,
    ) -> Option<DepNodeIndex> {
        self.try_mark_green(tcx, dep_node).map(|prev_index| {
            debug_assert!(self.is_green(&dep_node));
            self.read_index(prev_index);
            prev_index
        })
    }

    pub fn try_mark_green(
        &self,
        tcx: TyCtxt<'_>,
        dep_node: &DepNode,
    ) -> Option<DepNodeIndex> {
        debug_assert!(!dep_node.kind.is_eval_always());

        // Return None if the dep graph is disabled
        let data = self.data.as_ref()?;

        // Return None if the dep node didn't exist in the previous session
        let prev_index = data.previous.node_to_index_opt(dep_node)?;

        match data.colors.get(prev_index) {
            DepNodeState::Green => Some(prev_index),
            DepNodeState::Invalid |
            DepNodeState::Red => None,
            DepNodeState::Unknown |
            DepNodeState::WasUnknownWillBeGreen => {
                // This DepNode and the corresponding query invocation existed
                // in the previous compilation session too, so we can try to
                // mark it as green by recursively marking all of its
                // dependencies green.
                if self.try_mark_previous_green(
                    tcx.global_tcx(),
                    data,
                    prev_index,
                    &dep_node
                ) {
                    Some(prev_index)
                } else {
                    None
                }
            }
        }
    }

    /// Try to force a dep node to execute and see if it's green
    fn try_force_previous_green<'tcx>(
        &self,
        tcx: TyCtxt<'_, 'tcx, 'tcx>,
        data: &DepGraphData,
        dep_node_index: DepNodeIndex,
    ) -> bool {
        let dep_node = &data.previous.index_to_node(dep_node_index);

        match dep_node.kind {
            DepKind::Hir |
            DepKind::HirBody |
            DepKind::CrateMetadata => {
                if dep_node.extract_def_id(tcx).is_none() {
                    // If the node does not exist anymore, we
                    // just fail to mark green.
                    return false
                } else {
                    // If the node does exist, it should have
                    // been pre-allocated.
                    bug!("DepNode {:?} should have been \
                            pre-allocated but wasn't.",
                            dep_node)
                }
            }
            _ => {
                // For other kinds of nodes it's OK to be
                // forced.
            }
        }

        debug!("try_force_previous_green({:?}) --- trying to force", dep_node);
        if crate::ty::query::force_from_dep_node(tcx, dep_node) {
            match data.colors.get(dep_node_index) {
                DepNodeState::Green => {
                    debug!("try_force_previous_green({:?}) --- managed to \
                            FORCE to green",
                            dep_node);
                    true
                }
                DepNodeState::Red => {
                    debug!(
                        "try_force_previous_green({:?}) - END - was red after forcing",
                        dep_node
                    );
                    false
                }
                DepNodeState::Invalid |
                DepNodeState::Unknown |
                DepNodeState::WasUnknownWillBeGreen => {
                    bug!("try_force_previous_green() - Forcing the DepNode \
                            should have set its color")
                }
            }
        } else {
            // The DepNode could not be forced.
            debug!("try_force_previous_green({:?}) - END - could not be forced", dep_node);
            false
        }
    }

    /// Try to mark a dep-node which existed in the previous compilation session as green.
    fn try_mark_previous_green<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        data: &DepGraphData,
        dep_node_index: DepNodeIndex,
        // FIXME: Remove this, only used in debug statements
        dep_node: &DepNode,
    ) -> bool {
        debug!("try_mark_previous_green({:?}) - BEGIN", dep_node);

        #[cfg(not(parallel_compiler))]
        {
            debug_assert!(data.colors.get(dep_node_index).color().is_none());
        }

        // We cannot mark invalid results as green
        debug_assert_ne!(data.colors.get(dep_node_index), DepNodeState::Invalid);

        // We never try to mark eval_always nodes as green
        debug_assert!(!dep_node.kind.is_eval_always());

        debug_assert_eq!(data.previous.index_to_node(dep_node_index), *dep_node);

        let prev_deps = data.previous.edge_targets_from(dep_node_index);

        for &dep_dep_node_index in prev_deps {
            let dep_dep_node_color = data.colors.get(dep_dep_node_index);

            match dep_dep_node_color {
                DepNodeState::Green => {
                    // This dependency has been marked as green before, we are
                    // still fine and can continue with checking the other
                    // dependencies.
                    debug!("try_mark_previous_green({:?}) --- found dependency {:?} to \
                            be immediately green",
                            dep_node,
                            data.previous.index_to_node(dep_dep_node_index));
                }
                DepNodeState::Red => {
                    // We found a dependency the value of which has changed
                    // compared to the previous compilation session. We cannot
                    // mark the DepNode as green and also don't need to bother
                    // with checking any of the other dependencies.
                    debug!("try_mark_previous_green({:?}) - END - dependency {:?} was \
                            immediately red",
                            dep_node,
                            data.previous.index_to_node(dep_dep_node_index));
                    return false
                }
                // Either the previous result is too old or
                // this is a eval_always node. Try to force the node
                DepNodeState::Invalid => {
                    if !self.try_force_previous_green(tcx, data, dep_dep_node_index) {
                        return false;
                    }
                }
                DepNodeState::WasUnknownWillBeGreen |
                DepNodeState::Unknown => {
                    let dep_dep_node = &data.previous.index_to_node(dep_dep_node_index);

                    // We don't know the state of this dependency.
                    // We known it is not an eval_always node, since those get marked as `Invalid`.
                    // Let's try to mark it green recursively.
                    if self.try_mark_previous_green(
                        tcx,
                        data,
                        dep_dep_node_index,
                        dep_dep_node
                    ) {
                        debug!("try_mark_previous_green({:?}) --- managed to MARK \
                                dependency {:?} as green", dep_node, dep_dep_node);
                        continue;
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
        debug_assert_eq!(data.colors.get(dep_node_index), DepNodeState::Unknown,
                      "DepGraph::try_mark_previous_green() - Duplicate DepNodeState \
                      insertion for {:?}", dep_node);

        let did_mark = data.colors.mark_as_will_be_green(dep_node_index);

        // ... emitting any stored diagnostic ...

        let diagnostics = tcx.queries.on_disk_cache
                                .load_diagnostics(tcx, dep_node_index);

        if unlikely!(diagnostics.len() > 0) {
            self.emit_diagnostics(
                tcx,
                data,
                dep_node_index,
                did_mark,
                diagnostics
            );
        } else {
            // Avoid calling the destructor, since LLVM fails to optimize it away
            mem::forget(diagnostics);
        }

        // ... and finally storing a "Green" entry in the color map.
        // Multiple threads can all write the same color here
        // FIXME: Mark as green-promoted?
        data.colors.insert(dep_node_index, DepNodeState::Green);

        debug!("try_mark_previous_green({:?}) - END - successfully marked as green", dep_node);

        true
    }

    /// Atomically emits some loaded diagnotics, assuming that this only gets called with
    /// `did_mark` set to `true` on a single thread.
    #[cold]
    #[inline(never)]
    fn emit_diagnostics<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        data: &DepGraphData,
        dep_node_index: DepNodeIndex,
        did_mark: bool,
        diagnostics: Vec<Diagnostic>,
    ) {
        if did_mark || !cfg!(parallel_compiler) {
            // Only the thread which did the allocation emits the error messages
            let handle = tcx.sess.diagnostic();

            // Promote the previous diagnostics to the current session.
            tcx.queries.on_disk_cache
                .store_diagnostics(dep_node_index, diagnostics.clone().into());

            for diagnostic in diagnostics {
                DiagnosticBuilder::new_diagnostic(handle, diagnostic).emit();
            }

            #[cfg(parallel_compiler)]
            {
                // Mark the diagnostics aa emitted and wake up waiters
                data.emitted_diagnostics.lock().insert(dep_node_index);
                data.emitted_diagnostics_cond_var.notify_all();
            }
        } else {
            // The other threads will wait for the diagnostics to be emitted

            let mut emitted_diagnostics = data.emitted_diagnostics.lock();
            loop {
                if emitted_diagnostics.contains(&dep_node_index) {
                    break;
                }
                data.emitted_diagnostics_cond_var.wait(&mut emitted_diagnostics);
            }
        }
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
    pub fn exec_cache_promotions<'tcx>(&self, tcx: TyCtxt<'tcx>) {
        let green_nodes: Vec<DepNode> = {
            let data = self.data.as_ref().unwrap();
            data.colors.values.indices().filter_map(|prev_index| {
                match data.colors.get(prev_index) {
                    DepNodeState::Green => {
                        let dep_node = data.previous.index_to_node(prev_index);
                        if dep_node.cache_on_disk(tcx) {
                            Some(dep_node)
                        } else {
                            None
                        }
                    }
                    DepNodeState::WasUnknownWillBeGreen => bug!("no tasks should be in progress"),
                    DepNodeState::Invalid |
                    DepNodeState::Unknown |
                    DepNodeState::Red => {
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

    pub fn mark_loaded_from_cache(&self, dep_node: DepNode, state: bool) {
        debug!("mark_loaded_from_cache({:?}, {})", dep_node, state);

        self.data
            .as_ref()
            .unwrap()
            .loaded_from_cache
            .borrow_mut()
            .insert(dep_node, state);
    }

    pub fn was_loaded_from_cache(&self, dep_node: &DepNode) -> Option<bool> {
        let data = self.data.as_ref().unwrap();
        data.loaded_from_cache.borrow().get(&dep_node).cloned()
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

#[derive(Clone, Debug)]
pub(super) struct DepNodeData {
    pub(super) node: DepNode,
    pub(super) edges: TaskReads,
    pub(super) fingerprint: Fingerprint,
}

pub(super) struct CurrentDepGraph {
    //data: FxHashMap<DepNodeIndex, DepNodeData>,
    /// Used to map input nodes to a node index. Used by the `read` method.
    input_node_to_node_index: Lock<FxHashMap<DepNode, DepNodeIndex>>,

    /// Used to map input nodes to a node index
    anon_node_to_node_index: Lock<FxHashMap<DepNode, DepNodeIndex>>,

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

    total_read_count: AtomicU64,
    total_duplicate_read_count: AtomicU64,

    /// Produces the serialized dep graph for the next session,
    serializer: Lock<Serializer>,
}

impl CurrentDepGraph {
    fn new(prev_graph: Lrc<PreviousDepGraph>, file: File) -> CurrentDepGraph {
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

        // Pre-allocate the dep node structures. We over-allocate a little so
        // that we hopefully don't have to re-allocate during this compilation
        // session. The over-allocation is 2% plus a small constant to account
        // for the fact that in very small crates 2% might not be enough.
        //let new_node_count_estimate = (prev_graph.node_count() * 102) / 100 + 200;

        CurrentDepGraph {
            //data: FxHashMap::default(),
            anon_node_to_node_index: Default::default(),
            input_node_to_node_index: Default::default(),
            anon_id_seed: stable_hasher.finish(),
            forbidden_edge,
            total_read_count: AtomicU64::new(0),
            total_duplicate_read_count: AtomicU64::new(0),
            serializer: Lock::new(Serializer::new(file, prev_graph)),
        }
    }

    fn complete_anon_task(
        &self,
        kind: DepKind,
        task_deps: TaskDeps,
        previous: &PreviousDepGraph,
    ) -> DepNodeIndex {
        debug_assert!(!kind.is_eval_always());

        let mut hasher = StableHasher::new();

        // The dep node indices are hashed here instead of hashing the dep nodes of the
        // dependencies. These indices may refer to different nodes per session, but this isn't
        // a problem here because we that ensure the final dep node hash is per session only by
        // combining it with the per session random number `anon_id_seed`. This hash only need
        // to map the dependencies to a single value on a per session basis.
        task_deps.reads.hash(&mut hasher);

        let dep_node = DepNode {
            kind,

            // Fingerprint::combine() is faster than sending Fingerprint
            // through the StableHasher (at least as long as StableHasher
            // is so slow).
            hash: self.anon_id_seed.combine(hasher.finish()),
        };

        match self.anon_node_to_node_index.lock().entry(dep_node) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                // Make sure there's no collision with a previous dep node
                // FIXME: Ensure this by always allocating a new index for each
                // anon task instead of hashing a random seed?
                let dep_node_index = self.new_node(
                    dep_node,
                    task_deps.reads,
                    Fingerprint::ZERO,
                    previous,
                );
                entry.insert(dep_node_index);
                dep_node_index
            }
        }
    }

    fn new_node(
        &self,
        dep_node: DepNode,
        edges: TaskReads,
        fingerprint: Fingerprint,
        previous: &PreviousDepGraph,
    ) -> DepNodeIndex {
        debug_assert!(previous.node_to_index_opt(&dep_node).is_none());
        self.serializer.lock().serialize_new(DepNodeData {
            node: dep_node,
            edges,
            fingerprint
        })
    }

    fn update_node(
        &self,
        index: DepNodeIndex,
        dep_node: DepNode,
        edges: TaskReads,
        fingerprint: Fingerprint,
    ) {
        self.serializer.lock().serialize_updated(index, DepNodeData {
            node: dep_node,
            edges,
            fingerprint
        });
    }
}

impl DepGraphData {
    fn read_index(&self, source: DepNodeIndex) {
        ty::tls::with_context_opt(|icx| {
            let icx = if let Some(icx) = icx { icx } else {  return };
            if let Some(task_deps) = icx.task_deps {
                let mut task_deps = task_deps.lock();
                if cfg!(debug_assertions) {
                    self.current.total_read_count.fetch_add(1, SeqCst);
                }
                if task_deps.read_set.insert(source) {
                    task_deps.reads.push(source);

                    /*#[cfg(debug_assertions)]
                    {
                        if let Some(target) = task_deps.node {
                            let graph = self.current.lock();
                            if let Some(ref forbidden_edge) = graph.forbidden_edge {
                                let source = graph.data[&source].node;
                                if forbidden_edge.test(&source, &target) {
                                    bug!("forbidden edge {:?} -> {:?} created",
                                        source,
                                        target)
                                }
                            }
                        }
                    }*/
                } else if cfg!(debug_assertions) {
                    self.current.total_duplicate_read_count.fetch_add(1, SeqCst);
                }
            }
        })
    }
}

type TaskReads = SmallVec<[DepNodeIndex; 8]>;

pub struct TaskDeps {
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    node: Option<DepNode>,
    reads: TaskReads,
    read_set: FxHashSet<DepNodeIndex>,
}

struct DepNodeColorMap {
    values: IndexVec<DepNodeIndex, AtomicCell<DepNodeState>>,
}

impl DepNodeColorMap {
    /// Tries to mark the node as WasUnknownWillBeGreen.
    /// Returns false if another thread did it before us.
    #[inline]
    fn mark_as_will_be_green(&self, index: DepNodeIndex) -> bool {
        let actual_prev = self.values[index].compare_and_swap(
            DepNodeState::Unknown,
            DepNodeState::WasUnknownWillBeGreen,
        );
        actual_prev == DepNodeState::Unknown
    }

    #[inline]
    fn get(&self, index: DepNodeIndex) -> DepNodeState {
        self.values[index].load()
    }

    #[inline]
    fn insert(&self, index: DepNodeIndex, state: DepNodeState) {
        self.values[index].store(state)
    }
}
