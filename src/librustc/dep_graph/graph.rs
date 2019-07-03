use errors::{Diagnostic, DiagnosticBuilder};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use smallvec::SmallVec;
use rustc_data_structures::sync::{Lrc, Lock, AtomicU32, Ordering};
use std::env;
use std::hash::Hash;
use std::collections::hash_map::Entry;
use crate::ty::{self, TyCtxt};
use crate::util::common::{ProfileQueriesMsg, profq_msg};
use parking_lot::{Mutex, Condvar};

use crate::ich::{StableHashingContext, StableHashingContextProvider, Fingerprint};

use super::debug::EdgeFilter;
use super::dep_node::{DepNode, DepKind, WorkProductId};
use super::query::DepGraphQuery;
use super::safe::DepGraphSafe;
use super::serialized::{SerializedDepGraph, SerializedDepNodeIndex};
use super::prev::PreviousDepGraph;

#[derive(Clone)]
pub struct DepGraph {
    data: Option<Lrc<DepGraphData>>,
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
    loaded_from_cache: Lock<FxHashMap<DepNodeIndex, bool>>,
}

pub fn hash_result<R>(hcx: &mut StableHashingContext<'_>, result: &R) -> Option<Fingerprint>
where
    R: for<'a> HashStable<StableHashingContext<'a>>,
{
    let mut stable_hasher = StableHasher::new();
    result.hash_stable(hcx, &mut stable_hasher);

    Some(stable_hasher.finish())
}

impl DepGraph {
    pub fn new(prev_graph: PreviousDepGraph,
               prev_work_products: FxHashMap<WorkProductId, WorkProduct>) -> DepGraph {
        let prev_graph_node_count = prev_graph.node_count();

        DepGraph {
            data: Some(Lrc::new(DepGraphData {
                previous_work_products: prev_work_products,
                dep_node_debug: Default::default(),
                current: Lock::new(CurrentDepGraph::new(prev_graph_node_count)),
                emitted_diagnostics: Default::default(),
                emitted_diagnostics_cond_var: Condvar::new(),
                previous: prev_graph,
                colors: DepNodeColorMap::new(prev_graph_node_count),
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

        DepGraphQuery::new(&nodes[..], &edges[..])
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
            |data, key, fingerprint, task| {
                data.borrow_mut().complete_task(key, task.unwrap(), fingerprint)
            },
            hash_result)
    }

    /// Creates a new dep-graph input with value `input`
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

        self.with_task_impl(key, cx, input, true, identity_fn,
            |_| None,
            |data, key, fingerprint, _| {
                data.borrow_mut().alloc_node(key, SmallVec::new(), fingerprint)
            },
            hash_result::<R>)
    }

    fn with_task_impl<'a, C, A, R>(
        &self,
        key: DepNode,
        cx: C,
        arg: A,
        no_tcx: bool,
        task: fn(C, A) -> R,
        create_task: fn(DepNode) -> Option<TaskDeps>,
        finish_task_and_alloc_depnode: fn(&Lock<CurrentDepGraph>,
                                          DepNode,
                                          Fingerprint,
                                          Option<TaskDeps>) -> DepNodeIndex,
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

            let dep_node_index = finish_task_and_alloc_depnode(
                &data.current,
                key,
                current_fingerprint.unwrap_or(Fingerprint::ZERO),
                task_deps.map(|lock| lock.into_inner()),
            );

            let print_status = cfg!(debug_assertions) && hcx.sess().opts.debugging_opts.dep_tasks;

            // Determine the color of the new DepNode.
            if let Some(prev_index) = data.previous.node_to_index_opt(&key) {
                let prev_fingerprint = data.previous.fingerprint_by_index(prev_index);

                let color = if let Some(current_fingerprint) = current_fingerprint {
                    if current_fingerprint == prev_fingerprint {
                        if print_status {
                            eprintln!("[task::green] {:?}", key);
                        }
                        DepNodeColor::Green(dep_node_index)
                    } else {
                        if print_status {
                            eprintln!("[task::red] {:?}", key);
                        }
                        DepNodeColor::Red
                    }
                } else {
                    if print_status {
                        eprintln!("[task::unknown] {:?}", key);
                    }
                    // Mark the node as Red if we can't hash the result
                    DepNodeColor::Red
                };

                debug_assert!(data.colors.get(prev_index).is_none(),
                            "DepGraph::with_task() - Duplicate DepNodeColor \
                            insertion for {:?}", key);

                data.colors.insert(prev_index, color);
            } else {
                if print_status {
                    eprintln!("[task::new] {:?}", key);
                }
            }

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
                                     .borrow_mut()
                                     .complete_anon_task(dep_kind, task_deps);
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
            |data, key, fingerprint, _| {
                let mut current = data.borrow_mut();
                current.alloc_node(key, smallvec![], fingerprint)
            },
            hash_result)
    }

    #[inline]
    pub fn read(&self, v: DepNode) {
        if let Some(ref data) = self.data {
            let current = data.current.borrow_mut();
            if let Some(&dep_node_index) = current.node_to_node_index.get(&v) {
                std::mem::drop(current);
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
        let current = self.data.as_ref().expect("dep graph enabled").current.borrow_mut();
        current.data[dep_node_index].fingerprint
    }

    pub fn prev_fingerprint_of(&self, dep_node: &DepNode) -> Option<Fingerprint> {
        self.data.as_ref().unwrap().previous.fingerprint_of(dep_node)
    }

    #[inline]
    pub fn prev_dep_node_index_of(&self, dep_node: &DepNode) -> SerializedDepNodeIndex {
        self.data.as_ref().unwrap().previous.node_to_index(dep_node)
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
            let current_dep_graph = self.data.as_ref().unwrap().current.borrow();

            Some((current_dep_graph.total_read_count,
                  current_dep_graph.total_duplicate_read_count))
        } else {
            None
        }
    }

    pub fn serialize(&self) -> SerializedDepGraph {
        let current_dep_graph = self.data.as_ref().unwrap().current.borrow();

        let fingerprints: IndexVec<SerializedDepNodeIndex, _> =
            current_dep_graph.data.iter().map(|d| d.fingerprint).collect();
        let nodes: IndexVec<SerializedDepNodeIndex, _> =
            current_dep_graph.data.iter().map(|d| d.node).collect();

        let total_edge_count: usize = current_dep_graph.data.iter()
                                                            .map(|d| d.edges.len())
                                                            .sum();

        let mut edge_list_indices = IndexVec::with_capacity(nodes.len());
        let mut edge_list_data = Vec::with_capacity(total_edge_count);

        for (current_dep_node_index, edges) in current_dep_graph.data.iter_enumerated()
                                                                .map(|(i, d)| (i, &d.edges)) {
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
                return data.colors.get(prev_index)
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
    ) -> Option<(SerializedDepNodeIndex, DepNodeIndex)> {
        self.try_mark_green(tcx, dep_node).map(|(prev_index, dep_node_index)| {
            debug_assert!(self.is_green(&dep_node));
            self.read_index(dep_node_index);
            (prev_index, dep_node_index)
        })
    }

    pub fn try_mark_green(
        &self,
        tcx: TyCtxt<'_>,
        dep_node: &DepNode,
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
                self.try_mark_previous_green(
                    tcx.global_tcx(),
                    data,
                    prev_index,
                    &dep_node
                ).map(|dep_node_index| {
                    (prev_index, dep_node_index)
                })
            }
        }
    }

    /// Try to mark a dep-node which existed in the previous compilation session as green.
    fn try_mark_previous_green<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        data: &DepGraphData,
        prev_dep_node_index: SerializedDepNodeIndex,
        dep_node: &DepNode,
    ) -> Option<DepNodeIndex> {
        debug!("try_mark_previous_green({:?}) - BEGIN", dep_node);

        #[cfg(not(parallel_compiler))]
        {
            debug_assert!(!data.current.borrow().node_to_node_index.contains_key(dep_node));
            debug_assert!(data.colors.get(prev_dep_node_index).is_none());
        }

        // We never try to mark eval_always nodes as green
        debug_assert!(!dep_node.kind.is_eval_always());

        debug_assert_eq!(data.previous.index_to_node(prev_dep_node_index), *dep_node);

        let prev_deps = data.previous.edge_targets_from(prev_dep_node_index);

        let mut current_deps = SmallVec::new();

        for &dep_dep_node_index in prev_deps {
            let dep_dep_node_color = data.colors.get(dep_dep_node_index);

            match dep_dep_node_color {
                Some(DepNodeColor::Green(node_index)) => {
                    // This dependency has been marked as green before, we are
                    // still fine and can continue with checking the other
                    // dependencies.
                    debug!("try_mark_previous_green({:?}) --- found dependency {:?} to \
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
                    debug!("try_mark_previous_green({:?}) - END - dependency {:?} was \
                            immediately red",
                            dep_node,
                            data.previous.index_to_node(dep_dep_node_index));
                    return None
                }
                None => {
                    let dep_dep_node = &data.previous.index_to_node(dep_dep_node_index);

                    // We don't know the state of this dependency. If it isn't
                    // an eval_always node, let's try to mark it green recursively.
                    if !dep_dep_node.kind.is_eval_always() {
                         debug!("try_mark_previous_green({:?}) --- state of dependency {:?} \
                                 is unknown, trying to mark it green", dep_node,
                                 dep_dep_node);

                        let node_index = self.try_mark_previous_green(
                            tcx,
                            data,
                            dep_dep_node_index,
                            dep_dep_node
                        );
                        if let Some(node_index) = node_index {
                            debug!("try_mark_previous_green({:?}) --- managed to MARK \
                                    dependency {:?} as green", dep_node, dep_dep_node);
                            current_deps.push(node_index);
                            continue;
                        }
                    } else {
                        match dep_dep_node.kind {
                            DepKind::Hir |
                            DepKind::HirBody |
                            DepKind::CrateMetadata => {
                                if dep_dep_node.extract_def_id(tcx).is_none() {
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
                                // For other kinds of nodes it's OK to be
                                // forced.
                            }
                        }
                    }

                    // We failed to mark it green, so we try to force the query.
                    debug!("try_mark_previous_green({:?}) --- trying to force \
                            dependency {:?}", dep_node, dep_dep_node);
                    if crate::ty::query::force_from_dep_node(tcx, dep_dep_node) {
                        let dep_dep_node_color = data.colors.get(dep_dep_node_index);

                        match dep_dep_node_color {
                            Some(DepNodeColor::Green(node_index)) => {
                                debug!("try_mark_previous_green({:?}) --- managed to \
                                        FORCE dependency {:?} to green",
                                        dep_node, dep_dep_node);
                                current_deps.push(node_index);
                            }
                            Some(DepNodeColor::Red) => {
                                debug!("try_mark_previous_green({:?}) - END - \
                                        dependency {:?} was red after forcing",
                                       dep_node,
                                       dep_dep_node);
                                return None
                            }
                            None => {
                                if !tcx.sess.has_errors() {
                                    bug!("try_mark_previous_green() - Forcing the DepNode \
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
                        debug!("try_mark_previous_green({:?}) - END - dependency {:?} \
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

            // Copy the fingerprint from the previous graph,
            // so we don't have to recompute it
            let fingerprint = data.previous.fingerprint_by_index(prev_dep_node_index);

            // We allocating an entry for the node in the current dependency graph and
            // adding all the appropriate edges imported from the previous graph
            current.intern_node(*dep_node, current_deps, fingerprint)
        };

        // ... emitting any stored diagnostic ...

        let diagnostics = tcx.queries.on_disk_cache
                                .load_diagnostics(tcx, prev_dep_node_index);

        if unlikely!(diagnostics.len() > 0) {
            self.emit_diagnostics(
                tcx,
                data,
                dep_node_index,
                did_allocation,
                diagnostics
            );
        }

        // ... and finally storing a "Green" entry in the color map.
        // Multiple threads can all write the same color here
        #[cfg(not(parallel_compiler))]
        debug_assert!(data.colors.get(prev_dep_node_index).is_none(),
                      "DepGraph::try_mark_previous_green() - Duplicate DepNodeColor \
                      insertion for {:?}", dep_node);

        data.colors.insert(prev_dep_node_index, DepNodeColor::Green(dep_node_index));

        debug!("try_mark_previous_green({:?}) - END - successfully marked as green", dep_node);
        Some(dep_node_index)
    }

    /// Atomically emits some loaded diagnotics, assuming that this only gets called with
    /// `did_allocation` set to `true` on a single thread.
    #[cold]
    #[inline(never)]
    fn emit_diagnostics<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        data: &DepGraphData,
        dep_node_index: DepNodeIndex,
        did_allocation: bool,
        diagnostics: Vec<Diagnostic>,
    ) {
        if did_allocation || !cfg!(parallel_compiler) {
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
                // Mark the diagnostics and emitted and wake up waiters
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
    pub fn exec_cache_promotions(&self, tcx: TyCtxt<'_>) {
        let data = self.data.as_ref().unwrap();
        for prev_index in data.colors.values.indices() {
            match data.colors.get(prev_index) {
                Some(DepNodeColor::Green(_)) => {
                    let dep_node = data.previous.index_to_node(prev_index);
                    dep_node.try_load_from_on_disk_cache(tcx);
                }
                None |
                Some(DepNodeColor::Red) => {
                    // We can skip red nodes because a node can only be marked
                    // as red if the query result was recomputed and thus is
                    // already in memory.
                }
            }
        }
    }

    pub fn mark_loaded_from_cache(&self, dep_node_index: DepNodeIndex, state: bool) {
        debug!("mark_loaded_from_cache({:?}, {})",
               self.data.as_ref().unwrap().current.borrow().data[dep_node_index].node,
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
struct DepNodeData {
    node: DepNode,
    edges: SmallVec<[DepNodeIndex; 8]>,
    fingerprint: Fingerprint,
}

pub(super) struct CurrentDepGraph {
    data: IndexVec<DepNodeIndex, DepNodeData>,
    node_to_node_index: FxHashMap<DepNode, DepNodeIndex>,
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

    total_read_count: u64,
    total_duplicate_read_count: u64,
}

impl CurrentDepGraph {
    fn new(prev_graph_node_count: usize) -> CurrentDepGraph {
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
        let new_node_count_estimate = (prev_graph_node_count * 102) / 100 + 200;

        CurrentDepGraph {
            data: IndexVec::with_capacity(new_node_count_estimate),
            node_to_node_index: FxHashMap::with_capacity_and_hasher(
                new_node_count_estimate,
                Default::default(),
            ),
            anon_id_seed: stable_hasher.finish(),
            forbidden_edge,
            total_read_count: 0,
            total_duplicate_read_count: 0,
        }
    }

    fn complete_task(
        &mut self,
        node: DepNode,
        task_deps: TaskDeps,
        fingerprint: Fingerprint
    ) -> DepNodeIndex {
        self.alloc_node(node, task_deps.reads, fingerprint)
    }

    fn complete_anon_task(&mut self, kind: DepKind, task_deps: TaskDeps) -> DepNodeIndex {
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

        self.intern_node(target_dep_node, task_deps.reads, Fingerprint::ZERO).0
    }

    fn alloc_node(
        &mut self,
        dep_node: DepNode,
        edges: SmallVec<[DepNodeIndex; 8]>,
        fingerprint: Fingerprint
    ) -> DepNodeIndex {
        debug_assert!(!self.node_to_node_index.contains_key(&dep_node));
        self.intern_node(dep_node, edges, fingerprint).0
    }

    fn intern_node(
        &mut self,
        dep_node: DepNode,
        edges: SmallVec<[DepNodeIndex; 8]>,
        fingerprint: Fingerprint
    ) -> (DepNodeIndex, bool) {
        debug_assert_eq!(self.node_to_node_index.len(), self.data.len());

        match self.node_to_node_index.entry(dep_node) {
            Entry::Occupied(entry) => (*entry.get(), false),
            Entry::Vacant(entry) => {
                let dep_node_index = DepNodeIndex::new(self.data.len());
                self.data.push(DepNodeData {
                    node: dep_node,
                    edges,
                    fingerprint
                });
                entry.insert(dep_node_index);
                (dep_node_index, true)
            }
        }
    }
}

impl DepGraphData {
    fn read_index(&self, source: DepNodeIndex) {
        ty::tls::with_context_opt(|icx| {
            let icx = if let Some(icx) = icx { icx } else {  return };
            if let Some(task_deps) = icx.task_deps {
                let mut task_deps = task_deps.lock();
                if cfg!(debug_assertions) {
                    self.current.lock().total_read_count += 1;
                }
                if task_deps.read_set.insert(source) {
                    task_deps.reads.push(source);

                    #[cfg(debug_assertions)]
                    {
                        if let Some(target) = task_deps.node {
                            let graph = self.current.lock();
                            if let Some(ref forbidden_edge) = graph.forbidden_edge {
                                let source = graph.data[source].node;
                                if forbidden_edge.test(&source, &target) {
                                    bug!("forbidden edge {:?} -> {:?} created",
                                        source,
                                        target)
                                }
                            }
                        }
                    }
                } else if cfg!(debug_assertions) {
                    self.current.lock().total_duplicate_read_count += 1;
                }
            }
        })
    }
}

pub struct TaskDeps {
    #[cfg(debug_assertions)]
    node: Option<DepNode>,
    reads: SmallVec<[DepNodeIndex; 8]>,
    read_set: FxHashSet<DepNodeIndex>,
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
        DepNodeColorMap {
            values: (0..size).map(|_| AtomicU32::new(COMPRESSED_NONE)).collect(),
        }
    }

    fn get(&self, index: SerializedDepNodeIndex) -> Option<DepNodeColor> {
        match self.values[index].load(Ordering::Acquire) {
            COMPRESSED_NONE => None,
            COMPRESSED_RED => Some(DepNodeColor::Red),
            value => Some(DepNodeColor::Green(DepNodeIndex::from_u32(
                value - COMPRESSED_FIRST_GREEN
            )))
        }
    }

    fn insert(&self, index: SerializedDepNodeIndex, color: DepNodeColor) {
        self.values[index].store(match color {
            DepNodeColor::Red => COMPRESSED_RED,
            DepNodeColor::Green(index) => index.as_u32() + COMPRESSED_FIRST_GREEN,
        }, Ordering::Release)
    }
}
