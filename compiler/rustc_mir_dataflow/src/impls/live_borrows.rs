//! This module defines a more fine-grained analysis for `Local`s that are live due
//! to outstanding references or raw pointers than `MaybeBorrowedLocals`.
//!
//! The analysis consists of three steps:
//!     1. build a dependency graph that relates `Local`s based on their borrowing relationship.
//!        As an example if we have something like this (in a simplified MIR representation):
//!
//!        ```ignore(rust)
//!         _4 = Bar {}
//!         _5 = Ref(_4)
//!        ```
//!
//!         Then we add an edge from `_5` to `_4`.
//!     2. perform a liveness analysis for borrowed `Local`s.
//!        Continuing our example from step 1, if we later have a use of `_5`, `_5` is
//!        live at least from its definition to that use of it.
//!     3. Combine the two analyses from step 1 and 2. For any `Local` that corresponds
//!        to a borrow (`_5` in our example), we want to keep the `Local` (`_4`), which is actually
//!        borrowed through it, live over the range at which the borrow is live. Hence for any point
//!        in that range we traverse our dependency graph and look for leaf nodes. In our example
//!        we would find an edge from `_5` to `_4`, which is a leaf node and hence we keep `_4` live
//!        over that range.
//!
//! There are some corner cases we need to look out for to make this analysis sound. Let's look
//! at each of the three steps in more detail and elaborate how these steps deal with these corner
//! cases.
//!
//! 1. Dependency Graph
//!
//! The `Node`s in the dependency graph include data values of type `NodeKind`. `NodeKind` has
//! three variants: `Local`, `Borrow` and `LocalWithRefs`.
//!     * `NodeKind::Local` is used for `Local`s that are borrowed somewhere (`_4` in our example), but aren't
//!        themselves references or pointers.
//!     * `NodeKind::Borrow` is used for `Local`s that correspond to borrows (`_5` in our example) and
//!        also `Local`s that result from re-borrows.
//!     * `NodeKind::LocalWithRefs` is used for `Local`s that aren't themselves borrows, but contain
//!        `Local`s that correspond to references, pointers or other `Local`s with `Node`s of kind
//!        `NodeKind::LocalWithRef`s. Let's look at an example:
//!
//!        ```ignore(rust)
//!         _4 = Bar {}
//!         _5 = Ref(_4)
//!         _6 = Aggregate(..)(move _5)
//!         ...
//!         _7 = (_6.0)
//!         ```
//!
//!         In this example `_6` would be given `NodeKind::LocalWithRefs` and our graph would look
//!         as follows:
//!
//!         `_7` (NodeKind::Borrow) -> `_6` (NodeKind::LocalWithRefs) -> `_5` (NodeKind::Borrow) -> `_4` (NodeKind::Local)
//!
//!         On the one hand we need to treat `Local`s with `Node`s of kind `NodeKind::LocalWithRefs` similarly
//!         to how we treat `Local`s with `Node`s of kind `NodeKind::Local`, in the sense that if they are
//!         borrowed we want to keep them live over the live range of the borrow. But on the other hand we
//!         want to also treat them like `Local`s with `Node`s of kind `NodeKind::Borrow` as they ultimately
//!         could also contain references or pointers that refer to other `Local`s. So we want a
//!         path in the graph from a `NodeKind::LocalWithRef`s node to the `NodeKind::Local` nodes, whose borrows
//!         they might contain.
//!
//!         Additionally `NodeKind::LocalWithRefs` is also used for raw pointers that are cast to
//!         `usize`:
//!
//!         ```ignore(rust)
//!         _4 = Bar {}
//!         _5 = AddressOf(_4)
//!         _6 = _5 as usize
//!         _7 = Aggregate(..) (move _6)
//!         _8 = (_7.0)
//!         ```
//!
//!         In this example our graph would have the following edges:
//!             * `_5` (Borrow) -> `_4` (Local)
//!             * `_6` (LocalWithRefs) -> `_5` (Borrow)
//!             * `_7` (LocalWithRefs) -> `_6` (LocalWithRefs)
//!             * `_8` (LocalWithRefs) -> `_7` (LocalWithRefs)
//!
//!         We also have to be careful when dealing with `Terminator`s. Whenever we pass references,
//!         pointers or `Local`s with `NodeKind::LocalWithRefs` to a `TerminatorKind::Call` or
//!         `TerminatorKind::Yield`, the destination `Place` or resume place, resp., might contain
//!         these references, pointers or `NodeKind::LocalWithRefs` `Local`s, hence we have to be conservative
//!         and keep the `destination` `Local` and `resume_arg` `Local` live.
//!
//! 2. Liveness analysis for borrows
//!
//! We perform a standard liveness analysis on any outstanding references, pointers or `Local`s
//! with `NodeKind::LocalWithRefs`. So we `gen` at any use site, which are either direct uses
//! of these `Local`s or projections that contain these `Local`s. So e.g.:
//!
//! ```ignore(rust)
//! 1. _3 = Foo {}
//! 2. _4 = Bar {}
//! 3. _5 = Ref(_3)
//! 4. _6 = Ref(_4)
//! 5. _7 = Aggregate(..)(move _5)
//! 6. _8 = Call(..)(move _6)
//! 7. _9 = (_8.0)
//! 8. _10 = const 5
//! 9. (_7.0) = move _10
//! ```
//!
//! * `_5` is live from stmt 3 to stmt 5
//! * `_6` is live from stmt 4 to stmt 6
//! * `_7` is a `Local` of kind `LocalWithRefs` so needs to be taken into account in the
//!   analyis. It's live from stmt 5 to stmt 9
//! * `_8` is a `Local` of kind `LocalWithRefs`. It's live from 6. to 7.
//! * `_9` is a `Local` of kind `LocalWithRefs`
//!   it's live at 7.
//!
//! 3. Determining which `Local`s are borrowed
//!
//! Let's use our last example again. The dependency graph for that example looks as follows:
//!
//! `_5` (Borrow) -> `_3` (Local)
//! `_6` (Borrow) -> `_4` (Local)
//! `_7` (LocalWithRef) -> `_5` (Borrow)
//! `_8` (LocalWithRef) -> `_6` (Borrow)
//! `_9` (LocalWithRef) -> `_8` (LocalWithRef)
//! `_7` (LocalWithRef) -> `_10` (Local)
//!
//! So at each of those statements we have the following `Local`s that are live due to borrows:
//!
//! 1. {}
//! 2. {}
//! 3. {_3}
//! 4. {_3, _4}
//! 5. {_3, _4, _7}
//! 6. {_3, _4, _7, _8}
//! 7. {_3, _4, _7, _8}
//! 8. {_3, _7}
//! 9. {_3, _7}
//!

use super::*;

use crate::framework::{Analysis, Results, ResultsCursor};
use crate::{
    AnalysisDomain, Backward, CallReturnPlaces, GenKill, GenKillAnalysis, ResultsRefCursor,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::implementation::{Graph, NodeIndex};
use rustc_middle::mir::visit::PlaceContext;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty;

use either::Either;

#[derive(Copy, Clone, Debug)]
enum NodeKind {
    // An node corresponding to the place on the lhs of an assignment like `_3 = Ref(_, _, _4)`.
    Borrow(Local),

    // Nodes corresponding to the place on the lhs of a statement like
    // `_2 = Aggregate(Adt(..), _, _, _, _), [move _3, move _6])`,
    // where _3 and _6 are places corresponding to references or raw pointers.
    LocalWithRefs(Local),

    // Nodes corresponding to the borrowed place of an assignment like `_2 = Ref(_, _, borrowed_place)`,
    // if `borrowed_place` is a non-ref or non-ptr value.
    Local(Local),
}

impl NodeKind {
    fn get_local(&self) -> Local {
        match self {
            Self::Borrow(local) => *local,
            Self::LocalWithRefs(local) => *local,
            Self::Local(local) => *local,
        }
    }
}

/// Used to build a dependency graph between borrows/pointers and the `Local`s that
/// they reference.
/// We add edges to the graph in following situations:
///     * direct assignment of reference or raw pointer (e.g. `_4 = Ref(_, _ , borrowed_place)` or
///       `_4 = AddressOf(_, borrowed_place)`). For this case we create a `Node` of kind
///       `NodeKind::Borrow` for the `Local` being assigned to and an edge to either an existing
///       `Node` or if none exists yet to a new `Node` of type `NodeKind::Local` corresponding to
///       a non-ref/ptr `Local`.
///     * assignments to non-reference or non-pointer `Local`s, which themselves might contain
///       references or pointers (e.g. `_2 = Aggregate(Adt(..), _, _, _, _), [move _3, move _6])`,
///       where `_3` and `_6` are places corresponding to references or raw pointers). In this case
///       we create a `Node` of kind `NodeKind::LocalWithRefs` for `_2`. Since `_3` and `_6` are
///       `Local`s that correspond to references, pointers or composite types that might contain
///       references or pointers (`NodeKind::LocalWithRefs`), there already exist `Node`s for these
///       `Local`s. We then add edges from the `Node` for `_2` to both the `Node` for `_3` and the
///       `Node` for `_6`.
///     * `destination` places for `TerminatorKind::Call` and the `resume_arg` places for
///        `TerminatorKind::Yield` if we pass in any references, pointers or composite values that
///        might correspond to references, pointers or exposed pointers (`NodeKind::LocalWithRef`s).
///        The rationale for this is that the return values of both of these terminators might themselves
///        contain any of the references or pointers passed as arguments.
struct BorrowDependencies<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,

    // Maps `Local`s, for which we have nodes in the graph, to the `NodeIndex`es of those nodes.
    locals_to_node_indexes: FxHashMap<Local, NodeIndex>,

    // Tracks the dependencies of places and the references/pointers they may contain,
    // e.g. if we have `_3 = Ref(_, _, _2)` we add an edge from _3 to _2. We later use
    // this graph to allow us to infer which locals need to be kept live in the
    // liveness analysis.
    dep_graph: Graph<NodeKind, ()>,

    // Contains the `Local` to which we're currently assigning.
    current_local: Option<Local>,
}

impl<'a, 'tcx> BorrowDependencies<'a, 'tcx> {
    #[instrument(skip(local_decls, tcx), level = "debug")]
    fn new(local_decls: &'a LocalDecls<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        let num_nodes = local_decls.len();
        let approx_num_edges = 3 * num_nodes;

        BorrowDependencies {
            tcx,
            local_decls,
            dep_graph: Graph::with_capacity(num_nodes, approx_num_edges),
            current_local: None,
            locals_to_node_indexes: Default::default(),
        }
    }

    fn maybe_create_node(&mut self, node_kind: NodeKind) -> NodeIndex {
        let local = node_kind.get_local();
        if let Some(node_idx) = self.locals_to_node_indexes.get(&local) {
            *node_idx
        } else {
            let node_idx = self.dep_graph.add_node(node_kind);
            self.locals_to_node_indexes.insert(local, node_idx);
            node_idx
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BorrowDependencies<'a, 'tcx> {
    #[instrument(skip(self, body), level = "debug")]
    fn visit_body(&mut self, body: &Body<'tcx>) {
        // Add nodes for the arguments
        for i in 1..=body.arg_count {
            let local = Local::from_usize(i);
            let node_kind = NodeKind::Local(local);
            let node_idx = self.dep_graph.add_node(node_kind);
            self.locals_to_node_indexes.insert(local, node_idx);
        }

        self.super_body(body)
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(assign) => {
                let assign_place = assign.0;
                debug!("assign_place_ty: {:?}", assign_place.ty(self.local_decls, self.tcx).ty);
                let assign_local = assign_place.local;
                self.current_local = Some(assign_local);
                debug!("set current_local to {:?}", self.current_local);

                // Do not call `visit_place` here as this might introduce a self-edge, which our liveness analysis
                // assumes not to exist.
                self.visit_rvalue(&assign.1, location)
            }
            StatementKind::FakeRead(..)
            | StatementKind::StorageDead(_)
            | StatementKind::StorageLive(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Deinit(_)
            | StatementKind::Coverage(_) => {}
            _ => {
                self.super_statement(statement, location);
            }
        }

        self.current_local = None;
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        debug!("self.current_local: {:?}", self.current_local);
        match rvalue {
            Rvalue::Use(Operand::Move(place) | Operand::Copy(place))
                if matches!(
                    place.ty(self.local_decls, self.tcx).ty.kind(),
                    ty::Ref(..) | ty::RawPtr(..)
                ) =>
            {
                // these are just re-assignments of already outstanding refs or pointers,
                // hence we want to treat them as `NodeKind::Borrow`
                // FIXME Are these always Operand::Copy or is Operand::Move also possible for refs/ptrs?
                let Some(src_local) = self.current_local else {
                        bug!("Expected self.current_local to be set when encountering Rvalue");
                    };

                // These are just moves of refs/ptrs, hence `NodeKind::Borrow`.
                let src_node_kind = NodeKind::Borrow(src_local);
                let src_node_idx = self.maybe_create_node(src_node_kind);

                let node_kind = NodeKind::Borrow(place.local);
                let node_idx = self.maybe_create_node(node_kind);

                debug!(
                    "adding edge from {:?}({:?}) -> {:?}({:?})",
                    src_node_idx,
                    self.dep_graph.node(src_node_idx).data,
                    node_idx,
                    self.dep_graph.node(node_idx).data,
                );

                self.dep_graph.add_edge(src_node_idx, node_idx, ());
            }
            Rvalue::Ref(_, _, borrowed_place) | Rvalue::AddressOf(_, borrowed_place) => {
                let Some(src_local) = self.current_local else {
                    bug!("Expected self.current_local to be set with Rvalue::Ref|Rvalue::AddressOf");
                };

                // we're in a statement like `_4 = Ref(..)`, hence NodeKind::Borrow for `_4`
                let src_node_kind = NodeKind::Borrow(src_local);
                let src_node_idx = self.maybe_create_node(src_node_kind);

                // If we haven't previously added a node for `borrowed_place.local` then it can be neither
                // `NodeKind::Borrow` nor `NodeKind::LocalsWithRefs`.
                let borrowed_node_kind = NodeKind::Local(borrowed_place.local);
                let node_idx = self.maybe_create_node(borrowed_node_kind);

                debug!(
                    "adding edge from {:?}({:?}) -> {:?}({:?})",
                    src_node_idx,
                    self.dep_graph.node(src_node_idx).data,
                    node_idx,
                    self.dep_graph.node(node_idx).data,
                );

                self.dep_graph.add_edge(src_node_idx, node_idx, ());
            }
            Rvalue::Cast(..) => {
                // FIXME we probably should handle pointer casts here directly

                self.super_rvalue(rvalue, location)
            }
            _ => self.super_rvalue(rvalue, location),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // Add edges for places that correspond to references/raw pointers or nodes of kind `LocalWithRefs`
        let place_ty = place.ty(self.local_decls, self.tcx).ty;
        debug!(?place_ty);
        debug!("current local: {:?}", self.current_local);

        match self.current_local {
            Some(src_local) => {
                match place_ty.kind() {
                    ty::Ref(..) | ty::RawPtr(..) => {
                        // If we haven't created a node for this before, then this must be a
                        // `NodeKind::LocalWithRefs` as we would have handled the
                        // other possible assignment case (`NodeKind::Local`) previously in
                        // `visit_rvalue`.
                        let src_node_kind = NodeKind::LocalWithRefs(src_local);
                        let src_node_idx = self.maybe_create_node(src_node_kind);

                        let borrowed_node_kind = NodeKind::Borrow(place.local);
                        let node_idx = self.maybe_create_node(borrowed_node_kind);

                        debug!(
                            "adding edge from {:?}({:?}) -> {:?}({:?})",
                            src_node_idx,
                            self.dep_graph.node(src_node_idx).data,
                            node_idx,
                            place.local
                        );

                        self.dep_graph.add_edge(src_node_idx, node_idx, ());
                    }
                    _ => {
                        if let Some(node_idx) = self.locals_to_node_indexes.get(&place.local) {
                            // LocalsWithRefs -> LocalWithRefs

                            let node_idx = *node_idx;
                            let src_node_kind = NodeKind::LocalWithRefs(src_local);
                            let src_node_idx = self.maybe_create_node(src_node_kind);

                            debug!(
                                "adding edge from {:?}({:?}) -> {:?}({:?})",
                                src_node_idx,
                                self.dep_graph.node(src_node_idx).data,
                                node_idx,
                                self.dep_graph.node(node_idx).data,
                            );

                            self.dep_graph.add_edge(src_node_idx, node_idx, ());
                        }
                    }
                }
            }
            _ => {}
        }

        self.super_place(place, context, location)
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        self.super_operand(operand, location)
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Call { destination, func, args, .. } => {
                let dest_ty = destination.ty(self.local_decls, self.tcx).ty;
                debug!(?dest_ty);

                // To ensure safety we need to add `destination` to the graph as a `Node` with `NodeKind::LocalWithRefs`
                // if we pass in any refs/ptrs or `Local`s corresponding to `NodeKind::LocalWithRefs`. The reason for this
                // is that the function could include those refs/ptrs in its return value. It's not sufficient
                // to look for the existence of `ty::Ref` or `ty::RawPtr` in the type of the return type, since the
                // function could also cast pointers to integers e.g. .
                self.current_local = Some(destination.local);

                self.visit_operand(func, location);
                for arg in args {
                    self.visit_operand(arg, location);
                }

                self.current_local = None;
            }
            TerminatorKind::Yield { resume_arg, value, .. } => {
                let resume_arg_ty = resume_arg.ty(self.local_decls, self.tcx).ty;
                debug!(?resume_arg_ty);

                // We may need to add edges from `destination`, see the comment for this statement
                // in `TerminatorKind::Call` for the rationale behind this.
                self.current_local = Some(resume_arg.local);

                self.super_operand(value, location);
            }
            _ => self.super_terminator(terminator, location),
        }
    }
}

pub struct BorrowedLocalsResults<'mir, 'tcx> {
    // the results of the liveness analysis of `LiveBorrows`
    borrows_analysis_results: Results<'tcx, LiveBorrows<'mir, 'tcx>>,

    // Maps each `Local` that corresponds to a reference, pointer or a node of kind
    // `NodeKind::LocalWithRefs` (i.e. `Local`s which either correspond to refs, pointers or
    // exposed pointers or a composite value that might include refs, pointers or exposed pointers)
    // to the set of `Local`s that are borrowed through those references, pointers or composite values.
    borrowed_local_to_locals_to_keep_alive: FxHashMap<Local, Vec<Local>>,
}

impl<'mir, 'tcx> BorrowedLocalsResults<'mir, 'tcx>
where
    'tcx: 'mir,
{
    fn new(borrows_analysis_results: Results<'tcx, LiveBorrows<'mir, 'tcx>>) -> Self {
        let dep_graph = &borrows_analysis_results.analysis.borrow_deps.dep_graph;
        let borrowed_local_to_locals_to_keep_alive = Self::get_locals_to_keep_alive_map(dep_graph);
        Self { borrows_analysis_results, borrowed_local_to_locals_to_keep_alive }
    }

    /// Uses the dependency graph to find all locals that we need to keep live for a given
    /// `Node` (or more specically the `Local` corresponding to that `Node`).
    fn get_locals_to_keep_alive_map<'a>(
        dep_graph: &'a Graph<NodeKind, ()>,
    ) -> FxHashMap<Local, Vec<Local>> {
        let mut borrows_to_locals: FxHashMap<Local, Vec<Local>> = Default::default();
        let mut locals_visited = vec![];

        for (node_idx, node) in dep_graph.enumerated_nodes() {
            let current_local = node.data.get_local();
            if borrows_to_locals.get(&current_local).is_none() {
                Self::dfs_for_node(
                    node_idx,
                    &mut borrows_to_locals,
                    dep_graph,
                    &mut locals_visited,
                );
            }
        }

        debug!("borrows_to_locals: {:#?}", borrows_to_locals);
        borrows_to_locals
    }

    fn dfs_for_node(
        node_idx: NodeIndex,
        borrows_to_locals: &mut FxHashMap<Local, Vec<Local>>,
        dep_graph: &Graph<NodeKind, ()>,
        locals_visited: &mut Vec<Local>,
    ) -> Vec<Local> {
        let src_node = dep_graph.node(node_idx);
        let current_local = src_node.data.get_local();
        locals_visited.push(current_local);
        if let Some(locals_to_keep_alive) = borrows_to_locals.get(&current_local) {
            // already traversed this node
            return (*locals_to_keep_alive).clone();
        }

        let mut num_succs = 0;
        let mut locals_for_node = vec![];
        for (_, edge) in dep_graph.outgoing_edges(node_idx) {
            num_succs += 1;
            let target_node_idx = edge.target();
            let target_node = dep_graph.node(target_node_idx);
            let target_local = target_node.data.get_local();
            if locals_visited.contains(&target_local) {
                continue;
            }

            debug!(
                "edge {:?} ({:?}) -> {:?} ({:?})",
                node_idx, src_node.data, target_node_idx, target_node.data,
            );

            let mut locals_to_keep_alive_for_succ =
                Self::dfs_for_node(target_node_idx, borrows_to_locals, dep_graph, locals_visited);
            locals_for_node.append(&mut locals_to_keep_alive_for_succ);
        }

        match src_node.data {
            NodeKind::Local(_) => {
                assert!(num_succs == 0, "Local node with successors");
                return vec![src_node.data.get_local()];
            }
            NodeKind::LocalWithRefs(_) => {
                // These are locals that we need to keep alive, but that also contain
                // successors in the graph since they contain other references/pointers.
                locals_for_node.push(current_local);
            }
            NodeKind::Borrow(_) => {}
        }

        borrows_to_locals.insert(current_local, locals_for_node.clone());
        locals_for_node
    }
}

/// The function gets the results of the borrowed locals analysis in this module. See the module
/// doc-comment for information on what exactly this analysis does.
#[instrument(skip(tcx), level = "debug")]
pub fn get_borrowed_locals_results<'mir, 'tcx>(
    body: &'mir Body<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> BorrowedLocalsResults<'mir, 'tcx> {
    debug!("body: {:#?}", body);

    let mut borrow_deps = BorrowDependencies::new(body.local_decls(), tcx);
    borrow_deps.visit_body(body);

    if cfg!(debug_assertions) {
        let dep_graph = &borrow_deps.dep_graph;

        debug!(
            "nodes: {:#?}",
            dep_graph
                .all_nodes()
                .clone()
                .into_iter()
                .enumerate()
                .map(|(i, node)| (i, node.data))
                .collect::<Vec<_>>()
        );

        debug!("edges:");
        for edge in dep_graph.all_edges() {
            let src_node_idx = edge.source();
            let src_node = dep_graph.node(src_node_idx);
            let target_node_idx = edge.target();
            let target_node = dep_graph.node(target_node_idx);
            debug!(
                "{:?}({:?}) -> {:?}({:?}) ({:?})",
                src_node_idx, src_node.data, target_node_idx, target_node.data, edge.data
            )
        }
    }

    let live_borrows = LiveBorrows::new(body, tcx, borrow_deps);
    let results =
        live_borrows.into_engine(tcx, body).pass_name("borrowed_locals").iterate_to_fixpoint();

    BorrowedLocalsResults::new(results)
}

/// The `ResultsCursor` equivalent for the borrowed locals analysis. Since this analysis doesn't
/// require convergence, we expose the set of borrowed `Local`s for a `Location` directly via
/// the `get` method without the need for any prior 'seek' calls.
pub struct BorrowedLocalsResultsCursor<'a, 'mir, 'tcx> {
    body: &'mir Body<'tcx>,

    // The cursor for the liveness analysis performed by `LiveBorrows`
    borrows_analysis_cursor: ResultsRefCursor<'a, 'mir, 'tcx, LiveBorrows<'mir, 'tcx>>,

    // Maps each `Local` corresponding to a reference or pointer to the set of `Local`s
    // that are borrowed through the ref/ptr. Additionally contains entries for `Local`s
    // corresponding to `NodeKind::LocalWithRefs` since they might contain refs, ptrs or
    // exposed pointers and need to be treated equivalently to refs/ptrs
    borrowed_local_to_locals_to_keep_alive: &'a FxHashMap<Local, Vec<Local>>,
}

impl<'a, 'mir, 'tcx> BorrowedLocalsResultsCursor<'a, 'mir, 'tcx> {
    pub fn new(body: &'mir Body<'tcx>, results: &'a BorrowedLocalsResults<'mir, 'tcx>) -> Self {
        let mut cursor = ResultsCursor::new(body, &results.borrows_analysis_results);

        // We don't care about the order of the blocks, only about the result at a given location.
        // This statement is necessary since we're performing a backward analysis in `LiveBorrows`,
        // but want `Self::get` to be usable in forward analyses as well.
        cursor.allow_unreachable();

        Self {
            body,
            borrows_analysis_cursor: cursor,
            borrowed_local_to_locals_to_keep_alive: &results.borrowed_local_to_locals_to_keep_alive,
        }
    }

    /// Returns all `Local`s that need to be live at the given `Location` because of
    /// outstanding references or raw pointers.
    pub fn get(&mut self, loc: Location) -> BitSet<Local> {
        self.borrows_analysis_cursor.seek_before_primary_effect(loc);
        let live_borrows_at_loc = self.borrows_analysis_cursor.get();
        debug!(?live_borrows_at_loc);

        let mut borrowed_locals = BitSet::new_empty(live_borrows_at_loc.domain_size());
        for borrowed_local in live_borrows_at_loc.iter() {
            debug!(?borrowed_local);
            if let Some(locals_to_keep_alive) =
                self.borrowed_local_to_locals_to_keep_alive.get(&borrowed_local)
            {
                debug!(?locals_to_keep_alive);
                for local in locals_to_keep_alive.iter() {
                    borrowed_locals.insert(*local);
                }
            }
        }

        match self.body.stmt_at(loc) {
            Either::Right(terminator) => {
                match terminator.kind {
                    TerminatorKind::Drop { place: dropped_place, .. } => {
                        // Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut
                        // self` as a parameter. In the general case, a drop impl could launder that
                        // reference into the surrounding environment through a raw pointer, thus creating
                        // a valid `*mut` pointing to the dropped local. We are not yet willing to declare
                        // this particular case UB, so we must treat all dropped locals as mutably borrowed
                        // for now. See discussion on [#61069].
                        //
                        // [#61069]: https://github.com/rust-lang/rust/pull/61069
                        if !dropped_place.is_indirect() {
                            borrowed_locals.insert(dropped_place.local);
                        }
                    }
                    _ => {}
                }
            }
            Either::Left(_) => {}
        }

        debug!(?borrowed_locals);
        borrowed_locals
    }
}

/// Performs a liveness analysis for borrows and raw pointers. This analysis also tracks `Local`s
/// corresponding to `Node`s of kind `NodeKind::LocalWithRefs`, as these could potentially refer to
/// or include references, pointers or exposed pointers.
pub struct LiveBorrows<'mir, 'tcx> {
    body: &'mir Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    borrow_deps: BorrowDependencies<'mir, 'tcx>,
}

impl<'mir, 'tcx> LiveBorrows<'mir, 'tcx> {
    fn new(
        body: &'mir Body<'tcx>,
        tcx: TyCtxt<'tcx>,
        borrow_deps: BorrowDependencies<'mir, 'tcx>,
    ) -> Self {
        LiveBorrows { body, tcx, borrow_deps }
    }

    fn transfer_function<'b, T>(
        &self,
        trans: &'b mut T,
    ) -> TransferFunction<'mir, 'b, '_, 'tcx, T> {
        TransferFunction {
            body: self.body,
            tcx: self.tcx,
            _trans: trans,
            borrow_deps: &self.borrow_deps,
        }
    }
}

impl<'a, 'tcx> AnalysisDomain<'tcx> for LiveBorrows<'a, 'tcx> {
    type Domain = BitSet<Local>;
    type Direction = Backward;
    const NAME: &'static str = "live_borrows";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        BitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // Not supported for backward analyses
    }
}

impl<'a, 'tcx> GenKillAnalysis<'tcx> for LiveBorrows<'a, 'tcx> {
    type Idx = Local;

    #[instrument(skip(self, trans), level = "debug")]
    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    #[instrument(skip(self, trans), level = "debug")]
    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_terminator(terminator, location);
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }
}

/// A `Visitor` that defines the transfer function for `LiveBorrows`.
struct TransferFunction<'a, 'b, 'c, 'tcx, T> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    _trans: &'b mut T,
    borrow_deps: &'c BorrowDependencies<'a, 'tcx>,
}

impl<'a, 'tcx, T> Visitor<'tcx> for TransferFunction<'a, '_, '_, 'tcx, T>
where
    T: GenKill<Local>,
{
    #[instrument(skip(self), level = "debug")]
    fn visit_statement(&mut self, stmt: &Statement<'tcx>, location: Location) {
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let lhs_place = assign.0;
                let projection = lhs_place.projection;
                let lhs_place_ty = lhs_place.ty(self.body.local_decls(), self.tcx).ty;
                debug!(?lhs_place, ?lhs_place_ty);

                match projection.as_slice() {
                    &[] | &[ProjectionElem::OpaqueCast(_)] => {
                        // If there aren't any projections or just an OpaqueCast we need to
                        // kill the local.
                        match lhs_place_ty.kind() {
                            ty::Ref(..) | ty::RawPtr(..) => {
                                debug!("killing {:?}", lhs_place.local);
                                self._trans.kill(lhs_place.local);

                                self.visit_rvalue(&assign.1, location);
                            }
                            _ => {
                                if let Some(node_idx) =
                                    self.borrow_deps.locals_to_node_indexes.get(&lhs_place.local)
                                {
                                    let node = self.borrow_deps.dep_graph.node(*node_idx);
                                    if let NodeKind::LocalWithRefs(_) = node.data {
                                        debug!("killing {:?}", lhs_place.local);
                                        self._trans.kill(lhs_place.local);
                                    }
                                }
                                self.super_assign(&assign.0, &assign.1, location);
                            }
                        }
                    }
                    _ => {
                        // With any other projection elements, a projection of a local (of type ref/ptr)
                        // is actually a use-site, but we handle this in the call to `visit_place`.
                        self.super_assign(&assign.0, &assign.1, location);
                    }
                }
            }
            StatementKind::FakeRead(..)
            | StatementKind::StorageDead(_)
            | StatementKind::StorageLive(_) => {}
            _ => self.super_statement(stmt, location),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        context: PlaceContext,
        location: mir::Location,
    ) {
        let local = place.local;
        let local_ty = self.body.local_decls()[local].ty;
        debug!(?local_ty);

        match local_ty.kind() {
            ty::Ref(..) | ty::RawPtr(..) => {
                debug!("gen {:?}", local);
                self._trans.gen(local);
            }
            _ => {}
        }

        self.super_place(place, context, location);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: mir::Location) {
        self.super_operand(operand, location);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Call { destination, args, .. } => {
                match destination.projection.as_slice() {
                    &[] | &[ProjectionElem::OpaqueCast(_)] => {
                        debug!("killing {:?}", destination.local);
                        self._trans.kill(destination.local);

                        for arg in args {
                            match arg {
                                Operand::Copy(place) | Operand::Move(place) => {
                                    if let Some(node_idx) =
                                        self.borrow_deps.locals_to_node_indexes.get(&place.local)
                                    {
                                        let node = self.borrow_deps.dep_graph.node(*node_idx);

                                        // these are `Local`s that contain references/pointers or are raw pointers
                                        // that were assigned to raw pointers, which were cast to usize. Since the
                                        // function call is free to use these in any form, we need to gen them here.
                                        if let NodeKind::LocalWithRefs(_) = node.data {
                                            debug!("gen {:?}", place.local);
                                            self._trans.gen(place.local);
                                        }
                                    } else {
                                        self.super_operand(arg, location)
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => self.super_terminator(terminator, location),
                }
            }
            TerminatorKind::Yield { resume_arg, value, .. } => {
                match resume_arg.projection.as_slice() {
                    &[] | &[ProjectionElem::OpaqueCast(_)] => {
                        debug!("killing {:?}", resume_arg.local);
                        self._trans.kill(resume_arg.local);

                        match value {
                            Operand::Copy(place) | Operand::Move(place) => {
                                if let Some(node_idx) =
                                    self.borrow_deps.locals_to_node_indexes.get(&place.local)
                                {
                                    let node = self.borrow_deps.dep_graph.node(*node_idx);

                                    // these are `Local`s that contain references/pointers or are raw pointers
                                    // that were assigned to raw pointers, which were cast to usize. Since the
                                    // function call is free to use these in any form, we need to gen them here.
                                    if let NodeKind::LocalWithRefs(_) = node.data {
                                        debug!("gen {:?}", place.local);
                                        self._trans.gen(place.local);
                                    }
                                } else {
                                    self.super_operand(value, location)
                                }
                            }
                            _ => {}
                        }
                        self.visit_operand(value, location)
                    }
                    _ => self.super_terminator(terminator, location),
                }
            }
            _ => self.super_terminator(terminator, location),
        }
    }
}
