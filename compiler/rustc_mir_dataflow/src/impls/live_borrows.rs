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
use rustc_middle::ty::{self, Ty, TypeSuperVisitable, TypeVisitable};

use core::ops::ControlFlow;
use either::Either;

/// This module defines a more fine-grained analysis for `Local`s that are live due
/// to outstanding references or raw pointers.
/// The idea behind the analysis is that we first build a dependency graph between
/// `Local`s corresponding to references or pointers and the `Local`s that are borrowed.
/// This is done by the `BorrowDependencies` struct.
/// As a second step we perform a liveness analysis for references and pointers, which is
/// done by `LiveBorrows`.
/// Finally we combine the results of the liveness analysis and the dependency graph to
/// infer which borrowed locals need to be live at a given `Location`.

#[derive(Copy, Clone, Debug)]
enum NodeKind {
    // An node corresponding to the place on the lhs of a statement like
    // `_3 = Ref(_, _, _4)`
    Borrow(Local),

    // Nodes corresponding to the place on the lhs of a statement like
    // `_2 = Aggregate(Adt(..), _, _, _, _), [move _3, move _6])`,
    // where _3 and _6 are places corresponding to references or raw pointers
    // or locals that are borrowed (`_2 = Ref(..)`).
    LocalOrLocalWithRefs(Local),
}

impl NodeKind {
    fn get_local(&self) -> Local {
        match self {
            Self::Borrow(local) => *local,
            Self::LocalOrLocalWithRefs(local) => *local,
        }
    }
}

/// TypeVisitor that looks for `ty::Ref`, `ty::RawPtr`. Additionally looks for `ty::Param`,
/// which could themselves refer to references or raw pointers.
struct MaybeHasRefsOrPointersVisitor {
    has_refs_or_pointers: bool,
    has_params: bool,
}

impl MaybeHasRefsOrPointersVisitor {
    fn new() -> Self {
        MaybeHasRefsOrPointersVisitor { has_refs_or_pointers: false, has_params: false }
    }
}

impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for MaybeHasRefsOrPointersVisitor {
    type BreakTy = ();

    #[instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match ty.kind() {
            ty::Ref(..) => {
                self.has_refs_or_pointers = true;
                return ControlFlow::Break(());
            }
            ty::RawPtr(..) => {
                self.has_refs_or_pointers = true;
                return ControlFlow::Break(());
            }
            ty::Param(_) => {
                self.has_params = true;
            }
            _ => {}
        }

        ty.super_visit_with(self)
    }
}

/// Used to build a dependency graph between borrows/pointers and the `Local`s that
/// they reference.
/// We add edges to the graph in two kinds of situations:
///     * direct assignment of reference or raw pointer (e.g. `_4 = Ref(..)` or `_4 = AddressOf`)
///     * assignments to non-reference or non-pointer `Local`s, which themselves might contain
///       references or pointers (e.g. `_2 = Aggregate(Adt(..), _, _, _, _), [move _3, move _6])`,
///       where `_3` and `_6` are places corresponding to references or raw pointers).
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

    // Contains the `NodeIndex` corresponding to the local to which we're currently
    // assigning.
    current_local: Option<NodeIndex>,
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
}

impl<'a, 'tcx> Visitor<'tcx> for BorrowDependencies<'a, 'tcx> {
    #[instrument(skip(self, body), level = "debug")]
    fn visit_body(&mut self, body: &Body<'tcx>) {
        // Add nodes for the arguments
        for i in 1..=body.arg_count {
            let local = Local::from_usize(i);
            let node_kind = NodeKind::LocalOrLocalWithRefs(local);
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
                let assign_local = assign_place.local;

                let node_kind = match assign_place.ty(self.local_decls, self.tcx).ty.kind() {
                    ty::Ref(..) | ty::RawPtr(..) => NodeKind::Borrow(assign_local),
                    _ => NodeKind::LocalOrLocalWithRefs(assign_local),
                };

                let node_idx = self.dep_graph.add_node(node_kind);
                self.locals_to_node_indexes.insert(assign_local, node_idx);
                self.current_local = Some(node_idx);

                debug!(
                    "set current_local to {:?} for local {:?}",
                    self.current_local, assign_local
                );

                // Do not call `visit_place` here as this might introduce a self-edge, which our liveness analysis
                // assumes not to exist.
                self.visit_rvalue(&assign.1, location)
            }
            StatementKind::FakeRead(..)
            | StatementKind::StorageDead(_)
            | StatementKind::StorageLive(_) => {}
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
            Rvalue::Ref(_, _, borrowed_place) | Rvalue::AddressOf(_, borrowed_place) => {
                let Some(src_node_idx) = self.current_local else {
                    bug!("Expected self.current_local to be set with Rvalue::Ref|Rvalue::AddressOf");
                };

                let borrowed_local = borrowed_place.local;
                let node_idx =
                    if let Some(node_idx) = self.locals_to_node_indexes.get(&borrowed_local) {
                        *node_idx
                    } else {
                        self.dep_graph.add_node(NodeKind::Borrow(borrowed_place.local))
                    };

                debug!(
                    "adding edge from {:?}({:?}) -> {:?}({:?})",
                    src_node_idx,
                    self.dep_graph.node(src_node_idx).data,
                    node_idx,
                    self.dep_graph.node(node_idx).data,
                );

                self.dep_graph.add_edge(src_node_idx, node_idx, ());
            }
            _ => self.super_rvalue(rvalue, location),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // Add edges for places that correspond to references or raw pointers
        let place_ty = place.ty(self.local_decls, self.tcx).ty;
        debug!(?place_ty);
        debug!("current local: {:?}", self.current_local);
        match place_ty.kind() {
            ty::Ref(..) | ty::RawPtr(..) => match self.current_local {
                Some(src_node_idx) => {
                    let borrowed_local = place.local;
                    let node_idx =
                        if let Some(node_idx) = self.locals_to_node_indexes.get(&borrowed_local) {
                            *node_idx
                        } else {
                            self.dep_graph.add_node(NodeKind::Borrow(borrowed_local))
                        };

                    debug!(
                        "adding edge from {:?}({:?}) -> {:?}({:?})",
                        src_node_idx,
                        self.dep_graph.node(src_node_idx).data,
                        node_idx,
                        borrowed_local
                    );

                    self.dep_graph.add_edge(src_node_idx, node_idx, ());
                }
                None => {}
            },
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

                let mut has_refs_or_pointers_visitor = MaybeHasRefsOrPointersVisitor::new();
                dest_ty.visit_with(&mut has_refs_or_pointers_visitor);

                let dest_might_include_refs_or_pointers =
                    if has_refs_or_pointers_visitor.has_refs_or_pointers {
                        true
                    } else if has_refs_or_pointers_visitor.has_params {
                        // if we pass in any references or raw pointers as arguments and the
                        // return type includes type parameters, these type parameters could
                        // refer to those refs/ptrs, so we have to be conservative here and possibly
                        // include an edge.
                        true
                    } else {
                        false
                    };

                let node_idx =
                    self.dep_graph.add_node(NodeKind::LocalOrLocalWithRefs(destination.local));
                debug!("added local {:?} to graph with idx {:?}", destination.local, node_idx);
                self.locals_to_node_indexes.insert(destination.local, node_idx);

                if dest_might_include_refs_or_pointers {
                    self.current_local = Some(node_idx);
                    debug!("self.current_local: {:?}", self.current_local);
                }

                self.visit_operand(func, location);
                for arg in args {
                    self.visit_operand(arg, location);
                }

                self.current_local = None;
            }
            TerminatorKind::Yield { resume_arg, value, .. } => {
                let node_idx =
                    self.dep_graph.add_node(NodeKind::LocalOrLocalWithRefs(resume_arg.local));
                debug!("added local {:?} to graph with idx {:?}", resume_arg.local, node_idx);
                self.locals_to_node_indexes.insert(resume_arg.local, node_idx);

                self.super_operand(value, location);
            }
            _ => self.super_terminator(terminator, location),
        }
    }
}

pub struct BorrowedLocalsResults<'mir, 'tcx> {
    borrows_analysis_results: Results<'tcx, LiveBorrows<'mir, 'tcx>>,
    borrowed_local_to_locals_to_keep_alive: FxHashMap<Local, Vec<Local>>,
}

impl<'mir, 'tcx> BorrowedLocalsResults<'mir, 'tcx>
where
    'tcx: 'mir,
{
    fn new(
        tcx: TyCtxt<'tcx>,
        body: &'mir Body<'tcx>,
        borrows_analysis_results: Results<'tcx, LiveBorrows<'mir, 'tcx>>,
    ) -> Self {
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

        let dep_graph = &borrow_deps.dep_graph;
        let borrowed_local_to_locals_to_keep_alive = Self::get_locals_to_keep_alive_map(dep_graph);
        Self { borrows_analysis_results, borrowed_local_to_locals_to_keep_alive }
    }

    /// Uses the dependency graph to find all locals that we need to keep live for a given
    /// `Node` (or more specically the `Local` corresponding to that `Node`).
    fn get_locals_to_keep_alive_map<'a>(
        dep_graph: &'a Graph<NodeKind, ()>,
    ) -> FxHashMap<Local, Vec<Local>> {
        let mut borrows_to_locals: FxHashMap<Local, Vec<Local>> = Default::default();
        for (node_idx, node) in dep_graph.enumerated_nodes() {
            let current_local = node.data.get_local();
            if borrows_to_locals.get(&current_local).is_none() {
                Self::dfs_for_node(node_idx, &mut borrows_to_locals, dep_graph);
            }
        }

        debug!("borrows_to_locals: {:#?}", borrows_to_locals);
        borrows_to_locals
    }

    fn dfs_for_node(
        node_idx: NodeIndex,
        borrows_to_locals: &mut FxHashMap<Local, Vec<Local>>,
        dep_graph: &Graph<NodeKind, ()>,
    ) -> Vec<Local> {
        let src_node = dep_graph.node(node_idx);
        let current_local = src_node.data.get_local();
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

            debug!(
                "edge {:?} ({:?}) -> {:?} ({:?})",
                node_idx, src_node.data, target_node_idx, target_node.data,
            );

            let mut locals_to_keep_alive_for_succ =
                Self::dfs_for_node(target_node_idx, borrows_to_locals, dep_graph);
            locals_for_node.append(&mut locals_to_keep_alive_for_succ);
        }

        if num_succs == 0 {
            // base node to keep alive
            vec![src_node.data.get_local()]
        } else {
            if matches!(src_node.data, NodeKind::LocalOrLocalWithRefs(_)) {
                // These are locals that we need to keep alive, but that also contain
                // successors in the graph since they contain other references/pointers.
                locals_for_node.push(current_local);
            }

            borrows_to_locals.insert(current_local, locals_for_node.clone());
            locals_for_node
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
pub fn get_borrowed_locals_results<'mir, 'tcx>(
    body: &'mir Body<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> BorrowedLocalsResults<'mir, 'tcx> {
    debug!("body: {:#?}", body);
    let live_borrows = LiveBorrows::new(body, tcx);
    let results =
        live_borrows.into_engine(tcx, body).pass_name("borrowed_locals").iterate_to_fixpoint();

    BorrowedLocalsResults::new(tcx, body, results)
}

pub struct BorrowedLocalsResultsCursor<'a, 'mir, 'tcx> {
    body: &'mir Body<'tcx>,
    borrows_analysis_cursor: ResultsRefCursor<'a, 'mir, 'tcx, LiveBorrows<'mir, 'tcx>>,
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

/// Performs a liveness analysis for borrows and raw pointers.
pub struct LiveBorrows<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> LiveBorrows<'a, 'tcx> {
    fn new(body: &'a Body<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        LiveBorrows { body, tcx }
    }

    fn transfer_function<'b, T>(&self, trans: &'b mut T) -> TransferFunction<'a, 'b, 'tcx, T> {
        TransferFunction { body: self.body, tcx: self.tcx, _trans: trans }
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

/// A `Visitor` that defines the transfer function for `MaybeBorrowedLocals`.
struct TransferFunction<'a, 'b, 'tcx, T> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    _trans: &'b mut T,
}

impl<'a, 'tcx, T> Visitor<'tcx> for TransferFunction<'a, '_, 'tcx, T>
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
                        // kill the local if it's a ref or a pointer.
                        match lhs_place_ty.kind() {
                            ty::Ref(..) | ty::RawPtr(..) => {
                                debug!("killing {:?}", lhs_place.local);
                                self._trans.kill(lhs_place.local);

                                self.visit_rvalue(&assign.1, location);
                            }
                            _ => {
                                self.super_assign(&assign.0, &assign.1, location);
                            }
                        }
                    }
                    _ => {
                        // With any other projection elements a projection of a local (of type ref/ptr)
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
                self._trans.gen(place.local);
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
        match terminator.kind {
            TerminatorKind::Call { destination, .. } => {
                debug!("killing {:?}", destination.local);
                self._trans.kill(destination.local);
            }
            TerminatorKind::Yield { resume_arg, .. } => {
                debug!("killing {:?}", resume_arg.local);
                self._trans.kill(resume_arg.local);
            }
            _ => {}
        }

        self.super_terminator(terminator, location);
    }
}
