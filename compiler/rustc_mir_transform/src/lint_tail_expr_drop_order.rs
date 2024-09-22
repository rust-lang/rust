use rustc_hir::def_id::LocalDefId;
use rustc_hir::{ExprKind, HirId, HirIdSet, OwnerId};
use rustc_index::bit_set::BitSet;
use rustc_index::IndexVec;
use rustc_lint::{self as lint, Level};
use rustc_macros::LintDiagnostic;
use rustc_middle::mir::visit::{
    MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::{
    dump_mir, BasicBlock, Body, CallReturnPlaces, HasLocalDecls, Local, Location, Place,
    ProjectionElem, Statement, Terminator, TerminatorEdges, TerminatorKind,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::{Analysis, AnalysisDomain, GenKill, GenKillAnalysis};
use rustc_span::Span;
use rustc_type_ir::data_structures::IndexMap;
use tracing::debug;

fn blocks_reachable_from_tail_expr_rev(body: &Body<'_>, blocks: &mut BitSet<BasicBlock>) {
    let mut new_blocks = blocks.clone();
    let predecessors = body.basic_blocks.predecessors();
    while !new_blocks.is_empty() {
        let mut next_front = BitSet::new_empty(new_blocks.domain_size());
        for bb in new_blocks.iter() {
            for &pred in &predecessors[bb] {
                if body.basic_blocks[pred].is_cleanup {
                    continue;
                }
                if blocks.insert(pred) {
                    next_front.insert(pred);
                }
            }
        }
        new_blocks = next_front;
    }
}

fn blocks_reachable_from_tail_expr_fwd(body: &Body<'_>, blocks: &mut BitSet<BasicBlock>) {
    let mut new_blocks = blocks.clone();
    while !new_blocks.is_empty() {
        let mut next_front = BitSet::new_empty(new_blocks.domain_size());
        for bb in new_blocks.iter() {
            for succ in body.basic_blocks[bb].terminator.iter().flat_map(|term| term.successors()) {
                if body.basic_blocks[succ].is_cleanup {
                    continue;
                }
                if blocks.insert(succ) {
                    next_front.insert(succ);
                }
            }
        }
        new_blocks = next_front;
    }
}

fn is_descendent_of_hir_id(
    tcx: TyCtxt<'_>,
    id: HirId,
    root: HirId,
    descendants: &mut HirIdSet,
    filter: &mut HirIdSet,
) -> bool {
    let mut path = vec![];
    for id in [id].into_iter().chain(tcx.hir().parent_id_iter(id)) {
        if filter.contains(&id) {
            filter.extend(path);
            return false;
        } else if root == id || descendants.contains(&id) {
            descendants.extend(path);
            return true;
        }
        path.push(id);
    }
    filter.extend(path);
    false
}

pub(crate) fn run_lint<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::SyntheticCoroutineBody) {
        // Synthetic coroutine has no HIR body and it is enough to just analyse the original body
        return;
    }
    let (tail_expr_hir_id, tail_expr_span) = {
        let expr = tcx.hir().body_owned_by(def_id).value;
        match expr.kind {
            ExprKind::Block(block, _) => {
                if let Some(expr) = block.expr {
                    (expr.hir_id, expr.span)
                } else {
                    // There is no tail expression
                    return;
                }
            }
            _ => (expr.hir_id, expr.span),
        }
    };
    if tail_expr_span.edition().at_least_rust_2024() {
        // We should stop linting since Edition 2024
        return;
    }
    if let (Level::Allow, _) =
        tcx.lint_level_at_node(lint::builtin::TAIL_EXPR_DROP_ORDER, tail_expr_hir_id)
    {
        // Analysis is expensive, so let's skip if the lint is suppressed anyway
        return;
    }
    debug!(?def_id, ?tail_expr_span);
    dump_mir(tcx, false, "lint_drop_order", &0 as _, body, |pass_where, writer| match pass_where {
        rustc_middle::mir::PassWhere::AfterLocation(loc) => {
            if loc.statement_index < body.basic_blocks[loc.block].statements.len() {
                writeln!(
                    writer,
                    "@ {:?}",
                    body.basic_blocks[loc.block].statements[loc.statement_index].source_info.span
                )
            } else {
                Ok(())
            }
        }
        rustc_middle::mir::PassWhere::AfterTerminator(term) => {
            writeln!(writer, "@ {:?}", body.basic_blocks[term].terminator().source_info.span)
        }
        _ => Ok(()),
    });
    let mut non_tail_expr_parent: HirIdSet = tcx.hir().parent_id_iter(tail_expr_hir_id).collect();
    let mut tail_expr_descendants: HirIdSet = [tail_expr_hir_id].into_iter().collect();

    let param_env = tcx.param_env(def_id);
    let mut droppy_locals = IndexMap::default();
    let (nr_upvars, nr_locals, mut droppy_local_mask) = if tcx.is_closure_like(def_id.to_def_id()) {
        let captures = tcx.closure_captures(def_id);
        let nr_upvars = captures.len();
        let nr_body_locals = body.local_decls.len();
        let mut mask = BitSet::new_empty(nr_body_locals + nr_upvars);
        for (idx, &capture) in captures.iter().enumerate() {
            let ty = capture.place.ty();
            if ty.has_significant_drop(tcx, param_env) {
                let upvar = Local::from_usize(nr_body_locals + idx);
                mask.insert(upvar);
                droppy_locals.insert(upvar, (capture.var_ident.span, ty));
            }
        }
        (nr_upvars, nr_upvars + nr_body_locals, mask)
    } else {
        let nr_locals = body.local_decls.len();
        (0, nr_locals, BitSet::new_empty(nr_locals))
    };
    for (local, decl) in body.local_decls().iter_enumerated() {
        if local == Local::ZERO {
            // return place is ignored
            continue;
        }
        if decl.ty.has_significant_drop(tcx, param_env) {
            droppy_local_mask.insert(local);
            droppy_locals.insert(local, (decl.source_info.span, decl.ty));
        }
    }

    let nr_blocks = body.basic_blocks.len();
    let mut tail_expr_blocks = BitSet::new_empty(nr_blocks);
    let mut blocks_term_in_tail_expr = BitSet::new_empty(nr_blocks);
    let mut tail_expr_stmts = IndexVec::from_elem_n(0, nr_blocks);
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup {
            continue;
        }
        if let Some(terminator) = &data.terminator
            && let TerminatorKind::Drop { scope: Some((def_id, local_id)), .. } = terminator.kind
            && let Some(def_id) = def_id.as_local()
        {
            let hir_id = HirId { owner: OwnerId { def_id }, local_id };
            if is_descendent_of_hir_id(
                tcx,
                hir_id,
                tail_expr_hir_id,
                &mut tail_expr_descendants,
                &mut non_tail_expr_parent,
            ) {
                tail_expr_blocks.insert(bb);
                blocks_term_in_tail_expr.insert(bb);
                continue;
            }
        }
        for (idx, stmt) in data.statements.iter().enumerate().rev() {
            if tail_expr_span.contains(stmt.source_info.span) {
                tail_expr_blocks.insert(bb);
                tail_expr_stmts[bb] = idx;
                break;
            }
        }
    }
    let mut exit_tail_expr_blocks = tail_expr_blocks.clone();
    debug!(?tail_expr_blocks, "before reachability");
    blocks_reachable_from_tail_expr_rev(body, &mut tail_expr_blocks);
    debug!(?tail_expr_blocks, "reachable bb to tail expr");
    blocks_reachable_from_tail_expr_fwd(body, &mut exit_tail_expr_blocks);
    debug!(?exit_tail_expr_blocks, "exit reachability");
    exit_tail_expr_blocks.subtract(&tail_expr_blocks);
    debug!(?exit_tail_expr_blocks, "exit - rev");

    let mut tail_expr_exit_boundary_blocks = BitSet::new_empty(nr_blocks);
    let mut boundary_lies_on_statement = BitSet::new_empty(nr_blocks);
    'boundary_block: for (bb, block) in
        tail_expr_blocks.iter().map(|bb| (bb, &body.basic_blocks[bb]))
    {
        for succ in block.terminator.iter().flat_map(|term| term.successors()) {
            if exit_tail_expr_blocks.contains(succ) {
                if blocks_term_in_tail_expr.contains(bb) {
                    tail_expr_exit_boundary_blocks.insert(succ);
                } else {
                    boundary_lies_on_statement.insert(bb);
                    tail_expr_exit_boundary_blocks.insert(bb);
                    continue 'boundary_block;
                }
            }
        }
    }
    debug!(?tail_expr_exit_boundary_blocks);

    let mut maybe_live = LiveLocals { nr_upvars, nr_body_locals: body.local_decls.len() }
        .into_engine(tcx, body)
        .iterate_to_fixpoint()
        .into_results_cursor(body);
    let mut observables = BitSet::new_empty(nr_locals);
    for block in tail_expr_exit_boundary_blocks.iter() {
        debug!(?observables);
        if boundary_lies_on_statement.contains(block) {
            let target = Location { block, statement_index: tail_expr_stmts[block] };
            debug!(?target, "in block");
            maybe_live.seek_after_primary_effect(target);
        } else {
            maybe_live.seek_to_block_start(block);
        }
        let mut mask = droppy_local_mask.clone();
        let alive = maybe_live.get();
        debug!(?block, ?alive);
        mask.intersect(alive);
        observables.union(&mask);
    }
    if observables.is_empty() {
        debug!("no observable, bail");
        return;
    }
    debug!(?observables, ?droppy_locals);
    // A linted local has a span contained in the tail expression span
    let Some((linted_local, (linted_local_span, linted_local_ty))) = observables
        .iter()
        .filter_map(|local| droppy_locals.get(&local).map(|&info| (local, info)))
        .filter(|(_, (span, _))| tail_expr_span.contains(*span))
        .nth(0)
    else {
        debug!("nothing to lint");
        return;
    };
    debug!(?linted_local);
    let body_observables: Vec<_> = observables
        .iter()
        .filter(|&local| local != linted_local)
        .filter_map(|local| droppy_locals.get(&local))
        .map(|&(span, _)| span)
        .collect();
    if body_observables.is_empty() {
        debug!("nothing observable from body");
        return;
    }
    debug!(?linted_local, ?body_observables);
    let decorator = TailExprDropOrderLint { spans: body_observables, ty: linted_local_ty };
    tcx.emit_node_span_lint(
        lint::builtin::TAIL_EXPR_DROP_ORDER,
        tail_expr_hir_id,
        linted_local_span,
        decorator,
    );
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_tail_expr_drop_order)]
struct TailExprDropOrderLint<'a> {
    #[label]
    pub spans: Vec<Span>,
    pub ty: Ty<'a>,
}

struct LiveLocals {
    nr_upvars: usize,
    nr_body_locals: usize,
}

impl<'tcx> AnalysisDomain<'tcx> for LiveLocals {
    type Domain = BitSet<Local>;
    const NAME: &'static str = "liveness-by-use";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        debug_assert_eq!(self.nr_body_locals, body.local_decls.len());
        BitSet::new_empty(self.nr_body_locals + self.nr_upvars)
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        for arg in body.args_iter() {
            state.insert(arg);
        }
        debug_assert_eq!(self.nr_body_locals, body.local_decls.len());
        for upvar in 0..self.nr_upvars {
            state.insert(Local::from_usize(self.nr_body_locals + upvar));
        }
    }
}

impl<'tcx> GenKillAnalysis<'tcx> for LiveLocals {
    type Idx = Local;

    fn domain_size(&self, body: &Body<'tcx>) -> usize {
        body.local_decls.len()
    }

    fn statement_effect(
        &mut self,
        transfer: &mut impl GenKill<Self::Idx>,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        TransferFunction {
            nr_upvars: self.nr_upvars,
            nr_body_locals: self.nr_body_locals,
            transfer,
        }
        .visit_statement(statement, location);
    }

    fn terminator_effect<'mir>(
        &mut self,
        transfer: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        TransferFunction {
            nr_upvars: self.nr_upvars,
            nr_body_locals: self.nr_body_locals,
            transfer,
        }
        .visit_terminator(terminator, location);
        terminator.edges()
    }

    fn call_return_effect(
        &mut self,
        trans: &mut Self::Domain,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            if let Some(local) = place.as_local() {
                trans.gen_(local);
            }
        })
    }
}

struct TransferFunction<'a, T> {
    nr_upvars: usize,
    nr_body_locals: usize,
    transfer: &'a mut T,
}

impl<'tcx, T> Visitor<'tcx> for TransferFunction<'_, T>
where
    T: GenKill<Local>,
{
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        let local = if let Some(local) = place.as_local() {
            local
        } else if let Place { local: ty::CAPTURE_STRUCT_LOCAL, projection } = *place
            && let [ProjectionElem::Field(idx, _)] = **projection
            && idx.as_usize() < self.nr_upvars
        {
            Local::from_usize(self.nr_body_locals + idx.as_usize())
        } else {
            return;
        };

        match context {
            PlaceContext::NonUse(NonUseContext::StorageDead)
            | PlaceContext::MutatingUse(MutatingUseContext::Drop | MutatingUseContext::Deinit)
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => {
                self.transfer.kill(local);
                if matches!(local, ty::CAPTURE_STRUCT_LOCAL) && self.nr_upvars > 0 {
                    for upvar in 0..self.nr_upvars {
                        self.transfer.kill(Local::from_usize(self.nr_body_locals + upvar));
                    }
                }
            }
            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Call,
            ) => {
                self.transfer.gen_(local);
            }
            _ => {}
        }
    }
}
