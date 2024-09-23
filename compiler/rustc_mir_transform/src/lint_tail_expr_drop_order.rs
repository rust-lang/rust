use rustc_hir::def_id::LocalDefId;
use rustc_hir::{ExprKind, HirId, HirIdSet, OwnerId};
use rustc_index::IndexVec;
use rustc_index::bit_set::{BitSet, ChunkedBitSet};
use rustc_lint::{self as lint, Level};
use rustc_macros::LintDiagnostic;
use rustc_middle::mir::{BasicBlock, Body, Local, Location, TerminatorKind, dump_mir};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::{Analysis, MaybeReachable};
use rustc_span::Span;
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
        // A synthetic coroutine has no HIR body and it is enough to just analyse the original body
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
    let is_closure_like = tcx.is_closure_like(def_id.to_def_id());

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

    let move_data = MoveData::gather_moves(body, tcx, param_env, |_| true);
    let mut droppy_paths = ChunkedBitSet::new_empty(move_data.move_paths.len());
    let mut droppy_paths_not_in_tail = droppy_paths.clone();
    for (idx, path) in move_data.move_paths.iter_enumerated() {
        if path.place.ty(&body.local_decls, tcx).ty.has_significant_drop(tcx, param_env) {
            droppy_paths.insert(idx);
            if !tail_expr_span.contains(body.local_decls[path.place.local].source_info.span) {
                droppy_paths_not_in_tail.insert(idx);
            }
        }
    }
    let maybe_init = MaybeInitializedPlaces::new(tcx, body, &move_data);
    let mut maybe_init =
        maybe_init.into_engine(tcx, body).iterate_to_fixpoint().into_results_cursor(body);
    let mut observables = ChunkedBitSet::new_empty(move_data.move_paths.len());
    for block in tail_expr_exit_boundary_blocks.iter() {
        debug!(?observables);
        if boundary_lies_on_statement.contains(block) {
            let target = Location { block, statement_index: tail_expr_stmts[block] };
            debug!(?target, "in block");
            maybe_init.seek_after_primary_effect(target);
        } else {
            maybe_init.seek_to_block_start(block);
        }
        if let MaybeReachable::Reachable(alive) = maybe_init.get() {
            debug!(?block, ?alive);
            let mut mask = droppy_paths.clone();
            mask.intersect(alive);
            observables.union(&mask);
        }
    }
    debug!(?observables);
    // A linted local has a span contained in the tail expression span
    let Some(linted_move_path) = observables
        .iter()
        .map(|path| &move_data.move_paths[path])
        .filter(|move_path| {
            if move_path.place.local == Local::ZERO {
                return false;
            }
            if is_closure_like && matches!(move_path.place.local, ty::CAPTURE_STRUCT_LOCAL) {
                return false;
            }
            tail_expr_span.contains(body.local_decls[move_path.place.local].source_info.span)
        })
        .nth(0)
    else {
        debug!("nothing to lint");
        return;
    };
    debug!(?linted_move_path);
    let body_observables: Vec<_> = observables
        .iter()
        .filter_map(|idx| {
            // We want to lint on place/local in body, not another tail expression temporary.
            // Also, closures always drops their upvars last, so we don't have to lint about them.
            let base_local = move_data.base_local(idx);
            if base_local == linted_move_path.place.local || base_local == Local::ZERO {
                debug!(?base_local, "skip");
                return None;
            }
            if is_closure_like && matches!(base_local, ty::CAPTURE_STRUCT_LOCAL) {
                debug!(?base_local, "skip in closure");
                return None;
            }
            droppy_paths_not_in_tail
                .contains(idx)
                .then_some(body.local_decls[base_local].source_info.span)
        })
        .collect();
    if body_observables.is_empty() {
        debug!("nothing observable from body");
        return;
    }
    debug!(?body_observables);
    let linted_local_decl = &body.local_decls[linted_move_path.place.local];
    let decorator = TailExprDropOrderLint { spans: body_observables, ty: linted_local_decl.ty };
    tcx.emit_node_span_lint(
        lint::builtin::TAIL_EXPR_DROP_ORDER,
        tail_expr_hir_id,
        linted_local_decl.source_info.span,
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
