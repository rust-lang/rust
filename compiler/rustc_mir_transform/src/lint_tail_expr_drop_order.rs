use itertools::Itertools;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::BitSet;
use rustc_macros::LintDiagnostic;
use rustc_middle::mir::{
    BasicBlock, Body, ClearCrossCrate, Local, Location, Place, StatementKind, TerminatorKind,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::MoveData;
use rustc_mir_dataflow::{Analysis, MaybeReachable};
use rustc_session::lint;
use rustc_span::Span;
use tracing::debug;

fn drops_reachable_from_location<'tcx>(
    body: &Body<'tcx>,
    block: BasicBlock,
    place: &Place<'tcx>,
) -> BitSet<BasicBlock> {
    let mut reachable = BitSet::new_empty(body.basic_blocks.len());
    let mut visited = reachable.clone();
    let mut new_blocks = reachable.clone();
    new_blocks.insert(block);
    while !new_blocks.is_empty() {
        let mut next_front = BitSet::new_empty(new_blocks.domain_size());
        for bb in new_blocks.iter() {
            if !visited.insert(bb) {
                continue;
            }
            for succ in body.basic_blocks[bb].terminator.iter().flat_map(|term| term.successors()) {
                let target = &body.basic_blocks[succ];
                if target.is_cleanup {
                    continue;
                }
                if !next_front.contains(succ)
                    && let Some(terminator) = &target.terminator
                    && let TerminatorKind::Drop {
                        place: dropped_place,
                        target: _,
                        unwind: _,
                        replace: _,
                    } = &terminator.kind
                    && dropped_place.local == place.local
                    && dropped_place
                        .projection
                        .iter()
                        .zip(place.projection)
                        .all(|(dropped, linted)| dropped == linted)
                {
                    reachable.insert(succ);
                    // Now we have discovered a simple control flow path from a future drop point
                    // to the current drop point.
                    // We will not continue here.
                } else {
                    next_front.insert(succ);
                }
            }
        }
        new_blocks = next_front;
    }
    reachable
}

fn print_ty_without_trimming(ty: Ty<'_>) -> String {
    ty::print::with_no_trimmed_paths!(format!("{}", ty))
}

fn extract_component_with_significant_dtor<'tcx>(
    tcx: TyCtxt<'tcx>,
    _body_did: DefId,
    ty: Ty<'tcx>,
) -> (String, Vec<Span>) {
    let ty_def_span = |ty: Ty<'_>| match ty.kind() {
        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Float(_)
        | ty::Error(_)
        | ty::Str
        | ty::Never
        | ty::RawPtr(_, _)
        | ty::Ref(_, _, _)
        | ty::FnPtr(_, _)
        | ty::Tuple(_)
        | ty::Dynamic(_, _, _)
        | ty::Alias(_, _)
        | ty::Bound(_, _)
        | ty::Pat(_, _)
        | ty::Placeholder(_)
        | ty::Infer(_)
        | ty::Slice(_)
        | ty::Array(_, _) => None,
        ty::Adt(adt_def, _) => Some(tcx.def_span(adt_def.did())),
        ty::Coroutine(did, _)
        | ty::CoroutineWitness(did, _)
        | ty::CoroutineClosure(did, _)
        | ty::Closure(did, _)
        | ty::FnDef(did, _)
        | ty::Foreign(did) => Some(tcx.def_span(did)),
        // I honestly don't know how to extract the span reliably from a param arbitrarily nested
        ty::Param(_) => None,
    };
    let Some(adt_def) = ty.ty_adt_def() else {
        return (print_ty_without_trimming(ty), vec![]);
    };
    let Ok(tys) = tcx.adt_significant_drop_tys(adt_def.did()) else {
        return (print_ty_without_trimming(ty), vec![]);
    };
    let ty_names = tys.iter().map(print_ty_without_trimming).join(", ");
    let ty_spans = tys.iter().flat_map(ty_def_span).collect();
    (ty_names, ty_spans)
}

pub(crate) fn run_lint<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::SyntheticCoroutineBody) {
        // A synthetic coroutine has no HIR body and it is enough to just analyse the original body
        return;
    }
    if body.span.edition().at_least_rust_2024()
        || matches!(
            tcx.lint_level_at_node(
                lint::builtin::TAIL_EXPR_DROP_ORDER,
                tcx.hir().body_owned_by(def_id).id().hir_id
            ),
            (lint::Level::Allow, _)
        )
    {
        return;
    }
    let mut backwards_incompatible_drops = vec![];
    let mut bid_places = UnordSet::new();
    for (block, data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, stmt) in data.statements.iter().enumerate() {
            if let StatementKind::BackwardIncompatibleDropHint { place, reason: _ } = &stmt.kind {
                backwards_incompatible_drops.push((Location { block, statement_index }, &**place));
                bid_places.insert(&**place);
            }
        }
    }
    if backwards_incompatible_drops.is_empty() {
        return;
    }

    let param_env = tcx.param_env(def_id);
    let is_closure_like = tcx.is_closure_like(def_id.to_def_id());
    let move_data = MoveData::gather_moves(body, tcx, param_env, |_| true);
    let maybe_init = MaybeInitializedPlaces::new(tcx, body, &move_data);
    let mut maybe_init =
        maybe_init.into_engine(tcx, body).iterate_to_fixpoint().into_results_cursor(body);
    for &(candidate, place) in &backwards_incompatible_drops {
        maybe_init.seek_after_primary_effect(candidate);
        let MaybeReachable::Reachable(maybe_init_in_future) = maybe_init.get() else { continue };
        let maybe_init_in_future = maybe_init_in_future.clone();
        for block in drops_reachable_from_location(body, candidate.block, place).iter() {
            let data = &body.basic_blocks[block];
            maybe_init.seek_before_primary_effect(Location {
                block,
                statement_index: data.statements.len(),
            });
            let MaybeReachable::Reachable(maybe_init_now) = maybe_init.get() else { continue };
            let mut locals_dropped = maybe_init_in_future.clone();
            locals_dropped.subtract(maybe_init_now);
            for path_idx in locals_dropped.iter() {
                let move_path = &move_data.move_paths[path_idx];
                if bid_places.contains(&move_path.place) {
                    continue;
                }
                let dropped_local = move_path.place.local;
                if is_closure_like && matches!(dropped_local, ty::CAPTURE_STRUCT_LOCAL) {
                    debug!(?dropped_local, "skip closure captures");
                    continue;
                }
                if dropped_local == place.local || dropped_local == Local::ZERO {
                    debug!(?dropped_local, "skip candidate");
                    continue;
                }
                let observer_ty = move_path.place.ty(body, tcx).ty;
                if observer_ty.has_significant_drop(tcx, param_env) {
                    let linted_local_decl = &body.local_decls[place.local];
                    let lint_root =
                        match &body.source_scopes[linted_local_decl.source_info.scope].local_data {
                            ClearCrossCrate::Set(data) => data.lint_root,
                            _ => continue,
                        };
                    let observer_local_decl = &body.local_decls[move_path.place.local];
                    let (ty_drop_components, ty_spans) = extract_component_with_significant_dtor(
                        tcx,
                        def_id.to_def_id(),
                        linted_local_decl.ty,
                    );
                    let (observer_ty_drop_components, observer_ty_spans) =
                        extract_component_with_significant_dtor(
                            tcx,
                            def_id.to_def_id(),
                            observer_ty,
                        );
                    debug!(?candidate, ?place, ?move_path.place);
                    tcx.emit_node_span_lint(
                        lint::builtin::TAIL_EXPR_DROP_ORDER,
                        lint_root,
                        linted_local_decl.source_info.span,
                        TailExprDropOrderLint {
                            span: observer_local_decl.source_info.span,
                            ty: print_ty_without_trimming(linted_local_decl.ty),
                            ty_spans,
                            observer_ty: print_ty_without_trimming(observer_ty),
                            ty_drop_components,
                            observer_ty_drop_components,
                            observer_ty_spans,
                        },
                    );
                }
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_tail_expr_drop_order)]
struct TailExprDropOrderLint {
    #[label]
    pub span: Span,
    pub ty: String,
    pub observer_ty: String,
    pub ty_drop_components: String,
    #[note(mir_transform_note_ty)]
    pub ty_spans: Vec<Span>,
    pub observer_ty_drop_components: String,
    #[note(mir_transform_note_observer_ty)]
    pub observer_ty_spans: Vec<Span>,
}
