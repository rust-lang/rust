use itertools::Itertools;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::BitSet;
use rustc_macros::LintDiagnostic;
use rustc_middle::mir::{
    BasicBlock, Body, ClearCrossCrate, Local, Location, Place, ProjectionElem, StatementKind,
    TerminatorKind, dump_mir,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{MoveData, MovePathIndex};
use rustc_mir_dataflow::{Analysis, MaybeReachable};
use rustc_session::lint;
use rustc_span::Span;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

fn place_has_common_prefix<'tcx>(left: &Place<'tcx>, right: &Place<'tcx>) -> bool {
    left.local == right.local
        && left.projection.iter().zip(right.projection).all(|(left, right)| left == right)
}

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
                    && place_has_common_prefix(dropped_place, place)
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

#[instrument(level = "debug", skip(tcx, param_env))]
fn extract_component_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> (SmallVec<[Ty<'tcx>; 4]>, SmallVec<[Span; 4]>) {
    // Droppiness does not depend on regions, so let us erase them.
    let ty = tcx.try_normalize_erasing_regions(param_env, ty).unwrap_or(ty);

    let Ok(tys) = tcx.list_significant_drop_tys(param_env.and(ty)) else {
        return (smallvec![ty], smallvec![]);
    };
    debug!(?ty, "components");
    let mut out_tys = smallvec![];
    let mut out_spans = smallvec![];
    for ty in tys {
        if let ty::Coroutine(did, args) = ty.kind()
            && let Some(witness) = tcx.mir_coroutine_witnesses(did)
        {
            for field in &witness.field_tys {
                let (tys, spans) = extract_component_raw(
                    tcx,
                    param_env,
                    ty::EarlyBinder::bind(field.ty).instantiate(tcx, args),
                );
                if !tys.is_empty() {
                    debug!(?field.ty, "including span for field of type");
                    out_spans.extend([field.source_info.span].into_iter().chain(spans));
                } else {
                    assert!(spans.is_empty());
                }
                out_tys.extend(tys);
            }
        } else {
            out_tys.push(ty);
        }
    }
    (out_tys, out_spans)
}

#[instrument(level = "debug", skip(tcx, param_env))]
fn extract_component_with_significant_dtor<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
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
    let (tys, spans) = extract_component_raw(tcx, param_env, ty);
    let ty_names =
        tys.iter().copied().filter(|&oty| oty != ty).map(print_ty_without_trimming).join(", ");
    let ty_spans =
        tys.iter().copied().filter(|&oty| oty != ty).flat_map(ty_def_span).chain(spans).collect();
    (ty_names, ty_spans)
}

fn place_descendent_of_bids<'tcx>(
    mut idx: MovePathIndex,
    move_data: &MoveData<'tcx>,
    bids: &UnordSet<&Place<'tcx>>,
) -> bool {
    loop {
        let path = &move_data.move_paths[idx];
        if bids.contains(&path.place) {
            return true;
        }
        if let Some(parent) = path.parent {
            idx = parent;
        } else {
            return false;
        }
    }
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

    dump_mir(tcx, false, "lint_tail_expr_drop_order", &0 as _, body, |_, _| Ok(()));
    let param_env = tcx.param_env(def_id).with_reveal_all_normalized(tcx);
    let is_closure_like = tcx.is_closure_like(def_id.to_def_id());
    let move_data = MoveData::gather_moves(body, tcx, param_env, |_| true);
    let maybe_init = MaybeInitializedPlaces::new(tcx, body, &move_data);
    let mut maybe_init =
        maybe_init.into_engine(tcx, body).iterate_to_fixpoint().into_results_cursor(body);
    for &(candidate, place) in &backwards_incompatible_drops {
        maybe_init.seek_after_primary_effect(candidate);
        let MaybeReachable::Reachable(maybe_init_in_future) = maybe_init.get() else { continue };
        debug!(maybe_init_in_future = ?maybe_init_in_future.iter().map(|idx| &move_data.move_paths[idx]).collect::<Vec<_>>());
        let maybe_init_in_future = maybe_init_in_future.clone();
        for block in drops_reachable_from_location(body, candidate.block, place).iter() {
            let data = &body.basic_blocks[block];
            debug!(?candidate, ?block, "inspect");
            maybe_init.seek_before_primary_effect(Location {
                block,
                statement_index: data.statements.len(),
            });
            let MaybeReachable::Reachable(maybe_init_now) = maybe_init.get() else { continue };
            let mut locals_dropped = maybe_init_in_future.clone();
            debug!(maybe_init_now = ?maybe_init_now.iter().map(|idx| &move_data.move_paths[idx]).collect::<Vec<_>>());
            locals_dropped.subtract(maybe_init_now);
            debug!(locals_dropped = ?locals_dropped.iter().map(|idx| &move_data.move_paths[idx]).collect::<Vec<_>>());
            for path_idx in locals_dropped.iter() {
                let move_path = &move_data.move_paths[path_idx];
                if let [.., ProjectionElem::Downcast(_, _)] = **move_path.place.projection {
                    debug!(?move_path.place, "skip downcast which is not a real place");
                    continue;
                }
                if place_descendent_of_bids(path_idx, &move_data, &bid_places) {
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
                        param_env,
                        linted_local_decl.ty,
                    );
                    let (observer_ty_drop_components, observer_ty_spans) =
                        extract_component_with_significant_dtor(tcx, param_env, observer_ty);
                    debug!(?candidate, ?place, ?move_path.place);
                    tcx.emit_node_span_lint(
                        lint::builtin::TAIL_EXPR_DROP_ORDER,
                        lint_root,
                        linted_local_decl.source_info.span,
                        TailExprDropOrderLint {
                            span: observer_local_decl.source_info.span,
                            ty: print_ty_without_trimming(linted_local_decl.ty),
                            ty_drop_components_size: ty_drop_components.len(),
                            observer_ty_drop_components_size: observer_ty_drop_components.len(),
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
    pub ty_drop_components_size: usize,
    #[note(mir_transform_note_ty)]
    pub ty_spans: Vec<Span>,
    pub observer_ty_drop_components: String,
    pub observer_ty_drop_components_size: usize,
    #[note(mir_transform_note_observer_ty)]
    pub observer_ty_spans: Vec<Span>,
}
