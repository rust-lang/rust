use std::collections::BTreeSet;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::CRATE_HIR_ID;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::{BitSet, ChunkedBitSet};
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
use rustc_type_ir::data_structures::IndexMap;
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

#[instrument(level = "debug", skip(tcx, param_env))]
fn extract_component_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> SmallVec<[Ty<'tcx>; 4]> {
    // Droppiness does not depend on regions, so let us erase them.
    let ty = tcx.try_normalize_erasing_regions(param_env, ty).unwrap_or(ty);

    let Ok(tys) = tcx.list_significant_drop_tys(param_env.and(ty)) else {
        return smallvec![ty];
    };
    debug!(?ty, "components");
    let mut out_tys = smallvec![];
    for ty in tys {
        if let ty::Coroutine(did, args) = ty.kind()
            && let Some(witness) = tcx.mir_coroutine_witnesses(did)
        {
            for field in &witness.field_tys {
                out_tys.extend(extract_component_raw(
                    tcx,
                    param_env,
                    ty::EarlyBinder::bind(field.ty).instantiate(tcx, args),
                ));
            }
        } else {
            out_tys.push(ty);
        }
    }
    out_tys
}

#[instrument(level = "debug", skip(tcx, param_env))]
fn extract_component_with_significant_dtor<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> SmallVec<[Ty<'tcx>; 4]> {
    let mut tys = extract_component_raw(tcx, param_env, ty);
    let mut deduplicate = FxHashSet::default();
    tys.retain(|oty| deduplicate.insert(*oty));
    tys.into_iter().collect()
}

#[instrument(level = "debug", skip(tcx))]
fn ty_dtor_span<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<Span> {
    match ty.kind() {
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
        ty::Adt(adt_def, _) => {
            let did = adt_def.did();
            let try_local_did_span = |did: DefId| {
                if let Some(local) = did.as_local() {
                    tcx.source_span(local)
                } else {
                    tcx.def_span(did)
                }
            };
            let dtor = if let Some(dtor) = tcx.adt_destructor(did) {
                dtor.did
            } else if let Some(dtor) = tcx.adt_async_destructor(did) {
                dtor.future
            } else {
                return Some(try_local_did_span(did));
            };
            let def_key = tcx.def_key(dtor);
            let Some(parent_index) = def_key.parent else { return Some(try_local_did_span(dtor)) };
            let parent_did = DefId { index: parent_index, krate: dtor.krate };
            Some(try_local_did_span(parent_did))
        }
        ty::Coroutine(did, _)
        | ty::CoroutineWitness(did, _)
        | ty::CoroutineClosure(did, _)
        | ty::Closure(did, _)
        | ty::FnDef(did, _)
        | ty::Foreign(did) => Some(tcx.def_span(did)),
        ty::Param(_) => None,
    }
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
    // We are using blocks to identify locals with the same scope targeted by backwards-incompatible drops (BID)
    // because they tend to be scheduled in the same drop ladder block.
    let mut bid_per_block = IndexMap::default();
    let mut bid_places = UnordSet::new();
    for (block, data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, stmt) in data.statements.iter().enumerate() {
            if let StatementKind::BackwardIncompatibleDropHint { place, reason: _ } = &stmt.kind {
                bid_per_block
                    .entry(block)
                    .or_insert(vec![])
                    .push((Location { block, statement_index }, &**place));
                bid_places.insert(&**place);
            }
        }
    }
    if bid_per_block.is_empty() {
        return;
    }

    dump_mir(tcx, false, "lint_tail_expr_drop_order", &0 as _, body, |_, _| Ok(()));
    let param_env = tcx.param_env(def_id).with_reveal_all_normalized(tcx);
    let is_closure_like = tcx.is_closure_like(def_id.to_def_id());
    let move_data = MoveData::gather_moves(body, tcx, param_env, |_| true);
    let maybe_init = MaybeInitializedPlaces::new(tcx, body, &move_data);
    let mut maybe_init =
        maybe_init.into_engine(tcx, body).iterate_to_fixpoint().into_results_cursor(body);
    for (&block, candidates) in &bid_per_block {
        let mut all_locals_dropped = ChunkedBitSet::new_empty(move_data.move_paths.len());
        let mut linted_candidates = BTreeSet::new();
        let mut drop_span = None;
        for (idx, &(candidate, place)) in candidates.iter().enumerate() {
            maybe_init.seek_after_primary_effect(candidate);
            let MaybeReachable::Reachable(maybe_init_in_future) = maybe_init.get() else {
                continue;
            };
            debug!(maybe_init_in_future = ?maybe_init_in_future.iter().map(|idx| &move_data.move_paths[idx]).collect::<Vec<_>>());
            let maybe_init_in_future = maybe_init_in_future.clone();
            for block in drops_reachable_from_location(body, block, place).iter() {
                let data = &body.basic_blocks[block];
                if drop_span.is_none() {
                    drop_span = data
                        .terminator
                        .as_ref()
                        .map(|term| body.source_scopes[term.source_info.scope].span);
                }
                debug!(?candidate, "inspect");
                maybe_init.seek_before_primary_effect(Location {
                    block,
                    statement_index: data.statements.len(),
                });
                let MaybeReachable::Reachable(maybe_init_now) = maybe_init.get() else { continue };
                let mut locals_dropped = maybe_init_in_future.clone();
                debug!(maybe_init_now = ?maybe_init_now.iter().map(|idx| &move_data.move_paths[idx]).collect::<Vec<_>>());
                locals_dropped.subtract(maybe_init_now);
                debug!(locals_dropped = ?locals_dropped.iter().map(|idx| &move_data.move_paths[idx]).collect::<Vec<_>>());
                if all_locals_dropped.union(&locals_dropped) {
                    linted_candidates.insert(idx);
                }
            }
        }
        {
            let mut to_exclude = ChunkedBitSet::new_empty(all_locals_dropped.domain_size());
            for path_idx in all_locals_dropped.iter() {
                let move_path = &move_data.move_paths[path_idx];
                let dropped_local = move_path.place.local;
                if dropped_local == Local::ZERO {
                    debug!(?dropped_local, "skip return value");
                    to_exclude.insert(path_idx);
                    continue;
                }
                if is_closure_like && matches!(dropped_local, ty::CAPTURE_STRUCT_LOCAL) {
                    debug!(?dropped_local, "skip closure captures");
                    to_exclude.insert(path_idx);
                    continue;
                }
                if let [.., ProjectionElem::Downcast(_, _)] = **move_path.place.projection {
                    debug!(?move_path.place, "skip downcast which is not a real place");
                    to_exclude.insert(path_idx);
                    continue;
                }
                if place_descendent_of_bids(path_idx, &move_data, &bid_places) {
                    debug!(?dropped_local, "skip descendent of bids");
                    to_exclude.insert(path_idx);
                    continue;
                }
                let observer_ty = move_path.place.ty(body, tcx).ty;
                if !observer_ty.has_significant_drop(tcx, param_env) {
                    debug!(?dropped_local, "skip non-droppy types");
                    to_exclude.insert(path_idx);
                    continue;
                }
            }
            let mut iter = linted_candidates.iter();
            let Some(first_linted_local) = iter.next().map(|&idx| candidates[idx].1.local) else {
                continue;
            };
            if iter.all(|&idx| candidates[idx].1.local == first_linted_local) {
                for path_idx in all_locals_dropped.iter() {
                    if move_data.move_paths[path_idx].place.local == first_linted_local {
                        to_exclude.insert(path_idx);
                    }
                }
            }
            all_locals_dropped.subtract(&to_exclude);
        }
        if all_locals_dropped.is_empty() {
            continue;
        }

        let mut lint_root = None;
        let mut linted_spans = Vec::with_capacity(candidates.len());
        let mut tys = Vec::with_capacity(candidates.len());
        for &(_, place) in candidates {
            let linted_local_decl = &body.local_decls[place.local];
            linted_spans.push(linted_local_decl.source_info.span);
            if lint_root.is_none() {
                lint_root =
                    match &body.source_scopes[linted_local_decl.source_info.scope].local_data {
                        ClearCrossCrate::Set(data) => Some(data.lint_root),
                        _ => continue,
                    };
            }
            tys.extend(extract_component_with_significant_dtor(
                tcx,
                param_env,
                linted_local_decl.ty,
            ));
        }
        let linted_dtors = tys.into_iter().filter_map(|ty| ty_dtor_span(tcx, ty)).collect();

        let mut observer_spans = Vec::with_capacity(all_locals_dropped.count());
        let mut observer_tys = Vec::with_capacity(all_locals_dropped.count());
        for path_idx in all_locals_dropped.iter() {
            let move_path = &move_data.move_paths[path_idx];
            let observer_ty = move_path.place.ty(body, tcx).ty;

            let observer_local_decl = &body.local_decls[move_path.place.local];
            observer_spans.push(observer_local_decl.source_info.span);
            observer_tys.extend(extract_component_with_significant_dtor(
                tcx,
                param_env,
                observer_ty,
            ));
        }
        let observer_dtors =
            observer_tys.into_iter().filter_map(|ty| ty_dtor_span(tcx, ty)).collect();

        tcx.emit_node_span_lint(
            lint::builtin::TAIL_EXPR_DROP_ORDER,
            lint_root.unwrap_or(CRATE_HIR_ID),
            linted_spans[0],
            TailExprDropOrderLint {
                linted_spans,
                linted_dtors,
                observer_spans,
                observer_dtors,
                drop_span,
                _epilogue: (),
            },
        );
    }
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_tail_expr_drop_order)]
struct TailExprDropOrderLint {
    #[label(mir_transform_temporaries)]
    pub linted_spans: Vec<Span>,
    #[note(mir_transform_note_dtors)]
    pub linted_dtors: Vec<Span>,
    #[label(mir_transform_observers)]
    pub observer_spans: Vec<Span>,
    #[note(mir_transform_note_observer_dtors)]
    pub observer_dtors: Vec<Span>,
    #[label(mir_transform_drop_location)]
    pub drop_span: Option<Span>,
    #[note(mir_transform_note_epilogue)]
    pub _epilogue: (),
}
