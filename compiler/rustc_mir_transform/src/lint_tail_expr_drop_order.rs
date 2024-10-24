use std::cell::RefCell;
use std::collections::hash_map;
use std::rc::Rc;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::unord::UnordSet;
use rustc_hir::CRATE_HIR_ID;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::bit_set::ChunkedBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::LintDiagnostic;
use rustc_middle::mir::{
    BasicBlock, Body, ClearCrossCrate, Local, Location, Place, StatementKind, TerminatorKind,
    dump_mir,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData, MovePathIndex};
use rustc_mir_dataflow::{Analysis, MaybeReachable, ResultsCursor};
use rustc_session::lint;
use rustc_span::Span;
use rustc_type_ir::data_structures::IndexMap;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

fn place_has_common_prefix<'tcx>(left: &Place<'tcx>, right: &Place<'tcx>) -> bool {
    left.local == right.local
        && left.projection.iter().zip(right.projection).all(|(left, right)| left == right)
}

struct DropsReachable<'a, 'mir, 'tcx> {
    body: &'a Body<'tcx>,
    place: &'a Place<'tcx>,
    drop_span: &'a mut Option<Span>,
    move_data: &'a MoveData<'tcx>,
    maybe_init: &'a mut ResultsCursor<'mir, 'tcx, MaybeInitializedPlaces<'mir, 'tcx>>,
    block_drop_value_info: &'a mut IndexSlice<BasicBlock, Option<MovePathIndex>>,
    collected_drops: &'a mut ChunkedBitSet<MovePathIndex>,
    visited: FxHashMap<BasicBlock, Rc<RefCell<ChunkedBitSet<MovePathIndex>>>>,
}

impl<'a, 'mir, 'tcx> DropsReachable<'a, 'mir, 'tcx> {
    fn visit(&mut self, block: BasicBlock) {
        let move_set_size = self.move_data.move_paths.len();
        let make_new_path_set = || Rc::new(RefCell::new(ChunkedBitSet::new_empty(move_set_size)));

        let data = &self.body.basic_blocks[block];
        let Some(terminator) = &data.terminator else { return };
        // Given that we observe these dropped locals here at `block` so far,
        // we will try to update the successor blocks.
        // An occupied entry at `block` in `self.visited` signals that we have visited `block` before.
        let dropped_local_here =
            self.visited.entry(block).or_insert_with(make_new_path_set).clone();
        // We could have invoked reverse lookup for a `MovePathIndex` every time, but unfortunately it is expensive.
        // Let's cache them in `self.block_drop_value_info`.
        if let Some(dropped) = self.block_drop_value_info[block] {
            dropped_local_here.borrow_mut().insert(dropped);
        } else if let TerminatorKind::Drop { place, .. } = &terminator.kind
            && let LookupResult::Exact(idx) | LookupResult::Parent(Some(idx)) =
                self.move_data.rev_lookup.find(place.as_ref())
        {
            // Since we are working with MIRs at a very early stage,
            // observing a `drop` terminator is not indicative enough that
            // the drop will definitely happen.
            // That is decided in the drop elaboration pass instead.
            // Therefore, we need to consult with the maybe-initialization information.
            self.maybe_init.seek_before_primary_effect(Location {
                block,
                statement_index: data.statements.len(),
            });
            if let MaybeReachable::Reachable(maybe_init) = self.maybe_init.get()
                && maybe_init.contains(idx)
            {
                self.block_drop_value_info[block] = Some(idx);
                dropped_local_here.borrow_mut().insert(idx);
            }
        }

        for succ in terminator.successors() {
            let target = &self.body.basic_blocks[succ];
            if target.is_cleanup {
                continue;
            }

            // As long as we are passing through a new block, or new dropped places to propagate,
            // we will proceed with `succ`
            let dropped_local_there = match self.visited.entry(succ) {
                hash_map::Entry::Occupied(occupied_entry) => {
                    if !occupied_entry.get().borrow_mut().union(&*dropped_local_here.borrow()) {
                        // `succ` has been visited but no new drops observed so far,
                        // so we can bail on `succ` until new drop information arrives
                        continue;
                    }
                    occupied_entry.get().clone()
                }
                hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(dropped_local_here.clone()).clone()
                }
            };
            if let Some(terminator) = &target.terminator
                && let TerminatorKind::Drop {
                    place: dropped_place,
                    target: _,
                    unwind: _,
                    replace: _,
                } = &terminator.kind
                && place_has_common_prefix(dropped_place, self.place)
            {
                // We have now reached the current drop of the `place`.
                // Let's check the observed dropped places in.
                self.collected_drops.union(&*dropped_local_there.borrow());
                if self.drop_span.is_none() {
                    // FIXME(@dingxiangfei2009): it turns out that `self.body.source_scopes` are still a bit wonky.
                    // There is a high chance that this span still points to a block rather than a statement semicolon.
                    *self.drop_span =
                        Some(self.body.source_scopes[terminator.source_info.scope].span);
                }
                // Now we have discovered a simple control flow path from a future drop point
                // to the current drop point.
                // We will not continue from there.
            } else {
                self.visit(succ)
            }
        }
    }
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
    // ## About BIDs in blocks ##
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
    let mut block_drop_value_info = IndexVec::from_elem_n(None, body.basic_blocks.len());
    for (&block, candidates) in &bid_per_block {
        let mut all_locals_dropped = ChunkedBitSet::new_empty(move_data.move_paths.len());
        let mut drop_span = None;
        for &(_, place) in candidates.iter() {
            let mut collected_drops = ChunkedBitSet::new_empty(move_data.move_paths.len());
            DropsReachable {
                body,
                place,
                drop_span: &mut drop_span,
                move_data: &move_data,
                maybe_init: &mut maybe_init,
                block_drop_value_info: &mut block_drop_value_info,
                collected_drops: &mut collected_drops,
                visited: Default::default(),
            }
            .visit(block);

            all_locals_dropped.union(&collected_drops);
        }
        {
            let mut to_exclude = ChunkedBitSet::new_empty(all_locals_dropped.domain_size());
            // We will now do subtraction from the candidate dropped locals, because of the following reasons.
            for path_idx in all_locals_dropped.iter() {
                let move_path = &move_data.move_paths[path_idx];
                let dropped_local = move_path.place.local;
                // a) A return value _0 will eventually be used
                if dropped_local == Local::ZERO {
                    debug!(?dropped_local, "skip return value");
                    to_exclude.insert(path_idx);
                    continue;
                }
                // b) If we are analysing a closure, the captures are still dropped last.
                // This is part of the closure capture lifetime contract.
                if is_closure_like && matches!(dropped_local, ty::CAPTURE_STRUCT_LOCAL) {
                    debug!(?dropped_local, "skip closure captures");
                    to_exclude.insert(path_idx);
                    continue;
                }
                // c) Sometimes we collect places that are projections into the BID locals,
                // so they are considered dropped now.
                if place_descendent_of_bids(path_idx, &move_data, &bid_places) {
                    debug!(?dropped_local, "skip descendent of bids");
                    to_exclude.insert(path_idx);
                    continue;
                }
                let observer_ty = move_path.place.ty(body, tcx).ty;
                // d) The collect local has no custom destructor.
                if !observer_ty.has_significant_drop(tcx, param_env) {
                    debug!(?dropped_local, "skip non-droppy types");
                    to_exclude.insert(path_idx);
                    continue;
                }
            }
            // Suppose that all BIDs point into the same local,
            // we can remove the this local from the observed drops,
            // so that we can focus our diagnosis more on the others.
            if candidates.iter().all(|&(_, place)| candidates[0].1.local == place.local) {
                for path_idx in all_locals_dropped.iter() {
                    if move_data.move_paths[path_idx].place.local == candidates[0].1.local {
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
        // We now collect the types with custom destructors.
        for &(_, place) in candidates {
            let linted_local_decl = &body.local_decls[place.local];
            linted_spans.push(linted_local_decl.source_info.span);

            if lint_root.is_none()
                && let ClearCrossCrate::Set(data) =
                    &body.source_scopes[linted_local_decl.source_info.scope].local_data
            {
                lint_root = Some(data.lint_root);
            }

            tys.extend(extract_component_with_significant_dtor(
                tcx,
                param_env,
                linted_local_decl.ty,
            ));
        }
        // Collect spans of the custom destructors.
        let linted_dtors = tys.into_iter().filter_map(|ty| ty_dtor_span(tcx, ty)).collect();

        // Similarly, custom destructors of the observed drops.
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
