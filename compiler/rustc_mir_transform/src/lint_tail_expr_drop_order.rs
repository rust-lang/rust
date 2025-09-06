use std::cell::RefCell;
use std::collections::hash_map;
use std::rc::Rc;

use itertools::Itertools as _;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::Subdiagnostic;
use rustc_hir::CRATE_HIR_ID;
use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::MixedBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::bug;
use rustc_middle::mir::{
    self, BasicBlock, Body, ClearCrossCrate, Local, Location, MirDumper, Place, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::significant_drop_order::{
    extract_component_with_significant_dtor, ty_dtor_span,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{LookupResult, MoveData, MovePathIndex};
use rustc_mir_dataflow::{Analysis, MaybeReachable, ResultsCursor};
use rustc_session::lint::builtin::TAIL_EXPR_DROP_ORDER;
use rustc_session::lint::{self};
use rustc_span::{DUMMY_SP, Span, Symbol};
use tracing::debug;

fn place_has_common_prefix<'tcx>(left: &Place<'tcx>, right: &Place<'tcx>) -> bool {
    left.local == right.local
        && left.projection.iter().zip(right.projection).all(|(left, right)| left == right)
}

/// Cache entry of `drop` at a `BasicBlock`
#[derive(Debug, Clone, Copy)]
enum MovePathIndexAtBlock {
    /// We know nothing yet
    Unknown,
    /// We know that the `drop` here has no effect
    None,
    /// We know that the `drop` here will invoke a destructor
    Some(MovePathIndex),
}

struct DropsReachable<'a, 'mir, 'tcx> {
    body: &'a Body<'tcx>,
    place: &'a Place<'tcx>,
    drop_span: &'a mut Option<Span>,
    move_data: &'a MoveData<'tcx>,
    maybe_init: &'a mut ResultsCursor<'mir, 'tcx, MaybeInitializedPlaces<'mir, 'tcx>>,
    block_drop_value_info: &'a mut IndexSlice<BasicBlock, MovePathIndexAtBlock>,
    collected_drops: &'a mut MixedBitSet<MovePathIndex>,
    visited: FxHashMap<BasicBlock, Rc<RefCell<MixedBitSet<MovePathIndex>>>>,
}

impl<'a, 'mir, 'tcx> DropsReachable<'a, 'mir, 'tcx> {
    fn visit(&mut self, block: BasicBlock) {
        let move_set_size = self.move_data.move_paths.len();
        let make_new_path_set = || Rc::new(RefCell::new(MixedBitSet::new_empty(move_set_size)));

        let data = &self.body.basic_blocks[block];
        let Some(terminator) = &data.terminator else { return };
        // Given that we observe these dropped locals here at `block` so far, we will try to update
        // the successor blocks. An occupied entry at `block` in `self.visited` signals that we
        // have visited `block` before.
        let dropped_local_here =
            Rc::clone(self.visited.entry(block).or_insert_with(make_new_path_set));
        // We could have invoked reverse lookup for a `MovePathIndex` every time, but unfortunately
        // it is expensive. Let's cache them in `self.block_drop_value_info`.
        match self.block_drop_value_info[block] {
            MovePathIndexAtBlock::Some(dropped) => {
                dropped_local_here.borrow_mut().insert(dropped);
            }
            MovePathIndexAtBlock::Unknown => {
                if let TerminatorKind::Drop { place, .. } = &terminator.kind
                    && let LookupResult::Exact(idx) | LookupResult::Parent(Some(idx)) =
                        self.move_data.rev_lookup.find(place.as_ref())
                {
                    // Since we are working with MIRs at a very early stage, observing a `drop`
                    // terminator is not indicative enough that the drop will definitely happen.
                    // That is decided in the drop elaboration pass instead. Therefore, we need to
                    // consult with the maybe-initialization information.
                    self.maybe_init.seek_before_primary_effect(Location {
                        block,
                        statement_index: data.statements.len(),
                    });

                    // Check if the drop of `place` under inspection is really in effect. This is
                    // true only when `place` may have been initialized along a control flow path
                    // from a BID to the drop program point today. In other words, this is where
                    // the drop of `place` will happen in the future instead.
                    if let MaybeReachable::Reachable(maybe_init) = self.maybe_init.get()
                        && maybe_init.contains(idx)
                    {
                        // We also cache the drop information, so that we do not need to check on
                        // data-flow cursor again.
                        self.block_drop_value_info[block] = MovePathIndexAtBlock::Some(idx);
                        dropped_local_here.borrow_mut().insert(idx);
                    } else {
                        self.block_drop_value_info[block] = MovePathIndexAtBlock::None;
                    }
                }
            }
            MovePathIndexAtBlock::None => {}
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
                    if succ == block
                        || !occupied_entry.get().borrow_mut().union(&*dropped_local_here.borrow())
                    {
                        // `succ` has been visited but no new drops observed so far,
                        // so we can bail on `succ` until new drop information arrives
                        continue;
                    }
                    Rc::clone(occupied_entry.get())
                }
                hash_map::Entry::Vacant(vacant_entry) => Rc::clone(
                    vacant_entry.insert(Rc::new(RefCell::new(dropped_local_here.borrow().clone()))),
                ),
            };
            if let Some(terminator) = &target.terminator
                && let TerminatorKind::Drop {
                    place: dropped_place,
                    target: _,
                    unwind: _,
                    replace: _,
                    drop: _,
                    async_fut: _,
                } = &terminator.kind
                && place_has_common_prefix(dropped_place, self.place)
            {
                // We have now reached the current drop of the `place`.
                // Let's check the observed dropped places in.
                self.collected_drops.union(&*dropped_local_there.borrow());
                if self.drop_span.is_none() {
                    // FIXME(@dingxiangfei2009): it turns out that `self.body.source_scopes` are
                    // still a bit wonky. There is a high chance that this span still points to a
                    // block rather than a statement semicolon.
                    *self.drop_span = Some(terminator.source_info.span);
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

/// Check if a moved place at `idx` is a part of a BID.
/// The use of this check is that we will consider drops on these
/// as a drop of the overall BID and, thus, we can exclude it from the diagnosis.
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

/// The core of the lint `tail-expr-drop-order`
pub(crate) fn run_lint<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if matches!(tcx.def_kind(def_id), rustc_hir::def::DefKind::SyntheticCoroutineBody) {
        // A synthetic coroutine has no HIR body and it is enough to just analyse the original body
        return;
    }
    if body.span.edition().at_least_rust_2024()
        || tcx.lints_that_dont_need_to_run(()).contains(&lint::LintId::of(TAIL_EXPR_DROP_ORDER))
    {
        return;
    }

    // FIXME(typing_env): This should be able to reveal the opaques local to the
    // body using the typeck results.
    let typing_env = ty::TypingEnv::non_body_analysis(tcx, def_id);

    // ## About BIDs in blocks ##
    // Track the set of blocks that contain a backwards-incompatible drop (BID)
    // and, for each block, the vector of locations.
    //
    // We group them per-block because they tend to scheduled in the same drop ladder block.
    let mut bid_per_block = FxIndexMap::default();
    let mut bid_places = UnordSet::new();

    let mut ty_dropped_components = UnordMap::default();
    for (block, data) in body.basic_blocks.iter_enumerated() {
        for (statement_index, stmt) in data.statements.iter().enumerate() {
            if let StatementKind::BackwardIncompatibleDropHint { place, reason: _ } = &stmt.kind {
                let ty = place.ty(body, tcx).ty;
                if ty_dropped_components
                    .entry(ty)
                    .or_insert_with(|| extract_component_with_significant_dtor(tcx, typing_env, ty))
                    .is_empty()
                {
                    continue;
                }
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

    if let Some(dumper) = MirDumper::new(tcx, "lint_tail_expr_drop_order", body) {
        dumper.dump_mir(body);
    }

    let locals_with_user_names = collect_user_names(body);
    let is_closure_like = tcx.is_closure_like(def_id.to_def_id());

    // Compute the "maybe initialized" information for this body.
    // When we encounter a DROP of some place P we only care
    // about the drop if `P` may be initialized.
    let move_data = MoveData::gather_moves(body, tcx, |_| true);
    let mut maybe_init = MaybeInitializedPlaces::new(tcx, body, &move_data)
        .iterate_to_fixpoint(tcx, body, None)
        .into_results_cursor(body);
    let mut block_drop_value_info =
        IndexVec::from_elem_n(MovePathIndexAtBlock::Unknown, body.basic_blocks.len());
    for (&block, candidates) in &bid_per_block {
        // We will collect drops on locals on paths between BID points to their actual drop locations
        // into `all_locals_dropped`.
        let mut all_locals_dropped = MixedBitSet::new_empty(move_data.move_paths.len());
        let mut drop_span = None;
        for &(_, place) in candidates.iter() {
            let mut collected_drops = MixedBitSet::new_empty(move_data.move_paths.len());
            // ## On detecting change in relative drop order ##
            // Iterate through each BID-containing block `block`.
            // If the place `P` targeted by the BID is "maybe initialized",
            // then search forward to find the actual `DROP(P)` point.
            // Everything dropped between the BID and the actual drop point
            // is something whose relative drop order will change.
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
            // Compute the set `all_locals_dropped` of local variables that are dropped
            // after the BID point but before the current drop point.
            //
            // These are the variables whose drop impls will be reordered with respect
            // to `place`.
            all_locals_dropped.union(&collected_drops);
        }

        // We shall now exclude some local bindings for the following cases.
        {
            let mut to_exclude = MixedBitSet::new_empty(all_locals_dropped.domain_size());
            // We will now do subtraction from the candidate dropped locals, because of the
            // following reasons.
            for path_idx in all_locals_dropped.iter() {
                let move_path = &move_data.move_paths[path_idx];
                let dropped_local = move_path.place.local;
                // a) A return value _0 will eventually be used
                // Example:
                // fn f() -> Droppy {
                //     let _x = Droppy;
                //     Droppy
                // }
                // _0 holds the literal `Droppy` and rightfully `_x` has to be dropped first
                if dropped_local == Local::ZERO {
                    debug!(?dropped_local, "skip return value");
                    to_exclude.insert(path_idx);
                    continue;
                }
                // b) If we are analysing a closure, the captures are still dropped last.
                // This is part of the closure capture lifetime contract.
                // They are similar to the return value _0 with respect to lifetime rules.
                if is_closure_like && matches!(dropped_local, ty::CAPTURE_STRUCT_LOCAL) {
                    debug!(?dropped_local, "skip closure captures");
                    to_exclude.insert(path_idx);
                    continue;
                }
                // c) Sometimes we collect places that are projections into the BID locals,
                // so they are considered dropped now.
                // Example:
                // struct NotVeryDroppy(Droppy);
                // impl Drop for Droppy {..}
                // fn f() -> NotVeryDroppy {
                //    let x = NotVeryDroppy(droppy());
                //    {
                //        let y: Droppy = x.0;
                //        NotVeryDroppy(y)
                //    }
                // }
                // `y` takes `x.0`, which invalidates `x` as a complete `NotVeryDroppy`
                // so there is no point in linting against `x` any more.
                if place_descendent_of_bids(path_idx, &move_data, &bid_places) {
                    debug!(?dropped_local, "skip descendent of bids");
                    to_exclude.insert(path_idx);
                    continue;
                }
                let observer_ty = move_path.place.ty(body, tcx).ty;
                // d) The collected local has no custom destructor that passes our ecosystem filter.
                if ty_dropped_components
                    .entry(observer_ty)
                    .or_insert_with(|| {
                        extract_component_with_significant_dtor(tcx, typing_env, observer_ty)
                    })
                    .is_empty()
                {
                    debug!(?dropped_local, "skip non-droppy types");
                    to_exclude.insert(path_idx);
                    continue;
                }
            }
            // Suppose that all BIDs point into the same local,
            // we can remove the this local from the observed drops,
            // so that we can focus our diagnosis more on the others.
            if let Ok(local) = candidates.iter().map(|&(_, place)| place.local).all_equal_value() {
                for path_idx in all_locals_dropped.iter() {
                    if move_data.move_paths[path_idx].place.local == local {
                        to_exclude.insert(path_idx);
                    }
                }
            }
            all_locals_dropped.subtract(&to_exclude);
        }
        if all_locals_dropped.is_empty() {
            // No drop effect is observable, so let us move on.
            continue;
        }

        // ## The final work to assemble the diagnosis ##
        // First collect or generate fresh names for local variable bindings and temporary values.
        let local_names = assign_observables_names(
            all_locals_dropped
                .iter()
                .map(|path_idx| move_data.move_paths[path_idx].place.local)
                .chain(candidates.iter().map(|(_, place)| place.local)),
            &locals_with_user_names,
        );

        let mut lint_root = None;
        let mut local_labels = vec![];
        // We now collect the types with custom destructors.
        for &(_, place) in candidates {
            let linted_local_decl = &body.local_decls[place.local];
            let Some(&(ref name, is_generated_name)) = local_names.get(&place.local) else {
                bug!("a name should have been assigned")
            };
            let name = name.as_str();

            if lint_root.is_none()
                && let ClearCrossCrate::Set(data) =
                    &body.source_scopes[linted_local_decl.source_info.scope].local_data
            {
                lint_root = Some(data.lint_root);
            }

            // Collect spans of the custom destructors.
            let mut seen_dyn = false;
            let destructors = ty_dropped_components
                .get(&linted_local_decl.ty)
                .unwrap()
                .iter()
                .filter_map(|&ty| {
                    if let Some(span) = ty_dtor_span(tcx, ty) {
                        Some(DestructorLabel { span, name, dtor_kind: "concrete" })
                    } else if matches!(ty.kind(), ty::Dynamic(..)) {
                        if seen_dyn {
                            None
                        } else {
                            seen_dyn = true;
                            Some(DestructorLabel { span: DUMMY_SP, name, dtor_kind: "dyn" })
                        }
                    } else {
                        None
                    }
                })
                .collect();
            local_labels.push(LocalLabel {
                span: linted_local_decl.source_info.span,
                destructors,
                name,
                is_generated_name,
                is_dropped_first_edition_2024: true,
            });
        }

        // Similarly, custom destructors of the observed drops.
        for path_idx in all_locals_dropped.iter() {
            let place = &move_data.move_paths[path_idx].place;
            // We are not using the type of the local because the drop may be partial.
            let observer_ty = place.ty(body, tcx).ty;

            let observer_local_decl = &body.local_decls[place.local];
            let Some(&(ref name, is_generated_name)) = local_names.get(&place.local) else {
                bug!("a name should have been assigned")
            };
            let name = name.as_str();

            let mut seen_dyn = false;
            let destructors = extract_component_with_significant_dtor(tcx, typing_env, observer_ty)
                .into_iter()
                .filter_map(|ty| {
                    if let Some(span) = ty_dtor_span(tcx, ty) {
                        Some(DestructorLabel { span, name, dtor_kind: "concrete" })
                    } else if matches!(ty.kind(), ty::Dynamic(..)) {
                        if seen_dyn {
                            None
                        } else {
                            seen_dyn = true;
                            Some(DestructorLabel { span: DUMMY_SP, name, dtor_kind: "dyn" })
                        }
                    } else {
                        None
                    }
                })
                .collect();
            local_labels.push(LocalLabel {
                span: observer_local_decl.source_info.span,
                destructors,
                name,
                is_generated_name,
                is_dropped_first_edition_2024: false,
            });
        }

        let span = local_labels[0].span;
        tcx.emit_node_span_lint(
            lint::builtin::TAIL_EXPR_DROP_ORDER,
            lint_root.unwrap_or(CRATE_HIR_ID),
            span,
            TailExprDropOrderLint { local_labels, drop_span, _epilogue: () },
        );
    }
}

/// Extract binding names if available for diagnosis
fn collect_user_names(body: &Body<'_>) -> FxIndexMap<Local, Symbol> {
    let mut names = FxIndexMap::default();
    for var_debug_info in &body.var_debug_info {
        if let mir::VarDebugInfoContents::Place(place) = &var_debug_info.value
            && let Some(local) = place.local_or_deref_local()
        {
            names.entry(local).or_insert(var_debug_info.name);
        }
    }
    names
}

/// Assign names for anonymous or temporary values for diagnosis
fn assign_observables_names(
    locals: impl IntoIterator<Item = Local>,
    user_names: &FxIndexMap<Local, Symbol>,
) -> FxIndexMap<Local, (String, bool)> {
    let mut names = FxIndexMap::default();
    let mut assigned_names = FxHashSet::default();
    let mut idx = 0u64;
    let mut fresh_name = || {
        idx += 1;
        (format!("#{idx}"), true)
    };
    for local in locals {
        let name = if let Some(name) = user_names.get(&local) {
            let name = name.as_str();
            if assigned_names.contains(name) { fresh_name() } else { (name.to_owned(), false) }
        } else {
            fresh_name()
        };
        assigned_names.insert(name.0.clone());
        names.insert(local, name);
    }
    names
}

#[derive(LintDiagnostic)]
#[diag(mir_transform_tail_expr_drop_order)]
struct TailExprDropOrderLint<'a> {
    #[subdiagnostic]
    local_labels: Vec<LocalLabel<'a>>,
    #[label(mir_transform_drop_location)]
    drop_span: Option<Span>,
    #[note(mir_transform_note_epilogue)]
    _epilogue: (),
}

struct LocalLabel<'a> {
    span: Span,
    name: &'a str,
    is_generated_name: bool,
    is_dropped_first_edition_2024: bool,
    destructors: Vec<DestructorLabel<'a>>,
}

/// A custom `Subdiagnostic` implementation so that the notes are delivered in a specific order
impl Subdiagnostic for LocalLabel<'_> {
    fn add_to_diag<G: rustc_errors::EmissionGuarantee>(self, diag: &mut rustc_errors::Diag<'_, G>) {
        // Because parent uses this field , we need to remove it delay before adding it.
        diag.remove_arg("name");
        diag.arg("name", self.name);
        diag.remove_arg("is_generated_name");
        diag.arg("is_generated_name", self.is_generated_name);
        diag.remove_arg("is_dropped_first_edition_2024");
        diag.arg("is_dropped_first_edition_2024", self.is_dropped_first_edition_2024);
        let msg = diag.eagerly_translate(crate::fluent_generated::mir_transform_tail_expr_local);
        diag.span_label(self.span, msg);
        for dtor in self.destructors {
            dtor.add_to_diag(diag);
        }
        let msg =
            diag.eagerly_translate(crate::fluent_generated::mir_transform_label_local_epilogue);
        diag.span_label(self.span, msg);
    }
}

#[derive(Subdiagnostic)]
#[note(mir_transform_tail_expr_dtor)]
struct DestructorLabel<'a> {
    #[primary_span]
    span: Span,
    dtor_kind: &'static str,
    name: &'a str,
}
