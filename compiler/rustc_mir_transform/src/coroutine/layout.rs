//! Coroutine `StateTransform` inverts control flow in a coroutine from a function with yield
//! points to a state machine. Each yield point corresponds to a state variant, and each variant
//! stores the locals that are needed to continue the coroutine.
//!
//! The state transform creates a `poll` method such that calling the coroutine `f()` is equivalent
//! to:
//! ```ignore (example)
//! fn initial_mir(state: CoroutineState, mut resume_arg: ResumeTy) {
//!     // Repeatedly poll the state machine.
//!     loop {
//!         match final_mir(&mut state, resume_arg) {
//!             CoroutineState::Yielded(yield_value) => resume_arg = yield yield_value,
//!             CoroutineState::Complete(return_value) => return return_value,
//!         }
//!     }
//! }
//! ```
//!
//! This file compute for each yield point the set of locals that need to be saved in the coroutine
//! state. This is also used for borrowck to compute the set of types held inside that state, which
//! determine trait and region predicates that hold for this state.

use std::ops;

use itertools::izip;
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::pluralize;
use rustc_hir::{self as hir, find_attr};
use rustc_index::bit_set::{BitMatrix, DenseBitSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, CoroutineArgs, CoroutineArgsExt, Ty, TyCtxt, TypingMode};
use rustc_mir_dataflow::impls::{
    MaybeBorrowedLocals, MaybeLiveLocals, MaybeRequiresStorage, MaybeStorageLive,
    always_storage_live_locals,
};
use rustc_mir_dataflow::{
    Analysis, Results, ResultsCursor, ResultsVisitor, visit_reachable_results,
};
use rustc_session::config::PackCoroutineLayout;
use rustc_span::Span;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::TyCtxtInferExt as _;
use rustc_trait_selection::traits::{ObligationCause, ObligationCauseCode, ObligationCtxt};
use tracing::{debug, instrument, trace};

use crate::diagnostics::{MustNotSupend, MustNotSuspendReason};

const SELF_ARG: Local = Local::arg(0);

pub(super) struct LivenessInfo {
    /// Which locals are live across any suspension point.
    pub(super) saved_locals: CoroutineSavedLocals,

    /// The set of saved locals live at each suspension point.
    live_locals_at_suspension_points: Vec<DenseBitSet<CoroutineSavedLocal>>,

    /// Parallel vec to the above with SourceInfo for each yield terminator.
    source_info_at_suspension_points: Vec<SourceInfo>,

    /// For every saved local, the set of other saved locals that are
    /// storage-live at the same time as this local. We cannot overlap locals in
    /// the layout which have conflicting storage.
    pub(super) storage_conflicts: BitMatrix<CoroutineSavedLocal, CoroutineSavedLocal>,

    /// For every suspending block, the locals which are storage-live across
    /// that suspension point.
    storage_liveness: IndexVec<BasicBlock, Option<DenseBitSet<Local>>>,
}

/// Computes which locals have to be stored in the state-machine for the
/// given coroutine.
///
/// The basic idea is as follows:
/// - a local is live until we encounter a `StorageDead` statement. In
///   case none exist, the local is considered to be always live.
/// - a local has to be stored if it is either directly used after the
///   the suspend point, or if it is live and has been previously borrowed.
#[tracing::instrument(level = "trace", skip(tcx, body))]
pub(super) fn locals_live_across_suspend_points<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    always_live_locals: &DenseBitSet<Local>,
    movable: bool,
) -> LivenessInfo {
    // Calculate when MIR locals have live storage. This gives us an upper bound of their
    // lifetimes.
    let mut storage_live = MaybeStorageLive::new(std::borrow::Cow::Borrowed(always_live_locals))
        .iterate_to_fixpoint(tcx, body, None)
        .into_results_cursor(body);

    // Calculate the MIR locals that have been previously borrowed (even if they are still active).
    let borrowed_locals = MaybeBorrowedLocals.iterate_to_fixpoint(tcx, body, Some("coroutine"));
    let borrowed_locals_cursor1 = ResultsCursor::new_borrowing(body, &borrowed_locals);
    let mut borrowed_locals_cursor2 = ResultsCursor::new_borrowing(body, &borrowed_locals);

    // Calculate the MIR locals that we need to keep storage around for.
    let requires_storage =
        MaybeRequiresStorage::new(borrowed_locals_cursor1).iterate_to_fixpoint(tcx, body, None);
    let mut requires_storage_cursor = ResultsCursor::new_borrowing(body, &requires_storage);

    // Calculate the liveness of MIR locals ignoring borrows.
    let mut liveness =
        MaybeLiveLocals.iterate_to_fixpoint(tcx, body, Some("coroutine")).into_results_cursor(body);

    let mut storage_liveness_map = IndexVec::from_elem(None, &body.basic_blocks);
    let mut live_locals_at_suspension_points = Vec::new();
    let mut source_info_at_suspension_points = Vec::new();
    let mut live_locals_at_any_suspension_point = DenseBitSet::new_empty(body.local_decls.len());

    for (block, data) in body.basic_blocks.iter_enumerated() {
        let TerminatorKind::Yield { .. } = data.terminator().kind else { continue };

        let loc = Location { block, statement_index: data.statements.len() };

        liveness.seek_to_block_end(block);
        let mut live_locals = liveness.get().clone();

        if !movable {
            // The `liveness` variable contains the liveness of MIR locals ignoring borrows.
            // This is correct for movable coroutines since borrows cannot live across
            // suspension points. However for immovable coroutines we need to account for
            // borrows, so we conservatively assume that all borrowed locals are live until
            // we find a StorageDead statement referencing the locals.
            // To do this we just union our `liveness` result with `borrowed_locals`, which
            // contains all the locals which has been borrowed before this suspension point.
            // If a borrow is converted to a raw reference, we must also assume that it lives
            // forever. Note that the final liveness is still bounded by the storage liveness
            // of the local, which happens using the `intersect` operation below.
            borrowed_locals_cursor2.seek_before_primary_effect(loc);
            live_locals.union(borrowed_locals_cursor2.get());
        }

        // Store the storage liveness for later use so we can restore the state
        // after a suspension point
        storage_live.seek_before_primary_effect(loc);
        storage_liveness_map[block] = Some(storage_live.get().clone());

        // Locals live are live at this point only if they are used across
        // suspension points (the `liveness` variable)
        // and their storage is required (the `storage_required` variable)
        requires_storage_cursor.seek_before_primary_effect(loc);
        live_locals.intersect(requires_storage_cursor.get());

        // The coroutine argument is ignored.
        live_locals.remove(SELF_ARG);

        debug!(?loc, ?live_locals);

        // Add the locals live at this suspension point to the set of locals which live across
        // any suspension points
        live_locals_at_any_suspension_point.union(&live_locals);

        live_locals_at_suspension_points.push(live_locals);
        source_info_at_suspension_points.push(data.terminator().source_info);
    }

    debug!(?live_locals_at_any_suspension_point);
    let saved_locals = CoroutineSavedLocals(live_locals_at_any_suspension_point);

    // Renumber our liveness_map bitsets to include only the locals we are
    // saving.
    let live_locals_at_suspension_points = live_locals_at_suspension_points
        .iter()
        .map(|live_here| saved_locals.renumber_bitset(live_here))
        .collect();

    let storage_conflicts = compute_storage_conflicts(
        body,
        &saved_locals,
        always_live_locals.clone(),
        &requires_storage,
    );

    LivenessInfo {
        saved_locals,
        live_locals_at_suspension_points,
        source_info_at_suspension_points,
        storage_conflicts,
        storage_liveness: storage_liveness_map,
    }
}

/// The set of `Local`s that must be saved across yield points.
///
/// `CoroutineSavedLocal` is indexed in terms of the elements in this set;
/// i.e. `CoroutineSavedLocal::new(1)` corresponds to the second local
/// included in this set.
pub(super) struct CoroutineSavedLocals(DenseBitSet<Local>);

impl CoroutineSavedLocals {
    /// Returns an iterator over each `CoroutineSavedLocal` along with the `Local` it corresponds
    /// to.
    fn iter_enumerated(&self) -> impl '_ + Iterator<Item = (CoroutineSavedLocal, Local)> {
        self.iter().enumerate().map(|(i, l)| (CoroutineSavedLocal::from(i), l))
    }

    /// Transforms a `DenseBitSet<Local>` that contains only locals saved across yield points to the
    /// equivalent `DenseBitSet<CoroutineSavedLocal>`.
    fn renumber_bitset(&self, input: &DenseBitSet<Local>) -> DenseBitSet<CoroutineSavedLocal> {
        assert!(self.superset(input), "{:?} not a superset of {:?}", self.0, input);
        let mut out = DenseBitSet::new_empty(self.count());
        for (saved_local, local) in self.iter_enumerated() {
            if input.contains(local) {
                out.insert(saved_local);
            }
        }
        out
    }

    pub(super) fn get(&self, local: Local) -> Option<CoroutineSavedLocal> {
        if !self.contains(local) {
            return None;
        }

        let idx = self.iter().take_while(|&l| l < local).count();
        Some(CoroutineSavedLocal::new(idx))
    }
}

impl ops::Deref for CoroutineSavedLocals {
    type Target = DenseBitSet<Local>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// For every saved local, looks for which locals are StorageLive at the same
/// time. Generates a bitset for every local of all the other locals that may be
/// StorageLive simultaneously with that local. This is used in the layout
/// computation; see `CoroutineLayout` for more.
fn compute_storage_conflicts<'mir, 'tcx>(
    body: &'mir Body<'tcx>,
    saved_locals: &'mir CoroutineSavedLocals,
    always_live_locals: DenseBitSet<Local>,
    results: &Results<'tcx, MaybeRequiresStorage<'mir, 'tcx>>,
) -> BitMatrix<CoroutineSavedLocal, CoroutineSavedLocal> {
    assert_eq!(body.local_decls.len(), saved_locals.domain_size());

    debug!("compute_storage_conflicts({:?})", body.span);
    debug!("always_live = {:?}", always_live_locals);

    // Locals that are always live or ones that need to be stored across
    // suspension points are not eligible for overlap.
    let mut ineligible_locals = always_live_locals;
    ineligible_locals.intersect(&**saved_locals);

    // Compute the storage conflicts for all eligible locals.
    let mut visitor = StorageConflictVisitor {
        body,
        saved_locals,
        local_conflicts: BitMatrix::from_row_n(&ineligible_locals, body.local_decls.len()),
        eligible_storage_live: DenseBitSet::new_empty(body.local_decls.len()),
    };

    visit_reachable_results(body, results, &mut visitor);

    let local_conflicts = visitor.local_conflicts;

    // Compress the matrix using only stored locals (Local -> CoroutineSavedLocal).
    //
    // NOTE: Today we store a full conflict bitset for every local. Technically
    // this is twice as many bits as we need, since the relation is symmetric.
    // However, in practice these bitsets are not usually large. The layout code
    // also needs to keep track of how many conflicts each local has, so it's
    // simpler to keep it this way for now.
    let mut storage_conflicts = BitMatrix::new(saved_locals.count(), saved_locals.count());
    for (saved_local_a, local_a) in saved_locals.iter_enumerated() {
        if ineligible_locals.contains(local_a) {
            // Conflicts with everything.
            storage_conflicts.insert_all_into_row(saved_local_a);
        } else {
            // Keep overlap information only for stored locals.
            for (saved_local_b, local_b) in saved_locals.iter_enumerated() {
                if local_conflicts.contains(local_a, local_b) {
                    storage_conflicts.insert(saved_local_a, saved_local_b);
                }
            }
        }
    }
    storage_conflicts
}

struct StorageConflictVisitor<'a, 'tcx> {
    body: &'a Body<'tcx>,
    saved_locals: &'a CoroutineSavedLocals,
    // FIXME(tmandry): Consider using sparse bitsets here once we have good
    // benchmarks for coroutines.
    local_conflicts: BitMatrix<Local, Local>,
    // We keep this bitset as a buffer to avoid reallocating memory.
    eligible_storage_live: DenseBitSet<Local>,
}

impl<'a, 'tcx> ResultsVisitor<'tcx, MaybeRequiresStorage<'a, 'tcx>>
    for StorageConflictVisitor<'a, 'tcx>
{
    fn visit_after_early_statement_effect(
        &mut self,
        _analysis: &MaybeRequiresStorage<'a, 'tcx>,
        state: &DenseBitSet<Local>,
        _statement: &Statement<'tcx>,
        loc: Location,
    ) {
        self.apply_state(state, loc);
    }

    fn visit_after_early_terminator_effect(
        &mut self,
        _analysis: &MaybeRequiresStorage<'a, 'tcx>,
        state: &DenseBitSet<Local>,
        _terminator: &Terminator<'tcx>,
        loc: Location,
    ) {
        self.apply_state(state, loc);
    }
}

impl StorageConflictVisitor<'_, '_> {
    fn apply_state(&mut self, state: &DenseBitSet<Local>, loc: Location) {
        // Ignore unreachable blocks.
        if let TerminatorKind::Unreachable = self.body.basic_blocks[loc.block].terminator().kind {
            return;
        }

        self.eligible_storage_live.clone_from(state);
        self.eligible_storage_live.intersect(&**self.saved_locals);

        for local in self.eligible_storage_live.iter() {
            self.local_conflicts.union_row_with(&self.eligible_storage_live, local);
        }

        if self.eligible_storage_live.count() > 1 {
            trace!("at {:?}, eligible_storage_live={:?}", loc, self.eligible_storage_live);
        }
    }
}

#[tracing::instrument(level = "trace", skip(liveness, body))]
pub(super) fn compute_layout<'tcx>(
    liveness: LivenessInfo,
    body: &Body<'tcx>,
) -> (
    IndexVec<Local, Option<(Ty<'tcx>, VariantIdx, FieldIdx)>>,
    CoroutineLayout<'tcx>,
    IndexVec<BasicBlock, Option<DenseBitSet<Local>>>,
) {
    let LivenessInfo {
        saved_locals,
        live_locals_at_suspension_points,
        source_info_at_suspension_points,
        storage_conflicts,
        storage_liveness,
    } = liveness;

    // Gather live local types.
    let mut tys: IndexVec<CoroutineSavedLocal, CoroutineSavedTy<'_>> = saved_locals
        .iter_enumerated()
        .map(|(saved_local, local)| {
            debug!("coroutine saved local {:?} => {:?}", saved_local, local);

            let decl = &body.local_decls[local];

            // Do not `unwrap_crate_local` here, as post-borrowck cleanup may have already cleared
            // the information. This is alright, since `ignore_for_traits` is only relevant when
            // this code runs on pre-cleanup MIR, and `ignore_for_traits = false` is the safer
            // default.
            let ignore_for_traits = match decl.local_info {
                // Do not include raw pointers created from accessing `static` items, as those could
                // well be re-created by another access to the same static.
                ClearCrossCrate::Set(LocalInfo::StaticRef { is_thread_local, .. }) => {
                    !is_thread_local
                }
                // Fake borrows are only read by fake reads, so do not have any reality in
                // post-analysis MIR.
                ClearCrossCrate::Set(LocalInfo::FakeBorrow) => true,
                _ => false,
            };

            CoroutineSavedTy {
                ty: decl.ty,
                source_info: decl.source_info,
                ignore_for_traits,
                // Will be set later when walking debuginfo.
                debuginfo_name: None,
            }
        })
        .collect();

    // Leave empty variants for the UNRESUMED, RETURNED, and POISONED states.
    // In debuginfo, these will correspond to the beginning (UNRESUMED) or end
    // (RETURNED, POISONED) of the function.
    let body_span = body.source_scopes[OUTERMOST_SOURCE_SCOPE].span;
    let mut variant_source_info: IndexVec<VariantIdx, SourceInfo> = IndexVec::with_capacity(
        CoroutineArgs::RESERVED_VARIANTS + live_locals_at_suspension_points.len(),
    );
    variant_source_info.extend([
        SourceInfo::outermost(body_span.shrink_to_lo()),
        SourceInfo::outermost(body_span.shrink_to_hi()),
        SourceInfo::outermost(body_span.shrink_to_hi()),
    ]);

    // Simple map from new to old indices to avoid repeatedly counting bits.
    let reverse_local_map: IndexVec<CoroutineSavedLocal, Local> = saved_locals.iter().collect();

    // Build the coroutine variant field list.
    // Create a map from local indices to coroutine struct indices.
    let mut variant_fields: IndexVec<VariantIdx, _> = IndexVec::from_elem_n(
        IndexVec::new(),
        CoroutineArgs::RESERVED_VARIANTS + live_locals_at_suspension_points.len(),
    );
    let mut remap = IndexVec::from_elem_n(None, saved_locals.domain_size());
    for (live_locals, &source_info_at_suspension_point, (variant_index, fields)) in izip!(
        &live_locals_at_suspension_points,
        &source_info_at_suspension_points,
        variant_fields.iter_enumerated_mut().skip(CoroutineArgs::RESERVED_VARIANTS)
    ) {
        *fields = live_locals.iter().collect();
        for (idx, &saved_local) in fields.iter_enumerated() {
            // Note that if a field is included in multiple variants, we will
            // just use the first one here. That's fine; fields do not move
            // around inside coroutines, so it doesn't matter which variant
            // index we access them by.
            remap[reverse_local_map[saved_local]] = Some((tys[saved_local].ty, variant_index, idx));
        }
        variant_source_info.push(source_info_at_suspension_point);
    }
    debug!(?variant_fields);
    debug!(?storage_conflicts);

    for var in &body.var_debug_info {
        let VarDebugInfoContents::Place(place) = &var.value else { continue };
        let Some(local) = place.as_local() else { continue };
        let Some(&Some((_, variant, field))) = remap.get(local) else {
            continue;
        };

        let saved_local: CoroutineSavedLocal = variant_fields[variant][field];
        tys[saved_local].debuginfo_name.get_or_insert(var.name);
    }

    let layout = CoroutineLayout {
        field_tys: tys,
        variant_fields,
        variant_source_info,
        storage_conflicts,
        relocated_upvars: IndexVec::new(),
        pack: PackCoroutineLayout::No,
    };
    debug!(?remap);
    debug!(?layout);
    debug!(?storage_liveness);

    (remap, layout, storage_liveness)
}

#[instrument(level = "debug", skip(tcx), ret)]
pub(crate) fn mir_coroutine_witnesses<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> Option<CoroutineLayout<'tcx>> {
    let (body, _) = tcx.mir_promoted(def_id);
    let body = body.borrow();
    let body = &*body;

    // The first argument is the coroutine type passed by value
    let coroutine_ty = body.local_decls[ty::CAPTURE_STRUCT_LOCAL].ty;

    let movable = match *coroutine_ty.kind() {
        ty::Coroutine(def_id, _) => tcx.coroutine_movability(def_id) == hir::Movability::Movable,
        ty::Error(_) => return None,
        _ => span_bug!(body.span, "unexpected coroutine type {}", coroutine_ty),
    };

    // The witness simply contains all locals live across suspend points.

    let always_live_locals = always_storage_live_locals(body);
    let liveness_info = locals_live_across_suspend_points(tcx, body, &always_live_locals, movable);

    // Extract locals which are live across suspension point into `layout`
    // `remap` gives a mapping from local indices onto coroutine struct indices
    // `storage_liveness` tells us which locals have live storage at suspension points
    let (_, coroutine_layout, _) = compute_layout(liveness_info, body);

    check_suspend_tys(tcx, &coroutine_layout, body);
    check_field_tys_sized(tcx, &coroutine_layout, def_id);

    Some(coroutine_layout)
}

fn check_field_tys_sized<'tcx>(
    tcx: TyCtxt<'tcx>,
    coroutine_layout: &CoroutineLayout<'tcx>,
    def_id: LocalDefId,
) {
    // No need to check if unsized_fn_params is disabled,
    // since we will error during typeck.
    if !tcx.features().unsized_fn_params() {
        return;
    }

    // FIXME(#132279): @lcnr believes that we may want to support coroutines
    // whose `Sized`-ness relies on the hidden types of opaques defined by the
    // parent function. In this case we'd have to be able to reveal only these
    // opaques here.
    let infcx = tcx.infer_ctxt().ignoring_regions().build(TypingMode::non_body_analysis());
    let param_env = tcx.param_env(def_id);

    let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
    for field_ty in &coroutine_layout.field_tys {
        ocx.register_bound(
            ObligationCause::new(
                field_ty.source_info.span,
                def_id,
                ObligationCauseCode::SizedCoroutineInterior(def_id),
            ),
            param_env,
            field_ty.ty,
            tcx.require_lang_item(hir::LangItem::Sized, field_ty.source_info.span),
        );
    }

    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    debug!(?errors);
    if !errors.is_empty() {
        infcx.err_ctxt().report_fulfillment_errors(errors);
    }
}

fn check_suspend_tys<'tcx>(tcx: TyCtxt<'tcx>, layout: &CoroutineLayout<'tcx>, body: &Body<'tcx>) {
    let mut linted_tys = FxHashSet::default();

    for (variant, yield_source_info) in
        layout.variant_fields.iter().zip(&layout.variant_source_info)
    {
        debug!(?variant);
        for &local in variant {
            let decl = &layout.field_tys[local];
            debug!(?decl);

            if !decl.ignore_for_traits && linted_tys.insert(decl.ty) {
                let Some(hir_id) = decl.source_info.scope.lint_root(&body.source_scopes) else {
                    continue;
                };

                check_must_not_suspend_ty(
                    tcx,
                    decl.ty,
                    hir_id,
                    SuspendCheckData {
                        source_span: decl.source_info.span,
                        yield_span: yield_source_info.span,
                        plural_len: 1,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[derive(Default)]
struct SuspendCheckData<'a> {
    source_span: Span,
    yield_span: Span,
    descr_pre: &'a str,
    descr_post: &'a str,
    plural_len: usize,
}

// Returns whether it emitted a diagnostic or not
// Note that this fn and the proceeding one are based on the code
// for creating must_use diagnostics
//
// Note that this technique was chosen over things like a `Suspend` marker trait
// as it is simpler and has precedent in the compiler
fn check_must_not_suspend_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    hir_id: hir::HirId,
    data: SuspendCheckData<'_>,
) -> bool {
    if ty.is_unit() {
        return false;
    }

    let plural_suffix = pluralize!(data.plural_len);

    debug!("Checking must_not_suspend for {}", ty);

    match *ty.kind() {
        ty::Adt(_, args) if ty.is_box() => {
            let boxed_ty = args.type_at(0);
            let allocator_ty = args.type_at(1);
            check_must_not_suspend_ty(
                tcx,
                boxed_ty,
                hir_id,
                SuspendCheckData { descr_pre: &format!("{}boxed ", data.descr_pre), ..data },
            ) || check_must_not_suspend_ty(
                tcx,
                allocator_ty,
                hir_id,
                SuspendCheckData { descr_pre: &format!("{}allocator ", data.descr_pre), ..data },
            )
        }
        // FIXME(sized_hierarchy): This should be replaced with a requirement that types in
        // coroutines implement `const Sized`. Scalable vectors are temporarily `Sized` while
        // `feature(sized_hierarchy)` is not fully implemented, but in practice are
        // non-`const Sized` and so do not have a known size at compilation time. Layout computation
        // for a coroutine containing scalable vectors would be incorrect.
        ty::Adt(def, _) if def.repr().scalable() => {
            tcx.dcx()
                .span_err(data.source_span, "scalable vectors cannot be held over await points");
            true
        }
        ty::Adt(def, _) => check_must_not_suspend_def(tcx, def.did(), hir_id, data),
        // FIXME: support adding the attribute to TAITs
        ty::Alias(_, ty::AliasTy { kind: ty::Opaque { def_id: def }, .. }) => {
            let mut has_emitted = false;
            for &(predicate, _) in tcx.explicit_item_bounds(def).skip_binder() {
                // We only look at the `DefId`, so it is safe to skip the binder here.
                if let ty::ClauseKind::Trait(ref poly_trait_predicate) =
                    predicate.kind().skip_binder()
                {
                    let def_id = poly_trait_predicate.trait_ref.def_id;
                    let descr_pre = &format!("{}implementer{} of ", data.descr_pre, plural_suffix);
                    if check_must_not_suspend_def(
                        tcx,
                        def_id,
                        hir_id,
                        SuspendCheckData { descr_pre, ..data },
                    ) {
                        has_emitted = true;
                        break;
                    }
                }
            }
            has_emitted
        }
        ty::Dynamic(binder, _) => {
            let mut has_emitted = false;
            for predicate in binder.iter() {
                if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate.skip_binder() {
                    let def_id = trait_ref.def_id;
                    let descr_post = &format!(" trait object{}{}", plural_suffix, data.descr_post);
                    if check_must_not_suspend_def(
                        tcx,
                        def_id,
                        hir_id,
                        SuspendCheckData { descr_post, ..data },
                    ) {
                        has_emitted = true;
                        break;
                    }
                }
            }
            has_emitted
        }
        ty::Tuple(fields) => {
            let mut has_emitted = false;
            for (i, ty) in fields.iter().enumerate() {
                let descr_post = &format!(" in tuple element {i}");
                if check_must_not_suspend_ty(
                    tcx,
                    ty,
                    hir_id,
                    SuspendCheckData { descr_post, ..data },
                ) {
                    has_emitted = true;
                }
            }
            has_emitted
        }
        ty::Array(ty, len) => {
            let descr_pre = &format!("{}array{} of ", data.descr_pre, plural_suffix);
            check_must_not_suspend_ty(
                tcx,
                ty,
                hir_id,
                SuspendCheckData {
                    descr_pre,
                    // FIXME(must_not_suspend): This is wrong. We should handle printing unevaluated consts.
                    plural_len: len.try_to_target_usize(tcx).unwrap_or(0) as usize + 1,
                    ..data
                },
            )
        }
        // If drop tracking is enabled, we want to look through references, since the referent
        // may not be considered live across the await point.
        ty::Ref(_region, ty, _mutability) => {
            let descr_pre = &format!("{}reference{} to ", data.descr_pre, plural_suffix);
            check_must_not_suspend_ty(tcx, ty, hir_id, SuspendCheckData { descr_pre, ..data })
        }
        _ => false,
    }
}

fn check_must_not_suspend_def(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    hir_id: hir::HirId,
    data: SuspendCheckData<'_>,
) -> bool {
    if let Some(reason_str) = find_attr!(tcx, def_id, MustNotSupend {reason} => reason) {
        let reason = reason_str.map(|s| MustNotSuspendReason { span: data.source_span, reason: s });
        tcx.emit_node_span_lint(
            rustc_session::lint::builtin::MUST_NOT_SUSPEND,
            hir_id,
            data.source_span,
            MustNotSupend {
                tcx,
                yield_sp: data.yield_span,
                reason,
                src_sp: data.source_span,
                pre: data.descr_pre,
                def_id,
                post: data.descr_post,
            },
        );

        true
    } else {
        false
    }
}
