use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_index::IndexVec;
use rustc_index::bit_set::{DenseBitSet, MixedBitSet};
use rustc_index::interval::IntervalSet;
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::traits::TraitErrors;
use rustc_middle::mir::{BasicBlock, Body, ConstraintCategory, Local, Location};
use rustc_middle::traits::query::DropckOutlivesResult;
use rustc_middle::ty::{GenericArg, Ty, TypeVisitableExt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{HasMoveData, MoveData, MovePathIndex};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};
use rustc_mir_dataflow::{Analysis, MaybeReachable, ResultsCursor};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_trait_selection::traits::query::dropck_outlives;
use rustc_trait_selection::traits::query::type_op::{DropckOutlives, TypeOpOutput};
use tracing::debug;

use crate::polonius::{DeferredRegionLiveness, LiveRegionVariances, record_live_region_variance};
use crate::region_infer::values::LivenessValues;
use crate::type_check::liveness::local_use_map::LocalUseMap;
use crate::type_check::liveness::make_all_regions_live;
use crate::type_check::{NormalizeLocation, TypeChecker};
use crate::universal_regions::UniversalRegions;
use crate::{BorrowckInferCtxt, polonius};

/// This is the heart of the liveness computation. For each variable X
/// that requires a liveness computation, it walks over all the uses
/// of X and does a reverse depth-first search ("trace") through the
/// MIR. This search stops when we find a definition of that variable.
/// The points visited in this search is the USE-LIVE set for the variable;
/// of those points is added to all the regions that appear in the variable's
/// type.
///
/// We then also walks through each *drop* of those variables and does
/// another search, stopping when we reach a use or definition. This
/// is the DROP-LIVE set of points. Each of the points in the
/// DROP-LIVE set are to the liveness sets for regions found in the
/// `dropck_outlives` result of the variable's type (in particular,
/// this respects `#[may_dangle]` annotations).
pub(super) fn trace<'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
    location_map: &DenseLocationMap,
    move_data: &MoveData<'tcx>,
    relevant_live_locals: &[Local],
    boring_locals: &[Local],
    deferred_locals: &FxIndexSet<Local>,
) {
    let _timer = typeck.tcx().prof.generic_activity("borrowck_liveness_trace");

    // The use map must also cover the deferred locals: their liveness is computed later, from
    // this same map, when the loan liveness traversal first reaches one of their regions.
    let use_map_locals = relevant_live_locals.iter().chain(deferred_locals).copied();
    let local_use_map = LocalUseMap::build(use_map_locals, location_map, typeck.body);
    let comp = LivenessComputation::new(
        typeck.infcx,
        typeck.body,
        location_map,
        move_data,
        &local_use_map,
    );

    let mut results = LivenessResults::new(typeck, comp);

    results.record_legacy_polonius_drop_facts(relevant_live_locals, &deferred_locals);

    results.compute_for_all_locals(relevant_live_locals);

    let mut deferred_liveness = DeferredRegionLiveness::default();
    results.dropck_boring_locals(boring_locals, &deferred_locals, &mut deferred_liveness);

    if let Some(polonius_context) = &mut typeck.polonius_context {
        polonius_context.deferred_liveness = deferred_liveness;
        polonius_context.local_use_map = Some(local_use_map);
    }
}

pub(crate) struct LivenessComputation<'a, 'tcx> {
    pub(crate) infcx: &'a BorrowckInferCtxt<'tcx>,

    pub(crate) body: &'a Body<'tcx>,

    /// Defines the `PointIndex` mapping
    pub(crate) location_map: &'a DenseLocationMap,

    /// Mapping to/from the various indices used for initialization tracking.
    move_data: &'a MoveData<'tcx>,

    /// Results of dataflow tracking which variables (and paths) have been
    /// initialized. Computed lazily when needed by drop-liveness.
    flow_inits: Option<ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>>>,

    /// Index indicating where each variable is assigned, used, or
    /// dropped.
    local_use_map: &'a LocalUseMap,

    // Caches for the results of `initialized_at_terminator` and `initialized_at_exit`.
    term_states: IndexVec<BasicBlock, Option<MaybeReachable<MixedBitSet<MovePathIndex>>>>,
    exit_states: IndexVec<BasicBlock, Option<MaybeReachable<MixedBitSet<MovePathIndex>>>>,

    /// Set of points that define the current local.
    defs: DenseBitSet<PointIndex>,

    /// Points where the current variable is "use live" -- meaning
    /// that there is a future "full use" that may use its value.
    use_live_at: IntervalSet<PointIndex>,

    /// Points where the current variable is "drop live" -- meaning
    /// that there is no future "full use" that may use its value, but
    /// there is a future drop.
    drop_live_at: DenseBitSet<PointIndex>,

    /// Locations where drops may occur.
    drop_locations: Vec<Location>,

    /// Stack used when doing (reverse) DFS.
    stack: Vec<PointIndex>,
}

struct LivenessResults<'a, 'typeck, 'tcx> {
    /// Current type-checker, giving us our inference context etc.
    typeck: &'a mut TypeChecker<'typeck, 'tcx>,

    /// Cache for the results of `dropck_outlives` query.
    drop_data: FxIndexMap<Ty<'tcx>, DropData<'tcx>>,

    comp: LivenessComputation<'a, 'tcx>,
}

impl<'a, 'typeck, 'tcx> LivenessResults<'a, 'typeck, 'tcx> {
    fn new(
        typeck: &'a mut TypeChecker<'typeck, 'tcx>,
        comp: LivenessComputation<'a, 'tcx>,
    ) -> Self {
        LivenessResults { typeck, drop_data: FxIndexMap::default(), comp }
    }

    fn compute_for_all_locals(&mut self, relevant_live_locals: &[Local]) {
        for &local in relevant_live_locals {
            self.compute_for_local(local);
        }
    }

    fn compute_for_local(&mut self, local: Local) {
        // If we end up needing to compute the drop data (because there are
        // drop-live points), then we need to register region constraints and
        // emit drop facts.
        let mut computed_drop_data = None;

        self.comp.compute(
            local,
            self.typeck.universal_regions,
            self.typeck.polonius_context.as_mut().map(|c| &mut c.live_region_variances),
            &mut self.typeck.constraints.liveness_constraints,
            || {
                let local_ty = self.comp.body.local_decls[local].ty;
                let local_span = self.comp.body.local_decls[local].source_info.span;
                let drop_data =
                    dropck_local(&self.typeck.infcx, &mut self.drop_data, local_ty, local_span);
                let drop_data = computed_drop_data.insert(drop_data);
                &drop_data.dropck_result.kinds
            },
        );

        if let Some(drop_data) = computed_drop_data {
            if let Some(data) = &drop_data.region_constraint_data {
                for &drop_location in &self.comp.drop_locations {
                    self.typeck.push_region_constraints(
                        drop_location.to_locations(),
                        ConstraintCategory::Boring,
                        data,
                    );
                }
            }

            for &kind in &drop_data.dropck_result.kinds {
                polonius::legacy::emit_drop_facts(
                    self.typeck.tcx(),
                    local,
                    &kind,
                    self.typeck.universal_regions,
                    self.typeck.polonius_facts,
                );
            }
        }
    }

    /// Runs dropck for locals whose liveness isn't relevant. This is
    /// necessary to eagerly detect unbound recursion during drop glue computation.
    ///
    /// These are all the locals which do not potentially reference a region local
    /// to this body. Locals which only reference free regions are always drop-live
    /// and can therefore safely be dropped.
    fn dropck_boring_locals(
        &mut self,
        boring_locals: &[Local],
        deferred_locals: &FxIndexSet<Local>,
        deferred_liveness: &mut DeferredRegionLiveness<'tcx>,
    ) {
        for &local in boring_locals {
            let is_local_deferred = deferred_locals.contains(&local);
            self.dropck_boring_local(local, is_local_deferred, deferred_liveness);
        }
    }

    fn dropck_boring_local(
        &mut self,
        local: Local,
        is_local_deferred: bool,
        deferred_liveness: &mut DeferredRegionLiveness<'tcx>,
    ) {
        let typeck = &mut *self.typeck;
        let local_ty = self.comp.body.local_decls[local].ty;
        let local_span = self.comp.body.local_decls[local].source_info.span;

        // If we had treated this as "relevant", we would have run `compute_for_local`. This
        // in turn would have skipped calculating dropck *at all* for locals without drop-liveness.
        // Calculating drop-liveness is expensive, but we can skip it when we know that there
        // are *no* drops (which is relatively cheap).
        if is_local_deferred && self.comp.local_use_map.drops(local).next().is_none() {
            deferred_liveness.defer_local(
                typeck.infcx.tcx,
                typeck.universal_regions,
                local,
                local_ty,
                &[],
            );
            return;
        }

        // We need to compute dropck for *all* boring locals because we report overflows.
        //
        // FIXME: there is an argument to be made that we don't need to do this for boring locals
        // without drop-liveness, because we skip it for *relevant* locals without drop-liveness.
        // But, this is preexisting even on NLL, so leaving it for now.
        let drop_data = dropck_local(&typeck.infcx, &mut self.drop_data, local_ty, local_span);

        // We are done with *truly* boring locals.
        if !is_local_deferred {
            return;
        }

        // If this local is deferred and has drop region constraints, we need to register
        // them, but *only if the local is drop-live*.
        // It doesn't really make sense to only check drop-liveness but defer use-liveness,
        // so we just treat this as eager.
        if drop_data.region_constraint_data.is_some() {
            self.compute_for_local(local);
            return;
        }

        // The only other thing we need to do *eagerly* for deferred locals is to register
        // legacy drop facts (because these facts are on `typeck`).
        for &kind in &drop_data.dropck_result.kinds {
            polonius::legacy::emit_drop_facts(
                typeck.tcx(),
                local,
                &kind,
                typeck.universal_regions,
                typeck.polonius_facts,
            );
        }

        // Finally, we mark that this local is deferred, including the drop kinds.
        deferred_liveness.defer_local(
            typeck.infcx.tcx,
            typeck.universal_regions,
            local,
            local_ty,
            &drop_data.dropck_result.kinds,
        );
    }

    /// Add extra drop facts needed for Polonius Legacy.
    ///
    /// Add facts for all locals with free regions, since regions may outlive
    /// the function body only at certain nodes in the CFG.
    fn record_legacy_polonius_drop_facts(
        &mut self,
        nll_relevant_locals: &[Local],
        deferred_polonius_relevant_locals: &FxIndexSet<Local>,
    ) {
        // This is *all wonky* because this used to call a shared
        // `add_drop_live_facts_for` function that was also used for regular
        // relevant locals. Presumably, this can be cleaned up quite a bit.
        // FIXME for future hackers: investigate whether this is
        // actually necessary; these facts come from Polonius
        // and probably maybe plausibly does not need to go back in.
        // It may be necessary to just pick out the parts of
        // `add_drop_live_facts_for()` that make sense.
        let Some(facts) = self.typeck.polonius_facts.as_ref() else { return };
        let facts_to_add: Vec<_> = {
            let nll_relevant_locals: FxIndexSet<_> = nll_relevant_locals.iter().copied().collect();

            facts
                .var_dropped_at
                .iter()
                .filter_map(|&(local, location_index)| {
                    let local_ty = self.comp.body.local_decls[local].ty;
                    if nll_relevant_locals.contains(&local)
                        || deferred_polonius_relevant_locals.contains(&local)
                        || !local_ty.has_free_regions()
                    {
                        return None;
                    }

                    let location = self.typeck.location_table.to_location(location_index);
                    Some((local, local_ty, location))
                })
                .collect()
        };

        let live_at = IntervalSet::new(self.comp.location_map.num_points());
        for (local, local_ty, location) in facts_to_add {
            let local_span = self.comp.body.local_decls[local].source_info.span;
            let drop_data =
                dropck_local(&self.typeck.infcx, &mut self.drop_data, local_ty, local_span);

            if let Some(data) = &drop_data.region_constraint_data {
                self.typeck.push_region_constraints(
                    location.to_locations(),
                    ConstraintCategory::Boring,
                    data,
                );
            }

            for &kind in &drop_data.dropck_result.kinds {
                make_all_regions_live(
                    self.typeck.infcx,
                    self.typeck.universal_regions,
                    &mut self.typeck.constraints.liveness_constraints,
                    kind,
                    &live_at,
                );
                polonius::legacy::emit_drop_facts(
                    self.typeck.tcx(),
                    local,
                    &kind,
                    self.typeck.universal_regions,
                    self.typeck.polonius_facts,
                );
            }

            if let Some(polonius_context) = self.typeck.polonius_context.as_mut() {
                record_live_region_variance(
                    self.typeck.infcx.tcx,
                    &mut polonius_context.live_region_variances,
                    self.typeck.universal_regions,
                    local_ty,
                );
            }
        }
    }
}

enum InitAtLocation {
    Terminator,
    Exit,
}

impl<'a, 'tcx> LivenessComputation<'a, 'tcx> {
    pub(crate) fn new(
        infcx: &'a BorrowckInferCtxt<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
        move_data: &'a MoveData<'tcx>,
        local_use_map: &'a LocalUseMap,
    ) -> Self {
        let num_points = location_map.num_points();
        LivenessComputation {
            infcx,
            body,
            location_map,
            move_data,
            flow_inits: None,
            local_use_map,
            term_states: IndexVec::new(),
            exit_states: IndexVec::new(),
            defs: DenseBitSet::new_empty(num_points),
            use_live_at: IntervalSet::new(num_points),
            drop_live_at: DenseBitSet::new_empty(num_points),
            drop_locations: vec![],
            stack: vec![],
        }
    }

    /// Compute for a given local the use- and drop-live points
    pub(crate) fn compute<'drop_data>(
        &mut self,
        local: Local,
        universal_regions: &UniversalRegions<'tcx>,
        live_region_variances: Option<&mut LiveRegionVariances>,
        liveness_constraints: &mut LivenessValues,
        get_drop_args: impl FnOnce() -> &'drop_data [GenericArg<'tcx>],
    ) where
        'tcx: 'drop_data,
    {
        let _timer = self.infcx.tcx.prof.generic_activity("borrowck_liveness_compute");

        self.reset_local_state();
        self.add_defs_for(local);
        self.compute_use_live_points_for(local);
        self.compute_drop_live_points_for(local);

        let local_ty = self.body.local_decls[local].ty;

        // When using `-Zpolonius=next`, we also record the variance of regions in this live type.
        // For dropck in particular, note that we walk the type and not its live components seen in
        // the dropck results. See issue #160670.
        let is_live_anywhere = !self.use_live_at.is_empty() || !self.drop_live_at.is_empty();
        if is_live_anywhere && let Some(live_region_variances) = live_region_variances {
            record_live_region_variance(
                self.infcx.tcx,
                live_region_variances,
                universal_regions,
                local_ty,
            );
        }
        if !self.use_live_at.is_empty() {
            make_all_regions_live(
                self.infcx,
                universal_regions,
                liveness_constraints,
                local_ty,
                &self.use_live_at,
            );
        }
        if !self.drop_live_at.is_empty() {
            let drop_data = get_drop_args();

            // `compute_drop_live_points_for` computes `drop_live_at` as a `DenseBitSet`, but
            // `make_all_regions_live` expects an `IntervalSet`. We thus convert between those two
            // here. Using a `DenseBitSet` has better performance, but storing liveness as a dense
            // matrix has worse performance. There's probably room here for some cleanup, but this
            // works for now.
            let mut drop_live_at = IntervalSet::new(self.drop_live_at.domain_size());
            for point in self.drop_live_at.iter() {
                // We iterate the `drop_live_at` set from smallest to largest values, so
                // we can use append to add things to the interval set at the end.
                drop_live_at.append(point);
            }

            for &kind in drop_data {
                make_all_regions_live(
                    self.infcx,
                    universal_regions,
                    liveness_constraints,
                    kind,
                    &drop_live_at,
                );
            }
        }
    }

    /// Clear the value of fields that are "per local variable".
    fn reset_local_state(&mut self) {
        self.defs.clear();
        self.use_live_at.clear();
        self.drop_live_at.clear();
        self.drop_locations.clear();
        assert!(self.stack.is_empty());
    }

    /// Adds the definitions of `local` into `self.defs`.
    fn add_defs_for(&mut self, local: Local) {
        for def in self.local_use_map.defs(local) {
            debug!("- defined at {:?}", def);
            self.defs.insert(def);
        }
    }

    /// Computes all points where local is "use live" -- meaning its
    /// current value may be used later (except by a drop). This is
    /// done by walking backwards from each use of `local` until we
    /// find a `def` of local.
    ///
    /// Requires `add_defs_for(local)` to have been executed.
    fn compute_use_live_points_for(&mut self, local: Local) {
        debug!("compute_use_live_points_for(local={:?})", local);

        self.stack.extend(self.local_use_map.uses(local));
        while let Some(p) = self.stack.pop() {
            // We are live in this block from the closest to us of:
            //
            //  * Inclusively, the block start
            //  * Exclusively, the previous definition (if it's in this block)
            //  * Exclusively, the previous live_at setting (an optimization)
            let block_start = self.location_map.to_block_start(p);
            let previous_defs = self.defs.last_set_in(block_start..=p);
            let previous_live_at = self.use_live_at.last_set_in(block_start..=p);

            let exclusive_start = match (previous_defs, previous_live_at) {
                (Some(a), Some(b)) => Some(std::cmp::max(a, b)),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            };

            if let Some(exclusive) = exclusive_start {
                self.use_live_at.insert_range(exclusive + 1..=p);

                // If we have a bound after the start of the block, we should
                // not add the predecessors for this block.
                continue;
            } else {
                // Add all the elements of this block.
                self.use_live_at.insert_range(block_start..=p);

                // Then add the predecessors for this block, which are the
                // terminators of predecessor basic blocks. Push those onto the
                // stack so that the next iteration(s) will process them.

                let block = self.location_map.to_location(block_start).block;
                self.stack.extend(
                    self.body.basic_blocks.predecessors()[block]
                        .iter()
                        .map(|&pred_bb| self.body.terminator_loc(pred_bb))
                        .map(|pred_loc| self.location_map.point_from_location(pred_loc)),
                );
            }
        }
    }

    /// Computes all points where local is "drop live" -- meaning its
    /// current value may be dropped later (but not used). This is
    /// done by iterating over the drops of `local` where `local` (or
    /// some subpart of `local`) is initialized. For each such drop,
    /// we walk backwards until we find a point where `local` is
    /// either defined or use-live.
    ///
    /// Requires `compute_use_live_points_for` and `add_defs_for` to
    /// have been executed.
    fn compute_drop_live_points_for(&mut self, local: Local) {
        debug!("compute_drop_live_points_for(local={:?})", local);

        let Some(mpi) = self.move_data.rev_lookup.find_local(local) else { return };
        debug!("compute_drop_live_points_for: mpi = {:?}", mpi);

        // Find the drops where `local` is initialized.
        for drop_point in self.local_use_map.drops(local) {
            let location = self.location_map.to_location(drop_point);
            debug_assert_eq!(self.body.terminator_loc(location.block), location,);

            if self.initialized_at_terminator(location.block, mpi) {
                let inserted = self.drop_live_at.insert(drop_point);
                // Right now, we should not visit a drop_point twice.
                // If we do, this will trigger a debug assert so we know we can optimize.
                debug_assert!(inserted, "drop point should not have been visited yet");
                self.drop_locations.push(location);
                self.stack.push(drop_point);
            }
        }

        debug!("compute_drop_live_points_for: drop_locations={:?}", self.drop_locations);

        // Reverse DFS. But for drops, we do it a bit differently.
        // The stack only ever stores *terminators of blocks*. Within
        // a block, we walk back the statements in an inner loop.
        while let Some(term_point) = self.stack.pop() {
            self.compute_drop_live_points_for_block(mpi, term_point);
        }
    }

    /// Executes one iteration of the drop-live analysis loop.
    ///
    /// The parameter `mpi` is the `MovePathIndex` of the local variable
    /// we are currently analyzing.
    ///
    /// The point `term_point` represents some terminator in the MIR,
    /// where the local `mpi` is drop-live on entry to that terminator.
    ///
    /// This method adds all drop-live points within the block and --
    /// where applicable -- pushes the terminators of preceding blocks
    /// onto `self.stack`.
    fn compute_drop_live_points_for_block(&mut self, mpi: MovePathIndex, term_point: PointIndex) {
        debug!(
            "compute_drop_live_points_for_block(mpi={:?}, term_point={:?})",
            self.move_data.move_paths[mpi].place,
            self.location_map.to_location(term_point),
        );

        // We are only invoked with terminators where `mpi` is
        // drop-live on entry.
        debug_assert!(self.drop_live_at.contains(term_point));

        // Otherwise, scan backwards through the statements in the
        // block. One of them may be either a definition or use
        // live point.
        let term_location = self.location_map.to_location(term_point);
        debug_assert_eq!(self.body.terminator_loc(term_location.block), term_location,);
        let block = term_location.block;
        let entry_point = self.location_map.entry_point(term_location.block);
        for p in (entry_point..term_point).rev() {
            debug!(
                "compute_drop_live_points_for_block: p = {:?}",
                self.location_map.to_location(p)
            );

            if self.defs.contains(p) {
                debug!("compute_drop_live_points_for_block: def site");
                return;
            }

            if self.use_live_at.contains(p) {
                debug!("compute_drop_live_points_for_block: use-live at {:?}", p);
                return;
            }

            if !self.drop_live_at.insert(p) {
                debug!("compute_drop_live_points_for_block: already drop-live");
                return;
            }
        }

        let body = self.body;
        for &pred_block in body.basic_blocks.predecessors()[block].iter() {
            debug!("compute_drop_live_points_for_block: pred_block = {:?}", pred_block,);

            // Check whether the variable is (at least partially)
            // initialized at the exit of this predecessor. If so, we
            // want to enqueue it on our list. If not, go check the
            // next block.
            //
            // Note that we only need to check whether `live_local`
            // became de-initialized at basic block boundaries. If it
            // were to become de-initialized within the block, that
            // would have been a "use-live" transition in the earlier
            // loop, and we'd have returned already.
            //
            // NB. It's possible that the pred-block ends in a call
            // which stores to the variable; in that case, the
            // variable may be uninitialized "at exit" because this
            // call only considers the *unconditional effects* of the
            // terminator. *But*, in that case, the terminator is also
            // a *definition* of the variable, in which case we want
            // to stop the search anyhow. (But see Note 1 below.)
            if !self.initialized_at_exit(pred_block, mpi) {
                debug!("compute_drop_live_points_for_block: not initialized");
                continue;
            }

            let pred_term_loc = self.body.terminator_loc(pred_block);
            let pred_term_point = self.location_map.point_from_location(pred_term_loc);

            // If the terminator of this predecessor either *assigns*
            // our value or is a "normal use", then stop.
            if self.defs.contains(pred_term_point) {
                debug!("compute_drop_live_points_for_block: defined at {:?}", pred_term_loc);
                continue;
            }

            if self.use_live_at.contains(pred_term_point) {
                debug!("compute_drop_live_points_for_block: use-live at {:?}", pred_term_loc);
                continue;
            }

            // Otherwise, we are drop-live on entry to the terminator,
            // so walk it.
            if self.drop_live_at.insert(pred_term_point) {
                debug!("compute_drop_live_points_for_block: pushed to stack");
                self.stack.push(pred_term_point);
            }
        }

        // Note 1. There is a weird scenario that you might imagine
        // being problematic here, but which actually cannot happen.
        // The problem would be if we had a variable that *is* initialized
        // (but dead) on entry to the terminator, and where the current value
        // will be dropped in the case of unwind. In that case, we ought to
        // consider `X` to be drop-live in between the last use and call.
        // Here is the example:
        //
        // ```
        // BB0 {
        //   X = ...
        //   use(X); // last use
        //   ...     // <-- X ought to be drop-live here
        //   X = call() goto BB1 unwind BB2
        // }
        //
        // BB1 {
        //   DROP(X)
        // }
        //
        // BB2 {
        //   DROP(X)
        // }
        // ```
        //
        // However, the current code would, when walking back from BB2,
        // simply stop and never explore BB0. This seems bad! But it turns
        // out this code is flawed anyway -- note that the existing value of
        // `X` would leak in the case where unwinding did *not* occur.
        //
        // What we *actually* generate is a store to a temporary
        // for the call (`TMP = call()...`) and then a
        // `Drop(X)` followed by `X = TMP`  to swap that with `X`.
    }

    /// Returns `true` if the local variable (or some part of it) is initialized
    /// at the location defined by `init_at_location`.
    fn initialized_at(
        &mut self,
        block: BasicBlock,
        mpi: MovePathIndex,
        init_at_location: InitAtLocation,
    ) -> bool {
        // Computes the `MaybeInitializedPlaces` dataflow analysis if it hasn't been done already.
        //
        // In practice, the results of this dataflow analysis are rarely needed but can be expensive to
        // compute on big functions, so we compute them lazily as a fast path when:
        // - there are relevant live locals
        // - there are drop points for these relevant live locals.
        let flow_inits = self.flow_inits.get_or_insert_with(|| {
            let tcx = self.infcx.tcx;
            let body = self.body;
            // FIXME: reduce the `MaybeInitializedPlaces` domain to the useful `MovePath`s.
            //
            // This dataflow analysis computes maybe-initializedness of all move paths, which
            // explains why it can be expensive on big functions. But this data is only used in
            // drop-liveness. Therefore, most of the move paths computed here are ultimately unused,
            // even if the results are computed lazily and "no relevant live locals with drop
            // points" is the common case.
            //
            // So we only need the ones for 1) relevant live locals 2) that have drop points. That's
            // a much, much smaller domain: in our benchmarks, when it's not zero (the most likely
            // case), there are a few dozens compared to e.g. thousands or tens of thousands of
            // locals and move paths.
            let _timer = tcx.prof.generic_activity("borrowck_dataflow_maybe_inits");
            let flow_inits = MaybeInitializedPlaces::new(tcx, body, self.move_data)
                .iterate_to_fixpoint(tcx, body, Some("borrowck"))
                .into_results_cursor(body);
            flow_inits
        });
        let states = match init_at_location {
            InitAtLocation::Terminator => &mut self.term_states,
            InitAtLocation::Exit => &mut self.exit_states,
        };
        let state = states.get_or_insert_with(block, || {
            let terminator_location = self.body.terminator_loc(block);
            match init_at_location {
                InitAtLocation::Terminator => {
                    flow_inits.seek_before_primary_effect(terminator_location)
                }
                InitAtLocation::Exit => flow_inits.seek_after_primary_effect(terminator_location),
            }
            flow_inits.get().clone()
        });
        if state.contains(mpi) {
            return true;
        }

        let move_paths = &flow_inits.analysis().move_data().move_paths;
        move_paths[mpi].find_descendant(move_paths, |mpi| state.contains(mpi)).is_some()
    }

    /// Returns `true` if the local variable (or some part of it) is initialized in
    /// the terminator of `block`. We need to check this to determine if a
    /// DROP of some local variable will have an effect -- note that
    /// drops, as they may unwind, are always terminators.
    fn initialized_at_terminator(&mut self, block: BasicBlock, mpi: MovePathIndex) -> bool {
        self.initialized_at(block, mpi, InitAtLocation::Terminator)
    }

    /// Returns `true` if the path `mpi` (or some part of it) is initialized at
    /// the exit of `block`.
    ///
    /// **Warning:** Does not account for the result of `Call`
    /// instructions.
    fn initialized_at_exit(&mut self, block: BasicBlock, mpi: MovePathIndex) -> bool {
        self.initialized_at(block, mpi, InitAtLocation::Exit)
    }
}

/// Contains the results of computing dropck for a local. Namely, this includes
/// the dropped types, and overflows found, and the region constraints that must
/// hold at drop.
struct DropData<'tcx> {
    dropck_result: DropckOutlivesResult<'tcx>,
    region_constraint_data: Option<&'tcx QueryRegionConstraints<'tcx>>,
}

/// Computes the `DropData` for a given type, caching the result.
/// This also reports the overflow errors from the computation, if any.
fn dropck_local<'tcx, 'd>(
    infcx: &BorrowckInferCtxt<'tcx>,
    drop_data: &'d mut FxIndexMap<Ty<'tcx>, DropData<'tcx>>,
    local_ty: Ty<'tcx>,
    local_span: Span,
) -> &'d DropData<'tcx> {
    let compute_drop_data = || {
        let goal = DropckOutlives { dropped_ty: local_ty };
        match infcx.fully_perform(goal, DUMMY_SP) {
            Ok(TypeOpOutput { output, constraints, .. }) => {
                DropData { dropck_result: output, region_constraint_data: constraints }
            }
            Err(ErrorGuaranteed { .. }) => {
                // We don't run dropck on HIR, and dropck looks inside fields of
                // types, so there's no guarantee that it succeeds. We also
                // can't rely on the `ErrorGuaranteed` from `fully_perform` here
                // because it comes from delay_span_bug.
                //
                // Do this inside of a probe because we don't particularly care (or want)
                // any region side-effects of this operation in our infcx.
                infcx.probe(|_| {
                    let ocx = ObligationCtxt::new_with_diagnostics(infcx);
                    let errors = match dropck_outlives::compute_dropck_outlives_with_errors(
                        &ocx,
                        infcx.param_env.and(goal),
                        local_span,
                    ) {
                        Ok(_) => ocx.evaluate_obligations_error_on_ambiguity(),
                        Err(e) => TraitErrors::HasErrors(e),
                    };

                    // Could have no errors if a type lowering error, say, caused the query
                    // to fail.
                    if let TraitErrors::HasErrors(errors) = errors {
                        infcx.err_ctxt().report_fulfillment_errors(errors);
                    }
                });
                DropData { dropck_result: Default::default(), region_constraint_data: None }
            }
        }
    };

    let drop_data = drop_data.entry(local_ty).or_insert_with(compute_drop_data);
    drop_data.dropck_result.report_overflows(infcx.tcx, local_span, local_ty);
    drop_data
}
