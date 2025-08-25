use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_index::bit_set::DenseBitSet;
use rustc_index::interval::IntervalSet;
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::outlives::for_liveness;
use rustc_middle::mir::{BasicBlock, Body, ConstraintCategory, HasLocalDecls, Local, Location};
use rustc_middle::traits::query::DropckOutlivesResult;
use rustc_middle::ty::relate::Relate;
use rustc_middle::ty::{Ty, TyCtxt, TypeVisitable, TypeVisitableExt};
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{HasMoveData, MoveData, MovePathIndex};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_trait_selection::traits::query::dropck_outlives;
use rustc_trait_selection::traits::query::type_op::{DropckOutlives, TypeOp, TypeOpOutput};
use tracing::debug;

use crate::polonius;
use crate::region_infer::values;
use crate::type_check::liveness::local_use_map::LocalUseMap;
use crate::type_check::{NormalizeLocation, TypeChecker};

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
    relevant_live_locals: Vec<Local>,
    boring_locals: Vec<Local>,
) {
    let local_use_map = &LocalUseMap::build(&relevant_live_locals, location_map, typeck.body);
    let cx = LivenessContext {
        typeck,
        flow_inits: None,
        location_map,
        local_use_map,
        move_data,
        drop_data: FxIndexMap::default(),
    };

    let mut results = LivenessResults::new(cx);

    results.add_extra_drop_facts(&relevant_live_locals);

    results.compute_for_all_locals(relevant_live_locals);

    results.dropck_boring_locals(boring_locals);
}

/// Contextual state for the type-liveness coroutine.
struct LivenessContext<'a, 'typeck, 'tcx> {
    /// Current type-checker, giving us our inference context etc.
    ///
    /// This also stores the body we're currently analyzing.
    typeck: &'a mut TypeChecker<'typeck, 'tcx>,

    /// Defines the `PointIndex` mapping
    location_map: &'a DenseLocationMap,

    /// Mapping to/from the various indices used for initialization tracking.
    move_data: &'a MoveData<'tcx>,

    /// Cache for the results of `dropck_outlives` query.
    drop_data: FxIndexMap<Ty<'tcx>, DropData<'tcx>>,

    /// Results of dataflow tracking which variables (and paths) have been
    /// initialized. Computed lazily when needed by drop-liveness.
    flow_inits: Option<ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>>>,

    /// Index indicating where each variable is assigned, used, or
    /// dropped.
    local_use_map: &'a LocalUseMap,
}

struct DropData<'tcx> {
    dropck_result: DropckOutlivesResult<'tcx>,
    region_constraint_data: Option<&'tcx QueryRegionConstraints<'tcx>>,
}

struct LivenessResults<'a, 'typeck, 'tcx> {
    cx: LivenessContext<'a, 'typeck, 'tcx>,

    /// Set of points that define the current local.
    defs: DenseBitSet<PointIndex>,

    /// Points where the current variable is "use live" -- meaning
    /// that there is a future "full use" that may use its value.
    use_live_at: IntervalSet<PointIndex>,

    /// Points where the current variable is "drop live" -- meaning
    /// that there is no future "full use" that may use its value, but
    /// there is a future drop.
    drop_live_at: IntervalSet<PointIndex>,

    /// Locations where drops may occur.
    drop_locations: Vec<Location>,

    /// Stack used when doing (reverse) DFS.
    stack: Vec<PointIndex>,
}

impl<'a, 'typeck, 'tcx> LivenessResults<'a, 'typeck, 'tcx> {
    fn new(cx: LivenessContext<'a, 'typeck, 'tcx>) -> Self {
        let num_points = cx.location_map.num_points();
        LivenessResults {
            cx,
            defs: DenseBitSet::new_empty(num_points),
            use_live_at: IntervalSet::new(num_points),
            drop_live_at: IntervalSet::new(num_points),
            drop_locations: vec![],
            stack: vec![],
        }
    }

    fn compute_for_all_locals(&mut self, relevant_live_locals: Vec<Local>) {
        for local in relevant_live_locals {
            self.reset_local_state();
            self.add_defs_for(local);
            self.compute_use_live_points_for(local);
            self.compute_drop_live_points_for(local);

            let local_ty = self.cx.body().local_decls[local].ty;

            if !self.use_live_at.is_empty() {
                self.cx.add_use_live_facts_for(local_ty, &self.use_live_at);
            }

            if !self.drop_live_at.is_empty() {
                self.cx.add_drop_live_facts_for(
                    local,
                    local_ty,
                    &self.drop_locations,
                    &self.drop_live_at,
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
    fn dropck_boring_locals(&mut self, boring_locals: Vec<Local>) {
        for local in boring_locals {
            let local_ty = self.cx.body().local_decls[local].ty;
            let local_span = self.cx.body().local_decls[local].source_info.span;
            let drop_data = self.cx.drop_data.entry(local_ty).or_insert_with({
                let typeck = &self.cx.typeck;
                move || LivenessContext::compute_drop_data(typeck, local_ty, local_span)
            });

            drop_data.dropck_result.report_overflows(
                self.cx.typeck.infcx.tcx,
                self.cx.typeck.body.local_decls[local].source_info.span,
                local_ty,
            );
        }
    }

    /// Add extra drop facts needed for Polonius.
    ///
    /// Add facts for all locals with free regions, since regions may outlive
    /// the function body only at certain nodes in the CFG.
    fn add_extra_drop_facts(&mut self, relevant_live_locals: &[Local]) {
        // This collect is more necessary than immediately apparent
        // because these facts go into `add_drop_live_facts_for()`,
        // which also writes to `polonius_facts`, and so this is genuinely
        // a simultaneous overlapping mutable borrow.
        // FIXME for future hackers: investigate whether this is
        // actually necessary; these facts come from Polonius
        // and probably maybe plausibly does not need to go back in.
        // It may be necessary to just pick out the parts of
        // `add_drop_live_facts_for()` that make sense.
        let Some(facts) = self.cx.typeck.polonius_facts.as_ref() else { return };
        let facts_to_add: Vec<_> = {
            let relevant_live_locals: FxIndexSet<_> =
                relevant_live_locals.iter().copied().collect();

            facts
                .var_dropped_at
                .iter()
                .filter_map(|&(local, location_index)| {
                    let local_ty = self.cx.body().local_decls[local].ty;
                    if relevant_live_locals.contains(&local) || !local_ty.has_free_regions() {
                        return None;
                    }

                    let location = self.cx.typeck.location_table.to_location(location_index);
                    Some((local, local_ty, location))
                })
                .collect()
        };

        let live_at = IntervalSet::new(self.cx.location_map.num_points());
        for (local, local_ty, location) in facts_to_add {
            self.cx.add_drop_live_facts_for(local, local_ty, &[location], &live_at);
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
        for def in self.cx.local_use_map.defs(local) {
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

        self.stack.extend(self.cx.local_use_map.uses(local));
        while let Some(p) = self.stack.pop() {
            // We are live in this block from the closest to us of:
            //
            //  * Inclusively, the block start
            //  * Exclusively, the previous definition (if it's in this block)
            //  * Exclusively, the previous live_at setting (an optimization)
            let block_start = self.cx.location_map.to_block_start(p);
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

                let block = self.cx.location_map.to_location(block_start).block;
                self.stack.extend(
                    self.cx.body().basic_blocks.predecessors()[block]
                        .iter()
                        .map(|&pred_bb| self.cx.body().terminator_loc(pred_bb))
                        .map(|pred_loc| self.cx.location_map.point_from_location(pred_loc)),
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

        let Some(mpi) = self.cx.move_data.rev_lookup.find_local(local) else { return };
        debug!("compute_drop_live_points_for: mpi = {:?}", mpi);

        // Find the drops where `local` is initialized.
        for drop_point in self.cx.local_use_map.drops(local) {
            let location = self.cx.location_map.to_location(drop_point);
            debug_assert_eq!(self.cx.body().terminator_loc(location.block), location,);

            if self.cx.initialized_at_terminator(location.block, mpi)
                && self.drop_live_at.insert(drop_point)
            {
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
            self.cx.move_data.move_paths[mpi].place,
            self.cx.location_map.to_location(term_point),
        );

        // We are only invoked with terminators where `mpi` is
        // drop-live on entry.
        debug_assert!(self.drop_live_at.contains(term_point));

        // Otherwise, scan backwards through the statements in the
        // block. One of them may be either a definition or use
        // live point.
        let term_location = self.cx.location_map.to_location(term_point);
        debug_assert_eq!(self.cx.body().terminator_loc(term_location.block), term_location,);
        let block = term_location.block;
        let entry_point = self.cx.location_map.entry_point(term_location.block);
        for p in (entry_point..term_point).rev() {
            debug!(
                "compute_drop_live_points_for_block: p = {:?}",
                self.cx.location_map.to_location(p)
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

        let body = self.cx.typeck.body;
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
            if !self.cx.initialized_at_exit(pred_block, mpi) {
                debug!("compute_drop_live_points_for_block: not initialized");
                continue;
            }

            let pred_term_loc = self.cx.body().terminator_loc(pred_block);
            let pred_term_point = self.cx.location_map.point_from_location(pred_term_loc);

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
}

impl<'a, 'typeck, 'tcx> LivenessContext<'a, 'typeck, 'tcx> {
    /// Computes the `MaybeInitializedPlaces` dataflow analysis if it hasn't been done already.
    ///
    /// In practice, the results of this dataflow analysis are rarely needed but can be expensive to
    /// compute on big functions, so we compute them lazily as a fast path when:
    /// - there are relevant live locals
    /// - there are drop points for these relevant live locals.
    ///
    /// This happens as part of the drop-liveness computation: it's the only place checking for
    /// maybe-initializedness of `MovePathIndex`es.
    fn flow_inits(&mut self) -> &mut ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>> {
        self.flow_inits.get_or_insert_with(|| {
            let tcx = self.typeck.tcx();
            let body = self.typeck.body;
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
            let flow_inits = MaybeInitializedPlaces::new(tcx, body, self.move_data)
                .iterate_to_fixpoint(tcx, body, Some("borrowck"))
                .into_results_cursor(body);
            flow_inits
        })
    }
}

impl<'tcx> LivenessContext<'_, '_, 'tcx> {
    fn body(&self) -> &Body<'tcx> {
        self.typeck.body
    }

    /// Returns `true` if the local variable (or some part of it) is initialized at the current
    /// cursor position. Callers should call one of the `seek` methods immediately before to point
    /// the cursor to the desired location.
    fn initialized_at_curr_loc(&mut self, mpi: MovePathIndex) -> bool {
        let flow_inits = self.flow_inits();
        let state = flow_inits.get();
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
        let terminator_location = self.body().terminator_loc(block);
        self.flow_inits().seek_before_primary_effect(terminator_location);
        self.initialized_at_curr_loc(mpi)
    }

    /// Returns `true` if the path `mpi` (or some part of it) is initialized at
    /// the exit of `block`.
    ///
    /// **Warning:** Does not account for the result of `Call`
    /// instructions.
    fn initialized_at_exit(&mut self, block: BasicBlock, mpi: MovePathIndex) -> bool {
        let terminator_location = self.body().terminator_loc(block);
        self.flow_inits().seek_after_primary_effect(terminator_location);
        self.initialized_at_curr_loc(mpi)
    }

    /// Stores the result that all regions in `value` are live for the
    /// points `live_at`.
    fn add_use_live_facts_for(&mut self, value: Ty<'tcx>, live_at: &IntervalSet<PointIndex>) {
        debug!("add_use_live_facts_for(value={:?})", value);
        Self::make_all_regions_live(self.location_map, self.typeck, value, live_at);
    }

    /// Some variable with type `live_ty` is "drop live" at `location`
    /// -- i.e., it may be dropped later. This means that *some* of
    /// the regions in its type must be live at `location`. The
    /// precise set will depend on the dropck constraints, and in
    /// particular this takes `#[may_dangle]` into account.
    fn add_drop_live_facts_for(
        &mut self,
        dropped_local: Local,
        dropped_ty: Ty<'tcx>,
        drop_locations: &[Location],
        live_at: &IntervalSet<PointIndex>,
    ) {
        debug!(
            "add_drop_live_constraint(\
             dropped_local={:?}, \
             dropped_ty={:?}, \
             drop_locations={:?}, \
             live_at={:?})",
            dropped_local,
            dropped_ty,
            drop_locations,
            values::pretty_print_points(self.location_map, live_at.iter()),
        );

        let local_span = self.body().local_decls()[dropped_local].source_info.span;
        let drop_data = self.drop_data.entry(dropped_ty).or_insert_with({
            let typeck = &self.typeck;
            move || Self::compute_drop_data(typeck, dropped_ty, local_span)
        });

        if let Some(data) = &drop_data.region_constraint_data {
            for &drop_location in drop_locations {
                self.typeck.push_region_constraints(
                    drop_location.to_locations(),
                    ConstraintCategory::Boring,
                    data,
                );
            }
        }

        drop_data.dropck_result.report_overflows(
            self.typeck.infcx.tcx,
            self.typeck.body.source_info(*drop_locations.first().unwrap()).span,
            dropped_ty,
        );

        // All things in the `outlives` array may be touched by
        // the destructor and must be live at this point.
        for &kind in &drop_data.dropck_result.kinds {
            Self::make_all_regions_live(self.location_map, self.typeck, kind, live_at);
            polonius::legacy::emit_drop_facts(
                self.typeck.tcx(),
                dropped_local,
                &kind,
                self.typeck.universal_regions,
                self.typeck.polonius_facts,
            );
        }
    }

    fn make_all_regions_live(
        location_map: &DenseLocationMap,
        typeck: &mut TypeChecker<'_, 'tcx>,
        value: impl TypeVisitable<TyCtxt<'tcx>> + Relate<TyCtxt<'tcx>>,
        live_at: &IntervalSet<PointIndex>,
    ) {
        debug!("make_all_regions_live(value={:?})", value);
        debug!(
            "make_all_regions_live: live_at={}",
            values::pretty_print_points(location_map, live_at.iter()),
        );

        value.visit_with(&mut for_liveness::FreeRegionsVisitor {
            tcx: typeck.tcx(),
            param_env: typeck.infcx.param_env,
            op: |r| {
                let live_region_vid = typeck.universal_regions.to_region_vid(r);

                typeck.constraints.liveness_constraints.add_points(live_region_vid, live_at);
            },
        });

        // When using `-Zpolonius=next`, we record the variance of each live region.
        if let Some(polonius_liveness) = typeck.polonius_liveness.as_mut() {
            polonius_liveness.record_live_region_variance(
                typeck.infcx.tcx,
                typeck.universal_regions,
                value,
            );
        }
    }

    fn compute_drop_data(
        typeck: &TypeChecker<'_, 'tcx>,
        dropped_ty: Ty<'tcx>,
        span: Span,
    ) -> DropData<'tcx> {
        debug!("compute_drop_data(dropped_ty={:?})", dropped_ty);

        let op = typeck.infcx.param_env.and(DropckOutlives { dropped_ty });

        match op.fully_perform(typeck.infcx, typeck.root_cx.root_def_id(), DUMMY_SP) {
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
                typeck.infcx.probe(|_| {
                    let ocx = ObligationCtxt::new_with_diagnostics(&typeck.infcx);
                    let errors = match dropck_outlives::compute_dropck_outlives_with_errors(
                        &ocx, op, span,
                    ) {
                        Ok(_) => ocx.select_all_or_error(),
                        Err(e) => e,
                    };

                    // Could have no errors if a type lowering error, say, caused the query
                    // to fail.
                    if !errors.is_empty() {
                        typeck.infcx.err_ctxt().report_fulfillment_errors(errors);
                    }
                });
                DropData { dropck_result: Default::default(), region_constraint_data: None }
            }
        }
    }
}
