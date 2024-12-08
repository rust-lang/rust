use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_index::bit_set::BitSet;
use rustc_index::interval::IntervalSet;
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::outlives::for_liveness;
use rustc_middle::mir::{BasicBlock, Body, ConstraintCategory, Local, Location};
use rustc_middle::traits::query::DropckOutlivesResult;
use rustc_middle::ty::{Ty, TyCtxt, TypeVisitable, TypeVisitableExt};
use rustc_mir_dataflow::ResultsCursor;
use rustc_mir_dataflow::impls::MaybeInitializedPlaces;
use rustc_mir_dataflow::move_paths::{HasMoveData, MoveData, MovePathIndex};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits::query::type_op::{DropckOutlives, TypeOp, TypeOpOutput};
use tracing::debug;

use crate::location::RichLocation;
use crate::region_infer::values::{self, LiveLoans};
use crate::type_check::liveness::local_use_map::LocalUseMap;
use crate::type_check::liveness::polonius;
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
pub(super) fn trace<'a, 'tcx>(
    typeck: &mut TypeChecker<'_, 'tcx>,
    body: &Body<'tcx>,
    elements: &DenseLocationMap,
    flow_inits: ResultsCursor<'a, 'tcx, MaybeInitializedPlaces<'a, 'tcx>>,
    move_data: &MoveData<'tcx>,
    relevant_live_locals: Vec<Local>,
    boring_locals: Vec<Local>,
) {
    let local_use_map = &LocalUseMap::build(&relevant_live_locals, elements, body);

    // When using `-Zpolonius=next`, compute the set of loans that can reach a given region.
    if typeck.tcx().sess.opts.unstable_opts.polonius.is_next_enabled() {
        let borrow_set = &typeck.borrow_set;
        let mut live_loans = LiveLoans::new(borrow_set.len());
        let outlives_constraints = &typeck.constraints.outlives_constraints;
        let graph = outlives_constraints.graph(typeck.infcx.num_region_vars());
        let region_graph =
            graph.region_graph(outlives_constraints, typeck.universal_regions.fr_static);

        // Traverse each issuing region's constraints, and record the loan as flowing into the
        // outlived region.
        for (loan, issuing_region_data) in borrow_set.iter_enumerated() {
            for succ in rustc_data_structures::graph::depth_first_search(
                &region_graph,
                issuing_region_data.region,
            ) {
                // We don't need to mention that a loan flows into its issuing region.
                if succ == issuing_region_data.region {
                    continue;
                }

                live_loans.inflowing_loans.insert(succ, loan);
            }
        }

        // Store the inflowing loans in the liveness constraints: they will be used to compute live
        // loans when liveness data is recorded there.
        typeck.constraints.liveness_constraints.loans = Some(live_loans);
    };

    let cx = LivenessContext {
        typeck,
        body,
        flow_inits,
        elements,
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
struct LivenessContext<'a, 'typeck, 'b, 'tcx> {
    /// Current type-checker, giving us our inference context etc.
    typeck: &'a mut TypeChecker<'typeck, 'tcx>,

    /// Defines the `PointIndex` mapping
    elements: &'a DenseLocationMap,

    /// MIR we are analyzing.
    body: &'a Body<'tcx>,

    /// Mapping to/from the various indices used for initialization tracking.
    move_data: &'a MoveData<'tcx>,

    /// Cache for the results of `dropck_outlives` query.
    drop_data: FxIndexMap<Ty<'tcx>, DropData<'tcx>>,

    /// Results of dataflow tracking which variables (and paths) have been
    /// initialized.
    flow_inits: ResultsCursor<'b, 'tcx, MaybeInitializedPlaces<'b, 'tcx>>,

    /// Index indicating where each variable is assigned, used, or
    /// dropped.
    local_use_map: &'a LocalUseMap,
}

struct DropData<'tcx> {
    dropck_result: DropckOutlivesResult<'tcx>,
    region_constraint_data: Option<&'tcx QueryRegionConstraints<'tcx>>,
}

struct LivenessResults<'a, 'typeck, 'b, 'tcx> {
    cx: LivenessContext<'a, 'typeck, 'b, 'tcx>,

    /// Set of points that define the current local.
    defs: BitSet<PointIndex>,

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

impl<'a, 'typeck, 'b, 'tcx> LivenessResults<'a, 'typeck, 'b, 'tcx> {
    fn new(cx: LivenessContext<'a, 'typeck, 'b, 'tcx>) -> Self {
        let num_points = cx.elements.num_points();
        LivenessResults {
            cx,
            defs: BitSet::new_empty(num_points),
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

            let local_ty = self.cx.body.local_decls[local].ty;

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
            let local_ty = self.cx.body.local_decls[local].ty;
            let drop_data = self.cx.drop_data.entry(local_ty).or_insert_with({
                let typeck = &self.cx.typeck;
                move || LivenessContext::compute_drop_data(typeck, local_ty)
            });

            drop_data.dropck_result.report_overflows(
                self.cx.typeck.infcx.tcx,
                self.cx.body.local_decls[local].source_info.span,
                local_ty,
            );
        }
    }

    /// Add extra drop facts needed for Polonius.
    ///
    /// Add facts for all locals with free regions, since regions may outlive
    /// the function body only at certain nodes in the CFG.
    fn add_extra_drop_facts(&mut self, relevant_live_locals: &[Local]) -> Option<()> {
        // This collect is more necessary than immediately apparent
        // because these facts go into `add_drop_live_facts_for()`,
        // which also writes to `all_facts`, and so this is genuinely
        // a simultaneous overlapping mutable borrow.
        // FIXME for future hackers: investigate whether this is
        // actually necessary; these facts come from Polonius
        // and probably maybe plausibly does not need to go back in.
        // It may be necessary to just pick out the parts of
        // `add_drop_live_facts_for()` that make sense.
        let facts_to_add: Vec<_> = {
            let drop_used = &self.cx.typeck.all_facts.as_ref()?.var_dropped_at;

            let relevant_live_locals: FxIndexSet<_> =
                relevant_live_locals.iter().copied().collect();

            drop_used
                .iter()
                .filter_map(|(local, location_index)| {
                    let local_ty = self.cx.body.local_decls[*local].ty;
                    if relevant_live_locals.contains(local) || !local_ty.has_free_regions() {
                        return None;
                    }

                    let location = match self.cx.typeck.location_table.to_location(*location_index)
                    {
                        RichLocation::Start(l) => l,
                        RichLocation::Mid(l) => l,
                    };

                    Some((*local, local_ty, location))
                })
                .collect()
        };

        // FIXME: these locations seem to have a special meaning (e.g. everywhere, at the end,
        // ...), but I don't know which one. Please help me rename it to something descriptive!
        // Also, if this IntervalSet is used in many places, it maybe should have a newtype'd
        // name with a description of what it means for future mortals passing by.
        let locations = IntervalSet::new(self.cx.elements.num_points());

        for (local, local_ty, location) in facts_to_add {
            self.cx.add_drop_live_facts_for(local, local_ty, &[location], &locations);
        }
        Some(())
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
            let block_start = self.cx.elements.to_block_start(p);
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

                let block = self.cx.elements.to_location(block_start).block;
                self.stack.extend(
                    self.cx.body.basic_blocks.predecessors()[block]
                        .iter()
                        .map(|&pred_bb| self.cx.body.terminator_loc(pred_bb))
                        .map(|pred_loc| self.cx.elements.point_from_location(pred_loc)),
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
            let location = self.cx.elements.to_location(drop_point);
            debug_assert_eq!(self.cx.body.terminator_loc(location.block), location,);

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
            self.cx.elements.to_location(term_point),
        );

        // We are only invoked with terminators where `mpi` is
        // drop-live on entry.
        debug_assert!(self.drop_live_at.contains(term_point));

        // Otherwise, scan backwards through the statements in the
        // block. One of them may be either a definition or use
        // live point.
        let term_location = self.cx.elements.to_location(term_point);
        debug_assert_eq!(self.cx.body.terminator_loc(term_location.block), term_location,);
        let block = term_location.block;
        let entry_point = self.cx.elements.entry_point(term_location.block);
        for p in (entry_point..term_point).rev() {
            debug!("compute_drop_live_points_for_block: p = {:?}", self.cx.elements.to_location(p));

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

        let body = self.cx.body;
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

            let pred_term_loc = self.cx.body.terminator_loc(pred_block);
            let pred_term_point = self.cx.elements.point_from_location(pred_term_loc);

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

impl<'tcx> LivenessContext<'_, '_, '_, 'tcx> {
    /// Returns `true` if the local variable (or some part of it) is initialized at the current
    /// cursor position. Callers should call one of the `seek` methods immediately before to point
    /// the cursor to the desired location.
    fn initialized_at_curr_loc(&self, mpi: MovePathIndex) -> bool {
        let state = self.flow_inits.get();
        if state.contains(mpi) {
            return true;
        }

        let move_paths = &self.flow_inits.analysis().move_data().move_paths;
        move_paths[mpi].find_descendant(move_paths, |mpi| state.contains(mpi)).is_some()
    }

    /// Returns `true` if the local variable (or some part of it) is initialized in
    /// the terminator of `block`. We need to check this to determine if a
    /// DROP of some local variable will have an effect -- note that
    /// drops, as they may unwind, are always terminators.
    fn initialized_at_terminator(&mut self, block: BasicBlock, mpi: MovePathIndex) -> bool {
        self.flow_inits.seek_before_primary_effect(self.body.terminator_loc(block));
        self.initialized_at_curr_loc(mpi)
    }

    /// Returns `true` if the path `mpi` (or some part of it) is initialized at
    /// the exit of `block`.
    ///
    /// **Warning:** Does not account for the result of `Call`
    /// instructions.
    fn initialized_at_exit(&mut self, block: BasicBlock, mpi: MovePathIndex) -> bool {
        self.flow_inits.seek_after_primary_effect(self.body.terminator_loc(block));
        self.initialized_at_curr_loc(mpi)
    }

    /// Stores the result that all regions in `value` are live for the
    /// points `live_at`.
    fn add_use_live_facts_for(
        &mut self,
        value: impl TypeVisitable<TyCtxt<'tcx>>,
        live_at: &IntervalSet<PointIndex>,
    ) {
        debug!("add_use_live_facts_for(value={:?})", value);
        Self::make_all_regions_live(self.elements, self.typeck, value, live_at);
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
            values::pretty_print_points(self.elements, live_at.iter()),
        );

        let drop_data = self.drop_data.entry(dropped_ty).or_insert_with({
            let typeck = &self.typeck;
            move || Self::compute_drop_data(typeck, dropped_ty)
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
            self.body.source_info(*drop_locations.first().unwrap()).span,
            dropped_ty,
        );

        // All things in the `outlives` array may be touched by
        // the destructor and must be live at this point.
        for &kind in &drop_data.dropck_result.kinds {
            Self::make_all_regions_live(self.elements, self.typeck, kind, live_at);
            polonius::add_drop_of_var_derefs_origin(self.typeck, dropped_local, &kind);
        }
    }

    fn make_all_regions_live(
        elements: &DenseLocationMap,
        typeck: &mut TypeChecker<'_, 'tcx>,
        value: impl TypeVisitable<TyCtxt<'tcx>>,
        live_at: &IntervalSet<PointIndex>,
    ) {
        debug!("make_all_regions_live(value={:?})", value);
        debug!(
            "make_all_regions_live: live_at={}",
            values::pretty_print_points(elements, live_at.iter()),
        );

        value.visit_with(&mut for_liveness::FreeRegionsVisitor {
            tcx: typeck.tcx(),
            param_env: typeck.infcx.param_env,
            op: |r| {
                let live_region_vid = typeck.universal_regions.to_region_vid(r);

                typeck.constraints.liveness_constraints.add_points(live_region_vid, live_at);
            },
        });
    }

    fn compute_drop_data(typeck: &TypeChecker<'_, 'tcx>, dropped_ty: Ty<'tcx>) -> DropData<'tcx> {
        debug!("compute_drop_data(dropped_ty={:?})", dropped_ty,);

        match typeck
            .infcx
            .param_env
            .and(DropckOutlives { dropped_ty })
            .fully_perform(typeck.infcx, DUMMY_SP)
        {
            Ok(TypeOpOutput { output, constraints, .. }) => {
                DropData { dropck_result: output, region_constraint_data: constraints }
            }
            Err(_) => DropData { dropck_result: Default::default(), region_constraint_data: None },
        }
    }
}
