use std::fmt;

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::graph;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{
    self, BasicBlock, Body, CallReturnPlaces, Location, Place, TerminatorEdges,
};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::fmt::DebugWithContext;
use rustc_mir_dataflow::impls::{EverInitializedPlaces, MaybeUninitializedPlaces};
use rustc_mir_dataflow::{Analysis, GenKill, JoinSemiLattice, SwitchIntEdgeEffects};
use tracing::debug;

use crate::{BorrowSet, PlaceConflictBias, PlaceExt, RegionInferenceContext, places_conflict};

// This analysis is different to most others. Its results aren't computed with
// `iterate_to_fixpoint`, but are instead composed from the results of three sub-analyses that are
// computed individually with `iterate_to_fixpoint`.
pub(crate) struct Borrowck<'a, 'tcx> {
    pub(crate) borrows: Borrows<'a, 'tcx>,
    pub(crate) uninits: MaybeUninitializedPlaces<'a, 'tcx>,
    pub(crate) ever_inits: EverInitializedPlaces<'a, 'tcx>,
}

impl<'a, 'tcx> Analysis<'tcx> for Borrowck<'a, 'tcx> {
    type Domain = BorrowckDomain<'a, 'tcx>;

    const NAME: &'static str = "borrowck";

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        BorrowckDomain {
            borrows: self.borrows.bottom_value(body),
            uninits: self.uninits.bottom_value(body),
            ever_inits: self.ever_inits.bottom_value(body),
        }
    }

    fn initialize_start_block(&self, _body: &mir::Body<'tcx>, _state: &mut Self::Domain) {
        // This is only reachable from `iterate_to_fixpoint`, which this analysis doesn't use.
        unreachable!();
    }

    fn apply_before_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.borrows.apply_before_statement_effect(&mut state.borrows, stmt, loc);
        self.uninits.apply_before_statement_effect(&mut state.uninits, stmt, loc);
        self.ever_inits.apply_before_statement_effect(&mut state.ever_inits, stmt, loc);
    }

    fn apply_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.borrows.apply_statement_effect(&mut state.borrows, stmt, loc);
        self.uninits.apply_statement_effect(&mut state.uninits, stmt, loc);
        self.ever_inits.apply_statement_effect(&mut state.ever_inits, stmt, loc);
    }

    fn apply_before_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.borrows.apply_before_terminator_effect(&mut state.borrows, term, loc);
        self.uninits.apply_before_terminator_effect(&mut state.uninits, term, loc);
        self.ever_inits.apply_before_terminator_effect(&mut state.ever_inits, term, loc);
    }

    fn apply_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        term: &'mir mir::Terminator<'tcx>,
        loc: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        self.borrows.apply_terminator_effect(&mut state.borrows, term, loc);
        self.uninits.apply_terminator_effect(&mut state.uninits, term, loc);
        self.ever_inits.apply_terminator_effect(&mut state.ever_inits, term, loc);

        // This return value doesn't matter. It's only used by `iterate_to_fixpoint`, which this
        // analysis doesn't use.
        TerminatorEdges::None
    }

    fn apply_call_return_effect(
        &mut self,
        _state: &mut Self::Domain,
        _block: BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        // This is only reachable from `iterate_to_fixpoint`, which this analysis doesn't use.
        unreachable!();
    }

    fn apply_switch_int_edge_effects(
        &mut self,
        _block: BasicBlock,
        _discr: &mir::Operand<'tcx>,
        _apply_edge_effects: &mut impl SwitchIntEdgeEffects<Self::Domain>,
    ) {
        // This is only reachable from `iterate_to_fixpoint`, which this analysis doesn't use.
        unreachable!();
    }
}

impl JoinSemiLattice for BorrowckDomain<'_, '_> {
    fn join(&mut self, _other: &Self) -> bool {
        // This is only reachable from `iterate_to_fixpoint`, which this analysis doesn't use.
        unreachable!();
    }
}

impl<'tcx, C> DebugWithContext<C> for BorrowckDomain<'_, 'tcx>
where
    C: rustc_mir_dataflow::move_paths::HasMoveData<'tcx>,
{
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("borrows: ")?;
        self.borrows.fmt_with(ctxt, f)?;
        f.write_str(" uninits: ")?;
        self.uninits.fmt_with(ctxt, f)?;
        f.write_str(" ever_inits: ")?;
        self.ever_inits.fmt_with(ctxt, f)?;
        Ok(())
    }

    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == old {
            return Ok(());
        }

        if self.borrows != old.borrows {
            f.write_str("borrows: ")?;
            self.borrows.fmt_diff_with(&old.borrows, ctxt, f)?;
            f.write_str("\n")?;
        }

        if self.uninits != old.uninits {
            f.write_str("uninits: ")?;
            self.uninits.fmt_diff_with(&old.uninits, ctxt, f)?;
            f.write_str("\n")?;
        }

        if self.ever_inits != old.ever_inits {
            f.write_str("ever_inits: ")?;
            self.ever_inits.fmt_diff_with(&old.ever_inits, ctxt, f)?;
            f.write_str("\n")?;
        }

        Ok(())
    }
}

/// The transient state of the dataflow analyses used by the borrow checker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BorrowckDomain<'a, 'tcx> {
    pub(crate) borrows: <Borrows<'a, 'tcx> as Analysis<'tcx>>::Domain,
    pub(crate) uninits: <MaybeUninitializedPlaces<'a, 'tcx> as Analysis<'tcx>>::Domain,
    pub(crate) ever_inits: <EverInitializedPlaces<'a, 'tcx> as Analysis<'tcx>>::Domain,
}

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "bw{}"]
    pub struct BorrowIndex {}
}

/// `Borrows` stores the data used in the analyses that track the flow
/// of borrows.
///
/// It uniquely identifies every borrow (`Rvalue::Ref`) by a
/// `BorrowIndex`, and maps each such index to a `BorrowData`
/// describing the borrow. These indexes are used for representing the
/// borrows in compact bitvectors.
pub struct Borrows<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    borrow_set: &'a BorrowSet<'tcx>,
    borrows_out_of_scope_at_location: FxIndexMap<Location, Vec<BorrowIndex>>,
}

struct OutOfScopePrecomputer<'a, 'tcx> {
    visited: BitSet<mir::BasicBlock>,
    visit_stack: Vec<mir::BasicBlock>,
    body: &'a Body<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    borrows_out_of_scope_at_location: FxIndexMap<Location, Vec<BorrowIndex>>,
}

impl<'a, 'tcx> OutOfScopePrecomputer<'a, 'tcx> {
    fn new(body: &'a Body<'tcx>, regioncx: &'a RegionInferenceContext<'tcx>) -> Self {
        OutOfScopePrecomputer {
            visited: BitSet::new_empty(body.basic_blocks.len()),
            visit_stack: vec![],
            body,
            regioncx,
            borrows_out_of_scope_at_location: FxIndexMap::default(),
        }
    }
}

impl<'tcx> OutOfScopePrecomputer<'_, 'tcx> {
    fn precompute_borrows_out_of_scope(
        &mut self,
        borrow_index: BorrowIndex,
        borrow_region: RegionVid,
        first_location: Location,
    ) {
        let first_block = first_location.block;
        let first_bb_data = &self.body.basic_blocks[first_block];

        // This is the first block, we only want to visit it from the creation of the borrow at
        // `first_location`.
        let first_lo = first_location.statement_index;
        let first_hi = first_bb_data.statements.len();

        if let Some(kill_stmt) = self.regioncx.first_non_contained_inclusive(
            borrow_region,
            first_block,
            first_lo,
            first_hi,
        ) {
            let kill_location = Location { block: first_block, statement_index: kill_stmt };
            // If region does not contain a point at the location, then add to list and skip
            // successor locations.
            debug!("borrow {:?} gets killed at {:?}", borrow_index, kill_location);
            self.borrows_out_of_scope_at_location
                .entry(kill_location)
                .or_default()
                .push(borrow_index);

            // The borrow is already dead, there is no need to visit other blocks.
            return;
        }

        // The borrow is not dead. Add successor BBs to the work list, if necessary.
        for succ_bb in first_bb_data.terminator().successors() {
            if self.visited.insert(succ_bb) {
                self.visit_stack.push(succ_bb);
            }
        }

        // We may end up visiting `first_block` again. This is not an issue: we know at this point
        // that it does not kill the borrow in the `first_lo..=first_hi` range, so checking the
        // `0..first_lo` range and the `0..first_hi` range give the same result.
        while let Some(block) = self.visit_stack.pop() {
            let bb_data = &self.body[block];
            let num_stmts = bb_data.statements.len();
            if let Some(kill_stmt) =
                self.regioncx.first_non_contained_inclusive(borrow_region, block, 0, num_stmts)
            {
                let kill_location = Location { block, statement_index: kill_stmt };
                // If region does not contain a point at the location, then add to list and skip
                // successor locations.
                debug!("borrow {:?} gets killed at {:?}", borrow_index, kill_location);
                self.borrows_out_of_scope_at_location
                    .entry(kill_location)
                    .or_default()
                    .push(borrow_index);

                // We killed the borrow, so we do not visit this block's successors.
                continue;
            }

            // Add successor BBs to the work list, if necessary.
            for succ_bb in bb_data.terminator().successors() {
                if self.visited.insert(succ_bb) {
                    self.visit_stack.push(succ_bb);
                }
            }
        }

        self.visited.clear();
    }
}

// This is `pub` because it's used by unstable external borrowck data users, see `consumers.rs`.
pub fn calculate_borrows_out_of_scope_at_location<'tcx>(
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) -> FxIndexMap<Location, Vec<BorrowIndex>> {
    let mut prec = OutOfScopePrecomputer::new(body, regioncx);
    for (borrow_index, borrow_data) in borrow_set.iter_enumerated() {
        let borrow_region = borrow_data.region;
        let location = borrow_data.reserve_location;

        prec.precompute_borrows_out_of_scope(borrow_index, borrow_region, location);
    }

    prec.borrows_out_of_scope_at_location
}

struct PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    visited: BitSet<mir::BasicBlock>,
    visit_stack: Vec<mir::BasicBlock>,
    body: &'a Body<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,

    loans_out_of_scope_at_location: FxIndexMap<Location, Vec<BorrowIndex>>,
}

impl<'a, 'tcx> PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    fn new(body: &'a Body<'tcx>, regioncx: &'a RegionInferenceContext<'tcx>) -> Self {
        Self {
            visited: BitSet::new_empty(body.basic_blocks.len()),
            visit_stack: vec![],
            body,
            regioncx,
            loans_out_of_scope_at_location: FxIndexMap::default(),
        }
    }
}

impl<'tcx> PoloniusOutOfScopePrecomputer<'_, 'tcx> {
    /// Loans are in scope while they are live: whether they are contained within any live region.
    /// In the location-insensitive analysis, a loan will be contained in a region if the issuing
    /// region can reach it in the subset graph. So this is a reachability problem.
    fn precompute_loans_out_of_scope(
        &mut self,
        loan_idx: BorrowIndex,
        issuing_region: RegionVid,
        loan_issued_at: Location,
    ) {
        let sccs = self.regioncx.constraint_sccs();
        let universal_regions = self.regioncx.universal_regions();

        // We first handle the cases where the loan doesn't go out of scope, depending on the
        // issuing region's successors.
        for successor in graph::depth_first_search(&self.regioncx.region_graph(), issuing_region) {
            // 1. Via applied member constraints
            //
            // The issuing region can flow into the choice regions, and they are either:
            // - placeholders or free regions themselves,
            // - or also transitively outlive a free region.
            //
            // That is to say, if there are applied member constraints here, the loan escapes the
            // function and cannot go out of scope. We could early return here.
            //
            // For additional insurance via fuzzing and crater, we verify that the constraint's min
            // choice indeed escapes the function. In the future, we could e.g. turn this check into
            // a debug assert and early return as an optimization.
            let scc = sccs.scc(successor);
            for constraint in self.regioncx.applied_member_constraints(scc) {
                if universal_regions.is_universal_region(constraint.min_choice) {
                    return;
                }
            }

            // 2. Via regions that are live at all points: placeholders and free regions.
            //
            // If the issuing region outlives such a region, its loan escapes the function and
            // cannot go out of scope. We can early return.
            if self.regioncx.is_region_live_at_all_points(successor) {
                return;
            }
        }

        let first_block = loan_issued_at.block;
        let first_bb_data = &self.body.basic_blocks[first_block];

        // The first block we visit is the one where the loan is issued, starting from the statement
        // where the loan is issued: at `loan_issued_at`.
        let first_lo = loan_issued_at.statement_index;
        let first_hi = first_bb_data.statements.len();

        if let Some(kill_location) =
            self.loan_kill_location(loan_idx, loan_issued_at, first_block, first_lo, first_hi)
        {
            debug!("loan {:?} gets killed at {:?}", loan_idx, kill_location);
            self.loans_out_of_scope_at_location.entry(kill_location).or_default().push(loan_idx);

            // The loan dies within the first block, we're done and can early return.
            return;
        }

        // The loan is not dead. Add successor BBs to the work list, if necessary.
        for succ_bb in first_bb_data.terminator().successors() {
            if self.visited.insert(succ_bb) {
                self.visit_stack.push(succ_bb);
            }
        }

        // We may end up visiting `first_block` again. This is not an issue: we know at this point
        // that the loan is not killed in the `first_lo..=first_hi` range, so checking the
        // `0..first_lo` range and the `0..first_hi` range gives the same result.
        while let Some(block) = self.visit_stack.pop() {
            let bb_data = &self.body[block];
            let num_stmts = bb_data.statements.len();
            if let Some(kill_location) =
                self.loan_kill_location(loan_idx, loan_issued_at, block, 0, num_stmts)
            {
                debug!("loan {:?} gets killed at {:?}", loan_idx, kill_location);
                self.loans_out_of_scope_at_location
                    .entry(kill_location)
                    .or_default()
                    .push(loan_idx);

                // The loan dies within this block, so we don't need to visit its successors.
                continue;
            }

            // Add successor BBs to the work list, if necessary.
            for succ_bb in bb_data.terminator().successors() {
                if self.visited.insert(succ_bb) {
                    self.visit_stack.push(succ_bb);
                }
            }
        }

        self.visited.clear();
        assert!(self.visit_stack.is_empty(), "visit stack should be empty");
    }

    /// Returns the lowest statement in `start..=end`, where the loan goes out of scope, if any.
    /// This is the statement where the issuing region can't reach any of the regions that are live
    /// at this point.
    fn loan_kill_location(
        &self,
        loan_idx: BorrowIndex,
        loan_issued_at: Location,
        block: BasicBlock,
        start: usize,
        end: usize,
    ) -> Option<Location> {
        for statement_index in start..=end {
            let location = Location { block, statement_index };

            // Check whether the issuing region can reach local regions that are live at this point:
            // - a loan is always live at its issuing location because it can reach the issuing
            // region, which is always live at this location.
            if location == loan_issued_at {
                continue;
            }

            // - the loan goes out of scope at `location` if it's not contained within any regions
            // live at this point.
            //
            // FIXME: if the issuing region `i` can reach a live region `r` at point `p`, and `r` is
            // live at point `q`, then it's guaranteed that `i` would reach `r` at point `q`.
            // Reachability is location-insensitive, and we could take advantage of that, by jumping
            // to a further point than just the next statement: we can jump to the furthest point
            // within the block where `r` is live.
            if self.regioncx.is_loan_live_at(loan_idx, location) {
                continue;
            }

            // No live region is reachable from the issuing region: the loan is killed at this
            // point.
            return Some(location);
        }

        None
    }
}

impl<'a, 'tcx> Borrows<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        body: &'a Body<'tcx>,
        regioncx: &RegionInferenceContext<'tcx>,
        borrow_set: &'a BorrowSet<'tcx>,
    ) -> Self {
        let mut borrows_out_of_scope_at_location =
            calculate_borrows_out_of_scope_at_location(body, regioncx, borrow_set);

        // The in-tree polonius analysis computes loans going out of scope using the set-of-loans
        // model, and makes sure they're identical to the existing computation of the set-of-points
        // model.
        if tcx.sess.opts.unstable_opts.polonius.is_next_enabled() {
            let mut polonius_prec = PoloniusOutOfScopePrecomputer::new(body, regioncx);
            for (loan_idx, loan_data) in borrow_set.iter_enumerated() {
                let issuing_region = loan_data.region;
                let loan_issued_at = loan_data.reserve_location;

                polonius_prec.precompute_loans_out_of_scope(
                    loan_idx,
                    issuing_region,
                    loan_issued_at,
                );
            }

            assert_eq!(
                borrows_out_of_scope_at_location, polonius_prec.loans_out_of_scope_at_location,
                "polonius loan scopes differ from NLL borrow scopes, for body {:?}",
                body.span,
            );

            borrows_out_of_scope_at_location = polonius_prec.loans_out_of_scope_at_location;
        }

        Borrows { tcx, body, borrow_set, borrows_out_of_scope_at_location }
    }

    /// Add all borrows to the kill set, if those borrows are out of scope at `location`.
    /// That means they went out of a nonlexical scope
    fn kill_loans_out_of_scope_at_location(
        &self,
        trans: &mut <Self as Analysis<'tcx>>::Domain,
        location: Location,
    ) {
        // NOTE: The state associated with a given `location`
        // reflects the dataflow on entry to the statement.
        // Iterate over each of the borrows that we've precomputed
        // to have went out of scope at this location and kill them.
        //
        // We are careful always to call this function *before* we
        // set up the gen-bits for the statement or
        // terminator. That way, if the effect of the statement or
        // terminator *does* introduce a new loan of the same
        // region, then setting that gen-bit will override any
        // potential kill introduced here.
        if let Some(indices) = self.borrows_out_of_scope_at_location.get(&location) {
            trans.kill_all(indices.iter().copied());
        }
    }

    /// Kill any borrows that conflict with `place`.
    fn kill_borrows_on_place(
        &self,
        trans: &mut <Self as Analysis<'tcx>>::Domain,
        place: Place<'tcx>,
    ) {
        debug!("kill_borrows_on_place: place={:?}", place);

        let other_borrows_of_local = self
            .borrow_set
            .local_map
            .get(&place.local)
            .into_iter()
            .flat_map(|bs| bs.iter())
            .copied();

        // If the borrowed place is a local with no projections, all other borrows of this
        // local must conflict. This is purely an optimization so we don't have to call
        // `places_conflict` for every borrow.
        if place.projection.is_empty() {
            if !self.body.local_decls[place.local].is_ref_to_static() {
                trans.kill_all(other_borrows_of_local);
            }
            return;
        }

        // By passing `PlaceConflictBias::NoOverlap`, we conservatively assume that any given
        // pair of array indices are not equal, so that when `places_conflict` returns true, we
        // will be assured that two places being compared definitely denotes the same sets of
        // locations.
        let definitely_conflicting_borrows = other_borrows_of_local.filter(|&i| {
            places_conflict(
                self.tcx,
                self.body,
                self.borrow_set[i].borrowed_place,
                place,
                PlaceConflictBias::NoOverlap,
            )
        });

        trans.kill_all(definitely_conflicting_borrows);
    }
}

/// Forward dataflow computation of the set of borrows that are in scope at a particular location.
/// - we gen the introduced loans
/// - we kill loans on locals going out of (regular) scope
/// - we kill the loans going out of their region's NLL scope: in NLL terms, the frontier where a
///   region stops containing the CFG points reachable from the issuing location.
/// - we also kill loans of conflicting places when overwriting a shared path: e.g. borrows of
///   `a.b.c` when `a` is overwritten.
impl<'tcx> rustc_mir_dataflow::Analysis<'tcx> for Borrows<'_, 'tcx> {
    type Domain = BitSet<BorrowIndex>;

    const NAME: &'static str = "borrows";

    fn bottom_value(&self, _: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = nothing is reserved or activated yet;
        BitSet::new_empty(self.borrow_set.len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect.
    }

    fn apply_before_statement_effect(
        &mut self,
        trans: &mut Self::Domain,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn apply_statement_effect(
        &mut self,
        trans: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        location: Location,
    ) {
        match &stmt.kind {
            mir::StatementKind::Assign(box (lhs, rhs)) => {
                if let mir::Rvalue::Ref(_, _, place) = rhs {
                    if place.ignore_borrow(
                        self.tcx,
                        self.body,
                        &self.borrow_set.locals_state_at_exit,
                    ) {
                        return;
                    }
                    let index = self.borrow_set.get_index_of(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {location:?}");
                    });

                    trans.gen_(index);
                }

                // Make sure there are no remaining borrows for variables
                // that are assigned over.
                self.kill_borrows_on_place(trans, *lhs);
            }

            mir::StatementKind::StorageDead(local) => {
                // Make sure there are no remaining borrows for locals that
                // are gone out of scope.
                self.kill_borrows_on_place(trans, Place::from(*local));
            }

            mir::StatementKind::FakeRead(..)
            | mir::StatementKind::SetDiscriminant { .. }
            | mir::StatementKind::Deinit(..)
            | mir::StatementKind::StorageLive(..)
            | mir::StatementKind::Retag { .. }
            | mir::StatementKind::PlaceMention(..)
            | mir::StatementKind::AscribeUserType(..)
            | mir::StatementKind::Coverage(..)
            | mir::StatementKind::Intrinsic(..)
            | mir::StatementKind::ConstEvalCounter
            | mir::StatementKind::BackwardIncompatibleDropHint { .. }
            | mir::StatementKind::Nop => {}
        }
    }

    fn apply_before_terminator_effect(
        &mut self,
        trans: &mut Self::Domain,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn apply_terminator_effect<'mir>(
        &mut self,
        trans: &mut Self::Domain,
        terminator: &'mir mir::Terminator<'tcx>,
        _location: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        if let mir::TerminatorKind::InlineAsm { operands, .. } = &terminator.kind {
            for op in operands {
                if let mir::InlineAsmOperand::Out { place: Some(place), .. }
                | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } = *op
                {
                    self.kill_borrows_on_place(trans, place);
                }
            }
        }
        terminator.edges()
    }
}

impl<C> DebugWithContext<C> for BorrowIndex {}
