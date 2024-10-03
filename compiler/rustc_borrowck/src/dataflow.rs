use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::graph;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{
    self, BasicBlock, Body, CallReturnPlaces, Location, Place, TerminatorEdges,
};
use rustc_middle::ty::{RegionVid, TyCtxt};
use rustc_mir_dataflow::fmt::DebugWithContext;
use rustc_mir_dataflow::impls::{EverInitializedPlaces, MaybeUninitializedPlaces};
use rustc_mir_dataflow::{Analysis, AnalysisDomain, Forward, GenKill, Results, ResultsVisitable};
use tracing::debug;

use crate::{BorrowSet, PlaceConflictBias, PlaceExt, RegionInferenceContext, places_conflict};

/// The results of the dataflow analyses used by the borrow checker.
pub(crate) struct BorrowckResults<'a, 'tcx> {
    pub(crate) borrows: Results<'tcx, Borrows<'a, 'tcx>>,
    pub(crate) uninits: Results<'tcx, MaybeUninitializedPlaces<'a, 'tcx>>,
    pub(crate) ever_inits: Results<'tcx, EverInitializedPlaces<'a, 'tcx>>,
}

/// The transient state of the dataflow analyses used by the borrow checker.
#[derive(Debug)]
pub(crate) struct BorrowckDomain<'a, 'tcx> {
    pub(crate) borrows: <Borrows<'a, 'tcx> as AnalysisDomain<'tcx>>::Domain,
    pub(crate) uninits: <MaybeUninitializedPlaces<'a, 'tcx> as AnalysisDomain<'tcx>>::Domain,
    pub(crate) ever_inits: <EverInitializedPlaces<'a, 'tcx> as AnalysisDomain<'tcx>>::Domain,
}

impl<'a, 'tcx> ResultsVisitable<'tcx> for BorrowckResults<'a, 'tcx> {
    type Direction = Forward;
    type Domain = BorrowckDomain<'a, 'tcx>;

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        BorrowckDomain {
            borrows: self.borrows.analysis.bottom_value(body),
            uninits: self.uninits.analysis.bottom_value(body),
            ever_inits: self.ever_inits.analysis.bottom_value(body),
        }
    }

    fn reset_to_block_entry(&self, state: &mut Self::Domain, block: BasicBlock) {
        state.borrows.clone_from(self.borrows.entry_set_for_block(block));
        state.uninits.clone_from(self.uninits.entry_set_for_block(block));
        state.ever_inits.clone_from(self.ever_inits.entry_set_for_block(block));
    }

    fn reconstruct_before_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.borrows.analysis.apply_before_statement_effect(&mut state.borrows, stmt, loc);
        self.uninits.analysis.apply_before_statement_effect(&mut state.uninits, stmt, loc);
        self.ever_inits.analysis.apply_before_statement_effect(&mut state.ever_inits, stmt, loc);
    }

    fn reconstruct_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &mir::Statement<'tcx>,
        loc: Location,
    ) {
        self.borrows.analysis.apply_statement_effect(&mut state.borrows, stmt, loc);
        self.uninits.analysis.apply_statement_effect(&mut state.uninits, stmt, loc);
        self.ever_inits.analysis.apply_statement_effect(&mut state.ever_inits, stmt, loc);
    }

    fn reconstruct_before_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.borrows.analysis.apply_before_terminator_effect(&mut state.borrows, term, loc);
        self.uninits.analysis.apply_before_terminator_effect(&mut state.uninits, term, loc);
        self.ever_inits.analysis.apply_before_terminator_effect(&mut state.ever_inits, term, loc);
    }

    fn reconstruct_terminator_effect(
        &mut self,
        state: &mut Self::Domain,
        term: &mir::Terminator<'tcx>,
        loc: Location,
    ) {
        self.borrows.analysis.apply_terminator_effect(&mut state.borrows, term, loc);
        self.uninits.analysis.apply_terminator_effect(&mut state.uninits, term, loc);
        self.ever_inits.analysis.apply_terminator_effect(&mut state.ever_inits, term, loc);
    }
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

        // We first handle the cases where the loan doesn't go out of scope, depending on the issuing
        // region's successors.
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
        trans: &mut impl GenKill<BorrowIndex>,
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
    fn kill_borrows_on_place(&self, trans: &mut impl GenKill<BorrowIndex>, place: Place<'tcx>) {
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

impl<'tcx> rustc_mir_dataflow::AnalysisDomain<'tcx> for Borrows<'_, 'tcx> {
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
}

/// Forward dataflow computation of the set of borrows that are in scope at a particular location.
/// - we gen the introduced loans
/// - we kill loans on locals going out of (regular) scope
/// - we kill the loans going out of their region's NLL scope: in NLL terms, the frontier where a
///   region stops containing the CFG points reachable from the issuing location.
/// - we also kill loans of conflicting places when overwriting a shared path: e.g. borrows of
///   `a.b.c` when `a` is overwritten.
impl<'tcx> rustc_mir_dataflow::GenKillAnalysis<'tcx> for Borrows<'_, 'tcx> {
    type Idx = BorrowIndex;

    fn domain_size(&self, _: &mir::Body<'tcx>) -> usize {
        self.borrow_set.len()
    }

    fn before_statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
        _statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn statement_effect(
        &mut self,
        trans: &mut impl GenKill<Self::Idx>,
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
            | mir::StatementKind::Nop => {}
        }
    }

    fn before_terminator_effect(
        &mut self,
        trans: &mut Self::Domain,
        _terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.kill_loans_out_of_scope_at_location(trans, location);
    }

    fn terminator_effect<'mir>(
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

    fn call_return_effect(
        &mut self,
        _trans: &mut Self::Domain,
        _block: mir::BasicBlock,
        _return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
    }
}

impl<C> DebugWithContext<C> for BorrowIndex {}
