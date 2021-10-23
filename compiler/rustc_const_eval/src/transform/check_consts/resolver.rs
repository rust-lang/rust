//! Propagate `Qualif`s between locals and query the results.
//!
//! This contains the dataflow analysis used to track `Qualif`s on complex control-flow graphs.

use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, BasicBlock, Local, Location, Statement, StatementKind};

use std::marker::PhantomData;

use super::{qualifs, ConstCx, Qualif};

/// A `Visitor` that propagates qualifs between locals. This defines the transfer function of
/// `FlowSensitiveAnalysis`.
///
/// This transfer does nothing when encountering an indirect assignment. Consumers should rely on
/// the `MaybeMutBorrowedLocals` dataflow pass to see if a `Local` may have become qualified via
/// an indirect assignment or function call.
struct TransferFunction<'a, 'mir, 'tcx, Q> {
    ccx: &'a ConstCx<'mir, 'tcx>,
    qualifs_per_local: &'a mut BitSet<Local>,

    _qualif: PhantomData<Q>,
}

impl<Q> TransferFunction<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    fn new(ccx: &'a ConstCx<'mir, 'tcx>, qualifs_per_local: &'a mut BitSet<Local>) -> Self {
        TransferFunction { ccx, qualifs_per_local, _qualif: PhantomData }
    }

    fn initialize_state(&mut self) {
        self.qualifs_per_local.clear();

        for arg in self.ccx.body.args_iter() {
            let arg_ty = self.ccx.body.local_decls[arg].ty;
            if Q::in_any_value_of_ty(self.ccx, arg_ty) {
                self.qualifs_per_local.insert(arg);
            }
        }
    }

    fn assign_qualif_direct(&mut self, place: &mir::Place<'tcx>, value: bool) {
        debug_assert!(!place.is_indirect());

        match (value, place.as_ref()) {
            (true, mir::PlaceRef { local, .. }) => {
                self.qualifs_per_local.insert(local);
            }

            // For now, we do not clear the qualif if a local is overwritten in full by
            // an unqualified rvalue (e.g. `y = 5`). This is to be consistent
            // with aggregates where we overwrite all fields with assignments, which would not
            // get this feature.
            (false, mir::PlaceRef { local: _, projection: &[] }) => {
                // self.qualifs_per_local.remove(*local);
            }

            _ => {}
        }
    }

    fn apply_call_return_effect(
        &mut self,
        _block: BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        return_place: mir::Place<'tcx>,
    ) {
        // We cannot reason about another function's internals, so use conservative type-based
        // qualification for the result of a function call.
        let return_ty = return_place.ty(self.ccx.body, self.ccx.tcx).ty;
        let qualif = Q::in_any_value_of_ty(self.ccx, return_ty);

        if !return_place.is_indirect() {
            self.assign_qualif_direct(&return_place, qualif);
        }
    }
}

impl<Q> Visitor<'tcx> for TransferFunction<'_, '_, 'tcx, Q>
where
    Q: Qualif,
{
    fn visit_operand(&mut self, operand: &mir::Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);

        if !Q::IS_CLEARED_ON_MOVE {
            return;
        }

        // If a local with no projections is moved from (e.g. `x` in `y = x`), record that
        // it no longer needs to be dropped.
        if let mir::Operand::Move(place) = operand {
            if let Some(local) = place.as_local() {
                self.qualifs_per_local.remove(local);
            }
        }
    }

    fn visit_assign(
        &mut self,
        place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        let qualif = qualifs::in_rvalue::<Q, _>(
            self.ccx,
            &mut |l| self.qualifs_per_local.contains(l),
            rvalue,
        );
        if !place.is_indirect() {
            self.assign_qualif_direct(place, qualif);
        }

        // We need to assign qualifs to the left-hand side before visiting `rvalue` since
        // qualifs can be cleared on move.
        self.super_assign(place, rvalue, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::StorageDead(local) => {
                self.qualifs_per_local.remove(local);
            }
            _ => self.super_statement(statement, location),
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        // The effect of assignment to the return place in `TerminatorKind::Call` is not applied
        // here; that occurs in `apply_call_return_effect`.

        if let mir::TerminatorKind::DropAndReplace { value, place, .. } = &terminator.kind {
            let qualif = qualifs::in_operand::<Q, _>(
                self.ccx,
                &mut |l| self.qualifs_per_local.contains(l),
                value,
            );

            if !place.is_indirect() {
                self.assign_qualif_direct(place, qualif);
            }
        }

        // We need to assign qualifs to the dropped location before visiting the operand that
        // replaces it since qualifs can be cleared on move.
        self.super_terminator(terminator, location);
    }
}

/// The dataflow analysis used to propagate qualifs on arbitrary CFGs.
pub(super) struct FlowSensitiveAnalysis<'a, 'mir, 'tcx, Q> {
    ccx: &'a ConstCx<'mir, 'tcx>,
    _qualif: PhantomData<Q>,
}

impl<'a, 'mir, 'tcx, Q> FlowSensitiveAnalysis<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    pub(super) fn new(_: Q, ccx: &'a ConstCx<'mir, 'tcx>) -> Self {
        FlowSensitiveAnalysis { ccx, _qualif: PhantomData }
    }

    fn transfer_function(
        &self,
        state: &'a mut BitSet<Local>,
    ) -> TransferFunction<'a, 'mir, 'tcx, Q> {
        TransferFunction::<Q>::new(self.ccx, state)
    }
}

impl<Q> rustc_mir_dataflow::AnalysisDomain<'tcx> for FlowSensitiveAnalysis<'_, '_, 'tcx, Q>
where
    Q: Qualif,
{
    type Domain = BitSet<Local>;

    const NAME: &'static str = Q::ANALYSIS_NAME;

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        BitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, _body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        self.transfer_function(state).initialize_state();
    }
}

impl<Q> rustc_mir_dataflow::Analysis<'tcx> for FlowSensitiveAnalysis<'_, '_, 'tcx, Q>
where
    Q: Qualif,
{
    fn apply_statement_effect(
        &self,
        state: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(state).visit_statement(statement, location);
    }

    fn apply_terminator_effect(
        &self,
        state: &mut Self::Domain,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(state).visit_terminator(terminator, location);
    }

    fn apply_call_return_effect(
        &self,
        state: &mut Self::Domain,
        block: BasicBlock,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        return_place: mir::Place<'tcx>,
    ) {
        self.transfer_function(state).apply_call_return_effect(block, func, args, return_place)
    }
}
