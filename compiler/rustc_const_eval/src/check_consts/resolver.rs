//! Propagate `Qualif`s between locals and query the results.
//!
//! This contains the dataflow analysis used to track `Qualif`s on complex control-flow graphs.

use std::marker::PhantomData;

use rustc_index::bit_set::MixedBitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{
    self, BasicBlock, CallReturnPlaces, Local, Location, Statement, StatementKind,
};
use rustc_mir_dataflow::Analysis;

use super::{ConstCx, Qualif, qualifs};

/// A `Visitor` that propagates qualifs between locals. This defines the transfer function of
/// `FlowSensitiveAnalysis`.
///
/// To account for indirect assignments, data flow conservatively assumes that local becomes
/// qualified immediately after it is borrowed or its address escapes. The borrow must allow for
/// mutation, which includes shared borrows of places with interior mutability. The type of
/// borrowed place must contain the qualif.
struct TransferFunction<'mir, 'tcx, Q> {
    ccx: &'mir ConstCx<'mir, 'tcx>,
    state: &'mir mut MixedBitSet<Local>,
    _qualif: PhantomData<Q>,
}

impl<'mir, 'tcx, Q> TransferFunction<'mir, 'tcx, Q>
where
    Q: Qualif,
{
    fn new(ccx: &'mir ConstCx<'mir, 'tcx>, state: &'mir mut MixedBitSet<Local>) -> Self {
        TransferFunction { ccx, state, _qualif: PhantomData }
    }

    fn initialize_state(&mut self) {
        self.state.clear();

        for arg in self.ccx.body.args_iter() {
            let arg_ty = self.ccx.body.local_decls[arg].ty;
            if Q::in_any_value_of_ty(self.ccx, arg_ty) {
                self.state.insert(arg);
            }
        }
    }

    fn assign_qualif_direct(&mut self, place: &mir::Place<'tcx>, mut value: bool) {
        debug_assert!(!place.is_indirect());

        if !value {
            for (base, _elem) in place.iter_projections() {
                let base_ty = base.ty(self.ccx.body, self.ccx.tcx);
                if base_ty.ty.is_union() && Q::in_any_value_of_ty(self.ccx, base_ty.ty) {
                    value = true;
                    break;
                }
            }
        }

        match (value, place.as_ref()) {
            (true, mir::PlaceRef { local, .. }) => {
                self.state.insert(local);
            }

            // For now, we do not clear the qualif if a local is overwritten in full by
            // an unqualified rvalue (e.g. `y = 5`). This is to be consistent
            // with aggregates where we overwrite all fields with assignments, which would not
            // get this feature.
            (false, mir::PlaceRef { local: _, projection: &[] }) => {
                // self.state.remove(*local);
            }

            _ => {}
        }
    }

    fn apply_call_return_effect(
        &mut self,
        _block: BasicBlock,
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        return_places.for_each(|place| {
            // We cannot reason about another function's internals, so use conservative type-based
            // qualification for the result of a function call.
            let return_ty = place.ty(self.ccx.body, self.ccx.tcx).ty;
            let qualif = Q::in_any_value_of_ty(self.ccx, return_ty);

            if !place.is_indirect() {
                self.assign_qualif_direct(&place, qualif);
            }
        });
    }

    fn address_of_allows_mutation(&self) -> bool {
        // Exact set of permissions granted by RawPtr is undecided. Conservatively assume that
        // it might allow mutation until resolution of #56604.
        true
    }

    fn ref_allows_mutation(&self, kind: mir::BorrowKind, place: mir::Place<'tcx>) -> bool {
        match kind {
            mir::BorrowKind::Mut { .. } => true,
            mir::BorrowKind::Shared | mir::BorrowKind::Fake(_) => {
                self.shared_borrow_allows_mutation(place)
            }
        }
    }

    /// `&` only allow mutation if the borrowed place is `!Freeze`.
    ///
    /// This assumes that it is UB to take the address of a struct field whose type is
    /// `Freeze`, then use pointer arithmetic to derive a pointer to a *different* field of
    /// that same struct whose type is `!Freeze`. If we decide that this is not UB, we will
    /// have to check the type of the borrowed **local** instead of the borrowed **place**
    /// below. See [rust-lang/unsafe-code-guidelines#134].
    ///
    /// [rust-lang/unsafe-code-guidelines#134]: https://github.com/rust-lang/unsafe-code-guidelines/issues/134
    fn shared_borrow_allows_mutation(&self, place: mir::Place<'tcx>) -> bool {
        !place.ty(self.ccx.body, self.ccx.tcx).ty.is_freeze(self.ccx.tcx, self.ccx.typing_env)
    }
}

impl<'tcx, Q> Visitor<'tcx> for TransferFunction<'_, 'tcx, Q>
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
        if let mir::Operand::Move(place) = operand
            && let Some(local) = place.as_local()
        {
            // The local is no longer initialized so we can remove the qualif.
            self.state.remove(local);
        }
    }

    fn visit_assign(
        &mut self,
        place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        let qualif = qualifs::in_rvalue::<Q, _>(self.ccx, &mut |l| self.state.contains(l), rvalue);
        if !place.is_indirect() {
            self.assign_qualif_direct(place, qualif);
        }

        // We need to assign qualifs to the left-hand side before visiting `rvalue` since
        // qualifs can be cleared on move.
        self.super_assign(place, rvalue, location);
    }

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        match rvalue {
            mir::Rvalue::RawPtr(_mt, borrowed_place) => {
                if !borrowed_place.is_indirect() && self.address_of_allows_mutation() {
                    let place_ty = borrowed_place.ty(self.ccx.body, self.ccx.tcx).ty;
                    if Q::in_any_value_of_ty(self.ccx, place_ty) {
                        self.state.insert(borrowed_place.local);
                    }
                }
            }

            mir::Rvalue::Ref(_, kind, borrowed_place) => {
                if !borrowed_place.is_indirect() && self.ref_allows_mutation(*kind, *borrowed_place)
                {
                    let place_ty = borrowed_place.ty(self.ccx.body, self.ccx.tcx).ty;
                    if Q::in_any_value_of_ty(self.ccx, place_ty) {
                        self.state.insert(borrowed_place.local);
                    }
                }
            }

            mir::Rvalue::Reborrow(target, mutability, borrowed_place) => {
                // A Reborrow allows mutation if it is Reborrow or if the CoerceShared target isn't
                // Freeze.
                if !borrowed_place.is_indirect()
                    && (mutability.is_mut() || !target.is_freeze(self.ccx.tcx, self.ccx.typing_env))
                {
                    if Q::in_any_value_of_ty(self.ccx, *target) {
                        self.state.insert(borrowed_place.local);
                    }
                }
            }

            mir::Rvalue::Cast(..)
            | mir::Rvalue::Use(..)
            | mir::Rvalue::CopyForDeref(..)
            | mir::Rvalue::ThreadLocalRef(..)
            | mir::Rvalue::Repeat(..)
            | mir::Rvalue::BinaryOp(..)
            | mir::Rvalue::UnaryOp(..)
            | mir::Rvalue::Discriminant(..)
            | mir::Rvalue::Aggregate(..)
            | mir::Rvalue::WrapUnsafeBinder(..) => {}
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::StorageDead(local) => {
                self.state.remove(local);
            }
            _ => self.super_statement(statement, location),
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        // The effect of assignment to the return place in `TerminatorKind::Call` is not applied
        // here; that occurs in `apply_call_return_effect`.

        // We ignore borrow on drop because custom drop impls are not allowed in consts.
        // FIXME: Reconsider if accounting for borrows in drops is necessary for const drop.
        self.super_terminator(terminator, location);
    }
}

/// The dataflow analysis used to propagate qualifs on arbitrary CFGs.
pub(super) struct FlowSensitiveAnalysis<'mir, 'tcx, Q> {
    ccx: &'mir ConstCx<'mir, 'tcx>,
    _qualif: PhantomData<Q>,
}

impl<'mir, 'tcx, Q> FlowSensitiveAnalysis<'mir, 'tcx, Q>
where
    Q: Qualif,
{
    pub(super) fn new(ccx: &'mir ConstCx<'mir, 'tcx>) -> Self {
        FlowSensitiveAnalysis { ccx, _qualif: PhantomData }
    }

    fn transfer_function(
        &self,
        state: &'mir mut MixedBitSet<Local>,
    ) -> TransferFunction<'mir, 'tcx, Q> {
        TransferFunction::<Q>::new(self.ccx, state)
    }
}

impl<'tcx, Q> Analysis<'tcx> for FlowSensitiveAnalysis<'_, 'tcx, Q>
where
    Q: Qualif,
{
    /// Describes whether a local contains qualif. This domain is likely homogeneous, and has a big
    /// size, so we use a bitset that can be sparse (c.f. issue #134404).
    type Domain = MixedBitSet<Local>;

    const NAME: &'static str = Q::ANALYSIS_NAME;

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        MixedBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, _body: &mir::Body<'tcx>, state: &mut Self::Domain) {
        self.transfer_function(state).initialize_state();
    }

    fn apply_primary_statement_effect(
        &self,
        state: &mut Self::Domain,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(state).visit_statement(statement, location);
    }

    fn apply_primary_terminator_effect(
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
        return_places: CallReturnPlaces<'_, 'tcx>,
    ) {
        self.transfer_function(state).apply_call_return_effect(block, return_places)
    }
}
