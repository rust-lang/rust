//! Propagate `Qualif`s between locals and query the results.
//!
//! This contains the dataflow analysis used to track `Qualif`s on complex control-flow graphs.

use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, BasicBlock, Local, Location, Statement, StatementKind};
use rustc_mir_dataflow::fmt::DebugWithContext;
use rustc_mir_dataflow::JoinSemiLattice;
use rustc_span::DUMMY_SP;

use std::fmt;
use std::marker::PhantomData;

use super::{qualifs, ConstCx, Qualif};

/// A `Visitor` that propagates qualifs between locals. This defines the transfer function of
/// `FlowSensitiveAnalysis`.
///
/// To account for indirect assignments, data flow conservatively assumes that local becomes
/// qualified immediately after it is borrowed or its address escapes. The borrow must allow for
/// mutation, which includes shared borrows of places with interior mutability. The type of
/// borrowed place must contain the qualif.
struct TransferFunction<'a, 'mir, 'tcx, Q> {
    ccx: &'a ConstCx<'mir, 'tcx>,
    state: &'a mut State,
    _qualif: PhantomData<Q>,
}

impl<Q> TransferFunction<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    fn new(ccx: &'a ConstCx<'mir, 'tcx>, state: &'a mut State) -> Self {
        TransferFunction { ccx, state, _qualif: PhantomData }
    }

    fn initialize_state(&mut self) {
        self.state.qualif.clear();
        self.state.borrow.clear();

        for arg in self.ccx.body.args_iter() {
            let arg_ty = self.ccx.body.local_decls[arg].ty;
            if Q::in_any_value_of_ty(self.ccx, arg_ty) {
                self.state.qualif.insert(arg);
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
                self.state.qualif.insert(local);
            }

            // For now, we do not clear the qualif if a local is overwritten in full by
            // an unqualified rvalue (e.g. `y = 5`). This is to be consistent
            // with aggregates where we overwrite all fields with assignments, which would not
            // get this feature.
            (false, mir::PlaceRef { local: _, projection: &[] }) => {
                // self.state.qualif.remove(*local);
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

    fn address_of_allows_mutation(&self, _mt: mir::Mutability, _place: mir::Place<'tcx>) -> bool {
        // Exact set of permissions granted by AddressOf is undecided. Conservatively assume that
        // it might allow mutation until resolution of #56604.
        true
    }

    fn ref_allows_mutation(&self, kind: mir::BorrowKind, place: mir::Place<'tcx>) -> bool {
        match kind {
            mir::BorrowKind::Mut { .. } => true,
            mir::BorrowKind::Shared | mir::BorrowKind::Shallow | mir::BorrowKind::Unique => {
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
        !place
            .ty(self.ccx.body, self.ccx.tcx)
            .ty
            .is_freeze(self.ccx.tcx.at(DUMMY_SP), self.ccx.param_env)
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
                // For backward compatibility with the MaybeMutBorrowedLocals used in an earlier
                // implementation we retain qualif if a local had been borrowed before. This might
                // not be strictly necessary since the local is no longer initialized.
                if !self.state.borrow.contains(local) {
                    self.state.qualif.remove(local);
                }
            }
        }
    }

    fn visit_assign(
        &mut self,
        place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        let qualif =
            qualifs::in_rvalue::<Q, _>(self.ccx, &mut |l| self.state.qualif.contains(l), rvalue);
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
            mir::Rvalue::AddressOf(mt, borrowed_place) => {
                if !borrowed_place.is_indirect()
                    && self.address_of_allows_mutation(*mt, *borrowed_place)
                {
                    let place_ty = borrowed_place.ty(self.ccx.body, self.ccx.tcx).ty;
                    if Q::in_any_value_of_ty(self.ccx, place_ty) {
                        self.state.qualif.insert(borrowed_place.local);
                        self.state.borrow.insert(borrowed_place.local);
                    }
                }
            }

            mir::Rvalue::Ref(_, kind, borrowed_place) => {
                if !borrowed_place.is_indirect() && self.ref_allows_mutation(*kind, *borrowed_place)
                {
                    let place_ty = borrowed_place.ty(self.ccx.body, self.ccx.tcx).ty;
                    if Q::in_any_value_of_ty(self.ccx, place_ty) {
                        self.state.qualif.insert(borrowed_place.local);
                        self.state.borrow.insert(borrowed_place.local);
                    }
                }
            }

            mir::Rvalue::Cast(..)
            | mir::Rvalue::ShallowInitBox(..)
            | mir::Rvalue::Use(..)
            | mir::Rvalue::ThreadLocalRef(..)
            | mir::Rvalue::Repeat(..)
            | mir::Rvalue::Len(..)
            | mir::Rvalue::BinaryOp(..)
            | mir::Rvalue::CheckedBinaryOp(..)
            | mir::Rvalue::NullaryOp(..)
            | mir::Rvalue::UnaryOp(..)
            | mir::Rvalue::Discriminant(..)
            | mir::Rvalue::Aggregate(..) => {}
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::StorageDead(local) => {
                self.state.qualif.remove(local);
                self.state.borrow.remove(local);
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
                &mut |l| self.state.qualif.contains(l),
                value,
            );

            if !place.is_indirect() {
                self.assign_qualif_direct(place, qualif);
            }
        }

        // We ignore borrow on drop because custom drop impls are not allowed in consts.
        // FIXME: Reconsider if accounting for borrows in drops is necessary for const drop.

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

    fn transfer_function(&self, state: &'a mut State) -> TransferFunction<'a, 'mir, 'tcx, Q> {
        TransferFunction::<Q>::new(self.ccx, state)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct State {
    /// Describes whether a local contains qualif.
    pub qualif: BitSet<Local>,
    /// Describes whether a local's address escaped and it might become qualified as a result an
    /// indirect mutation.
    pub borrow: BitSet<Local>,
}

impl State {
    #[inline]
    pub(super) fn contains(&self, local: Local) -> bool {
        self.qualif.contains(local)
    }
}

impl<C> DebugWithContext<C> for State {
    fn fmt_with(&self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("qualif: ")?;
        self.qualif.fmt_with(ctxt, f)?;
        f.write_str(" borrow: ")?;
        self.borrow.fmt_with(ctxt, f)?;
        Ok(())
    }

    fn fmt_diff_with(&self, old: &Self, ctxt: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == old {
            return Ok(());
        }

        if self.qualif != old.qualif {
            f.write_str("qualif: ")?;
            self.qualif.fmt_diff_with(&old.qualif, ctxt, f)?;
            f.write_str("\n")?;
        }

        if self.borrow != old.borrow {
            f.write_str("borrow: ")?;
            self.qualif.fmt_diff_with(&old.borrow, ctxt, f)?;
            f.write_str("\n")?;
        }

        Ok(())
    }
}

impl JoinSemiLattice for State {
    fn join(&mut self, other: &Self) -> bool {
        self.qualif.join(&other.qualif) || self.borrow.join(&other.borrow)
    }
}

impl<Q> rustc_mir_dataflow::AnalysisDomain<'tcx> for FlowSensitiveAnalysis<'_, '_, 'tcx, Q>
where
    Q: Qualif,
{
    type Domain = State;

    const NAME: &'static str = Q::ANALYSIS_NAME;

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        State {
            qualif: BitSet::new_empty(body.local_decls.len()),
            borrow: BitSet::new_empty(body.local_decls.len()),
        }
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
