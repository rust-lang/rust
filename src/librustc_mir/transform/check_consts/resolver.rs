//! Propagate `Qualif`s between locals and query the results.
//!
//! This also contains the dataflow analysis used to track `Qualif`s on complex control-flow
//! graphs.

use rustc::mir::visit::Visitor;
use rustc::mir::{self, BasicBlock, Local, Location};
use rustc_index::bit_set::BitSet;

use std::cell::RefCell;
use std::marker::PhantomData;

use crate::dataflow::{self as old_dataflow, generic as dataflow};
use super::{Item, Qualif};
use self::old_dataflow::IndirectlyMutableLocals;

/// A `Visitor` that propagates qualifs between locals. This defines the transfer function of
/// `FlowSensitiveAnalysis` as well as the logic underlying `TempPromotionResolver`.
///
/// This transfer does nothing when encountering an indirect assignment. Consumers should rely on
/// the `IndirectlyMutableLocals` dataflow pass to see if a `Local` may have become qualified via
/// an indirect assignment or function call.
struct TransferFunction<'a, 'mir, 'tcx, Q> {
    item: &'a Item<'mir, 'tcx>,
    qualifs_per_local: &'a mut BitSet<Local>,

    _qualif: PhantomData<Q>,
}

impl<Q> TransferFunction<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    fn new(
        item: &'a Item<'mir, 'tcx>,
        qualifs_per_local: &'a mut BitSet<Local>,
    ) -> Self {
        TransferFunction {
            item,
            qualifs_per_local,
            _qualif: PhantomData,
        }
    }

    fn initialize_state(&mut self) {
        self.qualifs_per_local.clear();

        for arg in self.item.body.args_iter() {
            let arg_ty = self.item.body.local_decls[arg].ty;
            if Q::in_any_value_of_ty(self.item, arg_ty) {
                self.qualifs_per_local.insert(arg);
            }
        }
    }

    fn assign_qualif_direct(&mut self, place: &mir::Place<'tcx>, value: bool) {
        debug_assert!(!place.is_indirect());

        match (value, place) {
            (true, mir::Place { base: mir::PlaceBase::Local(local), .. }) => {
                self.qualifs_per_local.insert(*local);
            }

            // For now, we do not clear the qualif if a local is overwritten in full by
            // an unqualified rvalue (e.g. `y = 5`). This is to be consistent
            // with aggregates where we overwrite all fields with assignments, which would not
            // get this feature.
            (false, mir::Place { base: mir::PlaceBase::Local(_local), projection: box [] }) => {
                // self.qualifs_per_local.remove(*local);
            }

            _ => {}
        }
    }

    fn apply_call_return_effect(
        &mut self,
        _block: BasicBlock,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        return_place: &mir::Place<'tcx>,
    ) {
        let return_ty = return_place.ty(self.item.body, self.item.tcx).ty;
        let qualif = Q::in_call(self.item, &mut self.qualifs_per_local, func, args, return_ty);
        if !return_place.is_indirect() {
            self.assign_qualif_direct(return_place, qualif);
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
        if let mir::Operand::Move(mir::Place {
            base: mir::PlaceBase::Local(local),
            projection: box [],
        }) = *operand {
            self.qualifs_per_local.remove(local);
        }
    }

    fn visit_assign(
        &mut self,
        place: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: Location,
    ) {
        let qualif = Q::in_rvalue(self.item, self.qualifs_per_local, rvalue);
        if !place.is_indirect() {
            self.assign_qualif_direct(place, qualif);
        }

        // We need to assign qualifs to the left-hand side before visiting `rvalue` since
        // qualifs can be cleared on move.
        self.super_assign(place, rvalue, location);
    }

    fn visit_terminator_kind(&mut self, kind: &mir::TerminatorKind<'tcx>, location: Location) {
        // The effect of assignment to the return place in `TerminatorKind::Call` is not applied
        // here; that occurs in `apply_call_return_effect`.

        if let mir::TerminatorKind::DropAndReplace { value, location: dest, .. } = kind {
            let qualif = Q::in_operand(self.item, self.qualifs_per_local, value);
            if !dest.is_indirect() {
                self.assign_qualif_direct(dest, qualif);
            }
        }

        // We need to assign qualifs to the dropped location before visiting the operand that
        // replaces it since qualifs can be cleared on move.
        self.super_terminator_kind(kind, location);
    }
}

/// Types that can compute the qualifs of each local at each location in a `mir::Body`.
///
/// Code that wishes to use a `QualifResolver` must call `visit_{statement,terminator}` for each
/// statement or terminator, processing blocks in reverse post-order beginning from the
/// `START_BLOCK`. Calling code may optionally call `get` after visiting each statement or
/// terminator to query the qualification state immediately before that statement or terminator.
///
/// These conditions are much more restrictive than woud be required by `FlowSensitiveResolver`
/// alone. This is to allow a linear, on-demand `TempPromotionResolver` that can operate
/// efficiently on simple CFGs.
pub trait QualifResolver<Q> {
    /// Get the qualifs of each local at the last location visited.
    ///
    /// This takes `&mut self` so qualifs can be computed lazily.
    fn get(&mut self) -> &BitSet<Local>;

    /// A convenience method for `self.get().contains(local)`.
    fn contains(&mut self, local: Local) -> bool {
        self.get().contains(local)
    }

    /// Resets the resolver to the `START_BLOCK`. This allows a resolver to be reused
    /// for multiple passes over a `mir::Body`.
    fn reset(&mut self);
}

pub type IndirectlyMutableResults<'mir, 'tcx> =
    old_dataflow::DataflowResultsCursor<'mir, 'tcx, IndirectlyMutableLocals<'mir, 'tcx>>;

/// A resolver for qualifs that works on arbitrarily complex CFGs.
///
/// As soon as a `Local` becomes writable through a reference (as determined by the
/// `IndirectlyMutableLocals` dataflow pass), we must assume that it takes on all other qualifs
/// possible for its type. This is because no effort is made to track qualifs across indirect
/// assignments (e.g. `*p = x` or calls to opaque functions).
///
/// It is possible to be more precise here by waiting until an indirect assignment actually occurs
/// before marking a borrowed `Local` as qualified.
pub struct FlowSensitiveResolver<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    location: Location,
    indirectly_mutable_locals: &'a RefCell<IndirectlyMutableResults<'mir, 'tcx>>,
    cursor: dataflow::ResultsCursor<'mir, 'tcx, FlowSensitiveAnalysis<'a, 'mir, 'tcx, Q>>,
    qualifs_per_local: BitSet<Local>,

    /// The value of `Q::in_any_value_of_ty` for each local.
    qualifs_in_any_value_of_ty: BitSet<Local>,
}

impl<Q> FlowSensitiveResolver<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    pub fn new(
        _: Q,
        item: &'a Item<'mir, 'tcx>,
        indirectly_mutable_locals: &'a RefCell<IndirectlyMutableResults<'mir, 'tcx>>,
        dead_unwinds: &BitSet<BasicBlock>,
    ) -> Self {
        let analysis = FlowSensitiveAnalysis {
            item,
            _qualif: PhantomData,
        };
        let results =
            dataflow::Engine::new(item.body, dead_unwinds, analysis).iterate_to_fixpoint();
        let cursor = dataflow::ResultsCursor::new(item.body, results);

        let mut qualifs_in_any_value_of_ty = BitSet::new_empty(item.body.local_decls.len());
        for (local, decl) in item.body.local_decls.iter_enumerated() {
            if Q::in_any_value_of_ty(item, decl.ty) {
                qualifs_in_any_value_of_ty.insert(local);
            }
        }

        FlowSensitiveResolver {
            cursor,
            indirectly_mutable_locals,
            qualifs_per_local: BitSet::new_empty(item.body.local_decls.len()),
            qualifs_in_any_value_of_ty,
            location: Location { block: mir::START_BLOCK, statement_index: 0 },
        }
    }
}

impl<Q> Visitor<'tcx> for FlowSensitiveResolver<'_, '_, 'tcx, Q>
where
    Q: Qualif
{
    fn visit_statement(&mut self, _: &mir::Statement<'tcx>, location: Location) {
        self.location = location;
    }

    fn visit_terminator(&mut self, _: &mir::Terminator<'tcx>, location: Location) {
        self.location = location;
    }
}

impl<Q> QualifResolver<Q> for FlowSensitiveResolver<'_, '_, '_, Q>
where
    Q: Qualif
{
    fn get(&mut self) -> &BitSet<Local> {
        let mut indirectly_mutable_locals = self.indirectly_mutable_locals.borrow_mut();

        indirectly_mutable_locals.seek(self.location);
        self.cursor.seek_before(self.location);

        self.qualifs_per_local.overwrite(indirectly_mutable_locals.get());
        self.qualifs_per_local.union(self.cursor.get());
        self.qualifs_per_local.intersect(&self.qualifs_in_any_value_of_ty);
        &self.qualifs_per_local
    }

    fn contains(&mut self, local: Local) -> bool {
        // No need to update the cursor if we know that `Local` cannot possibly be qualified.
        if !self.qualifs_in_any_value_of_ty.contains(local) {
            return false;
        }

        // Otherwise, return `true` if this local is qualified or was indirectly mutable at any
        // point before this statement.
        self.cursor.seek_before(self.location);
        if self.cursor.get().contains(local) {
            return true;
        }

        let mut indirectly_mutable_locals = self.indirectly_mutable_locals.borrow_mut();
        indirectly_mutable_locals.seek(self.location);
        indirectly_mutable_locals.get().contains(local)
    }

    fn reset(&mut self)  {
        self.location = Location { block: mir::START_BLOCK, statement_index: 0 };
    }
}

/// The dataflow analysis used to propagate qualifs on arbitrary CFGs.
pub(super) struct FlowSensitiveAnalysis<'a, 'mir, 'tcx, Q> {
    item: &'a Item<'mir, 'tcx>,
    _qualif: PhantomData<Q>,
}

impl<'a, 'mir, 'tcx, Q> FlowSensitiveAnalysis<'a, 'mir, 'tcx, Q>
where
    Q: Qualif,
{
    fn transfer_function(
        &self,
        state: &'a mut BitSet<Local>,
    ) -> TransferFunction<'a, 'mir, 'tcx, Q> {
        TransferFunction::<Q>::new(self.item, state)
    }
}

impl<Q> old_dataflow::BottomValue for FlowSensitiveAnalysis<'_, '_, '_, Q> {
    const BOTTOM_VALUE: bool = false;
}

impl<Q> dataflow::Analysis<'tcx> for FlowSensitiveAnalysis<'_, '_, 'tcx, Q>
where
    Q: Qualif,
{
    type Idx = Local;

    const NAME: &'static str = "flow_sensitive_qualif";

    fn bits_per_block(&self, body: &mir::Body<'tcx>) -> usize {
        body.local_decls.len()
    }

    fn initialize_start_block(&self, _body: &mir::Body<'tcx>, state: &mut BitSet<Self::Idx>) {
        self.transfer_function(state).initialize_state();
    }

    fn apply_statement_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(state).visit_statement(statement, location);
    }

    fn apply_terminator_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(state).visit_terminator(terminator, location);
    }

    fn apply_call_return_effect(
        &self,
        state: &mut BitSet<Self::Idx>,
        block: BasicBlock,
        func: &mir::Operand<'tcx>,
        args: &[mir::Operand<'tcx>],
        return_place: &mir::Place<'tcx>,
    ) {
        self.transfer_function(state).apply_call_return_effect(block, func, args, return_place)
    }
}
