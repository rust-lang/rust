use super::*;

use crate::{AnalysisDomain, GenKill, GenKillAnalysis};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, TyCtxt};
use rustc_span::DUMMY_SP;

pub type MaybeMutBorrowedLocals<'mir, 'tcx> = MaybeBorrowedLocals<MutBorrow<'mir, 'tcx>>;

/// A dataflow analysis that tracks whether a pointer or reference could possibly exist that points
/// to a given local.
///
/// The `K` parameter determines what kind of borrows are tracked. By default,
/// `MaybeBorrowedLocals` looks for *any* borrow of a local. If you are only interested in borrows
/// that might allow mutation, use the `MaybeMutBorrowedLocals` type alias instead.
///
/// At present, this is used as a very limited form of alias analysis. For example,
/// `MaybeBorrowedLocals` is used to compute which locals are live during a yield expression for
/// immovable generators. `MaybeMutBorrowedLocals` is used during const checking to prove that a
/// local has not been mutated via indirect assignment (e.g., `*p = 42`), the side-effects of a
/// function call or inline assembly.
pub struct MaybeBorrowedLocals<K = AnyBorrow> {
    kind: K,
    ignore_borrow_on_drop: bool,
}

impl MaybeBorrowedLocals {
    /// A dataflow analysis that records whether a pointer or reference exists that may alias the
    /// given local.
    pub fn all_borrows() -> Self {
        MaybeBorrowedLocals { kind: AnyBorrow, ignore_borrow_on_drop: false }
    }
}

impl MaybeMutBorrowedLocals<'mir, 'tcx> {
    /// A dataflow analysis that records whether a pointer or reference exists that may *mutably*
    /// alias the given local.
    ///
    /// This includes `&mut` and pointers derived from an `&mut`, as well as shared borrows of
    /// types with interior mutability.
    pub fn mut_borrows_only(
        tcx: TyCtxt<'tcx>,
        body: &'mir mir::Body<'tcx>,
        param_env: ParamEnv<'tcx>,
    ) -> Self {
        MaybeBorrowedLocals {
            kind: MutBorrow { body, tcx, param_env },
            ignore_borrow_on_drop: false,
        }
    }
}

impl<K> MaybeBorrowedLocals<K> {
    /// During dataflow analysis, ignore the borrow that may occur when a place is dropped.
    ///
    /// Drop terminators may call custom drop glue (`Drop::drop`), which takes `&mut self` as a
    /// parameter. In the general case, a drop impl could launder that reference into the
    /// surrounding environment through a raw pointer, thus creating a valid `*mut` pointing to the
    /// dropped local. We are not yet willing to declare this particular case UB, so we must treat
    /// all dropped locals as mutably borrowed for now. See discussion on [#61069].
    ///
    /// In some contexts, we know that this borrow will never occur. For example, during
    /// const-eval, custom drop glue cannot be run. Code that calls this should document the
    /// assumptions that justify ignoring `Drop` terminators in this way.
    ///
    /// [#61069]: https://github.com/rust-lang/rust/pull/61069
    pub fn unsound_ignore_borrow_on_drop(self) -> Self {
        MaybeBorrowedLocals { ignore_borrow_on_drop: true, ..self }
    }

    fn transfer_function<'a, T>(&'a self, trans: &'a mut T) -> TransferFunction<'a, T, K> {
        TransferFunction {
            kind: &self.kind,
            trans,
            ignore_borrow_on_drop: self.ignore_borrow_on_drop,
        }
    }
}

impl<K> AnalysisDomain<'tcx> for MaybeBorrowedLocals<K>
where
    K: BorrowAnalysisKind<'tcx>,
{
    type Domain = BitSet<Local>;
    const NAME: &'static str = K::ANALYSIS_NAME;

    fn bottom_value(&self, body: &mir::Body<'tcx>) -> Self::Domain {
        // bottom = unborrowed
        BitSet::new_empty(body.local_decls().len())
    }

    fn initialize_start_block(&self, _: &mir::Body<'tcx>, _: &mut Self::Domain) {
        // No locals are aliased on function entry
    }
}

impl<K> GenKillAnalysis<'tcx> for MaybeBorrowedLocals<K>
where
    K: BorrowAnalysisKind<'tcx>,
{
    type Idx = Local;

    fn statement_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        statement: &mir::Statement<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_statement(statement, location);
    }

    fn terminator_effect(
        &self,
        trans: &mut impl GenKill<Self::Idx>,
        terminator: &mir::Terminator<'tcx>,
        location: Location,
    ) {
        self.transfer_function(trans).visit_terminator(terminator, location);
    }

    fn call_return_effect(
        &self,
        _trans: &mut impl GenKill<Self::Idx>,
        _block: mir::BasicBlock,
        _func: &mir::Operand<'tcx>,
        _args: &[mir::Operand<'tcx>],
        _dest_place: mir::Place<'tcx>,
    ) {
    }
}

/// A `Visitor` that defines the transfer function for `MaybeBorrowedLocals`.
struct TransferFunction<'a, T, K> {
    trans: &'a mut T,
    kind: &'a K,
    ignore_borrow_on_drop: bool,
}

impl<T, K> Visitor<'tcx> for TransferFunction<'a, T, K>
where
    T: GenKill<Local>,
    K: BorrowAnalysisKind<'tcx>,
{
    fn visit_statement(&mut self, stmt: &Statement<'tcx>, location: Location) {
        self.super_statement(stmt, location);

        // When we reach a `StorageDead` statement, we can assume that any pointers to this memory
        // are now invalid.
        if let StatementKind::StorageDead(local) = stmt.kind {
            self.trans.kill(local);
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);

        match rvalue {
            mir::Rvalue::AddressOf(mt, borrowed_place) => {
                if !borrowed_place.is_indirect() && self.kind.in_address_of(*mt, *borrowed_place) {
                    self.trans.gen(borrowed_place.local);
                }
            }

            mir::Rvalue::Ref(_, kind, borrowed_place) => {
                if !borrowed_place.is_indirect() && self.kind.in_ref(*kind, *borrowed_place) {
                    self.trans.gen(borrowed_place.local);
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

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            mir::TerminatorKind::Drop { place: dropped_place, .. }
            | mir::TerminatorKind::DropAndReplace { place: dropped_place, .. } => {
                // See documentation for `unsound_ignore_borrow_on_drop` for an explanation.
                if !self.ignore_borrow_on_drop {
                    self.trans.gen(dropped_place.local);
                }
            }

            TerminatorKind::Abort
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::InlineAsm { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. } => {}
        }
    }
}

pub struct AnyBorrow;

pub struct MutBorrow<'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir Body<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl MutBorrow<'mir, 'tcx> {
    /// `&` and `&raw` only allow mutation if the borrowed place is `!Freeze`.
    ///
    /// This assumes that it is UB to take the address of a struct field whose type is
    /// `Freeze`, then use pointer arithmetic to derive a pointer to a *different* field of
    /// that same struct whose type is `!Freeze`. If we decide that this is not UB, we will
    /// have to check the type of the borrowed **local** instead of the borrowed **place**
    /// below. See [rust-lang/unsafe-code-guidelines#134].
    ///
    /// [rust-lang/unsafe-code-guidelines#134]: https://github.com/rust-lang/unsafe-code-guidelines/issues/134
    fn shared_borrow_allows_mutation(&self, place: Place<'tcx>) -> bool {
        !place.ty(self.body, self.tcx).ty.is_freeze(self.tcx.at(DUMMY_SP), self.param_env)
    }
}

pub trait BorrowAnalysisKind<'tcx> {
    const ANALYSIS_NAME: &'static str;

    fn in_address_of(&self, mt: Mutability, place: Place<'tcx>) -> bool;
    fn in_ref(&self, kind: mir::BorrowKind, place: Place<'tcx>) -> bool;
}

impl BorrowAnalysisKind<'tcx> for AnyBorrow {
    const ANALYSIS_NAME: &'static str = "maybe_borrowed_locals";

    fn in_ref(&self, _: mir::BorrowKind, _: Place<'_>) -> bool {
        true
    }
    fn in_address_of(&self, _: Mutability, _: Place<'_>) -> bool {
        true
    }
}

impl BorrowAnalysisKind<'tcx> for MutBorrow<'mir, 'tcx> {
    const ANALYSIS_NAME: &'static str = "maybe_mut_borrowed_locals";

    fn in_ref(&self, kind: mir::BorrowKind, place: Place<'tcx>) -> bool {
        match kind {
            mir::BorrowKind::Mut { .. } => true,
            mir::BorrowKind::Shared | mir::BorrowKind::Shallow | mir::BorrowKind::Unique => {
                self.shared_borrow_allows_mutation(place)
            }
        }
    }

    fn in_address_of(&self, mt: Mutability, place: Place<'tcx>) -> bool {
        match mt {
            Mutability::Mut => true,
            Mutability::Not => self.shared_borrow_allows_mutation(place),
        }
    }
}
