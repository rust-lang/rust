//! Helpers for lowering generic reborrow expressions.

use rustc_middle::middle::region::TempLifetime;
use rustc_middle::mir::*;
use rustc_middle::thir::ExprId;
use rustc_middle::ty::Ty;
use rustc_span::Span;

use crate::builder::scope::DropKind;
use crate::builder::{BlockAnd, BlockAndExtension, Builder};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Build a reborrow from the source expression as a place.
    pub(in crate::builder::expr) fn reborrow_rvalue_from_source_place(
        &mut self,
        mut block: BasicBlock,
        source: ExprId,
        mutability: Mutability,
        target: Ty<'tcx>,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let source = unpack!(block = self.as_place(block, source));

        block.and(Rvalue::Reborrow(target, mutability, source))
    }

    /// Materialize the reborrow result before later user code, such as an assignment LHS, runs.
    pub(in crate::builder::expr) fn lower_reborrow_as_result_temp(
        &mut self,
        mut block: BasicBlock,
        source: ExprId,
        mutability: Mutability,
        target: Ty<'tcx>,
        temp_lifetime: TempLifetime,
        span: Span,
    ) -> BlockAnd<Rvalue<'tcx>> {
        let reborrow = unpack!(
            block = self.reborrow_rvalue_from_source_place(block, source, mutability, target)
        );
        let result = self.temp(target, span);
        let source_info = self.source_info(span);

        self.cfg.push(block, Statement::new(source_info, StatementKind::StorageLive(result.local)));
        if let Some(temp_lifetime) = temp_lifetime.temp_lifetime {
            self.schedule_drop(span, temp_lifetime, result.local, DropKind::Storage);
        }

        self.cfg.push_assign(block, source_info, result, reborrow);

        if let Some(temp_lifetime) = temp_lifetime.temp_lifetime {
            self.schedule_drop(span, temp_lifetime, result.local, DropKind::Value);
        }
        if let Some(backwards_incompatible) = temp_lifetime.backwards_incompatible {
            self.schedule_backwards_incompatible_drop(span, backwards_incompatible, result.local);
        }

        block.and(Rvalue::Use(Operand::Move(result), WithRetag::Yes))
    }
}
