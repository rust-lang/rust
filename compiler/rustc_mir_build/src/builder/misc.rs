//! Miscellaneous builder routines that are not specific to building any particular
//! kind of thing.

use rustc_middle::mir::*;
use rustc_middle::thir::{ExprId, ExprKind};
use rustc_middle::ty::Ty;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_span::Span;
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::debug;

use crate::builder::Builder;
use crate::builder::expr::category::Category;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Adds a new temporary value of type `ty` storing the result of
    /// evaluating `expr`.
    ///
    /// N.B., **No cleanup is scheduled for this temporary.** You should
    /// call `schedule_drop` once the temporary is initialized.
    pub(crate) fn temp(&mut self, ty: Ty<'tcx>, span: Span) -> Place<'tcx> {
        let temp = self.local_decls.push(LocalDecl::new(ty, span));
        let place = Place::from(temp);
        debug!("temp: created temp {:?} with type {:?}", place, self.local_decls[temp].ty);
        place
    }

    /// Convenience function for creating a literal operand, one
    /// without any user type annotation.
    pub(crate) fn literal_operand(&mut self, span: Span, const_: Const<'tcx>) -> Operand<'tcx> {
        let constant = Box::new(ConstOperand { span, user_ty: None, const_ });
        Operand::Constant(constant)
    }

    /// Returns a zero literal operand for the appropriate type, works for
    /// bool, char and integers.
    pub(crate) fn zero_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        let literal = Const::from_bits(self.tcx, 0, self.typing_env(), ty);

        self.literal_operand(span, literal)
    }

    pub(crate) fn push_usize(
        &mut self,
        block: BasicBlock,
        source_info: SourceInfo,
        value: u64,
    ) -> Place<'tcx> {
        let usize_ty = self.tcx.types.usize;
        let temp = self.temp(usize_ty, source_info.span);
        self.cfg.push_assign_constant(
            block,
            source_info,
            temp,
            ConstOperand {
                span: source_info.span,
                user_ty: None,
                const_: Const::from_usize(self.tcx, value),
            },
        );
        temp
    }

    pub(crate) fn consume_by_copy_or_move(&self, place: Place<'tcx>) -> Operand<'tcx> {
        let tcx = self.tcx;
        let ty = place.ty(&self.local_decls, tcx).ty;
        if self.infcx.type_is_copy_modulo_regions(self.param_env, ty) {
            Operand::Copy(place)
        } else {
            Operand::Move(place)
        }
    }

    /// Recursively inspect a THIR expression and probe through unsizing
    /// operations that can be const-folded today.
    pub(crate) fn check_constness(&self, mut kind: &'a ExprKind<'tcx>) -> bool {
        loop {
            debug!(?kind, "check_constness");
            match kind {
                &ExprKind::ValueTypeAscription { source: eid, user_ty: _, user_ty_span: _ }
                | &ExprKind::ValueExpr { source: eid }
                | &ExprKind::PointerCoercion {
                    cast: PointerCoercion::Unsize,
                    source: eid,
                    is_from_as_cast: _,
                }
                | &ExprKind::Scope { region_scope: _, hir_id: _, value: eid } => {
                    kind = &self.thir[eid].kind
                }
                _ => return matches!(Category::of(&kind), Some(Category::Constant)),
            }
        }
    }

    /// Whether `.use` applied to this expression needs to clone it.
    pub(crate) fn use_expr_must_clone(&self, expr: ExprId) -> bool {
        match self.thir[expr].kind {
            ExprKind::Scope { value: inner, .. }
            | ExprKind::PlaceTypeAscription { source: inner, .. }
            | ExprKind::PlaceUnwrapUnsafeBinder { source: inner }
            | ExprKind::Index { lhs: inner, .. } => self.use_expr_must_clone(inner),

            ExprKind::Deref { arg } if self.thir[arg].ty.is_box() => self.use_expr_must_clone(arg),

            ExprKind::Field { lhs, .. } => {
                if let Some(adt_def) = self.thir[lhs].ty.ty_adt_def()
                    && adt_def.has_dtor(self.tcx)
                {
                    // Can't move out of a field of a `Drop`-implementing type
                    true
                } else {
                    self.use_expr_must_clone(lhs)
                }
            }

            ExprKind::Deref { .. } | ExprKind::UpvarRef { .. } | ExprKind::VarRef { .. } => true,

            ExprKind::If { .. }
            | ExprKind::Call { .. }
            | ExprKind::ByUse { .. }
            | ExprKind::Binary { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Cast { .. }
            | ExprKind::ValueExpr { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::PointerCoercion { .. }
            | ExprKind::Loop { .. }
            | ExprKind::LoopMatch { .. }
            | ExprKind::Let { .. }
            | ExprKind::Match { .. }
            | ExprKind::Block { .. }
            | ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::RawBorrow { .. }
            | ExprKind::Break { .. }
            | ExprKind::Continue { .. }
            | ExprKind::ConstContinue { .. }
            | ExprKind::Return { .. }
            | ExprKind::Become { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::RepeatUse { .. }
            | ExprKind::Array { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Adt { .. }
            | ExprKind::ValueTypeAscription { .. }
            | ExprKind::ValueUnwrapUnsafeBinder { .. }
            | ExprKind::WrapUnsafeBinder { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Literal { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::ThreadLocalRef { .. }
            | ExprKind::Yield { .. }
            | ExprKind::Reborrow { .. } => false,
        }
    }
}
