//! Miscellaneous builder routines that are not specific to building any particular
//! kind of thing.

use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::debug;

use crate::builder::Builder;

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
        let literal = Const::from_bits(self.tcx, 0, ty::TypingEnv::fully_monomorphized(), ty);

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
}
