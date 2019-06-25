//! Miscellaneous builder routines that are not specific to building any particular
//! kind of thing.

use crate::build::Builder;

use rustc::ty::{self, Ty};

use rustc::mir::*;
use syntax_pos::{Span, DUMMY_SP};

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Adds a new temporary value of type `ty` storing the result of
    /// evaluating `expr`.
    ///
    /// N.B., **No cleanup is scheduled for this temporary.** You should
    /// call `schedule_drop` once the temporary is initialized.
    pub fn temp(&mut self, ty: Ty<'tcx>, span: Span) -> Place<'tcx> {
        let temp = self.local_decls.push(LocalDecl::new_temp(ty, span));
        let place = Place::from(temp);
        debug!("temp: created temp {:?} with type {:?}",
               place, self.local_decls[temp].ty);
        place
    }

    /// Convenience function for creating a literal operand, one
    /// without any user type annotation.
    pub fn literal_operand(&mut self,
                           span: Span,
                           ty: Ty<'tcx>,
                           literal: &'tcx ty::Const<'tcx>)
                           -> Operand<'tcx> {
        let constant = box Constant {
            span,
            ty,
            user_ty: None,
            literal,
        };
        Operand::Constant(constant)
    }

    pub fn unit_rvalue(&mut self) -> Rvalue<'tcx> {
        Rvalue::Aggregate(box AggregateKind::Tuple, vec![])
    }

    // Returns a zero literal operand for the appropriate type, works for
    // bool, char and integers.
    pub fn zero_literal(&mut self, span: Span, ty: Ty<'tcx>) -> Operand<'tcx> {
        let literal = ty::Const::from_bits(self.hir.tcx(), 0, ty::ParamEnv::empty().and(ty));

        self.literal_operand(span, ty, literal)
    }

    pub fn push_usize(&mut self,
                      block: BasicBlock,
                      source_info: SourceInfo,
                      value: u64)
                      -> Place<'tcx> {
        let usize_ty = self.hir.usize_ty();
        let temp = self.temp(usize_ty, source_info.span);
        self.cfg.push_assign_constant(
            block, source_info, &temp,
            Constant {
                span: source_info.span,
                ty: self.hir.usize_ty(),
                user_ty: None,
                literal: self.hir.usize_literal(value),
            });
        temp
    }

    pub fn consume_by_copy_or_move(&self, place: Place<'tcx>) -> Operand<'tcx> {
        let tcx = self.hir.tcx();
        let ty = place.ty(&self.local_decls, tcx).ty;
        if !self.hir.type_is_copy_modulo_regions(ty, DUMMY_SP) {
            Operand::Move(place)
        } else {
            Operand::Copy(place)
        }
    }
}
