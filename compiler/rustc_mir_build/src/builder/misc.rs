//! Miscellaneous builder routines that are not specific to building any particular
//! kind of thing.

use rustc_hir::LangItem;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::debug;

use super::{BlockAnd, BlockAndExtension};
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
        self.cfg.push_assign_constant(block, source_info, temp, ConstOperand {
            span: source_info.span,
            user_ty: None,
            const_: Const::from_usize(self.tcx, value),
        });
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

    pub(crate) fn call_intrinsic(
        &mut self,
        block: BasicBlock,
        span: Span,
        intrinsic: LangItem,
        type_args: &[Ty<'tcx>],
        args: Box<[Spanned<Operand<'tcx>>]>,
        output: Place<'tcx>,
    ) -> BlockAnd<()> {
        let tcx = self.tcx;
        let source_info = self.source_info(span);
        let func = Operand::function_handle(
            tcx,
            tcx.require_lang_item(intrinsic, Some(span)),
            type_args.iter().copied().map(Into::into),
            span,
        );

        let next_block = self.cfg.start_new_block();
        self.cfg.terminate(block, source_info, TerminatorKind::Call {
            func,
            args,
            destination: output,
            target: Some(next_block),
            unwind: UnwindAction::Continue,
            call_source: CallSource::Misc,
            fn_span: span,
        });

        next_block.unit()
    }
}

pub(crate) trait SpannedCallOperandsExt<'tcx> {
    fn args(&self, list: impl IntoIterator<Item = Operand<'tcx>>) -> Box<[Spanned<Operand<'tcx>>]>;
}

impl<'tcx> SpannedCallOperandsExt<'tcx> for Span {
    fn args(&self, list: impl IntoIterator<Item = Operand<'tcx>>) -> Box<[Spanned<Operand<'tcx>>]> {
        list.into_iter().map(move |arg| Spanned { node: arg, span: *self }).collect()
    }
}
