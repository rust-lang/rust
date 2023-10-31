use rustc_hir::LangItem;
use rustc_middle::mir;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext};
use rustc_span::Span;

use super::FunctionCx;
use crate::base;
use crate::common;
use crate::mir::OperandValue;
use crate::traits::*;

pub fn pointers_to_check<F>(
    statement: &mir::Statement<'_>,
    required_align_of: F,
) -> Vec<(mir::Local, u64)>
where
    F: Fn(mir::Local) -> Option<u64>,
{
    let mut finder = PointerFinder { required_align_of, pointers: Vec::new() };
    finder.visit_statement(statement, rustc_middle::mir::Location::START);
    finder.pointers
}

struct PointerFinder<F> {
    pointers: Vec<(mir::Local, u64)>,
    required_align_of: F,
}

impl<'tcx, F> Visitor<'tcx> for PointerFinder<F>
where
    F: Fn(mir::Local) -> Option<u64>,
{
    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        context: PlaceContext,
        location: mir::Location,
    ) {
        // We want to only check reads and writes to Places, so we specifically exclude
        // Borrows and AddressOf.
        match context {
            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::Drop,
            ) => {}
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
            ) => {}
            _ => {
                return;
            }
        }

        if !place.is_indirect() {
            return;
        }

        let pointer = place.local;
        let Some(required_alignment) = (self.required_align_of)(pointer) else {
            return;
        };

        if required_alignment == 1 {
            return;
        }

        // Ensure that this place is based on an aligned pointer.
        self.pointers.push((pointer, required_alignment));

        self.super_place(place, context, location);
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "debug", skip(self, bx))]
    pub fn codegen_alignment_check(
        &mut self,
        bx: &mut Bx,
        pointer: mir::Operand<'tcx>,
        required_alignment: u64,
        source_info: mir::SourceInfo,
    ) {
        // Compute the alignment mask
        let mask = bx.const_usize(required_alignment - 1);
        let zero = bx.const_usize(0);
        let required_alignment = bx.const_usize(required_alignment);

        // And the pointer with the mask
        let pointer = match self.codegen_operand(bx, &pointer).val {
            OperandValue::Immediate(imm) => imm,
            OperandValue::Pair(ptr, _) => ptr,
            _ => {
                unreachable!("{pointer:?}");
            }
        };
        let addr = bx.ptrtoint(pointer, bx.cx().type_isize());
        let masked = bx.and(addr, mask);

        // Branch on whether the masked value is zero
        let is_zero = bx.icmp(
            base::bin_op_to_icmp_predicate(mir::BinOp::Eq.to_hir_binop(), false),
            masked,
            zero,
        );

        // Create destination blocks, branching on is_zero
        let panic = bx.append_sibling_block("panic");
        let success = bx.append_sibling_block("success");
        bx.cond_br(is_zero, success, panic);

        // Switch to the failure block and codegen a call to the panic intrinsic
        bx.switch_to_block(panic);
        self.set_debug_loc(bx, source_info);
        let location = self.get_caller_location(bx, source_info).immediate();
        self.codegen_nounwind_panic(
            bx,
            LangItem::PanicMisalignedPointerDereference,
            &[required_alignment, addr, location],
            source_info.span,
        );

        // Continue codegen in the success block.
        bx.switch_to_block(success);
        self.set_debug_loc(bx, source_info);
    }

    /// Emit a call to a diverging and `rustc_nounwind` panic helper.
    #[instrument(level = "debug", skip(self, bx))]
    fn codegen_nounwind_panic(
        &mut self,
        bx: &mut Bx,
        lang_item: LangItem,
        args: &[Bx::Value],
        span: Span,
    ) {
        let (fn_abi, fn_ptr, instance) = common::build_langcall(bx, Some(span), lang_item);
        let fn_ty = bx.fn_decl_backend_type(&fn_abi);
        let fn_attrs = if bx.tcx().def_kind(self.instance.def_id()).has_codegen_attrs() {
            Some(bx.tcx().codegen_fn_attrs(self.instance.def_id()))
        } else {
            None
        };

        // bx.call requires that the call not unwind. Double-check that this LangItem can't unwind.
        assert!(!fn_abi.can_unwind);

        bx.call(
            fn_ty,
            fn_attrs,
            Some(&fn_abi),
            fn_ptr,
            args,
            None, /* funclet */
            Some(instance),
        );
        bx.unreachable();
    }
}
