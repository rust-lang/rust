//! Experimental support for emitting retags as function calls in generated code.

use rustc_middle::mir::{Rvalue, WithRetag};

use crate::mir::FunctionCx;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use crate::traits::BuilderMethods;

pub(crate) fn rvalue_needs_retag(rvalue: &Rvalue<'_>) -> bool {
    // `Ref` has its own internal retagging
    !matches!(rvalue, Rvalue::Ref(..)) && !matches!(rvalue, Rvalue::Use(.., WithRetag::No))
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'_, 'a, 'tcx, Bx> {
    /// Retags the pointers within an [`OperandRef`].
    pub(crate) fn codegen_retag_operand(
        &mut self,
        _bx: &mut Bx,
        operand: OperandRef<'tcx, Bx::Value>,
        _is_fn_entry: bool,
    ) -> OperandRef<'tcx, Bx::Value> {
        operand
    }

    /// Retags the pointers within a [`PlaceRef`].
    pub(crate) fn codegen_retag_place(
        &mut self,
        _bx: &mut Bx,
        _place_ref: PlaceRef<'tcx, Bx::Value>,
        _is_fn_entry: bool,
    ) {
    }
}
