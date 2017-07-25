//! This module contains everything needed to instantiate an interpreter.
//! This separation exists to ensure that no fancy miri features like
//! interpreting common C functions leak into CTFE.

use super::{
    EvalResult,
    EvalContext,
    Lvalue,
    PrimVal
};

use rustc::{mir, ty};

/// Methods of this trait signifies a point where CTFE evaluation would fail
/// and some use case dependent behaviour can instead be applied
pub trait Machine<'tcx>: Sized {
    /// Additional data that can be accessed via the EvalContext
    type Data;

    /// Additional data that can be accessed via the Memory
    type MemoryData;

    /// Called when a function's MIR is not found.
    /// This will happen for `extern "C"` functions.
    fn call_missing_fn<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        arg_operands: &[mir::Operand<'tcx>],
        sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx>;

    /// Called when operating on the value of pointers.
    ///
    /// Returns `None` if the operation should be handled by the integer
    /// op code
    ///
    /// Returns a (value, overflowed) pair otherwise
    fn ptr_op<'a>(
        ecx: &EvalContext<'a, 'tcx, Self>,
        bin_op: mir::BinOp,
        left: PrimVal,
        left_ty: ty::Ty<'tcx>,
        right: PrimVal,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>>;

    /// Called when adding a frame for a function that's not `const fn`
    ///
    /// Const eval returns `Err`, miri returns `Ok`
    fn check_non_const_fn_call(instance: ty::Instance<'tcx>) -> EvalResult<'tcx>;
}

