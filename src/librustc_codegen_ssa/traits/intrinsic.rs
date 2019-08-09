use super::BackendTypes;
use crate::mir::operand::OperandRef;
use rustc::ty::Ty;
use rustc_target::abi::call::FnType;
use syntax_pos::Span;

pub trait IntrinsicCallMethods<'tcx>: BackendTypes {
    /// Remember to add all intrinsics here, in librustc_typeck/check/mod.rs,
    /// and in libcore/intrinsics.rs; if you need access to any llvm intrinsics,
    /// add them to librustc_codegen_llvm/context.rs
    fn codegen_intrinsic_call(
        &mut self,
        callee_ty: Ty<'tcx>,
        fn_ty: &FnType<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: Span,
    );

    fn abort(&mut self);
    fn assume(&mut self, val: Self::Value);
    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value;
    /// Trait method used to inject `va_start` on the "spoofed" `VaListImpl` in
    /// Rust defined C-variadic functions.
    fn va_start(&mut self, val: Self::Value) -> Self::Value;
    /// Trait method used to inject `va_end` on the "spoofed" `VaListImpl` before
    /// Rust defined C-variadic functions return.
    fn va_end(&mut self, val: Self::Value) -> Self::Value;
}
