use super::BackendTypes;
use crate::mir::operand::OperandRef;
use rustc_middle::mir::Operand;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_target::abi::call::FnAbi;

pub trait IntrinsicCallMethods<'tcx>: BackendTypes {
    /// Remember to add all intrinsics here, in librustc_typeck/check/mod.rs,
    /// and in libcore/intrinsics.rs; if you need access to any llvm intrinsics,
    /// add them to librustc_codegen_llvm/context.rs
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: Span,
        caller_instance: ty::Instance<'tcx>,
    );

    /// Intrinsic-specific pre-codegen processing, if any is required. Some intrinsics are handled
    /// at compile time and do not generate code. Returns true if codegen is required or false if
    /// the intrinsic does not need code generation.
    fn is_codegen_intrinsic(
        &mut self,
        intrinsic: &str,
        args: &Vec<Operand<'tcx>>,
        caller_instance: ty::Instance<'tcx>,
    ) -> bool;

    fn abort(&mut self);
    fn assume(&mut self, val: Self::Value);
    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value;
    fn sideeffect(&mut self);
    /// Trait method used to inject `va_start` on the "spoofed" `VaListImpl` in
    /// Rust defined C-variadic functions.
    fn va_start(&mut self, val: Self::Value) -> Self::Value;
    /// Trait method used to inject `va_end` on the "spoofed" `VaListImpl` before
    /// Rust defined C-variadic functions return.
    fn va_end(&mut self, val: Self::Value) -> Self::Value;
}
