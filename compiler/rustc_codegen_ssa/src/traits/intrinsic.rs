use super::BackendTypes;
use crate::mir::operand::OperandRef;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use rustc_target::abi::call::FnAbi;

pub trait IntrinsicCallMethods<'tcx>: BackendTypes {
    /// Remember to add all intrinsics here, in `compiler/rustc_typeck/src/check/mod.rs`,
    /// and in `library/core/src/intrinsics.rs`; if you need access to any LLVM intrinsics,
    /// add them to `compiler/rustc_codegen_llvm/src/context.rs`.
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: Span,
    );

    fn abort(&mut self);
    fn assume(&mut self, val: Self::Value);
    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value;
    /// Emits a forced side effect.
    ///
    /// Currently has any effect only when LLVM versions prior to 12.0 are used as the backend.
    fn sideeffect(&mut self);
    /// Trait method used to test whether a given pointer is associated with a type identifier.
    fn type_test(&mut self, pointer: Self::Value, typeid: Self::Value) -> Self::Value;
    /// Trait method used to inject `va_start` on the "spoofed" `VaListImpl` in
    /// Rust defined C-variadic functions.
    fn va_start(&mut self, val: Self::Value) -> Self::Value;
    /// Trait method used to inject `va_end` on the "spoofed" `VaListImpl` before
    /// Rust defined C-variadic functions return.
    fn va_end(&mut self, val: Self::Value) -> Self::Value;
}
