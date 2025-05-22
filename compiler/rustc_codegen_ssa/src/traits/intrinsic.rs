use rustc_middle::ty;
use rustc_span::Span;

use super::BackendTypes;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;

pub trait IntrinsicCallBuilderMethods<'tcx>: BackendTypes {
    /// Remember to add all intrinsics here, in `compiler/rustc_hir_analysis/src/check/mod.rs`,
    /// and in `library/core/src/intrinsics.rs`; if you need access to any LLVM intrinsics,
    /// add them to `compiler/rustc_codegen_llvm/src/context.rs`.
    /// Returns `Err` if another instance should be called instead. This is used to invoke
    /// intrinsic default bodies in case an intrinsic is not implemented by the backend.
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, Self::Value>],
        result: PlaceRef<'tcx, Self::Value>,
        span: Span,
    ) -> Result<(), ty::Instance<'tcx>>;

    fn abort(&mut self);
    fn assume(&mut self, val: Self::Value);
    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value;
    /// Trait method used to test whether a given pointer is associated with a type identifier.
    fn type_test(&mut self, pointer: Self::Value, typeid: Self::Metadata) -> Self::Value;
    /// Trait method used to load a function while testing if it is associated with a type
    /// identifier.
    fn type_checked_load(
        &mut self,
        llvtable: Self::Value,
        vtable_byte_offset: u64,
        typeid: Self::Metadata,
    ) -> Self::Value;
    /// Trait method used to inject `va_start` on the "spoofed" `VaListImpl` in
    /// Rust defined C-variadic functions.
    fn va_start(&mut self, val: Self::Value) -> Self::Value;
    /// Trait method used to inject `va_end` on the "spoofed" `VaListImpl` before
    /// Rust defined C-variadic functions return.
    fn va_end(&mut self, val: Self::Value) -> Self::Value;
}
