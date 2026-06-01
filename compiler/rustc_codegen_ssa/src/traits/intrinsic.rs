use rustc_middle::ty;
use rustc_span::Span;

use super::BackendTypes;
use crate::RetagInfo;
use crate::mir::IntrinsicResult;
use crate::mir::operand::OperandRef;
use crate::mir::place::{PlaceRef, PlaceValue};

pub trait IntrinsicCallBuilderMethods<'tcx>: BackendTypes {
    /// Higher-level interface to emitting calls to intrinsics
    ///
    /// Remember to add all intrinsics here, in `compiler/rustc_hir_analysis/src/check/mod.rs`,
    /// and in `library/core/src/intrinsics.rs`; if you need access to any LLVM intrinsics,
    /// add them to `compiler/rustc_codegen_llvm/src/context.rs`.
    /// Returns `Fallback` if another instance should be called instead. This is used to invoke
    /// intrinsic default bodies in case an intrinsic is not implemented by the backend.
    ///
    /// The `result_place` will be provided for things that weren't `LocalKind::SSA`.
    /// If you need it for more things, see `intrinsic_call_expects_place_always`.
    ///
    /// NOTE: allowed to call [`BuilderMethods::call`]
    ///
    /// [`BuilderMethods::call`]: super::builder::BuilderMethods::call
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, Self::Value>],
        result_layout: ty::layout::TyAndLayout<'tcx>,
        result_place: Option<PlaceValue<Self::Value>>,
        span: Span,
    ) -> IntrinsicResult<'tcx, Self::Value>;

    fn codegen_offload_preload_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, Self::Value>],
        is_mut: bool,
    );

    fn codegen_offload_preload_mut_drop(
        &mut self,
        preload_ty: ty::Ty<'tcx>,
        place: PlaceRef<'tcx, Self::Value>,
    );

    fn codegen_llvm_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, Self::Value>],
        is_cleanup: bool,
    ) -> Self::Value;

    fn abort(&mut self);
    fn assume(&mut self, val: Self::Value);
    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value;
    /// Trait method used to load a function while testing if it is associated with a type
    /// identifier.
    fn type_checked_load(
        &mut self,
        llvtable: Self::Value,
        vtable_byte_offset: u64,
        typeid: &[u8],
    ) -> Self::Value;
    /// Trait method used to inject `va_start` on the "spoofed" `VaList` in
    /// Rust defined C-variadic functions.
    fn va_start(&mut self, val: Self::Value);
    /// Trait method used to inject `va_end` on the "spoofed" `VaList` before
    /// Rust defined C-variadic functions return.
    fn va_end(&mut self, val: Self::Value);
    /// Trait method used to retag a pointer stored within a place.
    fn retag_mem(&mut self, place: Self::Value, info: &RetagInfo<Self::Value>);
    /// Trait method used to retag a pointer that has been loaded into a register.
    fn retag_reg(&mut self, ptr: Self::Value, info: &RetagInfo<Self::Value>) -> Self::Value;
}
