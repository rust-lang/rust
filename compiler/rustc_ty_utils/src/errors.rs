//! Errors emitted by ty_utils

use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(ty_utils_needs_drop_overflow)]
pub struct NeedsDropOverflow<'tcx> {
    pub query_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(ty_utils_generic_constant_too_complex)]
#[help]
pub struct GenericConstantTooComplex {
    #[primary_span]
    pub span: Span,
    #[note(ty_utils_maybe_supported)]
    pub maybe_supported: Option<()>,
    #[subdiagnostic]
    pub sub: GenericConstantTooComplexSub,
}

#[derive(Subdiagnostic)]
pub enum GenericConstantTooComplexSub {
    #[label(ty_utils_borrow_not_supported)]
    BorrowNotSupported(#[primary_span] Span),
    #[label(ty_utils_address_and_deref_not_supported)]
    AddressAndDerefNotSupported(#[primary_span] Span),
    #[label(ty_utils_array_not_supported)]
    ArrayNotSupported(#[primary_span] Span),
    #[label(ty_utils_block_not_supported)]
    BlockNotSupported(#[primary_span] Span),
    #[label(ty_utils_never_to_any_not_supported)]
    NeverToAnyNotSupported(#[primary_span] Span),
    #[label(ty_utils_tuple_not_supported)]
    TupleNotSupported(#[primary_span] Span),
    #[label(ty_utils_index_not_supported)]
    IndexNotSupported(#[primary_span] Span),
    #[label(ty_utils_field_not_supported)]
    FieldNotSupported(#[primary_span] Span),
    #[label(ty_utils_const_block_not_supported)]
    ConstBlockNotSupported(#[primary_span] Span),
    #[label(ty_utils_adt_not_supported)]
    AdtNotSupported(#[primary_span] Span),
    #[label(ty_utils_pointer_not_supported)]
    PointerNotSupported(#[primary_span] Span),
    #[label(ty_utils_yield_not_supported)]
    YieldNotSupported(#[primary_span] Span),
    #[label(ty_utils_loop_not_supported)]
    LoopNotSupported(#[primary_span] Span),
    #[label(ty_utils_box_not_supported)]
    BoxNotSupported(#[primary_span] Span),
    #[label(ty_utils_binary_not_supported)]
    BinaryNotSupported(#[primary_span] Span),
    #[label(ty_utils_logical_op_not_supported)]
    LogicalOpNotSupported(#[primary_span] Span),
    #[label(ty_utils_assign_not_supported)]
    AssignNotSupported(#[primary_span] Span),
    #[label(ty_utils_closure_and_return_not_supported)]
    ClosureAndReturnNotSupported(#[primary_span] Span),
    #[label(ty_utils_control_flow_not_supported)]
    ControlFlowNotSupported(#[primary_span] Span),
    #[label(ty_utils_inline_asm_not_supported)]
    InlineAsmNotSupported(#[primary_span] Span),
    #[label(ty_utils_operation_not_supported)]
    OperationNotSupported(#[primary_span] Span),
}
