//! Errors emitted by ty_utils

use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(ty_utils::needs_drop_overflow)]
pub struct NeedsDropOverflow<'tcx> {
    pub query_ty: Ty<'tcx>,
}

#[derive(SessionDiagnostic)]
#[diag(ty_utils::generic_constant_too_complex)]
#[help]
pub struct GenericConstantTooComplex {
    #[primary_span]
    pub span: Span,
    #[note(ty_utils::maybe_supported)]
    pub maybe_supported: Option<()>,
    #[subdiagnostic]
    pub sub: GenericConstantTooComplexSub,
}

#[derive(SessionSubdiagnostic)]
pub enum GenericConstantTooComplexSub {
    #[label(ty_utils::borrow_not_supported)]
    BorrowNotSupported(#[primary_span] Span),
    #[label(ty_utils::address_and_deref_not_supported)]
    AddressAndDerefNotSupported(#[primary_span] Span),
    #[label(ty_utils::array_not_supported)]
    ArrayNotSupported(#[primary_span] Span),
    #[label(ty_utils::block_not_supported)]
    BlockNotSupported(#[primary_span] Span),
    #[label(ty_utils::never_to_any_not_supported)]
    NeverToAnyNotSupported(#[primary_span] Span),
    #[label(ty_utils::tuple_not_supported)]
    TupleNotSupported(#[primary_span] Span),
    #[label(ty_utils::index_not_supported)]
    IndexNotSupported(#[primary_span] Span),
    #[label(ty_utils::field_not_supported)]
    FieldNotSupported(#[primary_span] Span),
    #[label(ty_utils::const_block_not_supported)]
    ConstBlockNotSupported(#[primary_span] Span),
    #[label(ty_utils::adt_not_supported)]
    AdtNotSupported(#[primary_span] Span),
    #[label(ty_utils::pointer_not_supported)]
    PointerNotSupported(#[primary_span] Span),
    #[label(ty_utils::yield_not_supported)]
    YieldNotSupported(#[primary_span] Span),
    #[label(ty_utils::loop_not_supported)]
    LoopNotSupported(#[primary_span] Span),
    #[label(ty_utils::box_not_supported)]
    BoxNotSupported(#[primary_span] Span),
    #[label(ty_utils::binary_not_supported)]
    BinaryNotSupported(#[primary_span] Span),
    #[label(ty_utils::logical_op_not_supported)]
    LogicalOpNotSupported(#[primary_span] Span),
    #[label(ty_utils::assign_not_supported)]
    AssignNotSupported(#[primary_span] Span),
    #[label(ty_utils::closure_and_return_not_supported)]
    ClosureAndReturnNotSupported(#[primary_span] Span),
    #[label(ty_utils::control_flow_not_supported)]
    ControlFlowNotSupported(#[primary_span] Span),
    #[label(ty_utils::inline_asm_not_supported)]
    InlineAsmNotSupported(#[primary_span] Span),
    #[label(ty_utils::operation_not_supported)]
    OperationNotSupported(#[primary_span] Span),
}
