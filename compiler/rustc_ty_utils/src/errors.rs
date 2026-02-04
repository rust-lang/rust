//! Errors emitted by ty_utils

use rustc_errors::codes::*;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::{GenericArg, Ty};
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag("overflow while checking whether `{$query_ty}` requires drop")]
pub(crate) struct NeedsDropOverflow<'tcx> {
    pub query_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("overly complex generic constant")]
#[help("consider moving this anonymous constant into a `const` function")]
pub(crate) struct GenericConstantTooComplex {
    #[primary_span]
    pub span: Span,
    #[note("this operation may be supported in the future")]
    pub maybe_supported: bool,
    #[subdiagnostic]
    pub sub: GenericConstantTooComplexSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum GenericConstantTooComplexSub {
    #[label("borrowing is not supported in generic constants")]
    BorrowNotSupported(#[primary_span] Span),
    #[label("dereferencing or taking the address is not supported in generic constants")]
    AddressAndDerefNotSupported(#[primary_span] Span),
    #[label("array construction is not supported in generic constants")]
    ArrayNotSupported(#[primary_span] Span),
    #[label("blocks are not supported in generic constants")]
    BlockNotSupported(#[primary_span] Span),
    #[label("coercing the `never` type is not supported in generic constants")]
    NeverToAnyNotSupported(#[primary_span] Span),
    #[label("tuple construction is not supported in generic constants")]
    TupleNotSupported(#[primary_span] Span),
    #[label("indexing is not supported in generic constants")]
    IndexNotSupported(#[primary_span] Span),
    #[label("field access is not supported in generic constants")]
    FieldNotSupported(#[primary_span] Span),
    #[label("const blocks are not supported in generic constants")]
    ConstBlockNotSupported(#[primary_span] Span),
    #[label("struct/enum construction is not supported in generic constants")]
    AdtNotSupported(#[primary_span] Span),
    #[label("pointer casts are not allowed in generic constants")]
    PointerNotSupported(#[primary_span] Span),
    #[label("coroutine control flow is not allowed in generic constants")]
    YieldNotSupported(#[primary_span] Span),
    #[label("loops and loop control flow are not supported in generic constants")]
    LoopNotSupported(#[primary_span] Span),
    #[label("allocations are not allowed in generic constants")]
    BoxNotSupported(#[primary_span] Span),
    #[label("unsupported binary operation in generic constants")]
    BinaryNotSupported(#[primary_span] Span),
    #[label(".use is not allowed in generic constants")]
    ByUseNotSupported(#[primary_span] Span),
    #[label(
        "unsupported operation in generic constants, short-circuiting operations would imply control flow"
    )]
    LogicalOpNotSupported(#[primary_span] Span),
    #[label("assignment is not supported in generic constants")]
    AssignNotSupported(#[primary_span] Span),
    #[label("closures and function keywords are not supported in generic constants")]
    ClosureAndReturnNotSupported(#[primary_span] Span),
    #[label("control flow is not supported in generic constants")]
    ControlFlowNotSupported(#[primary_span] Span),
    #[label("assembly is not supported in generic constants")]
    InlineAsmNotSupported(#[primary_span] Span),
    #[label("unsupported operation in generic constants")]
    OperationNotSupported(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("`FnPtr` trait with unexpected associated item")]
pub(crate) struct UnexpectedFnPtrAssociatedItem {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "monomorphising SIMD type `{$ty}` with a non-primitive-scalar (integer/float/pointer) element type `{$e_ty}`"
)]
pub(crate) struct NonPrimitiveSimdType<'tcx> {
    pub ty: Ty<'tcx>,
    pub e_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("non-defining opaque type use in defining scope")]
pub(crate) struct DuplicateArg<'tcx> {
    pub arg: GenericArg<'tcx>,
    #[primary_span]
    #[label("generic argument `{$arg}` used twice")]
    pub span: Span,
    #[note("for this opaque type")]
    pub opaque_span: Span,
}

#[derive(Diagnostic)]
#[diag("non-defining opaque type use in defining scope", code = E0792)]
pub(crate) struct NotParam<'tcx> {
    pub arg: GenericArg<'tcx>,
    #[primary_span]
    #[label("argument `{$arg}` is not a generic parameter")]
    pub span: Span,
    #[note("for this opaque type")]
    pub opaque_span: Span,
}
