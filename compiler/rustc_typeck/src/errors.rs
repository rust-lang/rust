//! Errors emitted by typeck.
use rustc_errors::Applicability;
use rustc_macros::SessionDiagnostic;
use rustc_middle::ty::Ty;
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionDiagnostic)]
#[error(code = "E0062", slug = "typeck-field-multiply-specified-in-initializer")]
pub struct FieldMultiplySpecifiedInInitializer {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label = "previous-use-label"]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0092", slug = "typeck-unrecognized-atomic-operation")]
pub struct UnrecognizedAtomicOperation<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub op: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0094", slug = "typeck-wrong-number-of-generic-arguments-to-intrinsic")]
pub struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub descr: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0093", slug = "typeck-unrecognized-intrinsic-function")]
pub struct UnrecognizedIntrinsicFunction {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0195", slug = "typeck-lifetimes-or-bounds-mismatch-on-trait")]
pub struct LifetimesOrBoundsMismatchOnTrait {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label = "generics-label"]
    pub generics_span: Option<Span>,
    pub item_kind: &'static str,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0120", slug = "typeck-drop-impl-on-wrong-item")]
pub struct DropImplOnWrongItem {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0124", slug = "typeck-field-already-declared")]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[primary_span]
    #[label]
    pub span: Span,
    #[label = "previous-decl-label"]
    pub prev_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0184", slug = "typeck-copy-impl-on-type-with-dtor")]
pub struct CopyImplOnTypeWithDtor {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0203", slug = "typeck-multiple-relaxed-default-bounds")]
pub struct MultipleRelaxedDefaultBounds {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0206", slug = "typeck-copy-impl-on-non-adt")]
pub struct CopyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0224", slug = "typeck-trait-object-declared-with-no-traits")]
pub struct TraitObjectDeclaredWithNoTraits {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0227", slug = "typeck-ambiguous-lifetime-bound")]
pub struct AmbiguousLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0229", slug = "typeck-assoc-type-binding-not-allowed")]
pub struct AssocTypeBindingNotAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0436", slug = "typeck-functional-record-update-on-non-struct")]
pub struct FunctionalRecordUpdateOnNonStruct {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0516", slug = "typeck-typeof-reserved-keyword-used")]
pub struct TypeofReservedKeywordUsed<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion_verbose(message = "suggestion", code = "{ty}")]
    pub opt_sugg: Option<(Span, Applicability)>,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0572", slug = "typeck-return-stmt-outside-of-fn-body")]
pub struct ReturnStmtOutsideOfFnBody {
    #[primary_span]
    pub span: Span,
    #[label = "encl-body-label"]
    pub encl_body_span: Option<Span>,
    #[label = "encl-fn-label"]
    pub encl_fn_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0627", slug = "typeck-yield-expr-outside-of-generator")]
pub struct YieldExprOutsideOfGenerator {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0639", slug = "typeck-struct-expr-non-exhaustive")]
pub struct StructExprNonExhaustive {
    #[primary_span]
    pub span: Span,
    pub what: &'static str,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0699", slug = "typeck-method-call-on-unknown-type")]
pub struct MethodCallOnUnknownType {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0719", slug = "typeck-value-of-associated-struct-already-specified")]
pub struct ValueOfAssociatedStructAlreadySpecified {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label = "previous-bound-label"]
    pub prev_span: Span,
    pub item_name: Ident,
    pub def_path: String,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0745", slug = "typeck-address-of-temporary-taken")]
pub struct AddressOfTemporaryTaken {
    #[primary_span]
    #[label]
    pub span: Span,
}
