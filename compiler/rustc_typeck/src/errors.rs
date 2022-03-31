//! Errors emitted by typeck.
use rustc_macros::SessionDiagnostic;
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionDiagnostic)]
#[error(code = "E0062", slug = "typeck-field-multiply-specified-in-initializer")]
pub struct FieldMultiplySpecifiedInInitializer {
    #[message]
    #[label = "used more than once"]
    pub span: Span,
    #[label = "first use of `{ident}`"]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0092", slug = "typeck-unrecognized-atomic-operation")]
pub struct UnrecognizedAtomicOperation<'a> {
    #[message]
    #[label = "unrecognized atomic operation"]
    pub span: Span,
    pub op: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0094", slug = "typeck-wrong-number-of-generic-arguments-to-intrinsic")]
pub struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[message]
    #[label = "expected {expected} {descr} parameter{expected_pluralize}"]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub expected_pluralize: &'a str,
    pub descr: &'a str,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0093", slug = "typeck-unrecognized-intrinsic-function")]
pub struct UnrecognizedIntrinsicFunction {
    #[message]
    #[label = "unrecognized intrinsic"]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0195", slug = "typeck-lifetimes-or-bounds-mismatch-on-trait")]
pub struct LifetimesOrBoundsMismatchOnTrait {
    #[message]
    #[label = "lifetimes do not match {item_kind} in trait"]
    pub span: Span,
    #[label = "lifetimes in impl do not match this {item_kind} in trait"]
    pub generics_span: Option<Span>,
    pub item_kind: &'static str,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0120", slug = "typeck-drop-impl-on-wrong-item")]
pub struct DropImplOnWrongItem {
    #[message]
    #[label = "must be a struct, enum, or union"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0124", slug = "typeck-field-already-declared")]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[message]
    #[label = "field already declared"]
    pub span: Span,
    #[label = "`{field_name}` first declared here"]
    pub prev_span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0184", slug = "typeck-copy-impl-on-type-with-dtor")]
pub struct CopyImplOnTypeWithDtor {
    #[message]
    #[label = "Copy not allowed on types with destructors"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0203", slug = "typeck-multiple-relaxed-default-bounds")]
pub struct MultipleRelaxedDefaultBounds {
    #[message]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0206", slug = "typeck-copy-impl-on-non-adt")]
pub struct CopyImplOnNonAdt {
    #[message]
    #[label = "type is not a structure or enumeration"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0224", slug = "typeck-trait-object-declared-with-no-traits")]
pub struct TraitObjectDeclaredWithNoTraits {
    #[message]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0227", slug = "typeck-ambiguous-lifetime-bound")]
pub struct AmbiguousLifetimeBound {
    #[message]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0229", slug = "typeck-assoc-type-binding-not-allowed")]
pub struct AssocTypeBindingNotAllowed {
    #[message]
    #[label = "associated type not allowed here"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0436", slug = "typeck-functional-record-update-on-non-struct")]
pub struct FunctionalRecordUpdateOnNonStruct {
    #[message]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0516", slug = "typeck-typeof-reserved-keyword-used")]
pub struct TypeofReservedKeywordUsed {
    #[message]
    #[label = "reserved keyword"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0572", slug = "typeck-return-stmt-outside-of-fn-body")]
pub struct ReturnStmtOutsideOfFnBody {
    #[message]
    pub span: Span,
    #[label = "the return is part of this body..."]
    pub encl_body_span: Option<Span>,
    #[label = "...not the enclosing function body"]
    pub encl_fn_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0627", slug = "typeck-yield-expr-outside-of-generator")]
pub struct YieldExprOutsideOfGenerator {
    #[message]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0639", slug = "typeck-struct-expr-non-exhaustive")]
pub struct StructExprNonExhaustive {
    #[message]
    pub span: Span,
    pub what: &'static str,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0699", slug = "typeck-method-call-on-unknown-type")]
pub struct MethodCallOnUnknownType {
    #[message]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0719", slug = "typeck-value-of-associated-struct-already-specified")]
pub struct ValueOfAssociatedStructAlreadySpecified {
    #[message]
    #[label = "re-bound here"]
    pub span: Span,
    #[label = "`{item_name}` bound here first"]
    pub prev_span: Span,
    pub item_name: Ident,
    pub def_path: String,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0745", slug = "typeck-address-of-temporary-taken")]
pub struct AddressOfTemporaryTaken {
    #[message]
    #[label = "temporary value"]
    pub span: Span,
}
