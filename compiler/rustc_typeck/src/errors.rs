//! Errors emitted by typeck.
use rustc_macros::SessionDiagnostic;
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionDiagnostic)]
#[error = "E0062"]
pub struct FieldMultiplySpecifiedInInitializer {
    #[message = "field `{ident}` specified more than once"]
    #[label = "used more than once"]
    pub span: Span,
    #[label = "first use of `{ident}`"]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[error = "E0092"]
pub struct UnrecognizedAtomicOperation<'a> {
    #[message = "unrecognized atomic operation function: `{op}`"]
    #[label = "unrecognized atomic operation"]
    pub span: Span,
    pub op: &'a str,
}

#[derive(SessionDiagnostic)]
#[error = "E0094"]
pub struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[message = "intrinsic has wrong number of {descr} \
                         parameters: found {found}, expected {expected}"]
    #[label = "expected {expected} {descr} parameter{expected_pluralize}"]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub expected_pluralize: &'a str,
    pub descr: &'a str,
}

#[derive(SessionDiagnostic)]
#[error = "E0093"]
pub struct UnrecognizedIntrinsicFunction {
    #[message = "unrecognized intrinsic function: `{name}`"]
    #[label = "unrecognized intrinsic"]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[error = "E0195"]
pub struct LifetimesOrBoundsMismatchOnTrait {
    #[message = "lifetime parameters or bounds on {item_kind} `{ident}` do not match the trait declaration"]
    #[label = "lifetimes do not match {item_kind} in trait"]
    pub span: Span,
    #[label = "lifetimes in impl do not match this {item_kind} in trait"]
    pub generics_span: Option<Span>,
    pub item_kind: &'static str,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[error = "E0120"]
pub struct DropImplOnWrongItem {
    #[message = "the `Drop` trait may only be implemented for structs, enums, and unions"]
    #[label = "must be a struct, enum, or union"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0124"]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[message = "field `{field_name}` is already declared"]
    #[label = "field already declared"]
    pub span: Span,
    #[label = "`{field_name}` first declared here"]
    pub prev_span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0184"]
pub struct CopyImplOnTypeWithDtor {
    #[message = "the trait `Copy` may not be implemented for this type; the \
                              type has a destructor"]
    #[label = "Copy not allowed on types with destructors"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0203"]
pub struct MultipleRelaxedDefaultBounds {
    #[message = "type parameter has more than one relaxed default bound, only one is supported"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0206"]
pub struct CopyImplOnNonAdt {
    #[message = "the trait `Copy` may not be implemented for this type"]
    #[label = "type is not a structure or enumeration"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0224"]
pub struct TraitObjectDeclaredWithNoTraits {
    #[message = "at least one trait is required for an object type"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0227"]
pub struct AmbiguousLifetimeBound {
    #[message = "ambiguous lifetime bound, explicit lifetime bound required"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0229"]
pub struct AssocTypeBindingNotAllowed {
    #[message = "associated type bindings are not allowed here"]
    #[label = "associated type not allowed here"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0436"]
pub struct FunctionalRecordUpdateOnNonStruct {
    #[message = "functional record update syntax requires a struct"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0516"]
pub struct TypeofReservedKeywordUsed {
    #[message = "`typeof` is a reserved keyword but unimplemented"]
    #[label = "reserved keyword"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0572"]
pub struct ReturnStmtOutsideOfFnBody {
    #[message = "return statement outside of function body"]
    pub span: Span,
    #[label = "the return is part of this body..."]
    pub encl_body_span: Option<Span>,
    #[label = "...not the enclosing function body"]
    pub encl_fn_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[error = "E0627"]
pub struct YieldExprOutsideOfGenerator {
    #[message = "yield expression outside of generator literal"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0639"]
pub struct StructExprNonExhaustive {
    #[message = "cannot create non-exhaustive {what} using struct expression"]
    pub span: Span,
    pub what: &'static str,
}

#[derive(SessionDiagnostic)]
#[error = "E0699"]
pub struct MethodCallOnUnknownType {
    #[message = "the type of this value must be known to call a method on a raw pointer on it"]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0719"]
pub struct ValueOfAssociatedStructAlreadySpecified {
    #[message = "the value of the associated type `{item_name}` (from trait `{def_path}`) is already specified"]
    #[label = "re-bound here"]
    pub span: Span,
    #[label = "`{item_name}` bound here first"]
    pub prev_span: Span,
    pub item_name: Ident,
    pub def_path: String,
}

#[derive(SessionDiagnostic)]
#[error = "E0745"]
pub struct AddressOfTemporaryTaken {
    #[message = "cannot take address of a temporary"]
    #[label = "temporary value"]
    pub span: Span,
}
