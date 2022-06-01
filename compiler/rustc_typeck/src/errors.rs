//! Errors emitted by typeck.
use rustc_errors::{error_code, Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::Ty;
use rustc_session::{parse::ParseSess, SessionDiagnostic};
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
    #[label = "alias-span"]
    pub trait_alias_span: Option<Span>,
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
    #[suggestion_verbose(code = "{ty}")]
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

#[derive(SessionSubdiagnostic)]
pub enum AddReturnTypeSuggestion<'tcx> {
    #[suggestion(
        slug = "typeck-add-return-type-add",
        code = "-> {found} ",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
        found: Ty<'tcx>,
    },
    #[suggestion(
        slug = "typeck-add-return-type-missing-here",
        code = "-> _ ",
        applicability = "has-placeholders"
    )]
    MissingHere {
        #[primary_span]
        span: Span,
    },
}

#[derive(SessionSubdiagnostic)]
pub enum ExpectedReturnTypeLabel<'tcx> {
    #[label(slug = "typeck-expected-default-return-type")]
    Unit {
        #[primary_span]
        span: Span,
    },
    #[label(slug = "typeck-expected-return-type")]
    Other {
        #[primary_span]
        span: Span,
        expected: Ty<'tcx>,
    },
}

#[derive(SessionDiagnostic)]
#[error(slug = "typeck-unconstrained-opaque-type")]
#[note]
pub struct UnconstrainedOpaqueType {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[error(code = "E0632", slug = "typeck-explicit-generic-args-with-impl-trait")]
#[note]
pub struct ExplicitGenericArgsWithImplTrait {
    #[primary_span]
    #[label]
    pub spans: Vec<Span>,
    #[help]
    pub is_nightly_build: Option<()>,
}

pub struct MissingTypeParams {
    pub span: Span,
    pub def_span: Span,
    pub missing_type_params: Vec<String>,
    pub empty_generic_args: bool,
}

// Manual implementation of `SessionDiagnostic` to be able to call `span_to_snippet`.
impl<'a> SessionDiagnostic<'a> for MissingTypeParams {
    fn into_diagnostic(self, sess: &'a ParseSess) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut err = sess.span_diagnostic.struct_span_err_with_code(
            self.span,
            rustc_errors::fluent::typeck::missing_type_params,
            error_code!(E0393),
        );
        err.set_arg("parameterCount", self.missing_type_params.len());
        err.set_arg(
            "parameters",
            self.missing_type_params
                .iter()
                .map(|n| format!("`{}`", n))
                .collect::<Vec<_>>()
                .join(", "),
        );

        err.span_label(self.def_span, rustc_errors::fluent::typeck::label);

        let mut suggested = false;
        if let (Ok(snippet), true) = (
            sess.source_map().span_to_snippet(self.span),
            // Don't suggest setting the type params if there are some already: the order is
            // tricky to get right and the user will already know what the syntax is.
            self.empty_generic_args,
        ) {
            if snippet.ends_with('>') {
                // The user wrote `Trait<'a, T>` or similar. To provide an accurate suggestion
                // we would have to preserve the right order. For now, as clearly the user is
                // aware of the syntax, we do nothing.
            } else {
                // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Iterator<Type>`.
                err.span_suggestion(
                    self.span,
                    rustc_errors::fluent::typeck::suggestion,
                    format!("{}<{}>", snippet, self.missing_type_params.join(", ")),
                    Applicability::HasPlaceholders,
                );
                suggested = true;
            }
        }
        if !suggested {
            err.span_label(self.span, rustc_errors::fluent::typeck::no_suggestion_label);
        }

        err.note(rustc_errors::fluent::typeck::note);
        err
    }
}

#[derive(SessionDiagnostic)]
#[error(code = "E0183", slug = "typeck-manual-implementation")]
#[help]
pub struct ManualImplementation {
    #[primary_span]
    #[label]
    pub span: Span,
    pub trait_name: String,
}

#[derive(SessionDiagnostic)]
#[error(slug = "typeck-substs-on-overridden-impl")]
pub struct SubstsOnOverriddenImpl {
    #[primary_span]
    pub span: Span,
}
