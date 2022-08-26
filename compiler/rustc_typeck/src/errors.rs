//! Errors emitted by typeck.
use rustc_errors::{error_code, Applicability, DiagnosticBuilder, ErrorGuaranteed};
use rustc_macros::{LintDiagnostic, SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::Ty;
use rustc_session::{parse::ParseSess, SessionDiagnostic};
use rustc_span::{symbol::Ident, Span, Symbol};

#[derive(SessionDiagnostic)]
#[diag(typeck::field_multiply_specified_in_initializer, code = "E0062")]
pub struct FieldMultiplySpecifiedInInitializer {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(typeck::previous_use_label)]
    pub prev_span: Span,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::unrecognized_atomic_operation, code = "E0092")]
pub struct UnrecognizedAtomicOperation<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub op: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::wrong_number_of_generic_arguments_to_intrinsic, code = "E0094")]
pub struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub descr: &'a str,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::unrecognized_intrinsic_function, code = "E0093")]
pub struct UnrecognizedIntrinsicFunction {
    #[primary_span]
    #[label]
    pub span: Span,
    pub name: Symbol,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::lifetimes_or_bounds_mismatch_on_trait, code = "E0195")]
pub struct LifetimesOrBoundsMismatchOnTrait {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(typeck::generics_label)]
    pub generics_span: Option<Span>,
    pub item_kind: &'static str,
    pub ident: Ident,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::drop_impl_on_wrong_item, code = "E0120")]
pub struct DropImplOnWrongItem {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::field_already_declared, code = "E0124")]
pub struct FieldAlreadyDeclared {
    pub field_name: Ident,
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(typeck::previous_decl_label)]
    pub prev_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::copy_impl_on_type_with_dtor, code = "E0184")]
pub struct CopyImplOnTypeWithDtor {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::multiple_relaxed_default_bounds, code = "E0203")]
pub struct MultipleRelaxedDefaultBounds {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::copy_impl_on_non_adt, code = "E0206")]
pub struct CopyImplOnNonAdt {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::trait_object_declared_with_no_traits, code = "E0224")]
pub struct TraitObjectDeclaredWithNoTraits {
    #[primary_span]
    pub span: Span,
    #[label(typeck::alias_span)]
    pub trait_alias_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::ambiguous_lifetime_bound, code = "E0227")]
pub struct AmbiguousLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::assoc_type_binding_not_allowed, code = "E0229")]
pub struct AssocTypeBindingNotAllowed {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::functional_record_update_on_non_struct, code = "E0436")]
pub struct FunctionalRecordUpdateOnNonStruct {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::typeof_reserved_keyword_used, code = "E0516")]
pub struct TypeofReservedKeywordUsed<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
    #[suggestion_verbose(code = "{ty}")]
    pub opt_sugg: Option<(Span, Applicability)>,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::return_stmt_outside_of_fn_body, code = "E0572")]
pub struct ReturnStmtOutsideOfFnBody {
    #[primary_span]
    pub span: Span,
    #[label(typeck::encl_body_label)]
    pub encl_body_span: Option<Span>,
    #[label(typeck::encl_fn_label)]
    pub encl_fn_span: Option<Span>,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::yield_expr_outside_of_generator, code = "E0627")]
pub struct YieldExprOutsideOfGenerator {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::struct_expr_non_exhaustive, code = "E0639")]
pub struct StructExprNonExhaustive {
    #[primary_span]
    pub span: Span,
    pub what: &'static str,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::method_call_on_unknown_type, code = "E0699")]
pub struct MethodCallOnUnknownType {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::value_of_associated_struct_already_specified, code = "E0719")]
pub struct ValueOfAssociatedStructAlreadySpecified {
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(typeck::previous_bound_label)]
    pub prev_span: Span,
    pub item_name: Ident,
    pub def_path: String,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::address_of_temporary_taken, code = "E0745")]
pub struct AddressOfTemporaryTaken {
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionSubdiagnostic)]
pub enum AddReturnTypeSuggestion {
    #[suggestion(
        typeck::add_return_type_add,
        code = "-> {found} ",
        applicability = "machine-applicable"
    )]
    Add {
        #[primary_span]
        span: Span,
        found: String,
    },
    #[suggestion(
        typeck::add_return_type_missing_here,
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
    #[label(typeck::expected_default_return_type)]
    Unit {
        #[primary_span]
        span: Span,
    },
    #[label(typeck::expected_return_type)]
    Other {
        #[primary_span]
        span: Span,
        expected: Ty<'tcx>,
    },
}

#[derive(SessionDiagnostic)]
#[diag(typeck::unconstrained_opaque_type)]
#[note]
pub struct UnconstrainedOpaqueType {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

pub struct MissingTypeParams {
    pub span: Span,
    pub def_span: Span,
    pub missing_type_params: Vec<Symbol>,
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
                    format!(
                        "{}<{}>",
                        snippet,
                        self.missing_type_params
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
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
#[diag(typeck::manual_implementation, code = "E0183")]
#[help]
pub struct ManualImplementation {
    #[primary_span]
    #[label]
    pub span: Span,
    pub trait_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::substs_on_overridden_impl)]
pub struct SubstsOnOverriddenImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(typeck::unused_extern_crate)]
pub struct UnusedExternCrate {
    #[suggestion(applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(typeck::extern_crate_not_idiomatic)]
pub struct ExternCrateNotIdiomatic {
    #[suggestion_short(applicability = "machine-applicable", code = "{suggestion_code}")]
    pub span: Span,
    pub msg_code: String,
    pub suggestion_code: String,
}

#[derive(SessionDiagnostic)]
#[diag(typeck::expected_used_symbol)]
pub struct ExpectedUsedSymbol {
    #[primary_span]
    pub span: Span,
}
