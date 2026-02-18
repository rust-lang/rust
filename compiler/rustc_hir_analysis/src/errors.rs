//! Errors emitted by `rustc_hir_analysis`.

use rustc_abi::ExternAbi;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, DiagSymbolList, Diagnostic, EmissionGuarantee, Level,
    MultiSpan, listify, msg,
};
use rustc_hir::limit::Limit;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{self, Ty};
use rustc_span::{Ident, Span, Symbol};
pub(crate) mod wrong_number_of_generic_args;

mod precise_captures;
pub(crate) use precise_captures::*;

#[derive(Diagnostic)]
#[diag("ambiguous associated {$assoc_kind} `{$assoc_ident}` in bounds of `{$qself}`")]
pub(crate) struct AmbiguousAssocItem<'a> {
    #[primary_span]
    #[label("ambiguous associated {$assoc_kind} `{$assoc_ident}`")]
    pub span: Span,
    pub assoc_kind: &'static str,
    pub assoc_ident: Ident,
    pub qself: &'a str,
}

#[derive(Diagnostic)]
#[diag("expected {$expected}, found {$got}")]
pub(crate) struct AssocKindMismatch {
    #[primary_span]
    #[label("unexpected {$got}")]
    pub span: Span,
    pub expected: &'static str,
    pub got: &'static str,
    #[label("expected a {$expected} because of this associated {$expected}")]
    pub expected_because_label: Option<Span>,
    pub assoc_kind: &'static str,
    #[note("the associated {$assoc_kind} is defined here")]
    pub def_span: Span,
    #[label("bounds are not allowed on associated constants")]
    pub bound_on_assoc_const_label: Option<Span>,
    #[subdiagnostic]
    pub wrap_in_braces_sugg: Option<AssocKindMismatchWrapInBracesSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("consider adding braces here", applicability = "maybe-incorrect")]
pub(crate) struct AssocKindMismatchWrapInBracesSugg {
    #[suggestion_part(code = "{{ ")]
    pub lo: Span,
    #[suggestion_part(code = " }}")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("{$kind} `{$name}` is private", code = E0624)]
pub(crate) struct AssocItemIsPrivate {
    #[primary_span]
    #[label("private {$kind}")]
    pub span: Span,
    pub kind: &'static str,
    pub name: Ident,
    #[label("the {$kind} is defined here")]
    pub defined_here_label: Span,
}

#[derive(Diagnostic)]
#[diag("associated {$assoc_kind} `{$assoc_ident}` not found for `{$qself}`", code = E0220)]
pub(crate) struct AssocItemNotFound<'a> {
    #[primary_span]
    pub span: Span,
    pub assoc_ident: Ident,
    pub assoc_kind: &'static str,
    pub qself: &'a str,
    #[subdiagnostic]
    pub label: Option<AssocItemNotFoundLabel<'a>>,
    #[subdiagnostic]
    pub sugg: Option<AssocItemNotFoundSugg<'a>>,
    #[label("due to this macro variable")]
    pub within_macro_span: Option<Span>,
}

#[derive(Subdiagnostic)]
pub(crate) enum AssocItemNotFoundLabel<'a> {
    #[label("associated {$assoc_kind} `{$assoc_ident}` not found")]
    NotFound {
        #[primary_span]
        span: Span,
    },
    #[label(
        "there is {$identically_named ->
            [true] an
            *[false] a similarly named
            } associated {$assoc_kind} `{$suggested_name}` in the trait `{$trait_name}`"
    )]
    FoundInOtherTrait {
        #[primary_span]
        span: Span,
        assoc_kind: &'static str,
        trait_name: &'a str,
        suggested_name: Symbol,
        identically_named: bool,
    },
}

#[derive(Subdiagnostic)]

pub(crate) enum AssocItemNotFoundSugg<'a> {
    #[suggestion(
        "there is an associated {$assoc_kind} with a similar name",
        code = "{suggested_name}",
        applicability = "maybe-incorrect"
    )]
    Similar {
        #[primary_span]
        span: Span,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
    #[suggestion(
        "change the associated {$assoc_kind} name to use `{$suggested_name}` from `{$trait_name}`",
        code = "{suggested_name}",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    SimilarInOtherTrait {
        #[primary_span]
        span: Span,
        trait_name: &'a str,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
    #[multipart_suggestion(
        "consider fully qualifying{$identically_named ->
            [true] {\"\"}
            *[false] {\" \"}and renaming
        } the associated {$assoc_kind}",
        style = "verbose"
    )]
    SimilarInOtherTraitQPath {
        #[suggestion_part(code = "<")]
        lo: Span,
        #[suggestion_part(code = " as {trait_ref}>")]
        mi: Span,
        #[suggestion_part(code = "{suggested_name}")]
        hi: Option<Span>,
        trait_ref: String,
        suggested_name: Symbol,
        identically_named: bool,
        #[applicability]
        applicability: Applicability,
    },
    #[suggestion(
        "`{$qself}` has the following associated {$assoc_kind}",
        code = "{suggested_name}",
        applicability = "maybe-incorrect"
    )]
    Other {
        #[primary_span]
        span: Span,
        qself: &'a str,
        assoc_kind: &'static str,
        suggested_name: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("intrinsic has wrong number of {$descr} parameters: found {$found}, expected {$expected}", code = E0094)]
pub(crate) struct WrongNumberOfGenericArgumentsToIntrinsic<'a> {
    #[primary_span]
    #[label(
        "expected {$expected} {$descr} {$expected ->
            [one] parameter
            *[other] parameters
        }"
    )]
    pub span: Span,
    pub found: usize,
    pub expected: usize,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag("unrecognized intrinsic function: `{$name}`", code = E0093)]
#[help("if you're adding an intrinsic, be sure to update `check_intrinsic_type`")]
pub(crate) struct UnrecognizedIntrinsicFunction {
    #[primary_span]
    #[label("unrecognized intrinsic")]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("lifetime parameters or bounds on {$item_kind} `{$ident}` do not match the trait declaration", code = E0195)]
pub(crate) struct LifetimesOrBoundsMismatchOnTrait {
    #[primary_span]
    #[label("lifetimes do not match {$item_kind} in trait")]
    pub span: Span,
    #[label("lifetimes in impl do not match this {$item_kind} in trait")]
    pub generics_span: Span,
    #[label("this `where` clause might not match the one in the trait")]
    pub where_span: Option<Span>,
    #[label("this bound might be missing in the impl")]
    pub bounds_span: Vec<Span>,
    pub item_kind: &'static str,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag("the `{$trait_}` trait may only be implemented for local structs, enums, and unions", code = E0120)]
pub(crate) struct DropImplOnWrongItem {
    #[primary_span]
    #[label("must be a struct, enum, or union in the current crate")]
    pub span: Span,
    pub trait_: Symbol,
}

#[derive(Diagnostic)]
pub(crate) enum FieldAlreadyDeclared {
    #[diag("field `{$field_name}` is already declared", code = E0124)]
    NotNested {
        field_name: Ident,
        #[primary_span]
        #[label("field already declared")]
        span: Span,
        #[label("`{$field_name}` first declared here")]
        prev_span: Span,
    },
    #[diag("field `{$field_name}` is already declared")]
    CurrentNested {
        field_name: Ident,
        #[primary_span]
        #[label("field `{$field_name}` declared in this unnamed field")]
        span: Span,
        #[note("field `{$field_name}` declared here")]
        nested_field_span: Span,
        #[subdiagnostic]
        help: FieldAlreadyDeclaredNestedHelp,
        #[label("`{$field_name}` first declared here")]
        prev_span: Span,
    },
    #[diag("field `{$field_name}` is already declared")]
    PreviousNested {
        field_name: Ident,
        #[primary_span]
        #[label("field already declared")]
        span: Span,
        #[label("`{$field_name}` first declared here in this unnamed field")]
        prev_span: Span,
        #[note("field `{$field_name}` first declared here")]
        prev_nested_field_span: Span,
        #[subdiagnostic]
        prev_help: FieldAlreadyDeclaredNestedHelp,
    },
    #[diag("field `{$field_name}` is already declared")]
    BothNested {
        field_name: Ident,
        #[primary_span]
        #[label("field `{$field_name}` declared in this unnamed field")]
        span: Span,
        #[note("field `{$field_name}` declared here")]
        nested_field_span: Span,
        #[subdiagnostic]
        help: FieldAlreadyDeclaredNestedHelp,
        #[label("`{$field_name}` first declared here in this unnamed field")]
        prev_span: Span,
        #[note("field `{$field_name}` first declared here")]
        prev_nested_field_span: Span,
        #[subdiagnostic]
        prev_help: FieldAlreadyDeclaredNestedHelp,
    },
}

#[derive(Subdiagnostic)]
#[help("fields from the type of this unnamed field are considered fields of the outer type")]
pub(crate) struct FieldAlreadyDeclaredNestedHelp {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the trait `Copy` cannot be implemented for this type; the type has a destructor", code = E0184)]
pub(crate) struct CopyImplOnTypeWithDtor {
    #[primary_span]
    #[label("`Copy` not allowed on types with destructors")]
    pub span: Span,
    #[note("destructor declared here")]
    pub impl_: Span,
}

#[derive(Diagnostic)]
#[diag("the trait `Copy` cannot be implemented for this type", code = E0206)]
pub(crate) struct CopyImplOnNonAdt {
    #[primary_span]
    #[label("type is not a structure or enumeration")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the trait `ConstParamTy` may not be implemented for this type")]
pub(crate) struct ConstParamTyImplOnUnsized {
    #[primary_span]
    #[label("type is not `Sized`")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the trait `ConstParamTy` may not be implemented for this type")]
pub(crate) struct ConstParamTyImplOnNonAdt {
    #[primary_span]
    #[label("type is not a structure or enumeration")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("at least one trait is required for an object type", code = E0224)]
pub(crate) struct TraitObjectDeclaredWithNoTraits {
    #[primary_span]
    pub span: Span,
    #[label("this alias does not contain a trait")]
    pub trait_alias_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("ambiguous lifetime bound, explicit lifetime bound required", code = E0227)]
pub(crate) struct AmbiguousLifetimeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("associated item constraints are not allowed here", code = E0229)]
pub(crate) struct AssocItemConstraintsNotAllowedHere {
    #[primary_span]
    #[label("associated item constraint not allowed here")]
    pub span: Span,

    #[subdiagnostic]
    pub fn_trait_expansion: Option<ParenthesizedFnTraitExpansion>,
}

#[derive(Diagnostic)]
#[diag(
    "the type of the associated constant `{$assoc_const}` must not depend on {$param_category ->
        [self] `Self`
        [synthetic] `impl Trait`
        *[normal] generic parameters
    }"
)]
pub(crate) struct ParamInTyOfAssocConstBinding<'tcx> {
    #[primary_span]
    #[label(
        "its type must not depend on {$param_category ->
            [self] `Self`
            [synthetic] `impl Trait`
            *[normal] the {$param_def_kind} `{$param_name}`
        }"
    )]
    pub span: Span,
    pub assoc_const: Ident,
    pub param_name: Symbol,
    pub param_def_kind: &'static str,
    pub param_category: &'static str,
    #[label(
        "{$param_category ->
            [synthetic] the `impl Trait` is specified here
            *[normal] the {$param_def_kind} `{$param_name}` is defined here
        }"
    )]
    pub param_defined_here_label: Option<Span>,
    #[subdiagnostic]
    pub ty_note: Option<TyOfAssocConstBindingNote<'tcx>>,
}

#[derive(Subdiagnostic, Clone, Copy)]
#[note("`{$assoc_const}` has type `{$ty}`")]
pub(crate) struct TyOfAssocConstBindingNote<'tcx> {
    pub assoc_const: Ident,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(
    "the type of the associated constant `{$assoc_const}` cannot capture late-bound generic parameters"
)]
pub(crate) struct EscapingBoundVarInTyOfAssocConstBinding<'tcx> {
    #[primary_span]
    #[label("its type cannot capture the late-bound {$var_def_kind} `{$var_name}`")]
    pub span: Span,
    pub assoc_const: Ident,
    pub var_name: Symbol,
    pub var_def_kind: &'static str,
    #[label("the late-bound {$var_def_kind} `{$var_name}` is defined here")]
    pub var_defined_here_label: Span,
    #[subdiagnostic]
    pub ty_note: Option<TyOfAssocConstBindingNote<'tcx>>,
}

#[derive(Subdiagnostic)]
#[help("parenthesized trait syntax expands to `{$expanded_type}`")]
pub(crate) struct ParenthesizedFnTraitExpansion {
    #[primary_span]
    pub span: Span,

    pub expanded_type: String,
}

#[derive(Diagnostic)]
#[diag("the value of the associated type `{$item_name}` in trait `{$def_path}` is already specified", code = E0719)]
pub(crate) struct ValueOfAssociatedStructAlreadySpecified {
    #[primary_span]
    #[label("re-bound here")]
    pub span: Span,
    #[label("`{$item_name}` bound here first")]
    pub prev_span: Span,
    pub item_name: Ident,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag("unconstrained opaque type")]
#[note("`{$name}` must be used in combination with a concrete type within the same {$what}")]
pub(crate) struct UnconstrainedOpaqueType {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub what: &'static str,
}

pub(crate) struct MissingGenericParams {
    pub span: Span,
    pub def_span: Span,
    pub span_snippet: Option<String>,
    pub missing_generic_params: Vec<(Symbol, ty::GenericParamDefKind)>,
    pub empty_generic_args: bool,
}

// FIXME: This doesn't need to be a manual impl!
impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for MissingGenericParams {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut err = Diag::new(
            dcx,
            level,
            msg!(
                "the {$descr} {$parameterCount ->
                    [one] parameter
                    *[other] parameters
                } {$parameters} must be explicitly specified"
            ),
        );
        err.span(self.span);
        err.code(E0393);
        err.span_label(
            self.def_span,
            msg!(
                "{$descr} {$parameterCount ->
                    [one] parameter
                    *[other] parameters
                } {$parameters} must be specified for this"
            ),
        );

        enum Descr {
            Generic,
            Type,
            Const,
        }

        let mut descr = None;
        for (_, kind) in &self.missing_generic_params {
            descr = match (&descr, kind) {
                (None, ty::GenericParamDefKind::Type { .. }) => Some(Descr::Type),
                (None, ty::GenericParamDefKind::Const { .. }) => Some(Descr::Const),
                (Some(Descr::Type), ty::GenericParamDefKind::Const { .. })
                | (Some(Descr::Const), ty::GenericParamDefKind::Type { .. }) => {
                    Some(Descr::Generic)
                }
                _ => continue,
            }
        }

        err.arg(
            "descr",
            match descr.unwrap() {
                Descr::Generic => "generic",
                Descr::Type => "type",
                Descr::Const => "const",
            },
        );
        err.arg("parameterCount", self.missing_generic_params.len());
        err.arg(
            "parameters",
            listify(&self.missing_generic_params, |(n, _)| format!("`{n}`")).unwrap(),
        );

        let mut suggested = false;
        // Don't suggest setting the generic params if there are some already: The order is
        // tricky to get right and the user will already know what the syntax is.
        if let Some(snippet) = self.span_snippet
            && self.empty_generic_args
        {
            if snippet.ends_with('>') {
                // The user wrote `Trait<'a, T>` or similar. To provide an accurate suggestion
                // we would have to preserve the right order. For now, as clearly the user is
                // aware of the syntax, we do nothing.
            } else {
                // The user wrote `Trait`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Trait</* Term */>`.
                err.span_suggestion_verbose(
                    self.span.shrink_to_hi(),
                    msg!(
                        "explicitly specify the {$descr} {$parameterCount ->
                            [one] parameter
                            *[other] parameters
                        }"
                    ),
                    format!(
                        "<{}>",
                        self.missing_generic_params
                            .iter()
                            .map(|(n, _)| format!("/* {n} */"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    Applicability::HasPlaceholders,
                );
                suggested = true;
            }
        }
        if !suggested {
            err.span_label(
                self.span,
                msg!(
                    "missing {$parameterCount ->
                        [one] reference
                        *[other] references
                    } to {$parameters}"
                ),
            );
        }

        err.note(msg!(
            "because the parameter {$parameterCount ->
                [one] default references
                *[other] defaults reference
            } `Self`, the {$parameterCount ->
                [one] parameter
                *[other] parameters
            } must be specified on the trait object type"
        ));
        err
    }
}

#[derive(Diagnostic)]
#[diag("manual implementations of `{$trait_name}` are experimental", code = E0183)]
#[help("add `#![feature(unboxed_closures)]` to the crate attributes to enable")]
pub(crate) struct ManualImplementation {
    #[primary_span]
    #[label("manual implementations of `{$trait_name}` are experimental")]
    pub span: Span,
    pub trait_name: String,
}

#[derive(Diagnostic)]
#[diag("could not resolve generic parameters on overridden impl")]
pub(crate) struct GenericArgsOnOverriddenImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("const `impl` for trait `{$trait_name}` which is not `const`")]
pub(crate) struct ConstImplForNonConstTrait {
    #[primary_span]
    #[label("this trait is not `const`")]
    pub trait_ref_span: Span,
    pub trait_name: String,
    #[suggestion(
        "{$suggestion_pre}mark `{$trait_name}` as `const` to allow it to have `const` implementations",
        applicability = "machine-applicable",
        code = "const ",
        style = "verbose"
    )]
    pub suggestion: Option<Span>,
    pub suggestion_pre: &'static str,
    #[note("marking a trait with `const` ensures all default method bodies are `const`")]
    pub marking: (),
    #[note("adding a non-const method body in the future would be a breaking change")]
    pub adding: (),
}

#[derive(Diagnostic)]
#[diag("`{$modifier}` can only be applied to `const` traits")]
pub(crate) struct ConstBoundForNonConstTrait {
    #[primary_span]
    #[label("can't be applied to `{$trait_name}`")]
    pub span: Span,
    pub modifier: &'static str,
    #[note("`{$trait_name}` can't be used with `{$modifier}` because it isn't `const`")]
    pub def_span: Option<Span>,
    #[suggestion(
        "{$suggestion_pre}mark `{$trait_name}` as `const` to allow it to have `const` implementations",
        applicability = "machine-applicable",
        code = "const ",
        style = "verbose"
    )]
    pub suggestion: Option<Span>,
    pub suggestion_pre: &'static str,
    pub trait_name: String,
}

#[derive(Diagnostic)]
#[diag("`Self` is not valid in the self type of an impl block")]
pub(crate) struct SelfInImplSelf {
    #[primary_span]
    pub span: MultiSpan,
    #[note("replace `Self` with a different type")]
    pub note: (),
}

#[derive(Diagnostic)]
#[diag("invalid type for variable with `#[linkage]` attribute", code = E0791)]
pub(crate) struct LinkageType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]` attribute to your crate (`{$crate_name}`)"
)]
#[diag("reached the recursion limit while auto-dereferencing `{$ty}`", code = E0055)]
pub(crate) struct AutoDerefReachedRecursionLimit<'a> {
    #[primary_span]
    #[label("deref recursion limit reached")]
    pub span: Span,
    pub ty: Ty<'a>,
    pub suggested_limit: Limit,
    pub crate_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("`main` function is not allowed to have a `where` clause", code = E0646)]
pub(crate) struct WhereClauseOnMain {
    #[primary_span]
    pub span: Span,
    #[label("`main` cannot have a `where` clause")]
    pub generics_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("`main` function is not allowed to be `#[track_caller]`")]
pub(crate) struct TrackCallerOnMain {
    #[primary_span]
    #[suggestion("remove this annotation", applicability = "maybe-incorrect", code = "")]
    pub span: Span,
    #[label("`main` function is not allowed to be `#[track_caller]`")]
    pub annotated: Span,
}

#[derive(Diagnostic)]
#[diag("`main` function is not allowed to have `#[target_feature]`")]
pub(crate) struct TargetFeatureOnMain {
    #[primary_span]
    #[label("`main` function is not allowed to have `#[target_feature]`")]
    pub main: Span,
}

#[derive(Diagnostic)]
#[diag("`main` function return type is not allowed to have generic parameters", code = E0131)]
pub(crate) struct MainFunctionReturnTypeGeneric {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`main` function is not allowed to be `async`", code = E0752)]
pub(crate) struct MainFunctionAsync {
    #[primary_span]
    pub span: Span,
    #[label("`main` function is not allowed to be `async`")]
    pub asyncness: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("`main` function is not allowed to have generic parameters", code = E0131)]
pub(crate) struct MainFunctionGenericParameters {
    #[primary_span]
    pub span: Span,
    #[label("`main` cannot have generic parameters")]
    pub label_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("C-variadic functions with the {$convention} calling convention are not supported", code = E0045)]
pub(crate) struct VariadicFunctionCompatibleConvention<'a> {
    #[primary_span]
    #[label("C-variadic function must have a compatible calling convention")]
    pub span: Span,
    pub convention: &'a str,
}

#[derive(Diagnostic)]
pub(crate) enum CannotCaptureLateBound {
    #[diag("cannot capture late-bound type parameter in {$what}")]
    Type {
        #[primary_span]
        use_span: Span,
        #[label("parameter defined here")]
        def_span: Span,
        what: &'static str,
    },
    #[diag("cannot capture late-bound const parameter in {$what}")]
    Const {
        #[primary_span]
        use_span: Span,
        #[label("parameter defined here")]
        def_span: Span,
        what: &'static str,
    },
    #[diag("cannot capture late-bound lifetime in {$what}")]
    Lifetime {
        #[primary_span]
        use_span: Span,
        #[label("lifetime defined here")]
        def_span: Span,
        what: &'static str,
    },
}

#[derive(Diagnostic)]
#[diag("{$variances}")]
pub(crate) struct VariancesOf {
    #[primary_span]
    pub span: Span,
    pub variances: String,
}

#[derive(Diagnostic)]
#[diag("{$ty}")]
pub(crate) struct TypeOf<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union", code = E0740)]
pub(crate) struct InvalidUnionField {
    #[primary_span]
    pub field_span: Span,
    #[subdiagnostic]
    pub sugg: InvalidUnionFieldSuggestion,
    #[note(
        "union fields must not have drop side-effects, which is currently enforced via either `Copy` or `ManuallyDrop<...>`"
    )]
    pub note: (),
}

#[derive(Diagnostic)]
#[diag(
    "return type notation used on function that is not `async` and does not return `impl Trait`"
)]
pub(crate) struct ReturnTypeNotationOnNonRpitit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    #[label("this function must be `async` or return `impl Trait`")]
    pub fn_span: Option<Span>,
    #[note("function returns `{$ty}`, which is not compatible with associated type return bounds")]
    pub note: (),
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "wrap the field type in `ManuallyDrop<...>`",
    applicability = "machine-applicable"
)]
pub(crate) struct InvalidUnionFieldSuggestion {
    #[suggestion_part(code = "std::mem::ManuallyDrop<")]
    pub lo: Span,
    #[suggestion_part(code = ">")]
    pub hi: Span,
}

#[derive(Diagnostic)]
#[diag("return type notation is not allowed to use type equality")]
pub(crate) struct ReturnTypeNotationEqualityBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the placeholder `_` is not allowed within types on item signatures for {$kind}", code = E0121)]
pub(crate) struct PlaceholderNotAllowedItemSignatures {
    #[primary_span]
    #[label("not allowed in type signatures")]
    pub spans: Vec<Span>,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag("cannot use the {$what} of a trait with uninferred generic parameters", code = E0212)]
pub(crate) struct AssociatedItemTraitUninferredGenericParams {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use a fully qualified path with inferred lifetimes",
        style = "verbose",
        applicability = "maybe-incorrect",
        code = "{bound}"
    )]
    pub inferred_sugg: Option<Span>,
    pub bound: String,
    #[subdiagnostic]
    pub mpart_sugg: Option<AssociatedItemTraitUninferredGenericParamsMultipartSuggestion>,
    pub what: &'static str,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use a fully qualified path with explicit lifetimes",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedItemTraitUninferredGenericParamsMultipartSuggestion {
    #[suggestion_part(code = "{first}")]
    pub fspan: Span,
    pub first: String,
    #[suggestion_part(code = "{second}")]
    pub sspan: Span,
    pub second: String,
}

#[derive(Diagnostic)]
#[diag("enum discriminant overflowed", code = E0370)]
#[note("explicitly set `{$item_name} = {$wrapped_discr}` if that is desired outcome")]
pub(crate) struct EnumDiscriminantOverflowed {
    #[primary_span]
    #[label("overflowed on value after {$discr}")]
    pub span: Span,
    pub discr: String,
    pub item_name: Ident,
    pub wrapped_discr: String,
}

#[derive(Diagnostic)]
#[diag(
    "the `#[rustc_paren_sugar]` attribute is a temporary means of controlling which traits can use parenthetical notation"
)]
#[help("add `#![feature(unboxed_closures)]` to the crate attributes to use it")]
pub(crate) struct ParenSugarAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("use of SIMD type{$snip} in FFI is highly experimental and may result in invalid code")]
#[help("add `#![feature(simd_ffi)]` to the crate attributes to enable")]
pub(crate) struct SIMDFFIHighlyExperimental {
    #[primary_span]
    pub span: Span,
    pub snip: String,
}

#[derive(Diagnostic)]
pub(crate) enum ImplNotMarkedDefault {
    #[diag("`{$ident}` specializes an item from a parent `impl`, but that item is not marked `default`", code = E0520)]
    #[note("to specialize, `{$ident}` in the parent `impl` must be marked `default`")]
    Ok {
        #[primary_span]
        #[label("cannot specialize default item `{$ident}`")]
        span: Span,
        #[label("parent `impl` is here")]
        ok_label: Span,
        ident: Ident,
    },
    #[diag("`{$ident}` specializes an item from a parent `impl`, but that item is not marked `default`", code = E0520)]
    #[note("parent implementation is in crate `{$cname}`")]
    Err {
        #[primary_span]
        span: Span,
        cname: Symbol,
        ident: Ident,
    },
}

#[derive(LintDiagnostic)]
#[diag("this item cannot be used as its where bounds are not satisfied for the `Self` type")]
pub(crate) struct UselessImplItem;

#[derive(Diagnostic)]
#[diag("cannot override `{$ident}` because it already has a `final` definition in the trait")]
pub(crate) struct OverridingFinalTraitFunction {
    #[primary_span]
    pub impl_span: Span,
    #[note("`{$ident}` is marked final here")]
    pub trait_span: Span,
    pub ident: Ident,
}

#[derive(Diagnostic)]
#[diag("not all trait items implemented, missing: `{$missing_items_msg}`", code = E0046)]
pub(crate) struct MissingTraitItem {
    #[primary_span]
    #[label("missing `{$missing_items_msg}` in implementation")]
    pub span: Span,
    #[subdiagnostic]
    pub missing_trait_item_label: Vec<MissingTraitItemLabel>,
    #[subdiagnostic]
    pub missing_trait_item: Vec<MissingTraitItemSuggestion>,
    #[subdiagnostic]
    pub missing_trait_item_none: Vec<MissingTraitItemSuggestionNone>,
    pub missing_items_msg: String,
}

#[derive(Subdiagnostic)]
#[label("`{$item}` from trait")]
pub(crate) struct MissingTraitItemLabel {
    #[primary_span]
    pub span: Span,
    pub item: Symbol,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "implement the missing item: `{$snippet}`",
    style = "tool-only",
    applicability = "has-placeholders",
    code = "{code}"
)]
pub(crate) struct MissingTraitItemSuggestion {
    #[primary_span]
    pub span: Span,
    pub code: String,
    pub snippet: String,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "implement the missing item: `{$snippet}`",
    style = "hidden",
    applicability = "has-placeholders",
    code = "{code}"
)]
pub(crate) struct MissingTraitItemSuggestionNone {
    #[primary_span]
    pub span: Span,
    pub code: String,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag("not all trait items implemented, missing one of: `{$missing_items_msg}`", code = E0046)]
pub(crate) struct MissingOneOfTraitItem {
    #[primary_span]
    #[label("missing one of `{$missing_items_msg}` in implementation")]
    pub span: Span,
    #[note("required because of this annotation")]
    pub note: Option<Span>,
    pub missing_items_msg: String,
}

#[derive(Diagnostic)]
#[diag("not all trait items implemented, missing: `{$missing_item_name}`", code = E0046)]
#[note("default implementation of `{$missing_item_name}` is unstable")]
pub(crate) struct MissingTraitItemUnstable {
    #[primary_span]
    pub span: Span,
    #[note("use of unstable library feature `{$feature}`: {$reason}")]
    pub some_note: bool,
    #[note("use of unstable library feature `{$feature}`")]
    pub none_note: bool,
    pub missing_item_name: Ident,
    pub feature: Symbol,
    pub reason: String,
}

#[derive(Diagnostic)]
#[diag("transparent enum needs exactly one variant, but has {$number}", code = E0731)]
pub(crate) struct TransparentEnumVariant {
    #[primary_span]
    #[label("needs exactly one variant, but has {$number}")]
    pub span: Span,
    #[label("variant here")]
    pub spans: Vec<Span>,
    #[label("too many variants in `{$path}`")]
    pub many: Option<Span>,
    pub number: usize,
    pub path: String,
}

#[derive(Diagnostic)]
#[diag("the variant of a transparent {$desc} needs at most one field with non-trivial size or alignment, but has {$field_count}", code = E0690)]
pub(crate) struct TransparentNonZeroSizedEnum<'a> {
    #[primary_span]
    #[label("needs at most one field with non-trivial size or alignment, but has {$field_count}")]
    pub span: Span,
    #[label("this field has non-zero size or requires alignment")]
    pub spans: Vec<Span>,
    pub field_count: usize,
    pub desc: &'a str,
}

#[derive(Diagnostic)]
#[diag("transparent {$desc} needs at most one field with non-trivial size or alignment, but has {$field_count}", code = E0690)]
pub(crate) struct TransparentNonZeroSized<'a> {
    #[primary_span]
    #[label("needs at most one field with non-trivial size or alignment, but has {$field_count}")]
    pub span: Span,
    #[label("this field has non-zero size or requires alignment")]
    pub spans: Vec<Span>,
    pub field_count: usize,
    pub desc: &'a str,
}

#[derive(Diagnostic)]
#[diag("extern static is too large for the target architecture")]
pub(crate) struct TooLargeStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("implementing `rustc_specialization_trait` traits is unstable")]
#[help("add `#![feature(min_specialization)]` to the crate attributes to enable")]
pub(crate) struct SpecializationTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("implicit types in closure signatures are forbidden when `for<...>` is present")]
pub(crate) struct ClosureImplicitHrtb {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label("`for<...>` is here")]
    pub for_sp: Span,
}

#[derive(Diagnostic)]
#[diag("specialization impl does not specialize any associated items")]
pub(crate) struct EmptySpecialization {
    #[primary_span]
    pub span: Span,
    #[note("impl is a specialization of this impl")]
    pub base_impl_span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot specialize on `'static` lifetime")]
pub(crate) struct StaticSpecialize {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum DropImplPolarity {
    #[diag("negative `Drop` impls are not supported")]
    Negative {
        #[primary_span]
        span: Span,
    },
    #[diag("reservation `Drop` impls are not supported")]
    Reservation {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
pub(crate) enum ReturnTypeNotationIllegalParam {
    #[diag("return type notation is not allowed for functions that have type parameters")]
    Type {
        #[primary_span]
        span: Span,
        #[label("type parameter declared here")]
        param_span: Span,
    },
    #[diag("return type notation is not allowed for functions that have const parameters")]
    Const {
        #[primary_span]
        span: Span,
        #[label("const parameter declared here")]
        param_span: Span,
    },
}

#[derive(Diagnostic)]
pub(crate) enum LateBoundInApit {
    #[diag("`impl Trait` can only mention type parameters from an fn or impl")]
    Type {
        #[primary_span]
        span: Span,
        #[label("type parameter declared here")]
        param_span: Span,
    },
    #[diag("`impl Trait` can only mention const parameters from an fn or impl")]
    Const {
        #[primary_span]
        span: Span,
        #[label("const parameter declared here")]
        param_span: Span,
    },
    #[diag("`impl Trait` can only mention lifetimes from an fn or impl")]
    Lifetime {
        #[primary_span]
        span: Span,
        #[label("lifetime declared here")]
        param_span: Span,
    },
}

#[derive(LintDiagnostic)]
#[diag("unnecessary associated type bound for dyn-incompatible associated type")]
#[note(
    "this associated type has a `where Self: Sized` bound, and while the associated type can be specified, it cannot be used because trait objects are never `Sized`"
)]
pub(crate) struct UnusedAssociatedTypeBounds {
    #[suggestion("remove this bound", code = "")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("impl trait in impl method signature does not match trait method signature")]
#[note(
    "add `#[allow(refining_impl_trait)]` if it is intended for this to be part of the public API of this crate"
)]
#[note(
    "we are soliciting feedback, see issue #121718 <https://github.com/rust-lang/rust/issues/121718> for more information"
)]
pub(crate) struct ReturnPositionImplTraitInTraitRefined<'tcx> {
    #[suggestion(
        "replace the return type so that it matches the trait",
        applicability = "maybe-incorrect",
        code = "{pre}{return_ty}{post}"
    )]
    pub impl_return_span: Span,
    #[label("return type from trait method defined here")]
    pub trait_return_span: Option<Span>,
    #[label("this bound is stronger than that defined on the trait")]
    pub unmatched_bound: Option<Span>,

    pub pre: &'static str,
    pub post: &'static str,
    pub return_ty: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag("impl trait in impl method captures fewer lifetimes than in trait")]
#[note(
    "add `#[allow(refining_impl_trait)]` if it is intended for this to be part of the public API of this crate"
)]
#[note(
    "we are soliciting feedback, see issue #121718 <https://github.com/rust-lang/rust/issues/121718> for more information"
)]
pub(crate) struct ReturnPositionImplTraitInTraitRefinedLifetimes {
    #[suggestion(
        "modify the `use<..>` bound to capture the same lifetimes that the trait does",
        applicability = "maybe-incorrect",
        code = "{suggestion}"
    )]
    pub suggestion_span: Span,
    pub suggestion: String,
}

#[derive(Diagnostic)]
#[diag("cannot define inherent `impl` for a type outside of the crate where the type is defined", code = E0390)]
#[help("consider moving this inherent impl into the crate defining the type if possible")]
pub(crate) struct InherentTyOutside {
    #[primary_span]
    #[help(
        "alternatively add `#[rustc_has_incoherent_inherent_impls]` to the type and `#[rustc_allow_incoherent_impl]` to the relevant impl items"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("structs implementing `DispatchFromDyn` may not have `#[repr(packed)]` or `#[repr(C)]`", code = E0378)]
pub(crate) struct DispatchFromDynRepr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`derive(CoercePointee)` is only applicable to `struct`, instead of `{$kind}`", code = E0802)]
pub(crate) struct CoercePointeeNotStruct {
    #[primary_span]
    pub span: Span,
    pub kind: String,
}

#[derive(Diagnostic)]
#[diag("`derive(CoercePointee)` is only applicable to `struct`", code = E0802)]
pub(crate) struct CoercePointeeNotConcreteType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("asserting applicability of `derive(CoercePointee)` on a target data is forbidden", code = E0802)]
pub(crate) struct CoercePointeeNoUserValidityAssertion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`derive(CoercePointee)` is only applicable to `struct` with `repr(transparent)` layout", code = E0802)]
pub(crate) struct CoercePointeeNotTransparent {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`CoercePointee` can only be derived on `struct`s with at least one field", code = E0802)]
pub(crate) struct CoercePointeeNoField {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot define inherent `impl` for a type outside of the crate where the type is defined", code = E0390)]
#[help("consider moving this inherent impl into the crate defining the type if possible")]
pub(crate) struct InherentTyOutsideRelevant {
    #[primary_span]
    pub span: Span,
    #[help("alternatively add `#[rustc_allow_incoherent_impl]` to the relevant impl items")]
    pub help_span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot define inherent `impl` for a type outside of the crate where the type is defined", code = E0116)]
#[help(
    "consider defining a trait and implementing it for the type or using a newtype wrapper like `struct MyType(ExternalType);` and implement it"
)]
#[note(
    "for more details about the orphan rules, see <https://doc.rust-lang.org/reference/items/implementations.html?highlight=orphan#orphan-rules>"
)]
pub(crate) struct InherentTyOutsideNew {
    #[primary_span]
    #[label("impl for type defined outside of crate")]
    pub span: Span,
    #[subdiagnostic]
    pub note: Option<InherentTyOutsideNewAliasNote>,
}

#[derive(Subdiagnostic)]
#[note("`{$ty_name}` does not define a new type, only an alias of `{$alias_ty_name}` defined here")]
pub(crate) struct InherentTyOutsideNewAliasNote {
    #[primary_span]
    pub span: Span,
    pub ty_name: String,
    pub alias_ty_name: String,
}

#[derive(Diagnostic)]
#[diag("cannot define inherent `impl` for primitive types outside of `core`", code = E0390)]
#[help("consider moving this inherent impl into `core` if possible")]
pub(crate) struct InherentTyOutsidePrimitive {
    #[primary_span]
    pub span: Span,
    #[help("alternatively add `#[rustc_allow_incoherent_impl]` to the relevant impl items")]
    pub help_span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot define inherent `impl` for primitive types", code = E0390)]
#[help("consider using an extension trait instead")]
pub(crate) struct InherentPrimitiveTy<'a> {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: Option<InherentPrimitiveTyNote<'a>>,
}

#[derive(Subdiagnostic)]
#[note(
    "you could also try moving the reference to uses of `{$subty}` (such as `self`) within the implementation"
)]
pub(crate) struct InherentPrimitiveTyNote<'a> {
    pub subty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("cannot define inherent `impl` for a dyn auto trait", code = E0785)]
#[note("define and implement a new trait or type instead")]
pub(crate) struct InherentDyn {
    #[primary_span]
    #[label("impl requires at least one non-auto trait")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("no nominal type found for inherent implementation", code = E0118)]
#[note("either implement a trait on it or create a newtype to wrap it instead")]
pub(crate) struct InherentNominal {
    #[primary_span]
    #[label("impl requires a nominal type")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the trait `DispatchFromDyn` may only be implemented for structs containing the field being coerced, ZST fields with 1 byte alignment that don't mention type/const generics, and nothing else", code = E0378)]
#[note("extra field `{$name}` of type `{$ty}` is not allowed")]
pub(crate) struct DispatchFromDynZST<'a> {
    #[primary_span]
    pub span: Span,
    pub name: Ident,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("implementing `{$trait_name}` requires a field to be coerced", code = E0374)]
pub(crate) struct CoerceNoField {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'static str,
    #[note("expected a single field to be coerced, none found")]
    pub note: bool,
}

#[derive(Diagnostic)]
#[diag("implementing `{$trait_name}` does not allow multiple fields to be coerced", code = E0375)]
pub(crate) struct CoerceMulti {
    pub trait_name: &'static str,
    #[primary_span]
    pub span: Span,
    pub number: usize,
    #[note(
        "the trait `{$trait_name}` may only be implemented when a single field is being coerced"
    )]
    pub fields: MultiSpan,
}

#[derive(Diagnostic)]
#[diag("the trait `{$trait_name}` may only be implemented for a coercion between structures", code = E0377)]
pub(crate) struct CoerceUnsizedNonStruct {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'static str,
}

#[derive(Diagnostic)]
#[diag("only pattern types with the same pattern can be coerced between each other")]
pub(crate) struct CoerceSamePatKind {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'static str,
    pub pat_a: String,
    pub pat_b: String,
}

#[derive(Diagnostic)]
#[diag("the trait `{$trait_name}` may only be implemented for a coercion between structures", code = E0377)]
pub(crate) struct CoerceSameStruct {
    #[primary_span]
    pub span: Span,
    pub trait_name: &'static str,
    #[note(
        "expected coercion between the same definition; expected `{$source_path}`, found `{$target_path}`"
    )]
    pub note: bool,
    pub source_path: String,
    pub target_path: String,
}

#[derive(Diagnostic)]
#[diag(
    "for `{$ty}` to have a valid implementation of `{$trait_name}`, it must be possible to coerce the field of type `{$field_ty}`"
)]
pub(crate) struct CoerceFieldValidity<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub trait_name: &'static str,
    #[label(
        "`{$field_ty}` must be a pointer, reference, or smart pointer that is allowed to be unsized"
    )]
    pub field_span: Span,
    pub field_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("the trait `{$trait_name}` cannot be implemented for this type", code = E0204)]
pub(crate) struct TraitCannotImplForTy {
    #[primary_span]
    pub span: Span,
    pub trait_name: String,
    #[label("this field does not implement `{$trait_name}`")]
    pub label_spans: Vec<Span>,
    #[subdiagnostic]
    pub notes: Vec<ImplForTyRequires>,
}

#[derive(Subdiagnostic)]
#[note("the `{$trait_name}` impl for `{$ty}` requires that `{$error_predicate}`")]
pub(crate) struct ImplForTyRequires {
    #[primary_span]
    pub span: MultiSpan,
    pub error_predicate: String,
    pub trait_name: String,
    pub ty: String,
}

#[derive(Diagnostic)]
#[diag("traits with a default impl, like `{$traits}`, cannot be implemented for {$problematic_kind} `{$self_ty}`", code = E0321)]
#[note(
    "a trait object implements `{$traits}` if and only if `{$traits}` is one of the trait object's trait bounds"
)]
pub(crate) struct TraitsWithDefaultImpl<'a> {
    #[primary_span]
    pub span: Span,
    pub traits: String,
    pub problematic_kind: &'a str,
    pub self_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("cross-crate traits with a default impl, like `{$traits}`, can only be implemented for a struct/enum type, not `{$self_ty}`", code = E0321)]
pub(crate) struct CrossCrateTraits<'a> {
    #[primary_span]
    #[label("can't implement cross-crate trait with a default impl for non-struct/enum type")]
    pub span: Span,
    pub traits: String,
    pub self_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("cross-crate traits with a default impl, like `{$traits}`, can only be implemented for a struct/enum type defined in the current crate", code = E0321)]
pub(crate) struct CrossCrateTraitsDefined {
    #[primary_span]
    #[label("can't implement cross-crate trait for type in another crate")]
    pub span: Span,
    pub traits: String,
}

#[derive(Diagnostic)]
#[diag("no variant named `{$ident}` found for enum `{$ty}`", code = E0599)]
pub struct NoVariantNamed<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
    pub ty: Ty<'tcx>,
}

// FIXME(fmease): Deduplicate:

#[derive(Diagnostic)]
#[diag("type parameter `{$param}` must be covered by another type when it appears before the first local type (`{$local_type}`)", code = E0210)]
#[note(
    "implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type"
)]
pub(crate) struct TyParamFirstLocal<'tcx> {
    #[primary_span]
    #[label(
        "type parameter `{$param}` must be covered by another type when it appears before the first local type (`{$local_type}`)"
    )]
    pub span: Span,
    #[note(
        "in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last"
    )]
    pub note: (),
    pub param: Ident,
    pub local_type: Ty<'tcx>,
}

#[derive(LintDiagnostic)]
#[diag("type parameter `{$param}` must be covered by another type when it appears before the first local type (`{$local_type}`)", code = E0210)]
#[note(
    "implementing a foreign trait is only possible if at least one of the types for which it is implemented is local, and no uncovered type parameters appear before that first local type"
)]
pub(crate) struct TyParamFirstLocalLint<'tcx> {
    #[label(
        "type parameter `{$param}` must be covered by another type when it appears before the first local type (`{$local_type}`)"
    )]
    pub span: Span,
    #[note(
        "in this case, 'before' refers to the following order: `impl<..> ForeignTrait<T1, ..., Tn> for T0`, where `T0` is the first and `Tn` is the last"
    )]
    pub note: (),
    pub param: Ident,
    pub local_type: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("type parameter `{$param}` must be used as the type parameter for some local type (e.g., `MyStruct<{$param}>`)", code = E0210)]
#[note(
    "implementing a foreign trait is only possible if at least one of the types for which it is implemented is local"
)]
pub(crate) struct TyParamSome {
    #[primary_span]
    #[label("type parameter `{$param}` must be used as the type parameter for some local type")]
    pub span: Span,
    #[note("only traits defined in the current crate can be implemented for a type parameter")]
    pub note: (),
    pub param: Ident,
}

#[derive(LintDiagnostic)]
#[diag("type parameter `{$param}` must be used as the type parameter for some local type (e.g., `MyStruct<{$param}>`)", code = E0210)]
#[note(
    "implementing a foreign trait is only possible if at least one of the types for which it is implemented is local"
)]
pub(crate) struct TyParamSomeLint {
    #[label("type parameter `{$param}` must be used as the type parameter for some local type")]
    pub span: Span,
    #[note("only traits defined in the current crate can be implemented for a type parameter")]
    pub note: (),
    pub param: Ident,
}

#[derive(Diagnostic)]
pub(crate) enum OnlyCurrentTraits {
    #[diag("only traits defined in the current crate can be implemented for types defined outside of the crate", code = E0117)]
    Outside {
        #[primary_span]
        span: Span,
        #[note("impl doesn't have any local type before any uncovered type parameters")]
        #[note(
            "for more information see https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules"
        )]
        #[note("define and implement a trait or new type instead")]
        note: (),
    },
    #[diag("only traits defined in the current crate can be implemented for primitive types", code = E0117)]
    Primitive {
        #[primary_span]
        span: Span,
        #[note("impl doesn't have any local type before any uncovered type parameters")]
        #[note(
            "for more information see https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules"
        )]
        #[note("define and implement a trait or new type instead")]
        note: (),
    },
    #[diag("only traits defined in the current crate can be implemented for arbitrary types", code = E0117)]
    Arbitrary {
        #[primary_span]
        span: Span,
        #[note("impl doesn't have any local type before any uncovered type parameters")]
        #[note(
            "for more information see https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules"
        )]
        #[note("define and implement a trait or new type instead")]
        note: (),
    },
}

#[derive(Subdiagnostic)]
#[label(
    "type alias impl trait is treated as if it were foreign, because its hidden type could be from a foreign crate"
)]
pub(crate) struct OnlyCurrentTraitsOpaque {
    #[primary_span]
    pub span: Span,
}
#[derive(Subdiagnostic)]
#[label("this is not defined in the current crate because this is a foreign trait")]
pub(crate) struct OnlyCurrentTraitsForeign {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[label("this is not defined in the current crate because {$name} are always foreign")]
pub(crate) struct OnlyCurrentTraitsName<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Subdiagnostic)]
#[label("`{$pointer}` is not defined in the current crate because raw pointers are always foreign")]
pub(crate) struct OnlyCurrentTraitsPointer<'a> {
    #[primary_span]
    pub span: Span,
    pub pointer: Ty<'a>,
}

#[derive(Subdiagnostic)]
#[label("`{$ty}` is not defined in the current crate")]
pub(crate) struct OnlyCurrentTraitsTy<'a> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'a>,
}

#[derive(Subdiagnostic)]
#[label("`{$name}` is not defined in the current crate")]
pub(crate) struct OnlyCurrentTraitsAdt {
    #[primary_span]
    pub span: Span,
    pub name: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider introducing a new wrapper type",
    applicability = "maybe-incorrect"
)]
pub(crate) struct OnlyCurrentTraitsPointerSugg<'a> {
    #[suggestion_part(code = "WrapperType")]
    pub wrapper_span: Span,
    #[suggestion_part(code = "struct WrapperType(*{mut_key}{ptr_ty});\n\n")]
    pub(crate) struct_span: Span,
    pub mut_key: &'a str,
    pub ptr_ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("{$descr}")]
pub(crate) struct UnsupportedDelegation<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
    #[label("callee defined here")]
    pub callee_span: Span,
}

#[derive(Diagnostic)]
#[diag("method should be `async` or return a future, but it is synchronous")]
pub(crate) struct MethodShouldReturnFuture {
    #[primary_span]
    pub span: Span,
    pub method_name: Ident,
    #[note("this method is `async` so it expects a future to be returned")]
    pub trait_item_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("{$param_def_kind} `{$param_name}` is never used")]
pub(crate) struct UnusedGenericParameter {
    #[primary_span]
    #[label("unused {$param_def_kind}")]
    pub span: Span,
    pub param_name: Ident,
    pub param_def_kind: &'static str,
    #[label("`{$param_name}` is named here, but is likely unused in the containing type")]
    pub usage_spans: Vec<Span>,
    #[subdiagnostic]
    pub help: UnusedGenericParameterHelp,
    #[help(
        "if you intended `{$param_name}` to be a const parameter, use `const {$param_name}: /* Type */` instead"
    )]
    pub const_param_help: bool,
}

#[derive(Diagnostic)]
#[diag("{$param_def_kind} `{$param_name}` is only used recursively")]
pub(crate) struct RecursiveGenericParameter {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label("{$param_def_kind} must be used non-recursively in the definition")]
    pub param_span: Span,
    pub param_name: Ident,
    pub param_def_kind: &'static str,
    #[subdiagnostic]
    pub help: UnusedGenericParameterHelp,
    #[note(
        "all type parameters must be used in a non-recursive way in order to constrain their variance"
    )]
    pub note: (),
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedGenericParameterHelp {
    #[help(
        "consider removing `{$param_name}`, referring to it in a field, or using a marker such as `{$phantom_data}`"
    )]
    Adt { param_name: Ident, phantom_data: String },
    #[help("consider removing `{$param_name}` or referring to it in a field")]
    AdtNoPhantomData { param_name: Ident },
    #[help("consider removing `{$param_name}` or referring to it in the body of the type alias")]
    TyAlias { param_name: Ident },
}

#[derive(Diagnostic)]
#[diag(
    "the {$param_def_kind} `{$param_name}` is not constrained by the impl trait, self type, or predicates"
)]
pub(crate) struct UnconstrainedGenericParameter {
    #[primary_span]
    #[label("unconstrained {$param_def_kind}")]
    pub span: Span,
    pub param_name: Ident,
    pub param_def_kind: &'static str,
    #[note("expressions using a const parameter must map each value to a distinct output value")]
    pub const_param_note: bool,
    #[note(
        "proving the result of expressions other than the parameter are unique is not supported"
    )]
    pub const_param_note2: bool,
}

#[derive(Diagnostic)]
#[diag("`impl Trait` cannot capture {$bad_place}", code = E0657)]
pub(crate) struct OpaqueCapturesHigherRankedLifetime {
    #[primary_span]
    pub span: MultiSpan,
    #[label("`impl Trait` implicitly captures all lifetimes in scope")]
    pub label: Option<Span>,
    #[note("lifetime declared here")]
    pub decl_span: MultiSpan,
    pub bad_place: &'static str,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidReceiverTyHint {
    #[note(
        "`Weak` does not implement `Receiver` because it has methods that may shadow the referent; consider wrapping your `Weak` in a newtype wrapper for which you implement `Receiver`"
    )]
    Weak,
    #[note(
        "`NonNull` does not implement `Receiver` because it has methods that may shadow the referent; consider wrapping your `NonNull` in a newtype wrapper for which you implement `Receiver`"
    )]
    NonNull,
}

#[derive(Diagnostic)]
#[diag("invalid `self` parameter type: `{$receiver_ty}`", code = E0307)]
#[note("type of `self` must be `Self` or a type that dereferences to it")]
#[help(
    "consider changing to `self`, `&self`, `&mut self`, `self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one of the previous types except `Self`)"
)]
pub(crate) struct InvalidReceiverTyNoArbitrarySelfTypes<'tcx> {
    #[primary_span]
    pub span: Span,
    pub receiver_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("invalid `self` parameter type: `{$receiver_ty}`", code = E0307)]
#[note("type of `self` must be `Self` or some type implementing `Receiver`")]
#[help(
    "consider changing to `self`, `&self`, `&mut self`, or a type implementing `Receiver` such as `self: Box<Self>`, `self: Rc<Self>`, or `self: Arc<Self>`"
)]
pub(crate) struct InvalidReceiverTy<'tcx> {
    #[primary_span]
    pub span: Span,
    pub receiver_ty: Ty<'tcx>,
    #[subdiagnostic]
    pub hint: Option<InvalidReceiverTyHint>,
}

#[derive(Diagnostic)]
#[diag("invalid generic `self` parameter type: `{$receiver_ty}`", code = E0801)]
#[note("type of `self` must not be a method generic parameter type")]
#[help(
    "use a concrete type such as `self`, `&self`, `&mut self`, `self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one of the previous types except `Self`)"
)]
pub(crate) struct InvalidGenericReceiverTy<'tcx> {
    #[primary_span]
    pub span: Span,
    pub receiver_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("arguments for `{$abi}` function too large to pass via registers", code = E0798)]
#[note(
    "functions with the `{$abi}` ABI must pass all their arguments via the 4 32-bit argument registers"
)]
pub(crate) struct CmseInputsStackSpill {
    #[primary_span]
    #[label("does not fit in the available registers")]
    pub spans: Vec<Span>,
    pub abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("return value of `{$abi}` function too large to pass via registers", code = E0798)]
#[note("functions with the `{$abi}` ABI must pass their result via the available return registers")]
#[note(
    "the result must either be a (transparently wrapped) i64, u64 or f64, or be at most 4 bytes in size"
)]
pub(crate) struct CmseOutputStackSpill {
    #[primary_span]
    #[label("this type doesn't fit in the available registers")]
    pub span: Span,
    pub abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("generics are not allowed in `extern {$abi}` signatures", code = E0798)]
pub(crate) struct CmseGeneric {
    #[primary_span]
    pub span: Span,
    pub abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("`impl Trait` is not allowed in `extern {$abi}` signatures", code = E0798)]
pub(crate) struct CmseImplTrait {
    #[primary_span]
    pub span: Span,
    pub abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("return type notation not allowed in this position yet")]
pub(crate) struct BadReturnTypeNotation {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("trait item `{$item}` from `{$subtrait}` shadows identically named item from supertrait")]
pub(crate) struct SupertraitItemShadowing {
    pub item: Symbol,
    pub subtrait: Symbol,
    #[subdiagnostic]
    pub shadowee: SupertraitItemShadowee,
}

#[derive(Subdiagnostic)]
pub(crate) enum SupertraitItemShadowee {
    #[note("item from `{$supertrait}` is shadowed by a subtrait item")]
    Labeled {
        #[primary_span]
        span: Span,
        supertrait: Symbol,
    },
    #[note("items from several supertraits are shadowed: {$traits}")]
    Several {
        #[primary_span]
        spans: MultiSpan,
        traits: DiagSymbolList,
    },
}

#[derive(Diagnostic)]
#[diag("{$kind} binding in trait object type mentions `Self`")]
pub(crate) struct DynTraitAssocItemBindingMentionsSelf {
    #[primary_span]
    #[label("contains a mention of `Self`")]
    pub span: Span,
    pub kind: &'static str,
    #[label("this binding mentions `Self`")]
    pub binding: Span,
}

#[derive(Diagnostic)]
#[diag(
    "items with the \"custom\" ABI can only be declared externally or defined via naked functions"
)]
pub(crate) struct AbiCustomClothedFunction {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "convert this to an `#[unsafe(naked)]` function",
        applicability = "maybe-incorrect",
        code = "#[unsafe(naked)]\n",
        style = "short"
    )]
    pub naked_span: Span,
}

#[derive(Diagnostic)]
#[diag("`AsyncDrop` impl without `Drop` impl")]
#[help(
    "type implementing `AsyncDrop` trait must also implement `Drop` trait to be used in sync context and unwinds"
)]
pub(crate) struct AsyncDropWithoutSyncDrop {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("lifetime parameters or bounds of `{$ident}` do not match the declaration")]
pub(crate) struct LifetimesOrBoundsMismatchOnEii {
    #[primary_span]
    #[label("lifetimes do not match")]
    pub span: Span,
    #[label("lifetimes in impl do not match this signature")]
    pub generics_span: Span,
    #[label("this `where` clause might not match the one in the trait")]
    pub where_span: Option<Span>,
    #[label("this bound might be missing in the impl")]
    pub bounds_span: Vec<Span>,
    pub ident: Symbol,
}

#[derive(Diagnostic)]
#[diag("`{$impl_name}` cannot have generic parameters other than lifetimes")]
#[help("`#[{$eii_name}]` marks the implementation of an \"externally implementable item\"")]
pub(crate) struct EiiWithGenerics {
    #[primary_span]
    pub span: Span,
    #[label("required by this attribute")]
    pub attr: Span,
    pub eii_name: Symbol,
    pub impl_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("explicit impls for the `Unpin` trait are not permitted for structurally pinned types")]
pub(crate) struct ImplUnpinForPinProjectedType {
    #[primary_span]
    #[label("impl of `Unpin` not allowed")]
    pub span: Span,
    #[help("`{$adt_name}` is structurally pinned because it is marked as `#[pin_v2]`")]
    pub adt_span: Span,
    pub adt_name: Symbol,
}
