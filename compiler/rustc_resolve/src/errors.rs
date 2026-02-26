use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, DiagMessage, Diagnostic, ElidedLifetimeInPathSubdiag,
    EmissionGuarantee, IntoDiagArg, Level, LintDiagnostic, MultiSpan, Subdiagnostic, msg,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span, Symbol};

use crate::Res;
use crate::late::PatternSource;

#[derive(Diagnostic)]
#[diag("can't use {$is_self ->
        [true] `Self`
        *[false] generic parameters
    } from outer item", code = E0401)]
#[note(
    "nested items are independent from their parent item for everything except for privacy and name resolution"
)]
pub(crate) struct GenericParamsFromOuterItem {
    #[primary_span]
    #[label(
        "use of {$is_self ->
            [true] `Self`
            *[false] generic parameter
        } from outer item"
    )]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) label: Option<GenericParamsFromOuterItemLabel>,
    #[subdiagnostic]
    pub(crate) refer_to_type_directly: Option<UseTypeDirectly>,
    #[subdiagnostic]
    pub(crate) sugg: Option<GenericParamsFromOuterItemSugg>,
    #[subdiagnostic]
    pub(crate) static_or_const: Option<GenericParamsFromOuterItemStaticOrConst>,
    pub(crate) is_self: bool,
    #[subdiagnostic]
    pub(crate) item: Option<GenericParamsFromOuterItemInnerItem>,
}

#[derive(Subdiagnostic)]
#[label(
    "{$is_self ->
        [true] `Self`
        *[false] generic parameter
    } used in this inner {$descr}"
)]
pub(crate) struct GenericParamsFromOuterItemInnerItem {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) descr: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum GenericParamsFromOuterItemStaticOrConst {
    #[note("a `static` is a separate item from the item that contains it")]
    Static,
    #[note("a `const` is a separate item from the item that contains it")]
    Const,
}

#[derive(Subdiagnostic)]
pub(crate) enum GenericParamsFromOuterItemLabel {
    #[label("can't use `Self` here")]
    SelfTyParam(#[primary_span] Span),
    #[label("`Self` type implicitly declared here, by this `impl`")]
    SelfTyAlias(#[primary_span] Span),
    #[label("type parameter from outer item")]
    TyParam(#[primary_span] Span),
    #[label("const parameter from outer item")]
    ConstParam(#[primary_span] Span),
}

#[derive(Subdiagnostic)]
#[suggestion(
    "try introducing a local generic parameter here",
    code = "{snippet}",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct GenericParamsFromOuterItemSugg {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) snippet: String,
}
#[derive(Subdiagnostic)]
#[suggestion(
    "refer to the type directly here instead",
    code = "{snippet}",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct UseTypeDirectly {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) snippet: String,
}

#[derive(Diagnostic)]
#[diag("the name `{$name}` is already used for a generic parameter in this item's generic parameters", code = E0403)]
pub(crate) struct NameAlreadyUsedInParameterList {
    #[primary_span]
    #[label("already used")]
    pub(crate) span: Span,
    #[label("first use of `{$name}`")]
    pub(crate) first_use_span: Span,
    pub(crate) name: Ident,
}

#[derive(Diagnostic)]
#[diag("method `{$method}` is not a member of trait `{$trait_}`", code = E0407)]
pub(crate) struct MethodNotMemberOfTrait {
    #[primary_span]
    #[label("not a member of trait `{$trait_}`")]
    pub(crate) span: Span,
    pub(crate) method: Ident,
    pub(crate) trait_: String,
    #[subdiagnostic]
    pub(crate) sub: Option<AssociatedFnWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "there is an associated function with a similar name",
    code = "{candidate}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedFnWithSimilarNameExists {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) candidate: Symbol,
}

#[derive(Diagnostic)]
#[diag("type `{$type_}` is not a member of trait `{$trait_}`", code = E0437)]
pub(crate) struct TypeNotMemberOfTrait {
    #[primary_span]
    #[label("not a member of trait `{$trait_}`")]
    pub(crate) span: Span,
    pub(crate) type_: Ident,
    pub(crate) trait_: String,
    #[subdiagnostic]
    pub(crate) sub: Option<AssociatedTypeWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "there is an associated type with a similar name",
    code = "{candidate}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedTypeWithSimilarNameExists {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) candidate: Symbol,
}

#[derive(Diagnostic)]
#[diag("const `{$const_}` is not a member of trait `{$trait_}`", code = E0438)]
pub(crate) struct ConstNotMemberOfTrait {
    #[primary_span]
    #[label("not a member of trait `{$trait_}`")]
    pub(crate) span: Span,
    pub(crate) const_: Ident,
    pub(crate) trait_: String,
    #[subdiagnostic]
    pub(crate) sub: Option<AssociatedConstWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "there is an associated constant with a similar name",
    code = "{candidate}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedConstWithSimilarNameExists {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) candidate: Symbol,
}

#[derive(Diagnostic)]
#[diag("variable `{$variable_name}` is bound inconsistently across alternatives separated by `|`", code = E0409)]
pub(crate) struct VariableBoundWithDifferentMode {
    #[primary_span]
    #[label("bound in different ways")]
    pub(crate) span: Span,
    #[label("first binding")]
    pub(crate) first_binding_span: Span,
    pub(crate) variable_name: Ident,
}

#[derive(Diagnostic)]
#[diag("identifier `{$identifier}` is bound more than once in this parameter list", code = E0415)]
pub(crate) struct IdentifierBoundMoreThanOnceInParameterList {
    #[primary_span]
    #[label("used as parameter more than once")]
    pub(crate) span: Span,
    pub(crate) identifier: Ident,
}

#[derive(Diagnostic)]
#[diag("identifier `{$identifier}` is bound more than once in the same pattern", code = E0416)]
pub(crate) struct IdentifierBoundMoreThanOnceInSamePattern {
    #[primary_span]
    #[label("used in a pattern more than once")]
    pub(crate) span: Span,
    pub(crate) identifier: Ident,
}

#[derive(Diagnostic)]
#[diag("use of undeclared label `{$name}`", code = E0426)]
pub(crate) struct UndeclaredLabel {
    #[primary_span]
    #[label("undeclared label `{$name}`")]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) sub_reachable: Option<LabelWithSimilarNameReachable>,
    #[subdiagnostic]
    pub(crate) sub_reachable_suggestion: Option<TryUsingSimilarlyNamedLabel>,
    #[subdiagnostic]
    pub(crate) sub_unreachable: Option<UnreachableLabelWithSimilarNameExists>,
}

#[derive(Subdiagnostic)]
#[label("a label with a similar name is reachable")]
pub(crate) struct LabelWithSimilarNameReachable(#[primary_span] pub(crate) Span);

#[derive(Subdiagnostic)]
#[suggestion(
    "try using similarly named label",
    code = "{ident_name}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct TryUsingSimilarlyNamedLabel {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident_name: Symbol,
}

#[derive(Subdiagnostic)]
#[label("a label with a similar name exists but is unreachable")]
pub(crate) struct UnreachableLabelWithSimilarNameExists {
    #[primary_span]
    pub(crate) ident_span: Span,
}

#[derive(Diagnostic)]
#[diag("can't capture dynamic environment in a fn item", code = E0434)]
#[help("use the `|| {\"{\"} ... {\"}\"}` closure form instead")]
pub(crate) struct CannotCaptureDynamicEnvironmentInFnItem {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("attempt to use a non-constant value in a constant", code = E0435)]
pub(crate) struct AttemptToUseNonConstantValueInConstant<'a> {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) with: Option<AttemptToUseNonConstantValueInConstantWithSuggestion<'a>>,
    #[subdiagnostic]
    pub(crate) with_label: Option<AttemptToUseNonConstantValueInConstantLabelWithSuggestion>,
    #[subdiagnostic]
    pub(crate) without: Option<AttemptToUseNonConstantValueInConstantWithoutSuggestion<'a>>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider using `{$suggestion}` instead of `{$current}`",
    style = "verbose",
    applicability = "has-placeholders"
)]
pub(crate) struct AttemptToUseNonConstantValueInConstantWithSuggestion<'a> {
    // #[primary_span]
    #[suggestion_part(code = "{suggestion} ")]
    pub(crate) span: Span,
    pub(crate) suggestion: &'a str,
    #[suggestion_part(code = ": /* Type */")]
    pub(crate) type_span: Option<Span>,
    pub(crate) current: &'a str,
}

#[derive(Subdiagnostic)]
#[label("non-constant value")]
pub(crate) struct AttemptToUseNonConstantValueInConstantLabelWithSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[label("this would need to be a `{$suggestion}`")]
pub(crate) struct AttemptToUseNonConstantValueInConstantWithoutSuggestion<'a> {
    #[primary_span]
    pub(crate) ident_span: Span,
    pub(crate) suggestion: &'a str,
}

#[derive(Diagnostic)]
#[diag("`self` imports are only allowed within a {\"{\"} {\"}\"} list", code = E0429)]
pub(crate) struct SelfImportsOnlyAllowedWithin {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) suggestion: Option<SelfImportsOnlyAllowedWithinSuggestion>,
    #[subdiagnostic]
    pub(crate) mpart_suggestion: Option<SelfImportsOnlyAllowedWithinMultipartSuggestion>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "consider importing the module directly",
    code = "",
    applicability = "machine-applicable"
)]
pub(crate) struct SelfImportsOnlyAllowedWithinSuggestion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "alternatively, use the multi-path `use` syntax to import `self`",
    applicability = "machine-applicable"
)]
pub(crate) struct SelfImportsOnlyAllowedWithinMultipartSuggestion {
    #[suggestion_part(code = "{{")]
    pub(crate) multipart_start: Span,
    #[suggestion_part(code = "}}")]
    pub(crate) multipart_end: Span,
}

#[derive(Diagnostic)]
#[diag("{$shadowing_binding}s cannot shadow {$shadowed_binding}s", code = E0530)]
pub(crate) struct BindingShadowsSomethingUnacceptable<'a> {
    #[primary_span]
    #[label("cannot be named the same as {$article} {$shadowed_binding}")]
    pub(crate) span: Span,
    pub(crate) shadowing_binding: PatternSource,
    pub(crate) shadowed_binding: Res,
    pub(crate) article: &'a str,
    #[subdiagnostic]
    pub(crate) sub_suggestion: Option<BindingShadowsSomethingUnacceptableSuggestion>,
    #[label("the {$shadowed_binding} `{$name}` is {$participle} here")]
    pub(crate) shadowed_binding_span: Span,
    pub(crate) participle: &'a str,
    pub(crate) name: Symbol,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "try specify the pattern arguments",
    code = "{name}(..)",
    applicability = "unspecified"
)]
pub(crate) struct BindingShadowsSomethingUnacceptableSuggestion {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag("generic parameter defaults cannot reference parameters before they are declared", code = E0128)]
pub(crate) struct ForwardDeclaredGenericParam {
    #[primary_span]
    #[label("cannot reference `{$param}` before it is declared")]
    pub(crate) span: Span,
    pub(crate) param: Symbol,
}

#[derive(Diagnostic)]
#[diag("const parameter types cannot reference parameters before they are declared")]
pub(crate) struct ForwardDeclaredGenericInConstParamTy {
    #[primary_span]
    #[label("const parameter type cannot reference `{$param}` before it is declared")]
    pub(crate) span: Span,
    pub(crate) param: Symbol,
}

#[derive(Diagnostic)]
#[diag("the type of const parameters must not depend on other generic parameters", code = E0770)]
pub(crate) struct ParamInTyOfConstParam {
    #[primary_span]
    #[label("the type must not depend on the parameter `{$name}`")]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag("generic parameters cannot use `Self` in their defaults", code = E0735)]
pub(crate) struct SelfInGenericParamDefault {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot use `Self` in const parameter type")]
pub(crate) struct SelfInConstGenericTy {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "{$is_ogca ->
    [true] generic parameters in const blocks are only allowed as the direct value of a `type const`
    *[false] generic parameters may not be used in const operations
}"
)]
pub(crate) struct ParamInNonTrivialAnonConst {
    #[primary_span]
    #[label("cannot perform const operation using `{$name}`")]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) param_kind: ParamKindInNonTrivialAnonConst,
    #[help("add `#![feature(generic_const_exprs)]` to allow generic const expressions")]
    pub(crate) help: bool,
    pub(crate) is_ogca: bool,
    #[help(
        "consider factoring the expression into a `type const` item and use it as the const argument instead"
    )]
    pub(crate) help_ogca: bool,
}

#[derive(Debug)]
#[derive(Subdiagnostic)]
pub(crate) enum ParamKindInNonTrivialAnonConst {
    #[note("type parameters may not be used in const expressions")]
    Type,
    #[help("const parameters may only be used as standalone arguments here, i.e. `{$name}`")]
    Const { name: Symbol },
    #[note("lifetime parameters may not be used in const expressions")]
    Lifetime,
}

#[derive(Diagnostic)]
#[diag("use of unreachable label `{$name}`", code = E0767)]
#[note("labels are unreachable through functions, closures, async blocks and modules")]
pub(crate) struct UnreachableLabel {
    #[primary_span]
    #[label("unreachable label `{$name}`")]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[label("unreachable label defined here")]
    pub(crate) definition_span: Span,
    #[subdiagnostic]
    pub(crate) sub_suggestion: Option<UnreachableLabelSubSuggestion>,
    #[subdiagnostic]
    pub(crate) sub_suggestion_label: Option<UnreachableLabelSubLabel>,
    #[subdiagnostic]
    pub(crate) sub_unreachable_label: Option<UnreachableLabelSubLabelUnreachable>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "try using similarly named label",
    code = "{ident_name}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct UnreachableLabelSubSuggestion {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident_name: Symbol,
}

#[derive(Subdiagnostic)]
#[label("a label with a similar name is reachable")]
pub(crate) struct UnreachableLabelSubLabel {
    #[primary_span]
    pub(crate) ident_span: Span,
}

#[derive(Subdiagnostic)]
#[label("a label with a similar name exists but is also unreachable")]
pub(crate) struct UnreachableLabelSubLabelUnreachable {
    #[primary_span]
    pub(crate) ident_span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid `sym` operand")]
#[help("`sym` operands must refer to either a function or a static")]
pub(crate) struct InvalidAsmSym {
    #[primary_span]
    #[label("is a local variable")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("attempt to use a non-constant value in a constant")]
pub(crate) struct LowercaseSelf {
    #[primary_span]
    #[suggestion(
        "try using `Self`",
        code = "Self",
        applicability = "maybe-incorrect",
        style = "short"
    )]
    pub(crate) span: Span,
}

#[derive(Debug)]
#[derive(Diagnostic)]
#[diag("never patterns cannot contain variable bindings")]
pub(crate) struct BindingInNeverPattern {
    #[primary_span]
    #[suggestion(
        "use a wildcard `_` instead",
        code = "_",
        applicability = "machine-applicable",
        style = "short"
    )]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("duplicate definitions with name `{$name}`:", code = E0201)]
pub(crate) struct TraitImplDuplicate {
    #[primary_span]
    #[label("duplicate definition")]
    pub(crate) span: Span,
    #[label("previous definition here")]
    pub(crate) old_span: Span,
    #[label("item in trait")]
    pub(crate) trait_item_span: Span,
    pub(crate) name: Ident,
}

#[derive(Diagnostic)]
#[diag("relative paths are not supported in visibilities in 2018 edition or later")]
pub(crate) struct Relative2018 {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion("try", code = "crate::{path_str}", applicability = "maybe-incorrect")]
    pub(crate) path_span: Span,
    pub(crate) path_str: String,
}

#[derive(Diagnostic)]
#[diag("visibilities can only be restricted to ancestor modules", code = E0742)]
pub(crate) struct AncestorOnly(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag("expected module, found {$res} `{$path_str}`", code = E0577)]
pub(crate) struct ExpectedModuleFound {
    #[primary_span]
    #[label("not a module")]
    pub(crate) span: Span,
    pub(crate) res: Res,
    pub(crate) path_str: String,
}

#[derive(Diagnostic)]
#[diag("cannot determine resolution for the visibility", code = E0578)]
pub(crate) struct Indeterminate(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag("cannot use a tool module through an import")]
pub(crate) struct ToolModuleImported {
    #[primary_span]
    pub(crate) span: Span,
    #[note("the tool module imported here")]
    pub(crate) import: Span,
}

#[derive(Diagnostic)]
#[diag("visibility must resolve to a module")]
pub(crate) struct ModuleOnly(#[primary_span] pub(crate) Span);

#[derive(Diagnostic)]
#[diag("expected {$expected}, found {$found} `{$macro_path}`")]
pub(crate) struct MacroExpectedFound<'a> {
    #[primary_span]
    #[label("not {$article} {$expected}")]
    pub(crate) span: Span,
    pub(crate) found: &'a str,
    pub(crate) article: &'static str,
    pub(crate) expected: &'a str,
    pub(crate) macro_path: &'a str,
    #[subdiagnostic]
    pub(crate) remove_surrounding_derive: Option<RemoveSurroundingDerive>,
    #[subdiagnostic]
    pub(crate) add_as_non_derive: Option<AddAsNonDerive<'a>>,
}

#[derive(Subdiagnostic)]
#[help("remove from the surrounding `derive()`")]
pub(crate) struct RemoveSurroundingDerive {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[help(
    "
    add as non-Derive macro
    `#[{$macro_path}]`"
)]
pub(crate) struct AddAsNonDerive<'a> {
    pub(crate) macro_path: &'a str,
}

#[derive(Diagnostic)]
#[diag("can't use a procedural macro from the same crate that defines it")]
pub(crate) struct ProcMacroSameCrate {
    #[primary_span]
    pub(crate) span: Span,
    #[help("you can define integration tests in a directory named `tests`")]
    pub(crate) is_test: bool,
}

#[derive(LintDiagnostic)]
#[diag("cannot find {$ns_descr} `{$ident}` in this scope")]
pub(crate) struct ProcMacroDeriveResolutionFallback {
    #[label("names from parent modules are not accessible without an explicit import")]
    pub span: Span,
    pub ns_descr: &'static str,
    pub ident: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(
    "macro-expanded `macro_export` macros from the current crate cannot be referred to by absolute paths"
)]
pub(crate) struct MacroExpandedMacroExportsAccessedByAbsolutePaths {
    #[note("the macro is defined here")]
    pub definition: Span,
}

#[derive(Diagnostic)]
#[diag("`#[macro_use]` is not supported on `extern crate self`")]
pub(crate) struct MacroUseExternCrateSelf {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("not sure whether the path is accessible or not")]
#[note("the type may have associated items, but we are currently not checking them")]
pub(crate) struct CfgAccessibleUnsure {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Debug)]
#[derive(Diagnostic)]
#[diag("generic parameters may not be used in enum discriminant values")]
pub(crate) struct ParamInEnumDiscriminant {
    #[primary_span]
    #[label("cannot perform const operation using `{$name}`")]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    #[subdiagnostic]
    pub(crate) param_kind: ParamKindInEnumDiscriminant,
}

#[derive(Debug)]
#[derive(Subdiagnostic)]
pub(crate) enum ParamKindInEnumDiscriminant {
    #[note("type parameters may not be used in enum discriminant values")]
    Type,
    #[note("const parameters may not be used in enum discriminant values")]
    Const,
    #[note("lifetime parameters may not be used in enum discriminant values")]
    Lifetime,
}

#[derive(Subdiagnostic)]
#[label("you can use `as` to change the binding name of the import")]
pub(crate) struct ChangeImportBinding {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "you can use `as` to change the binding name of the import",
    code = "{suggestion}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ChangeImportBindingSuggestion {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) suggestion: String,
}

#[derive(Diagnostic)]
#[diag("imports cannot refer to {$what}")]
pub(crate) struct ImportsCannotReferTo<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) what: &'a str,
}

#[derive(Diagnostic)]
#[diag("cannot find {$expected} `{$ident}` in this scope")]
pub(crate) struct CannotFindIdentInThisScope<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) expected: &'a str,
    pub(crate) ident: Ident,
}

#[derive(Subdiagnostic)]
#[note("unsafe traits like `{$ident}` should be implemented explicitly")]
pub(crate) struct ExplicitUnsafeTraits {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Subdiagnostic)]
#[note("a macro with the same name exists, but it appears later")]
pub(crate) struct MacroDefinedLater {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[label("consider moving the definition of `{$ident}` before this call")]
pub(crate) struct MacroSuggMovePosition {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Subdiagnostic)]
pub(crate) enum MacroRulesNot {
    #[label("`{$ident}` exists, but has no rules for function-like invocation")]
    Func {
        #[primary_span]
        span: Span,
        ident: Ident,
    },
    #[label("`{$ident}` exists, but has no `attr` rules")]
    Attr {
        #[primary_span]
        span: Span,
        ident: Ident,
    },
    #[label("`{$ident}` exists, but has no `derive` rules")]
    Derive {
        #[primary_span]
        span: Span,
        ident: Ident,
    },
}

#[derive(Subdiagnostic)]
#[note("maybe you have forgotten to define a name for this `macro_rules!`")]
pub(crate) struct MaybeMissingMacroRulesName {
    #[primary_span]
    pub(crate) spans: MultiSpan,
}

#[derive(Subdiagnostic)]
#[help("have you added the `#[macro_use]` on the module/import?")]
pub(crate) struct AddedMacroUse;

#[derive(Subdiagnostic)]
#[suggestion("consider adding a derive", code = "{suggestion}", applicability = "maybe-incorrect")]
pub(crate) struct ConsiderAddingADerive {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) suggestion: String,
}

#[derive(Diagnostic)]
#[diag("cannot determine resolution for the import")]
pub(crate) struct CannotDetermineImportResolution {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot determine resolution for the {$kind} `{$path}`")]
#[note("import resolution is stuck, try simplifying macro imports")]
pub(crate) struct CannotDetermineMacroResolution {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) kind: &'static str,
    pub(crate) path: String,
}

#[derive(Diagnostic)]
#[diag("`{$ident}` is private, and cannot be re-exported", code = E0364)]
pub(crate) struct CannotBeReexportedPrivate {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("`{$ident}` is only public within the crate, and cannot be re-exported outside", code = E0364)]
pub(crate) struct CannotBeReexportedCratePublic {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("`{$ident}` is private, and cannot be re-exported", code = E0365)]
#[note("consider declaring type or module `{$ident}` with `pub`")]
pub(crate) struct CannotBeReexportedPrivateNS {
    #[primary_span]
    #[label("re-export of private `{$ident}`")]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("`{$ident}` is only public within the crate, and cannot be re-exported outside", code = E0365)]
#[note("consider declaring type or module `{$ident}` with `pub`")]
pub(crate) struct CannotBeReexportedCratePublicNS {
    #[primary_span]
    #[label("re-export of crate public `{$ident}`")]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag("extern crate `{$ident}` is private and cannot be re-exported", code = E0365)]
pub(crate) struct PrivateExternCrateReexport {
    pub ident: Ident,
    #[suggestion(
        "consider making the `extern crate` item publicly accessible",
        code = "pub ",
        style = "verbose",
        applicability = "maybe-incorrect"
    )]
    pub sugg: Span,
}

#[derive(Subdiagnostic)]
#[help("consider adding a `#[macro_export]` to the macro in the imported module")]
pub(crate) struct ConsiderAddingMacroExport {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "in case you want to use the macro within this crate only, reduce the visibility to `pub(crate)`",
    code = "pub(crate)",
    applicability = "maybe-incorrect"
)]
pub(crate) struct ConsiderMarkingAsPubCrate {
    #[primary_span]
    pub(crate) vis_span: Span,
}

#[derive(Subdiagnostic)]
#[note("consider marking `{$ident}` as `pub` in the imported module")]
pub(crate) struct ConsiderMarkingAsPub {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("cannot glob-import all possible crates")]
pub(crate) struct CannotGlobImportAllCrates {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "you might have meant to write a const parameter here",
    code = "const ",
    style = "verbose"
)]
pub(crate) struct UnexpectedResChangeTyToConstParamSugg {
    #[primary_span]
    pub span: Span,
    #[applicability]
    pub applicability: Applicability,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "you might have meant to write a const parameter here",
    applicability = "has-placeholders",
    style = "verbose"
)]
pub(crate) struct UnexpectedResChangeTyParamToConstParamSugg {
    #[suggestion_part(code = "const ")]
    pub before: Span,
    #[suggestion_part(code = ": /* Type */")]
    pub after: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "if you meant to collect the rest of the slice in `{$ident}`, use the at operator",
    code = "{snippet}",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct UnexpectedResUseAtOpInSlicePatWithRangeSugg {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
    pub snippet: String,
}

#[derive(Diagnostic)]
#[diag("an `extern crate` loading macros must be at the crate root", code = E0468)]
pub(crate) struct ExternCrateLoadingMacroNotAtCrateRoot {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`extern crate self;` requires renaming")]
pub(crate) struct ExternCrateSelfRequiresRenaming {
    #[primary_span]
    #[suggestion(
        "rename the `self` crate to be able to import it",
        code = "extern crate self as name;",
        applicability = "has-placeholders"
    )]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$name}` is already in scope")]
#[note("macro-expanded `#[macro_use]`s may not shadow existing macros (see RFC 1560)")]
pub(crate) struct MacroUseNameAlreadyInUse {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag("imported macro not found", code = E0469)]
pub(crate) struct ImportedMacroNotFound {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[macro_escape]` is a deprecated synonym for `#[macro_use]`")]
pub(crate) struct MacroExternDeprecated {
    #[primary_span]
    pub(crate) span: Span,
    #[help("try an outer attribute: `#[macro_use]`")]
    pub inner_attribute: bool,
}

#[derive(Diagnostic)]
#[diag("arguments to `macro_use` are not allowed here")]
pub(crate) struct ArgumentsMacroUseNotAllowed {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "try renaming it with a name",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct UnnamedImportSugg {
    #[suggestion_part(code = "{ident} as name")]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("imports need to be explicitly named")]
pub(crate) struct UnnamedImport {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) sugg: Option<UnnamedImportSugg>,
}

#[derive(Diagnostic)]
#[diag("macro-expanded `extern crate` items cannot shadow names passed with `--extern`")]
pub(crate) struct MacroExpandedExternCrateCannotShadowExternArguments {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`&` without an explicit lifetime name cannot be used here", code = E0637)]
pub(crate) struct ElidedAnonymousLifetimeReportError {
    #[primary_span]
    #[label("explicit lifetime name needed here")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) suggestion: Option<ElidedAnonymousLifetimeReportErrorSuggestion>,
}

#[derive(Diagnostic)]
#[diag(
    "associated type `Iterator::Item` is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type"
)]
pub(crate) struct LendingIteratorReportError {
    #[primary_span]
    pub(crate) lifetime: Span,
    #[note(
        "you can't create an `Iterator` that borrows each `Item` from itself, but you can instead create a new type that borrows your existing type and implement `Iterator` for that new type"
    )]
    pub(crate) ty: Span,
}

#[derive(Diagnostic)]
#[diag("missing lifetime in associated type")]
pub(crate) struct AnonymousLifetimeNonGatReportError {
    #[primary_span]
    #[label("this lifetime must come from the implemented type")]
    pub(crate) lifetime: Span,
    #[note(
        "in the trait the associated type is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type"
    )]
    pub(crate) decl: MultiSpan,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider introducing a higher-ranked lifetime here",
    applicability = "machine-applicable"
)]
pub(crate) struct ElidedAnonymousLifetimeReportErrorSuggestion {
    #[suggestion_part(code = "for<'a> ")]
    pub(crate) lo: Span,
    #[suggestion_part(code = "'a ")]
    pub(crate) hi: Span,
}

#[derive(Diagnostic)]
#[diag("`'_` cannot be used here", code = E0637)]
pub(crate) struct ExplicitAnonymousLifetimeReportError {
    #[primary_span]
    #[label("`'_` is a reserved lifetime name")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("implicit elided lifetime not allowed here", code = E0726)]
pub(crate) struct ImplicitElidedLifetimeNotAllowedHere {
    #[primary_span]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub(crate) subdiag: ElidedLifetimeInPathSubdiag,
}

#[derive(Diagnostic)]
#[diag("`'_` cannot be used here", code = E0637)]
#[help("use another lifetime specifier")]
pub(crate) struct UnderscoreLifetimeIsReserved {
    #[primary_span]
    #[label("`'_` is a reserved lifetime name")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("invalid lifetime parameter name: `{$lifetime}`", code = E0262)]
pub(crate) struct StaticLifetimeIsReserved {
    #[primary_span]
    #[label("'static is a reserved lifetime name")]
    pub(crate) span: Span,
    pub(crate) lifetime: Ident,
}

#[derive(Diagnostic)]
#[diag("variable `{$name}` is not bound in all patterns", code = E0408)]
pub(crate) struct VariableIsNotBoundInAllPatterns {
    #[primary_span]
    pub(crate) multispan: MultiSpan,
    pub(crate) name: Ident,
}

#[derive(Subdiagnostic, Debug, Clone)]
#[label("pattern doesn't bind `{$name}`")]
pub(crate) struct PatternDoesntBindName {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Ident,
}

#[derive(Subdiagnostic, Debug, Clone)]
#[label("variable not in all patterns")]
pub(crate) struct VariableNotInAllPatterns {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "you might have meant to use the similarly named previously used binding `{$typo}`",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct PatternBindingTypo {
    #[suggestion_part(code = "{typo}")]
    pub(crate) spans: Vec<Span>,
    pub(crate) typo: Symbol,
}

#[derive(Diagnostic)]
#[diag("the name `{$name}` is defined multiple times")]
#[note("`{$name}` must be defined only once in the {$descr} namespace of this {$container}")]
pub(crate) struct NameDefinedMultipleTime {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
    pub(crate) descr: &'static str,
    pub(crate) container: &'static str,
    #[subdiagnostic]
    pub(crate) label: NameDefinedMultipleTimeLabel,
    #[subdiagnostic]
    pub(crate) old_binding_label: Option<NameDefinedMultipleTimeOldBindingLabel>,
}

#[derive(Subdiagnostic)]
pub(crate) enum NameDefinedMultipleTimeLabel {
    #[label("`{$name}` reimported here")]
    Reimported {
        #[primary_span]
        span: Span,
    },
    #[label("`{$name}` redefined here")]
    Redefined {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum NameDefinedMultipleTimeOldBindingLabel {
    #[label("previous import of the {$old_kind} `{$name}` here")]
    Import {
        #[primary_span]
        span: Span,
        old_kind: &'static str,
    },
    #[label("previous definition of the {$old_kind} `{$name}` here")]
    Definition {
        #[primary_span]
        span: Span,
        old_kind: &'static str,
    },
}

#[derive(Diagnostic)]
#[diag("{$ident_descr} `{$ident}` is private", code = E0603)]
pub(crate) struct IsPrivate<'a> {
    #[primary_span]
    #[label("private {$ident_descr}")]
    pub(crate) span: Span,
    pub(crate) ident_descr: &'a str,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("generic arguments in macro path")]
pub(crate) struct GenericArgumentsInMacroPath {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("attributes starting with `rustc` are reserved for use by the `rustc` compiler")]
pub(crate) struct AttributesStartingWithRustcAreReserved {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot use {$article} {$descr} through an import")]
pub(crate) struct CannotUseThroughAnImport {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) article: &'static str,
    pub(crate) descr: &'static str,
    #[note("the {$descr} imported here")]
    pub(crate) binding_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("name `{$ident}` is reserved in attribute namespace")]
pub(crate) struct NameReservedInAttributeNamespace {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Symbol,
}

#[derive(Diagnostic)]
#[diag("cannot find a built-in macro with name `{$ident}`")]
pub(crate) struct CannotFindBuiltinMacroWithName {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("tool `{$tool}` was already registered")]
pub(crate) struct ToolWasAlreadyRegistered {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) tool: Ident,
    #[label("already registered here")]
    pub(crate) old_ident_span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum DefinedHere {
    #[label("similarly named {$candidate_descr} `{$candidate}` defined here")]
    SimilarlyNamed {
        #[primary_span]
        span: Span,
        candidate_descr: &'static str,
        candidate: Symbol,
    },
    #[label("{$candidate_descr} `{$candidate}` defined here")]
    SingleItem {
        #[primary_span]
        span: Span,
        candidate_descr: &'static str,
        candidate: Symbol,
    },
}

#[derive(Subdiagnostic)]
#[label("{$outer_ident_descr} `{$outer_ident}` is not publicly re-exported")]
pub(crate) struct OuterIdentIsNotPubliclyReexported {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) outer_ident_descr: &'static str,
    pub(crate) outer_ident: Ident,
}

#[derive(Subdiagnostic)]
#[label("a constructor is private if any of the fields is private")]
pub(crate) struct ConstructorPrivateIfAnyFieldPrivate {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "{ $number_of_fields ->
        [one] consider making the field publicly accessible
        *[other] consider making the fields publicly accessible
    }",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct ConsiderMakingTheFieldPublic {
    #[suggestion_part(code = "pub ")]
    pub(crate) spans: Vec<Span>,
    pub(crate) number_of_fields: usize,
}

#[derive(Subdiagnostic)]
pub(crate) enum ImportIdent {
    #[suggestion(
        "import `{$ident}` through the re-export",
        code = "{path}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    ThroughReExport {
        #[primary_span]
        span: Span,
        ident: Ident,
        path: String,
    },
    #[suggestion(
        "import `{$ident}` directly",
        code = "{path}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    Directly {
        #[primary_span]
        span: Span,
        ident: Ident,
        path: String,
    },
}

#[derive(Subdiagnostic)]
#[note(
    "{$first ->
        [true] {$dots ->
            [true] the {$binding_descr} `{$binding_name}` is defined here...
            *[false] the {$binding_descr} `{$binding_name}` is defined here
        }
        *[false] {$dots ->
            [true] ...and refers to the {$binding_descr} `{$binding_name}` which is defined here...
            *[false] ...and refers to the {$binding_descr} `{$binding_name}` which is defined here
        }
    }"
)]
pub(crate) struct NoteAndRefersToTheItemDefinedHere<'a> {
    #[primary_span]
    pub(crate) span: MultiSpan,
    pub(crate) binding_descr: &'a str,
    pub(crate) binding_name: Ident,
    pub(crate) first: bool,
    pub(crate) dots: bool,
}

#[derive(Subdiagnostic)]
#[suggestion("remove unnecessary import", code = "", applicability = "maybe-incorrect")]
pub(crate) struct RemoveUnnecessaryImport {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "remove unnecessary import",
    code = "",
    applicability = "maybe-incorrect",
    style = "tool-only"
)]
pub(crate) struct ToolOnlyRemoveUnnecessaryImport {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Subdiagnostic)]
#[note("`{$imported_ident}` is imported here, but it is {$imported_ident_desc}")]
pub(crate) struct IdentImporterHereButItIsDesc<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) imported_ident: Ident,
    pub(crate) imported_ident_desc: &'a str,
}

#[derive(Subdiagnostic)]
#[note("`{$imported_ident}` is in scope, but it is {$imported_ident_desc}")]
pub(crate) struct IdentInScopeButItIsDesc<'a> {
    pub(crate) imported_ident: Ident,
    pub(crate) imported_ident_desc: &'a str,
}

pub(crate) struct FoundItemConfigureOut {
    pub(crate) span: Span,
    pub(crate) item_was: ItemWas,
}

pub(crate) enum ItemWas {
    BehindFeature { feature: Symbol, span: Span },
    CfgOut { span: Span },
}

impl Subdiagnostic for FoundItemConfigureOut {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut multispan: MultiSpan = self.span.into();
        match self.item_was {
            ItemWas::BehindFeature { feature, span } => {
                let key = "feature".into();
                let value = feature.into_diag_arg(&mut None);
                let msg = diag.dcx.eagerly_translate_to_string(
                    msg!("the item is gated behind the `{$feature}` feature"),
                    [(&key, &value)].into_iter(),
                );
                multispan.push_span_label(span, msg);
            }
            ItemWas::CfgOut { span } => {
                multispan.push_span_label(span, msg!("the item is gated here"));
            }
        }
        diag.span_note(multispan, msg!("found an item that was configured out"));
    }
}

#[derive(Diagnostic)]
#[diag("item `{$name}` is an associated {$kind}, which doesn't match its trait `{$trait_path}`")]
pub(crate) struct TraitImplMismatch {
    #[primary_span]
    #[label("does not match trait")]
    pub(crate) span: Span,
    pub(crate) name: Ident,
    pub(crate) kind: &'static str,
    pub(crate) trait_path: String,
    #[label("item in trait")]
    pub(crate) trait_item_span: Span,
}

#[derive(LintDiagnostic)]
#[diag("derive helper attribute is used before it is introduced")]
pub(crate) struct LegacyDeriveHelpers {
    #[label("the attribute is introduced here")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("unused extern crate")]
pub(crate) struct UnusedExternCrate {
    #[label("unused")]
    pub span: Span,
    #[suggestion(
        "remove the unused `extern crate`",
        code = "",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub removal_span: Span,
}

#[derive(LintDiagnostic)]
#[diag("{$kind} `{$name}` from private dependency '{$krate}' is re-exported")]
pub(crate) struct ReexportPrivateDependency {
    pub name: Symbol,
    pub kind: &'static str,
    pub krate: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("unused label")]
pub(crate) struct UnusedLabel;

#[derive(LintDiagnostic)]
#[diag("unused `#[macro_use]` import")]
pub(crate) struct UnusedMacroUse;

#[derive(LintDiagnostic)]
#[diag("applying the `#[macro_use]` attribute to an `extern crate` item is deprecated")]
#[help("remove it and import macros at use sites with a `use` item instead")]
pub(crate) struct MacroUseDeprecated;

#[derive(LintDiagnostic)]
#[diag("macro `{$ident}` is private")]
pub(crate) struct MacroIsPrivate {
    pub ident: Ident,
}

#[derive(LintDiagnostic)]
#[diag("unused macro definition: `{$name}`")]
pub(crate) struct UnusedMacroDefinition {
    pub name: Symbol,
}

#[derive(LintDiagnostic)]
#[diag("rule #{$n} of macro `{$name}` is never used")]
pub(crate) struct MacroRuleNeverUsed {
    pub n: usize,
    pub name: Symbol,
}

pub(crate) struct UnstableFeature {
    pub msg: DiagMessage,
}

impl<'a> LintDiagnostic<'a, ()> for UnstableFeature {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.primary_message(self.msg);
    }
}

#[derive(LintDiagnostic)]
#[diag("`extern crate` is not idiomatic in the new edition")]
pub(crate) struct ExternCrateNotIdiomatic {
    #[suggestion(
        "convert it to a `use`",
        style = "verbose",
        code = "{code}",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub code: &'static str,
}

#[derive(LintDiagnostic)]
#[diag("cannot find macro `{$path}` in the current scope when looking from {$location}")]
#[help("import `macro_rules` with `use` to make it callable above its definition")]
pub(crate) struct OutOfScopeMacroCalls {
    #[label("not found from {$location}")]
    pub span: Span,
    pub path: String,
    pub location: String,
}

#[derive(LintDiagnostic)]
#[diag(
    "glob import doesn't reexport anything with visibility `{$import_vis}` because no imported item is public enough"
)]
pub(crate) struct RedundantImportVisibility {
    #[note("the most public imported item is `{$max_vis}`")]
    pub span: Span,
    #[help("reduce the glob import's visibility or increase visibility of imported items")]
    pub help: (),
    pub import_vis: String,
    pub max_vis: String,
}

#[derive(LintDiagnostic)]
#[diag("unknown diagnostic attribute")]
pub(crate) struct UnknownDiagnosticAttribute {
    #[subdiagnostic]
    pub typo: Option<UnknownDiagnosticAttributeTypoSugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "an attribute with a similar name exists",
    style = "verbose",
    code = "{typo_name}",
    applicability = "machine-applicable"
)]
pub(crate) struct UnknownDiagnosticAttributeTypoSugg {
    #[primary_span]
    pub span: Span,
    pub typo_name: Symbol,
}

// FIXME: Make this properly translatable.
pub(crate) struct Ambiguity {
    pub ident: Ident,
    pub ambig_vis: Option<String>,
    pub kind: &'static str,
    pub help: Option<&'static [&'static str]>,
    pub b1_note: Spanned<String>,
    pub b1_help_msgs: Vec<String>,
    pub b2_note: Spanned<String>,
    pub b2_help_msgs: Vec<String>,
}

impl Ambiguity {
    fn decorate<'a>(self, diag: &mut Diag<'a, impl EmissionGuarantee>) {
        if let Some(ambig_vis) = self.ambig_vis {
            diag.primary_message(format!("ambiguous import visibility: {ambig_vis}"));
        } else {
            diag.primary_message(format!("`{}` is ambiguous", self.ident));
            diag.span_label(self.ident.span, "ambiguous name");
        }
        diag.note(format!("ambiguous because of {}", self.kind));
        diag.span_note(self.b1_note.span, self.b1_note.node);
        if let Some(help) = self.help {
            for help in help {
                diag.help(*help);
            }
        }
        for help_msg in self.b1_help_msgs {
            diag.help(help_msg);
        }
        diag.span_note(self.b2_note.span, self.b2_note.node);
        for help_msg in self.b2_help_msgs {
            diag.help(help_msg);
        }
    }
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for Ambiguity {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(dcx, level, "").with_span(self.ident.span).with_code(E0659);
        self.decorate(&mut diag);
        diag
    }
}

impl<'a> LintDiagnostic<'a, ()> for Ambiguity {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        self.decorate(diag);
    }
}
