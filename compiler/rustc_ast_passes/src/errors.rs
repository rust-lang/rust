//! Errors emitted by ast_passes.

use rustc_abi::ExternAbi;
use rustc_ast::ParamKindOrd;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, EmissionGuarantee, Subdiagnostic};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag("visibility qualifiers are not permitted here", code = E0449)]
pub(crate) struct VisibilityNotPermitted {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub note: VisibilityNotPermittedNote,
    #[suggestion("remove the qualifier", code = "", applicability = "machine-applicable")]
    pub remove_qualifier_sugg: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum VisibilityNotPermittedNote {
    #[note("enum variants and their fields always share the visibility of the enum they are in")]
    EnumVariant,
    #[note("trait items always share the visibility of their trait")]
    TraitImpl,
    #[note("place qualifiers on individual impl items instead")]
    IndividualImplItems,
    #[note("place qualifiers on individual foreign items instead")]
    IndividualForeignItems,
}
#[derive(Diagnostic)]
#[diag("redundant `const` fn marker in const impl")]
pub(crate) struct ImplFnConst {
    #[primary_span]
    #[suggestion("remove the `const`", code = "", applicability = "machine-applicable")]
    pub span: Span,
    #[label("this declares all associated functions implicitly const")]
    pub parent_constness: Span,
}

#[derive(Diagnostic)]
#[diag("functions in {$in_impl ->
        [true] trait impls
        *[false] traits
    } cannot be declared const", code = E0379)]
pub(crate) struct TraitFnConst {
    #[primary_span]
    #[label(
        "functions in {$in_impl ->
            [true] trait impls
            *[false] traits
        } cannot be const"
    )]
    pub span: Span,
    pub in_impl: bool,
    #[label("this declares all associated functions implicitly const")]
    pub const_context_label: Option<Span>,
    #[suggestion(
        "remove the `const`{$requires_multiple_changes ->
            [true] {\" ...\"}
            *[false] {\"\"}
        }",
        code = ""
    )]
    pub remove_const_sugg: (Span, Applicability),
    pub requires_multiple_changes: bool,
    #[suggestion(
        "... and declare the impl to be const instead",
        code = "const ",
        applicability = "maybe-incorrect"
    )]
    pub make_impl_const_sugg: Option<Span>,
    #[suggestion(
        "... and declare the trait to be const instead",
        code = "const ",
        applicability = "maybe-incorrect"
    )]
    pub make_trait_const_sugg: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(
    "async functions are not allowed in `const` {$context ->
        [trait_impl] trait impls
        [impl] impls
        *[trait] traits
    }"
)]
pub(crate) struct AsyncFnInConstTraitOrTraitImpl {
    #[primary_span]
    pub async_keyword: Span,
    pub context: &'static str,
    #[label("associated functions of `const` cannot be declared `async`")]
    pub const_keyword: Span,
}

#[derive(Diagnostic)]
#[diag("bounds cannot be used in this context")]
pub(crate) struct ForbiddenBound {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("late-bound const parameters cannot be used currently")]
pub(crate) struct ForbiddenConstParam {
    #[primary_span]
    pub const_param_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("function can not have more than {$max_num_args} arguments")]
pub(crate) struct FnParamTooMany {
    #[primary_span]
    pub span: Span,
    pub max_num_args: usize,
}

#[derive(Diagnostic)]
#[diag("`...` must be the last argument of a C-variadic function")]
pub(crate) struct FnParamCVarArgsNotLast {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("documentation comments cannot be applied to function parameters")]
pub(crate) struct FnParamDocComment {
    #[primary_span]
    #[label("doc comments are not allowed here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters"
)]
pub(crate) struct FnParamForbiddenAttr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`self` parameter is only allowed in associated functions")]
#[note("associated functions are those in `impl` or `trait` definitions")]
pub(crate) struct FnParamForbiddenSelf {
    #[primary_span]
    #[label("not semantically valid as function parameter")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`default` is only allowed on items in trait impls")]
pub(crate) struct ForbiddenDefault {
    #[primary_span]
    pub span: Span,
    #[label("`default` because of this")]
    pub def_span: Span,
}

#[derive(Diagnostic)]
#[diag("associated constant in `impl` without body")]
pub(crate) struct AssocConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the constant",
        code = " = <expr>;",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag("associated function in `impl` without body")]
pub(crate) struct AssocFnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the function",
        code = " {{ <body> }}",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag("associated type in `impl` without body")]
pub(crate) struct AssocTypeWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the type",
        code = " = <type>;",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag("free constant item without body")]
pub(crate) struct ConstWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the constant",
        code = " = <expr>;",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag("free static item without body")]
pub(crate) struct StaticWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the static",
        code = " = <expr>;",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag("free type alias without body")]
pub(crate) struct TyAliasWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the type",
        code = " = <type>;",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
}

#[derive(Diagnostic)]
#[diag("free function without a body")]
pub(crate) struct FnWithoutBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "provide a definition for the function",
        code = " {{ <body> }}",
        applicability = "has-placeholders"
    )]
    pub replace_span: Span,
    #[subdiagnostic]
    pub extern_block_suggestion: Option<ExternBlockSuggestion>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ExternBlockSuggestion {
    #[multipart_suggestion(
        "if you meant to declare an externally defined function, use an `extern` block",
        applicability = "maybe-incorrect"
    )]
    Implicit {
        #[suggestion_part(code = "extern {{")]
        start_span: Span,
        #[suggestion_part(code = " }}")]
        end_span: Span,
    },
    #[multipart_suggestion(
        "if you meant to declare an externally defined function, use an `extern` block",
        applicability = "maybe-incorrect"
    )]
    Explicit {
        #[suggestion_part(code = "extern \"{abi}\" {{")]
        start_span: Span,
        #[suggestion_part(code = " }}")]
        end_span: Span,
        abi: Symbol,
    },
}

#[derive(Diagnostic)]
#[diag("items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers")]
pub(crate) struct InvalidSafetyOnExtern {
    #[primary_span]
    pub item_span: Span,
    #[suggestion(
        "add `unsafe` to this `extern` block",
        code = "unsafe ",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub block: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(
    "items outside of `unsafe extern {\"{ }\"}` cannot be declared with `safe` safety qualifier"
)]
pub(crate) struct InvalidSafetyOnItem {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("function pointers cannot be declared with `safe` safety qualifier")]
pub(crate) struct InvalidSafetyOnFnPtr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("static items cannot be declared with `unsafe` safety qualifier outside of `extern` block")]
pub(crate) struct UnsafeStatic {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("bounds on `type`s in {$ctx} have no effect")]
pub(crate) struct BoundInContext<'a> {
    #[primary_span]
    pub span: Span,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[diag("`type`s inside `extern` blocks cannot have {$descr}")]
#[note("for more information, visit https://doc.rust-lang.org/std/keyword.extern.html")]
pub(crate) struct ExternTypesCannotHave<'a> {
    #[primary_span]
    #[suggestion("remove the {$remove_descr}", code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    pub descr: &'a str,
    pub remove_descr: &'a str,
    #[label("`extern` block begins here")]
    pub block_span: Span,
}

#[derive(Diagnostic)]
#[diag("incorrect `{$kind}` inside `extern` block")]
#[note("for more information, visit https://doc.rust-lang.org/std/keyword.extern.html")]
pub(crate) struct BodyInExtern<'a> {
    #[primary_span]
    #[label("cannot have a body")]
    pub span: Span,
    #[label("the invalid body")]
    pub body: Span,
    #[label(
        "`extern` blocks define existing foreign {$kind}s and {$kind}s inside of them cannot have a body"
    )]
    pub block: Span,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag("incorrect function inside `extern` block")]
#[help(
    "you might have meant to write a function accessible through FFI, which can be done by writing `extern fn` outside of the `extern` block"
)]
#[note("for more information, visit https://doc.rust-lang.org/std/keyword.extern.html")]
pub(crate) struct FnBodyInExtern {
    #[primary_span]
    #[label("cannot have a body")]
    pub span: Span,
    #[suggestion("remove the invalid body", code = ";", applicability = "maybe-incorrect")]
    pub body: Span,
    #[label(
        "`extern` blocks define existing foreign functions and functions inside of them cannot have a body"
    )]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag("functions in `extern` blocks cannot have `{$kw}` qualifier")]
pub(crate) struct FnQualifierInExtern {
    #[primary_span]
    #[suggestion("remove the `{$kw}` qualifier", code = "", applicability = "maybe-incorrect")]
    pub span: Span,
    #[label("in this `extern` block")]
    pub block: Span,
    pub kw: &'static str,
}

#[derive(Diagnostic)]
#[diag("items in `extern` blocks cannot use non-ascii identifiers")]
#[note(
    "this limitation may be lifted in the future; see issue #83942 <https://github.com/rust-lang/rust/issues/83942> for more information"
)]
pub(crate) struct ExternItemAscii {
    #[primary_span]
    pub span: Span,
    #[label("in this `extern` block")]
    pub block: Span,
}

#[derive(Diagnostic)]
#[diag("`...` is not supported for non-extern functions")]
#[help(
    "only `extern \"C\"` and `extern \"C-unwind\"` functions may have a C variable argument list"
)]
pub(crate) struct CVariadicNoExtern {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("functions with a C variable argument list must be unsafe")]
pub(crate) struct CVariadicMustBeUnsafe {
    #[primary_span]
    pub span: Span,

    #[suggestion(
        "add the `unsafe` keyword to this definition",
        applicability = "maybe-incorrect",
        code = "unsafe ",
        style = "verbose"
    )]
    pub unsafe_span: Span,
}

#[derive(Diagnostic)]
#[diag("`...` is not supported for `extern \"{$abi}\"` functions")]
#[help(
    "only `extern \"C\"` and `extern \"C-unwind\"` functions may have a C variable argument list"
)]
pub(crate) struct CVariadicBadExtern {
    #[primary_span]
    pub span: Span,
    pub abi: &'static str,
    #[label("`extern \"{$abi}\"` because of this")]
    pub extern_span: Span,
}

#[derive(Diagnostic)]
#[diag("`...` is not supported for `extern \"{$abi}\"` naked functions")]
#[help("C-variadic function must have a compatible calling convention")]
pub(crate) struct CVariadicBadNakedExtern {
    #[primary_span]
    pub span: Span,
    pub abi: &'static str,
    #[label("`extern \"{$abi}\"` because of this")]
    pub extern_span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$kind}` items in this context need a name")]
pub(crate) struct ItemUnderscore<'a> {
    #[primary_span]
    #[label("`_` is not a valid name for this `{$kind}` item")]
    pub span: Span,
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag("`#[no_mangle]` requires ASCII identifier", code = E0754)]
pub(crate) struct NoMangleAscii {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("trying to load file for module `{$name}` with non-ascii identifier name", code = E0754)]
#[help("consider using the `#[path]` attribute to specify filesystem path")]
pub(crate) struct ModuleNonAscii {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("auto traits cannot have generic parameters", code = E0567)]
pub(crate) struct AutoTraitGeneric {
    #[primary_span]
    #[suggestion(
        "remove the parameters",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub span: Span,
    #[label("auto trait cannot have generic parameters")]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag("auto traits cannot have super traits or lifetime bounds", code = E0568)]
pub(crate) struct AutoTraitBounds {
    #[primary_span]
    pub span: Vec<Span>,
    #[suggestion(
        "remove the super traits or lifetime bounds",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub removal: Span,
    #[label("auto traits cannot have super traits or lifetime bounds")]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag("auto traits cannot have associated items", code = E0380)]
pub(crate) struct AutoTraitItems {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(
        "remove the associated items",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub total: Span,
    #[label("auto traits cannot have associated items")]
    pub ident: Span,
}

#[derive(Diagnostic)]
#[diag("auto traits cannot be const")]
#[help("remove the `const` keyword")]
pub(crate) struct ConstAutoTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("generic arguments must come before the first constraint")]
pub(crate) struct ArgsBeforeConstraint {
    #[primary_span]
    pub arg_spans: Vec<Span>,
    #[label(
        "{$constraint_len ->
            [one] constraint
            *[other] constraints
        }"
    )]
    pub constraints: Span,
    #[label(
        "generic {$args_len ->
            [one] argument
            *[other] arguments
        }"
    )]
    pub args: Span,
    #[suggestion(
        "move the {$constraint_len ->
            [one] constraint
            *[other] constraints
        } after the generic {$args_len ->
            [one] argument
            *[other] arguments
        }",
        code = "{suggestion}",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    pub data: Span,
    pub suggestion: String,
    pub constraint_len: usize,
    pub args_len: usize,
    #[subdiagnostic]
    pub constraint_spans: EmptyLabelManySpans,
    #[subdiagnostic]
    pub arg_spans2: EmptyLabelManySpans,
}

pub(crate) struct EmptyLabelManySpans(pub Vec<Span>);

// The derive for `Vec<Span>` does multiple calls to `span_label`, adding commas between each
impl Subdiagnostic for EmptyLabelManySpans {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_labels(self.0, "");
    }
}

#[derive(Diagnostic)]
#[diag("patterns aren't allowed in function pointer types", code = E0561)]
pub(crate) struct PatternFnPointer {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("only a single explicit lifetime bound is permitted", code = E0226)]
pub(crate) struct TraitObjectBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("nested `impl Trait` is not allowed", code = E0666)]
pub(crate) struct NestedImplTrait {
    #[primary_span]
    pub span: Span,
    #[label("outer `impl Trait`")]
    pub outer: Span,
    #[label("nested `impl Trait` here")]
    pub inner: Span,
}

#[derive(Diagnostic)]
#[diag("at least one trait must be specified")]
pub(crate) struct AtLeastOneTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$param_ord} parameters must be declared prior to {$max_param} parameters")]
pub(crate) struct OutOfOrderParams<'a> {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(
        "reorder the parameters: lifetimes, then consts and types",
        code = "{ordered_params}",
        applicability = "machine-applicable"
    )]
    pub sugg_span: Span,
    pub param_ord: &'a ParamKindOrd,
    pub max_param: &'a ParamKindOrd,
    pub ordered_params: &'a str,
}

#[derive(Diagnostic)]
#[diag("`impl Trait for .. {\"{}\"}` is an obsolete syntax")]
#[help("use `auto trait Trait {\"{}\"}` instead")]
pub(crate) struct ObsoleteAuto {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("negative impls cannot be unsafe", code = E0198)]
pub(crate) struct UnsafeNegativeImpl {
    #[primary_span]
    pub span: Span,
    #[label("negative because of this")]
    pub negative: Span,
    #[label("unsafe because of this")]
    pub r#unsafe: Span,
}

#[derive(Diagnostic)]
#[diag("{$kind} cannot be declared unsafe")]
pub(crate) struct UnsafeItem {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag("extern blocks must be unsafe")]
pub(crate) struct MissingUnsafeOnExtern {
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("extern blocks should be unsafe")]
pub(crate) struct MissingUnsafeOnExternLint {
    #[suggestion(
        "needs `unsafe` before the extern keyword",
        code = "unsafe ",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("unions cannot have zero fields")]
pub(crate) struct FieldlessUnion {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("where clauses are not allowed after the type for type aliases")]
#[note("see issue #112792 <https://github.com/rust-lang/rust/issues/112792> for more information")]
pub(crate) struct WhereClauseAfterTypeAlias {
    #[primary_span]
    pub span: Span,
    #[help("add `#![feature(lazy_type_alias)]` to the crate attributes to enable")]
    pub help: bool,
}

#[derive(Diagnostic)]
#[diag("where clauses are not allowed before the type for type aliases")]
#[note("see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information")]
pub(crate) struct WhereClauseBeforeTypeAlias {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sugg: WhereClauseBeforeTypeAliasSugg,
}

#[derive(Subdiagnostic)]
pub(crate) enum WhereClauseBeforeTypeAliasSugg {
    #[suggestion("remove this `where`", applicability = "machine-applicable", code = "")]
    Remove {
        #[primary_span]
        span: Span,
    },
    #[multipart_suggestion(
        "move it to the end of the type declaration",
        applicability = "machine-applicable",
        style = "verbose"
    )]
    Move {
        #[suggestion_part(code = "")]
        left: Span,
        snippet: String,
        #[suggestion_part(code = "{snippet}")]
        right: Span,
    },
}

#[derive(Diagnostic)]
#[diag("generic parameters with a default must be trailing")]
pub(crate) struct GenericDefaultTrailing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("nested quantification of lifetimes", code = E0316)]
pub(crate) struct NestedLifetimes {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("const trait bounds are not allowed in trait object types")]
pub(crate) struct ConstBoundTraitObject {
    #[primary_span]
    pub span: Span,
}

// FIXME(const_trait_impl): Consider making the note/reason the message of the diagnostic.
// FIXME(const_trait_impl): Provide structured suggestions (e.g., add `const` here).
#[derive(Diagnostic)]
#[diag("`[const]` is not allowed here")]
pub(crate) struct TildeConstDisallowed {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub reason: TildeConstReason,
}

#[derive(Subdiagnostic, Copy, Clone)]
pub(crate) enum TildeConstReason {
    #[note("closures cannot have `[const]` trait bounds")]
    Closure,
    #[note("this function is not `const`, so it cannot have `[const]` trait bounds")]
    Function {
        #[primary_span]
        ident: Span,
    },
    #[note("this trait is not `const`, so it cannot have `[const]` trait bounds")]
    Trait {
        #[primary_span]
        span: Span,
    },
    #[note("this impl is not `const`, so it cannot have `[const]` trait bounds")]
    TraitImpl {
        #[primary_span]
        span: Span,
    },
    #[note("inherent impls cannot have `[const]` trait bounds")]
    Impl {
        #[primary_span]
        span: Span,
    },
    #[note("associated types in non-`const` traits cannot have `[const]` trait bounds")]
    TraitAssocTy {
        #[primary_span]
        span: Span,
    },
    #[note("associated types in non-const impls cannot have `[const]` trait bounds")]
    TraitImplAssocTy {
        #[primary_span]
        span: Span,
    },
    #[note("inherent associated types cannot have `[const]` trait bounds")]
    InherentAssocTy {
        #[primary_span]
        span: Span,
    },
    #[note("structs cannot have `[const]` trait bounds")]
    Struct {
        #[primary_span]
        span: Span,
    },
    #[note("enums cannot have `[const]` trait bounds")]
    Enum {
        #[primary_span]
        span: Span,
    },
    #[note("unions cannot have `[const]` trait bounds")]
    Union {
        #[primary_span]
        span: Span,
    },
    #[note("anonymous constants cannot have `[const]` trait bounds")]
    AnonConst {
        #[primary_span]
        span: Span,
    },
    #[note("trait objects cannot have `[const]` trait bounds")]
    TraitObject,
    #[note("this item cannot have `[const]` trait bounds")]
    Item,
}

#[derive(Diagnostic)]
#[diag("functions cannot be both `const` and `{$coroutine_kind}`")]
pub(crate) struct ConstAndCoroutine {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label("`const` because of this")]
    pub const_span: Span,
    #[label("`{$coroutine_kind}` because of this")]
    pub coroutine_span: Span,
    #[label("{\"\"}")]
    pub span: Span,
    pub coroutine_kind: &'static str,
}

#[derive(Diagnostic)]
#[diag("functions cannot be both `const` and C-variadic")]
pub(crate) struct ConstAndCVariadic {
    #[primary_span]
    pub spans: Vec<Span>,
    #[label("`const` because of this")]
    pub const_span: Span,
    #[label("C-variadic because of this")]
    pub variadic_span: Span,
}

#[derive(Diagnostic)]
#[diag("functions cannot be both `{$coroutine_kind}` and C-variadic")]
pub(crate) struct CoroutineAndCVariadic {
    #[primary_span]
    pub spans: Vec<Span>,
    pub coroutine_kind: &'static str,
    #[label("`{$coroutine_kind}` because of this")]
    pub coroutine_span: Span,
    #[label("C-variadic because of this")]
    pub variadic_span: Span,
}

#[derive(Diagnostic)]
#[diag("the `{$target}` target does not support c-variadic functions")]
pub(crate) struct CVariadicNotSupported<'a> {
    #[primary_span]
    pub variadic_span: Span,
    pub target: &'a str,
}

#[derive(Diagnostic)]
#[diag("patterns aren't allowed in foreign function declarations", code = E0130)]
// FIXME: deduplicate with rustc_lint (`BuiltinLintDiag::PatternsInFnsWithoutBody`)
pub(crate) struct PatternInForeign {
    #[primary_span]
    #[label("pattern not allowed in foreign function")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("patterns aren't allowed in functions without bodies", code = E0642)]
// FIXME: deduplicate with rustc_lint (`BuiltinLintDiag::PatternsInFnsWithoutBody`)
pub(crate) struct PatternInBodiless {
    #[primary_span]
    #[label("pattern not allowed in function without body")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("equality constraints are not yet supported in `where` clauses")]
#[note("see issue #20041 <https://github.com/rust-lang/rust/issues/20041> for more information")]
pub(crate) struct EqualityInWhere {
    #[primary_span]
    #[label("not supported")]
    pub span: Span,
    #[subdiagnostic]
    pub assoc: Option<AssociatedSuggestion>,
    #[subdiagnostic]
    pub assoc2: Option<AssociatedSuggestion2>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "if `{$ident}` is an associated type you're trying to set, use the associated type binding syntax",
    code = "{param}: {path}",
    style = "verbose",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedSuggestion {
    #[primary_span]
    pub span: Span,
    pub ident: Ident,
    pub param: Ident,
    pub path: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "if `{$trait_segment}::{$potential_assoc}` is an associated type you're trying to set, use the associated type binding syntax",
    applicability = "maybe-incorrect"
)]
pub(crate) struct AssociatedSuggestion2 {
    #[suggestion_part(code = "{args}")]
    pub span: Span,
    pub args: String,
    #[suggestion_part(code = "")]
    pub predicate: Span,
    pub trait_segment: Ident,
    pub potential_assoc: Ident,
}

#[derive(Diagnostic)]
#[diag("`#![feature]` may not be used on the {$channel} release channel", code = E0554)]
pub(crate) struct FeatureOnNonNightly {
    #[primary_span]
    pub span: Span,
    pub channel: &'static str,
    #[subdiagnostic]
    pub stable_features: Vec<StableFeature>,
    #[suggestion("remove the attribute", code = "", applicability = "machine-applicable")]
    pub sugg: Option<Span>,
}

#[derive(Subdiagnostic)]
#[help(
    "the feature `{$name}` has been stable since `{$since}` and no longer requires an attribute to enable"
)]
pub(crate) struct StableFeature {
    pub name: Symbol,
    pub since: Symbol,
}

#[derive(Diagnostic)]
#[diag("`{$f1}` and `{$f2}` are incompatible, using them at the same time is not allowed")]
#[help("remove one of these features")]
pub(crate) struct IncompatibleFeatures {
    #[primary_span]
    pub spans: Vec<Span>,
    pub f1: Symbol,
    pub f2: Symbol,
}

#[derive(Diagnostic)]
#[diag("`{$parent}` requires {$missing} to be enabled")]
#[help("enable all of these features")]
pub(crate) struct MissingDependentFeatures {
    #[primary_span]
    pub parent_span: Span,
    pub parent: Symbol,
    pub missing: String,
}

#[derive(Diagnostic)]
#[diag("negative bounds are not supported")]
pub(crate) struct NegativeBoundUnsupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("associated type constraints not allowed on negative bounds")]
pub(crate) struct ConstraintOnNegativeBound {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("parenthetical notation may not be used for negative bounds")]
pub(crate) struct NegativeBoundWithParentheticalNotation {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`match` arm with no body")]
pub(crate) struct MatchArmWithNoBody {
    #[primary_span]
    pub span: Span,
    // We include the braces around `todo!()` so that a comma is optional, and we don't have to have
    // any logic looking at the arm being replaced if there was a comma already or not for the
    // resulting code to be correct.
    #[suggestion(
        "add a body after the pattern",
        code = " => {{ todo!() }}",
        applicability = "has-placeholders",
        style = "verbose"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("`use<...>` precise capturing syntax not allowed in {$loc}")]
pub(crate) struct PreciseCapturingNotAllowedHere {
    #[primary_span]
    pub span: Span,
    pub loc: &'static str,
}

#[derive(Diagnostic)]
#[diag("duplicate `use<...>` precise capturing syntax")]
pub(crate) struct DuplicatePreciseCapturing {
    #[primary_span]
    pub bound1: Span,
    #[label("second `use<...>` here")]
    pub bound2: Span,
}

#[derive(Diagnostic)]
#[diag("`extern` declarations without an explicit ABI are disallowed")]
#[help("prior to Rust 2024, a default ABI was inferred")]
pub(crate) struct MissingAbi {
    #[primary_span]
    #[suggestion("specify an ABI", code = "extern \"<abi>\"", applicability = "has-placeholders")]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("`extern` declarations without an explicit ABI are deprecated")]
pub(crate) struct MissingAbiSugg {
    #[suggestion(
        "explicitly specify the {$default_abi} ABI",
        code = "extern {default_abi}",
        applicability = "machine-applicable"
    )]
    pub span: Span,
    pub default_abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("foreign functions with the \"custom\" ABI cannot be safe")]
pub(crate) struct AbiCustomSafeForeignFunction {
    #[primary_span]
    pub span: Span,

    #[suggestion(
        "remove the `safe` keyword from this definition",
        applicability = "maybe-incorrect",
        code = "",
        style = "verbose"
    )]
    pub safe_span: Span,
}

#[derive(Diagnostic)]
#[diag("functions with the \"custom\" ABI must be unsafe")]
pub(crate) struct AbiCustomSafeFunction {
    #[primary_span]
    pub span: Span,
    pub abi: ExternAbi,

    #[suggestion(
        "add the `unsafe` keyword to this definition",
        applicability = "maybe-incorrect",
        code = "unsafe ",
        style = "verbose"
    )]
    pub unsafe_span: Span,
}

#[derive(Diagnostic)]
#[diag("functions with the {$abi} ABI cannot be `{$coroutine_kind_str}`")]
pub(crate) struct AbiCannotBeCoroutine {
    #[primary_span]
    pub span: Span,
    pub abi: ExternAbi,

    #[suggestion(
        "remove the `{$coroutine_kind_str}` keyword from this definition",
        applicability = "maybe-incorrect",
        code = "",
        style = "verbose"
    )]
    pub coroutine_kind_span: Span,
    pub coroutine_kind_str: &'static str,
}

#[derive(Diagnostic)]
#[diag("invalid signature for `extern {$abi}` function")]
#[note("functions with the {$abi} ABI cannot have any parameters or return type")]
pub(crate) struct AbiMustNotHaveParametersOrReturnType {
    #[primary_span]
    pub spans: Vec<Span>,
    pub abi: ExternAbi,

    #[suggestion(
        "remove the parameters and return type",
        applicability = "maybe-incorrect",
        code = "{padding}fn {symbol}()",
        style = "verbose"
    )]
    pub suggestion_span: Span,
    pub symbol: Symbol,
    pub padding: &'static str,
}

#[derive(Diagnostic)]
#[diag("invalid signature for `extern {$abi}` function")]
#[note("functions with the {$abi} ABI cannot have a return type")]
pub(crate) struct AbiMustNotHaveReturnType {
    #[primary_span]
    #[help("remove the return type")]
    pub span: Span,
    pub abi: ExternAbi,
}

#[derive(Diagnostic)]
#[diag("invalid signature for `extern \"x86-interrupt\"` function")]
#[note(
    "functions with the \"x86-interrupt\" ABI must be have either 1 or 2 parameters (but found {$param_count})"
)]
pub(crate) struct AbiX86Interrupt {
    #[primary_span]
    pub spans: Vec<Span>,
    pub param_count: usize,
}

#[derive(Diagnostic)]
#[diag("scalable vectors must be tuple structs")]
pub(crate) struct ScalableVectorNotTupleStruct {
    #[primary_span]
    pub span: Span,
}
