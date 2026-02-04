use rustc_errors::DiagArgFromDisplay;
use rustc_errors::codes::*;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Ident, Span, Symbol};

#[derive(Diagnostic)]
#[diag("parenthesized type parameters may only be used with a `Fn` trait", code = E0214)]
pub(crate) struct GenericTypeWithParentheses {
    #[primary_span]
    #[label("only `Fn` traits may use parentheses")]
    pub span: Span,
    #[subdiagnostic]
    pub sub: Option<UseAngleBrackets>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("use angle brackets instead", applicability = "maybe-incorrect")]
pub(crate) struct UseAngleBrackets {
    #[suggestion_part(code = "<")]
    pub open_param: Span,
    #[suggestion_part(code = ">")]
    pub close_param: Span,
}

#[derive(Diagnostic)]
#[diag("invalid ABI: found `{$abi}`", code = E0703)]
#[note("invoke `{$command}` for a full list of supported calling conventions")]
pub(crate) struct InvalidAbi {
    #[primary_span]
    #[label("invalid ABI")]
    pub span: Span,
    pub abi: Symbol,
    pub command: String,
    #[subdiagnostic]
    pub suggestion: Option<InvalidAbiSuggestion>,
}

#[derive(Diagnostic)]
#[diag("default fields are not supported in tuple structs")]
pub(crate) struct TupleStructWithDefault {
    #[primary_span]
    #[label("default fields are only supported on structs")]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "there's a similarly named valid ABI `{$suggestion}`",
    code = "\"{suggestion}\"",
    applicability = "maybe-incorrect",
    style = "verbose"
)]
pub(crate) struct InvalidAbiSuggestion {
    #[primary_span]
    pub span: Span,
    pub suggestion: String,
}

#[derive(Diagnostic)]
#[diag("parenthesized generic arguments cannot be used in associated type constraints")]
pub(crate) struct AssocTyParentheses {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub sub: AssocTyParenthesesSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum AssocTyParenthesesSub {
    #[multipart_suggestion("remove these parentheses")]
    Empty {
        #[suggestion_part(code = "")]
        parentheses_span: Span,
    },
    #[multipart_suggestion("use angle brackets instead")]
    NotEmpty {
        #[suggestion_part(code = "<")]
        open_param: Span,
        #[suggestion_part(code = ">")]
        close_param: Span,
    },
}

#[derive(Diagnostic)]
#[diag("`impl Trait` is not allowed in {$position}", code = E0562)]
#[note("`impl Trait` is only allowed in arguments and return types of functions and methods")]
pub(crate) struct MisplacedImplTrait<'a> {
    #[primary_span]
    pub span: Span,
    pub position: DiagArgFromDisplay<'a>,
}

#[derive(Diagnostic)]
#[diag("associated type bounds are not allowed in `dyn` types")]
pub(crate) struct MisplacedAssocTyBinding {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use `impl Trait` to introduce a type instead",
        code = " = impl",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub suggestion: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("in expressions, `_` can only be used on the left-hand side of an assignment")]
pub(crate) struct UnderscoreExprLhsAssign {
    #[primary_span]
    #[label("`_` not allowed here")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`await` is only allowed inside `async` functions and blocks", code = E0728)]
pub(crate) struct AwaitOnlyInAsyncFnAndBlocks {
    #[primary_span]
    #[label("only allowed inside `async` functions and blocks")]
    pub await_kw_span: Span,
    #[label("this is not `async`")]
    pub item_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("too many parameters for a coroutine (expected 0 or 1 parameters)", code = E0628)]
pub(crate) struct CoroutineTooManyParameters {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(Diagnostic)]
#[diag("closures cannot be static", code = E0697)]
pub(crate) struct ClosureCannotBeStatic {
    #[primary_span]
    pub fn_decl_span: Span,
}

#[derive(Diagnostic)]
#[diag("functional record updates are not allowed in destructuring assignments")]
pub(crate) struct FunctionalRecordUpdateDestructuringAssignment {
    #[primary_span]
    #[suggestion(
        "consider removing the trailing pattern",
        code = "",
        applicability = "machine-applicable"
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`async` coroutines are not yet supported", code = E0727)]
pub(crate) struct AsyncCoroutinesNotSupported {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("inline assembly is unsupported on this target", code = E0472)]
pub(crate) struct InlineAsmUnsupportedTarget {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("the `att_syntax` option is only supported on x86")]
pub(crate) struct AttSyntaxOnlyX86 {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$prev_name}` ABI specified multiple times")]
pub(crate) struct AbiSpecifiedMultipleTimes {
    #[primary_span]
    pub abi_span: Span,
    pub prev_name: Symbol,
    #[label("previously specified here")]
    pub prev_span: Span,
    #[note("these ABIs are equivalent on the current target")]
    pub equivalent: bool,
}

#[derive(Diagnostic)]
#[diag("`clobber_abi` is not supported on this target")]
pub(crate) struct ClobberAbiNotSupported {
    #[primary_span]
    pub abi_span: Span,
}

#[derive(Diagnostic)]
#[note("the following ABIs are supported on this target: {$supported_abis}")]
#[diag("invalid ABI for `clobber_abi`")]
pub(crate) struct InvalidAbiClobberAbi {
    #[primary_span]
    pub abi_span: Span,
    pub supported_abis: String,
}

#[derive(Diagnostic)]
#[diag("invalid register `{$reg}`: {$error}")]
pub(crate) struct InvalidRegister<'a> {
    #[primary_span]
    pub op_span: Span,
    pub reg: Symbol,
    pub error: &'a str,
}

#[derive(Diagnostic)]
#[note(
    "the following register classes are supported on this target: {$supported_register_classes}"
)]
#[diag("invalid register class `{$reg_class}`: unknown register class")]
pub(crate) struct InvalidRegisterClass {
    #[primary_span]
    pub op_span: Span,
    pub reg_class: Symbol,
    pub supported_register_classes: String,
}

#[derive(Diagnostic)]
#[diag("invalid asm template modifier for this register class")]
pub(crate) struct InvalidAsmTemplateModifierRegClass {
    #[primary_span]
    #[label("template modifier")]
    pub placeholder_span: Span,
    #[label("argument")]
    pub op_span: Span,
    #[subdiagnostic]
    pub sub: InvalidAsmTemplateModifierRegClassSub,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidAsmTemplateModifierRegClassSub {
    #[note(
        "the `{$class_name}` register class supports the following template modifiers: {$modifiers}"
    )]
    SupportModifier { class_name: Symbol, modifiers: String },
    #[note("the `{$class_name}` register class does not support template modifiers")]
    DoesNotSupportModifier { class_name: Symbol },
}

#[derive(Diagnostic)]
#[diag("asm template modifiers are not allowed for `const` arguments")]
pub(crate) struct InvalidAsmTemplateModifierConst {
    #[primary_span]
    #[label("template modifier")]
    pub placeholder_span: Span,
    #[label("argument")]
    pub op_span: Span,
}

#[derive(Diagnostic)]
#[diag("asm template modifiers are not allowed for `sym` arguments")]
pub(crate) struct InvalidAsmTemplateModifierSym {
    #[primary_span]
    #[label("template modifier")]
    pub placeholder_span: Span,
    #[label("argument")]
    pub op_span: Span,
}

#[derive(Diagnostic)]
#[diag("asm template modifiers are not allowed for `label` arguments")]
pub(crate) struct InvalidAsmTemplateModifierLabel {
    #[primary_span]
    #[label("template modifier")]
    pub placeholder_span: Span,
    #[label("argument")]
    pub op_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "register class `{$reg_class_name}` can only be used as a clobber, not as an input or output"
)]
pub(crate) struct RegisterClassOnlyClobber {
    #[primary_span]
    pub op_span: Span,
    pub reg_class_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("register class `{$reg_class_name}` can only be used as a clobber in stable")]
pub(crate) struct RegisterClassOnlyClobberStable {
    #[primary_span]
    pub op_span: Span,
    pub reg_class_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("register `{$reg1_name}` conflicts with register `{$reg2_name}`")]
pub(crate) struct RegisterConflict<'a> {
    #[primary_span]
    #[label("register `{$reg1_name}`")]
    pub op_span1: Span,
    #[label("register `{$reg2_name}`")]
    pub op_span2: Span,
    pub reg1_name: &'a str,
    pub reg2_name: &'a str,
    #[help("use `lateout` instead of `out` to avoid conflict")]
    pub in_out: Option<Span>,
}

#[derive(Diagnostic)]
#[help("remove this and bind each tuple field independently")]
#[diag("`{$ident_name} @` is not allowed in a {$ctx}")]
pub(crate) struct SubTupleBinding<'a> {
    #[primary_span]
    #[label("this is only allowed in slice patterns")]
    #[suggestion(
        "if you don't need to use the contents of {$ident}, discard the tuple's remaining fields",
        style = "verbose",
        code = "..",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
    pub ident: Ident,
    pub ident_name: Symbol,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[diag("`..` can only be used once per {$ctx} pattern")]
pub(crate) struct ExtraDoubleDot<'a> {
    #[primary_span]
    #[label("can only be used once per {$ctx} pattern")]
    pub span: Span,
    #[label("previously used here")]
    pub prev_span: Span,
    pub ctx: &'a str,
}

#[derive(Diagnostic)]
#[note("only allowed in tuple, tuple struct, and slice patterns")]
#[diag("`..` patterns are not allowed here")]
pub(crate) struct MisplacedDoubleDot {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`match` arm with no body")]
pub(crate) struct MatchArmWithNoBody {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "add a body after the pattern",
        code = " => todo!(),",
        applicability = "has-placeholders"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("a never pattern is always unreachable")]
pub(crate) struct NeverPatternWithBody {
    #[primary_span]
    #[label("this will never be executed")]
    #[suggestion("remove this expression", code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("a guard on a never pattern will never be run")]
pub(crate) struct NeverPatternWithGuard {
    #[primary_span]
    #[suggestion("remove this guard", code = "", applicability = "maybe-incorrect")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("arbitrary expressions aren't allowed in patterns")]
pub(crate) struct ArbitraryExpressionInPattern {
    #[primary_span]
    pub span: Span,
    #[note("the `expr` fragment specifier forces the metavariable's content to be an expression")]
    pub pattern_from_macro_note: bool,
    #[help("use a named `const`-item or an `if`-guard (`x if x == const {\"{ ... }\"}`) instead")]
    pub const_block_in_pattern_help: bool,
}

#[derive(Diagnostic)]
#[diag("inclusive range with no end")]
pub(crate) struct InclusiveRangeWithNoEnd {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "use the right argument notation and remove the return type",
    applicability = "machine-applicable",
    style = "verbose"
)]
/// Given `T: Tr<m() -> Ret>` or `T: Tr<m(Ty) -> Ret>`, suggest `T: Tr<m(..)>`.
pub(crate) struct RTNSuggestion {
    #[suggestion_part(code = "")]
    pub output: Span,
    #[suggestion_part(code = "(..)")]
    pub input: Span,
}

#[derive(Diagnostic)]
pub(crate) enum BadReturnTypeNotation {
    #[diag("argument types not allowed with return type notation")]
    Inputs {
        #[primary_span]
        #[suggestion(
            "remove the input types",
            code = "(..)",
            applicability = "machine-applicable",
            style = "verbose"
        )]
        span: Span,
    },
    #[diag("return type not allowed with return type notation")]
    Output {
        #[primary_span]
        span: Span,
        #[subdiagnostic]
        suggestion: RTNSuggestion,
    },
    #[diag("return type notation arguments must be elided with `..`")]
    NeedsDots {
        #[primary_span]
        #[suggestion(
            "use the correct syntax by adding `..` to the arguments",
            code = "(..)",
            applicability = "machine-applicable",
            style = "verbose"
        )]
        span: Span,
    },
    #[diag("return type notation not allowed in this position yet")]
    Position {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("defaults for generic parameters are not allowed in `for<...>` binders")]
pub(crate) struct GenericParamDefaultInBinder {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`async` bound modifier only allowed on trait, not `{$descr}`")]
pub(crate) struct AsyncBoundNotOnTrait {
    #[primary_span]
    pub span: Span,
    pub descr: &'static str,
}

#[derive(Diagnostic)]
#[diag("`async` bound modifier only allowed on `Fn`/`FnMut`/`FnOnce` traits")]
pub(crate) struct AsyncBoundOnlyForFnTraits {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`use<...>` precise capturing syntax not allowed in argument-position `impl Trait`")]
pub(crate) struct NoPreciseCapturesOnApit {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`yield` can only be used in `#[coroutine]` closures, or `gen` blocks")]
pub(crate) struct YieldInClosure {
    #[primary_span]
    pub span: Span,
    #[suggestion(
        "use `#[coroutine]` to make this closure a coroutine",
        code = "#[coroutine] ",
        applicability = "maybe-incorrect",
        style = "verbose"
    )]
    pub suggestion: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(
    "invalid argument to a legacy const generic: cannot have const blocks, closures, async blocks or items"
)]
pub(crate) struct InvalidLegacyConstGenericArg {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub suggestion: UseConstGenericArg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "try using a const generic argument instead",
    applicability = "maybe-incorrect"
)]
pub(crate) struct UseConstGenericArg {
    #[suggestion_part(code = "::<{const_args}>")]
    pub end_of_fn: Span,
    pub const_args: String,
    pub other_args: String,
    #[suggestion_part(code = "{other_args}")]
    pub call_args: Span,
}

#[derive(Diagnostic)]
#[diag("unions cannot have default field values")]
pub(crate) struct UnionWithDefault {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("failed to resolve delegation callee")]
pub(crate) struct UnresolvedDelegationCallee {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("encountered a cycle during delegation signature resolution")]
pub(crate) struct CycleInDelegationSignatureResolution {
    #[primary_span]
    pub span: Span,
}
