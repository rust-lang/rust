use rustc_errors::codes::*;
use rustc_errors::{
    Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level, MultiSpan, SingleLabelManySpans,
    Subdiagnostic, inline_fluent,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Ident, Span, Symbol};

#[derive(LintDiagnostic)]
#[diag("avoid using `.intel_syntax`, Intel syntax is the default")]
pub(crate) struct AvoidIntelSyntax;

#[derive(LintDiagnostic)]
#[diag("avoid using `.att_syntax`, prefer using `options(att_syntax)` instead")]
pub(crate) struct AvoidAttSyntax;

#[derive(LintDiagnostic)]
#[diag("include macro expected single expression in source")]
pub(crate) struct IncompleteInclude;

#[derive(LintDiagnostic)]
#[diag("cannot test inner items")]
pub(crate) struct UnnameableTestItems;

#[derive(LintDiagnostic)]
#[diag("duplicated attribute")]
pub(crate) struct DuplicateMacroAttribute;

#[derive(Diagnostic)]
#[diag("macro requires a cfg-pattern as an argument")]
pub(crate) struct RequiresCfgPattern {
    #[primary_span]
    #[label("cfg-pattern required")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected 1 cfg-pattern")]
pub(crate) struct OneCfgPattern {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("alloc_error_handler must be a function")]
pub(crate) struct AllocErrorMustBeFn {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("macro requires a boolean expression as an argument")]
pub(crate) struct AssertRequiresBoolean {
    #[primary_span]
    #[label("boolean expression required")]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("macro requires an expression as an argument")]
pub(crate) struct AssertRequiresExpression {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion("try removing semicolon", code = "", applicability = "maybe-incorrect")]
    pub(crate) token: Span,
}

#[derive(Diagnostic)]
#[diag("unexpected string literal")]
pub(crate) struct AssertMissingComma {
    #[primary_span]
    pub(crate) span: Span,
    #[suggestion(
        "try adding a comma",
        code = ", ",
        applicability = "maybe-incorrect",
        style = "short"
    )]
    pub(crate) comma: Span,
}

#[derive(Diagnostic)]
pub(crate) enum CfgAccessibleInvalid {
    #[diag("`cfg_accessible` path is not specified")]
    UnspecifiedPath(#[primary_span] Span),
    #[diag("multiple `cfg_accessible` paths are specified")]
    MultiplePaths(#[primary_span] Span),
    #[diag("`cfg_accessible` path cannot be a literal")]
    LiteralPath(#[primary_span] Span),
    #[diag("`cfg_accessible` path cannot accept arguments")]
    HasArguments(#[primary_span] Span),
}

#[derive(Diagnostic)]
#[diag("cannot determine whether the path is accessible or not")]
pub(crate) struct CfgAccessibleIndeterminate {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a literal")]
#[note("only literals (like `\"foo\"`, `-42` and `3.14`) can be passed to `concat!()`")]
pub(crate) struct ConcatMissingLiteral {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("cannot concatenate a byte string literal")]
pub(crate) struct ConcatBytestr {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot concatenate a C string literal")]
pub(crate) struct ConcatCStrLit {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("cannot export macro_rules! macros from a `proc-macro` crate type currently")]
pub(crate) struct ExportMacroRules {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "`proc-macro` crate types currently cannot export any items other than functions tagged with `#[proc_macro]`, `#[proc_macro_derive]`, or `#[proc_macro_attribute]`"
)]
pub(crate) struct ProcMacro {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("trace_macros! accepts only `true` or `false`")]
pub(crate) struct TraceMacros {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("functions used as benches must have signature `fn(&mut Bencher) -> impl Termination`")]
pub(crate) struct BenchSig {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("allocators must be statics")]
pub(crate) struct AllocMustStatics {
    #[primary_span]
    pub(crate) span: Span,
}

pub(crate) use autodiff::*;

mod autodiff {
    use super::*;
    #[derive(Diagnostic)]
    #[diag("autodiff requires at least a name and mode")]
    pub(crate) struct AutoDiffMissingConfig {
        #[primary_span]
        pub(crate) span: Span,
    }
    #[derive(Diagnostic)]
    #[diag("did not recognize Activity: `{$act}`")]
    pub(crate) struct AutoDiffUnknownActivity {
        #[primary_span]
        pub(crate) span: Span,
        pub(crate) act: String,
    }
    #[derive(Diagnostic)]
    #[diag("{$act} can not be used for this type")]
    pub(crate) struct AutoDiffInvalidTypeForActivity {
        #[primary_span]
        pub(crate) span: Span,
        pub(crate) act: String,
    }
    #[derive(Diagnostic)]
    #[diag("expected {$expected} activities, but found {$found}")]
    pub(crate) struct AutoDiffInvalidNumberActivities {
        #[primary_span]
        pub(crate) span: Span,
        pub(crate) expected: usize,
        pub(crate) found: usize,
    }
    #[derive(Diagnostic)]
    #[diag("{$act} can not be used in {$mode} Mode")]
    pub(crate) struct AutoDiffInvalidApplicationModeAct {
        #[primary_span]
        pub(crate) span: Span,
        pub(crate) mode: String,
        pub(crate) act: String,
    }

    #[derive(Diagnostic)]
    #[diag("invalid return activity {$act} in {$mode} Mode")]
    pub(crate) struct AutoDiffInvalidRetAct {
        #[primary_span]
        pub(crate) span: Span,
        pub(crate) mode: String,
        pub(crate) act: String,
    }

    #[derive(Diagnostic)]
    #[diag("autodiff width must fit u32, but is {$width}")]
    pub(crate) struct AutoDiffInvalidWidth {
        #[primary_span]
        pub(crate) span: Span,
        pub(crate) width: u128,
    }

    #[derive(Diagnostic)]
    #[diag("autodiff must be applied to function")]
    pub(crate) struct AutoDiffInvalidApplication {
        #[primary_span]
        pub(crate) span: Span,
    }
}

#[derive(Diagnostic)]
#[diag("cannot concatenate {$lit_kind} literals")]
pub(crate) struct ConcatBytesInvalid {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) lit_kind: &'static str,
    #[subdiagnostic]
    pub(crate) sugg: Option<ConcatBytesInvalidSuggestion>,
    #[note("concatenating C strings is ambiguous about including the '\\0'")]
    pub(crate) cs_note: Option<()>,
}

#[derive(Subdiagnostic)]
pub(crate) enum ConcatBytesInvalidSuggestion {
    #[suggestion(
        "try using a byte character",
        code = "b{snippet}",
        applicability = "machine-applicable"
    )]
    CharLit {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[suggestion(
        "try using a byte string",
        code = "b{snippet}",
        applicability = "machine-applicable"
    )]
    StrLit {
        #[primary_span]
        span: Span,
        snippet: String,
    },
    #[note("concatenating C strings is ambiguous about including the '\\0'")]
    #[suggestion(
        "try using a null-terminated byte string",
        code = "{as_bstr}",
        applicability = "machine-applicable"
    )]
    CStrLit {
        #[primary_span]
        span: Span,
        as_bstr: String,
    },
    #[suggestion(
        "try wrapping the number in an array",
        code = "[{snippet}]",
        applicability = "machine-applicable"
    )]
    IntLit {
        #[primary_span]
        span: Span,
        snippet: String,
    },
}

#[derive(Diagnostic)]
#[diag("numeric literal is out of bounds")]
pub(crate) struct ConcatBytesOob {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("numeric literal is not a `u8`")]
pub(crate) struct ConcatBytesNonU8 {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected a byte literal")]
#[note(
    "only byte literals (like `b\"foo\"`, `b's'` and `[3, 4, 5]`) can be passed to `concat_bytes!()`"
)]
pub(crate) struct ConcatBytesMissingLiteral {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("cannot concatenate doubly nested array")]
pub(crate) struct ConcatBytesArray {
    #[primary_span]
    pub(crate) span: Span,
    #[note("byte strings are treated as arrays of bytes")]
    #[help("try flattening the array")]
    pub(crate) bytestr: bool,
}

#[derive(Diagnostic)]
#[diag("repeat count is not a positive number")]
pub(crate) struct ConcatBytesBadRepeat {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`derive` may only be applied to `struct`s, `enum`s and `union`s", code = E0774)]
pub(crate) struct BadDeriveTarget {
    #[primary_span]
    #[label("not applicable here")]
    pub(crate) span: Span,
    #[label("not a `struct`, `enum` or `union`")]
    pub(crate) item: Span,
}

#[derive(Diagnostic)]
#[diag("building tests with panic=abort is not supported without `-Zpanic_abort_tests`")]
pub(crate) struct TestsNotSupport {}

#[derive(Diagnostic)]
#[diag("expected path to a trait, found literal", code = E0777)]
pub(crate) struct BadDeriveLit {
    #[primary_span]
    #[label("not a trait")]
    pub(crate) span: Span,
    #[subdiagnostic]
    pub help: BadDeriveLitHelp,
}

#[derive(Subdiagnostic)]
pub(crate) enum BadDeriveLitHelp {
    #[help("try using `#[derive({$sym})]`")]
    StrLit { sym: Symbol },
    #[help("for example, write `#[derive(Debug)]` for `Debug`")]
    Other,
}

#[derive(Diagnostic)]
#[diag("traits in `#[derive(...)]` don't accept arguments")]
pub(crate) struct DerivePathArgsList {
    #[suggestion("remove the arguments", code = "", applicability = "machine-applicable")]
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("traits in `#[derive(...)]` don't accept values")]
pub(crate) struct DerivePathArgsValue {
    #[suggestion("remove the value", code = "", applicability = "machine-applicable")]
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[derive(Default)]` on enum with no `#[default]`", code = E0665)]
pub(crate) struct NoDefaultVariant {
    #[primary_span]
    pub(crate) span: Span,
    #[label("this enum needs a unit variant marked with `#[default]`")]
    pub(crate) item_span: Span,
    #[subdiagnostic]
    pub(crate) suggs: Vec<NoDefaultVariantSugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "make this unit variant default by placing `#[default]` on it",
    code = "#[default] ",
    applicability = "maybe-incorrect"
)]
pub(crate) struct NoDefaultVariantSugg {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("multiple declared defaults")]
#[note("only one variant can be default")]
pub(crate) struct MultipleDefaults {
    #[primary_span]
    pub(crate) span: Span,
    #[label("first default")]
    pub(crate) first: Span,
    #[label("additional default")]
    pub additional: Vec<Span>,
    #[subdiagnostic]
    pub suggs: Vec<MultipleDefaultsSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "make `{$ident}` default",
    applicability = "maybe-incorrect",
    style = "tool-only"
)]
pub(crate) struct MultipleDefaultsSugg {
    #[suggestion_part(code = "")]
    pub(crate) spans: Vec<Span>,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("the `#[default]` attribute may only be used on unit enum variants{$post}")]
#[help("consider a manual implementation of `Default`")]
pub(crate) struct NonUnitDefault {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) post: &'static str,
}

#[derive(Diagnostic)]
#[diag("default variant must be exhaustive")]
#[help("consider a manual implementation of `Default`")]
pub(crate) struct NonExhaustiveDefault {
    #[primary_span]
    pub(crate) span: Span,
    #[label("declared `#[non_exhaustive]` here")]
    pub(crate) non_exhaustive: Span,
}

#[derive(Diagnostic)]
#[diag("multiple `#[default]` attributes")]
#[note("only one `#[default]` attribute is needed")]
pub(crate) struct MultipleDefaultAttrs {
    #[primary_span]
    pub(crate) span: Span,
    #[label("`#[default]` used here")]
    pub(crate) first: Span,
    #[label("`#[default]` used again here")]
    pub(crate) first_rest: Span,
    #[help(
        "try removing {$only_one ->
            [true] this
            *[false] these
        }"
    )]
    pub(crate) rest: MultiSpan,
    pub(crate) only_one: bool,
    #[subdiagnostic]
    pub(crate) sugg: MultipleDefaultAttrsSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "consider a manual implementation of `Default`",
    applicability = "machine-applicable",
    style = "tool-only"
)]
pub(crate) struct MultipleDefaultAttrsSugg {
    #[suggestion_part(code = "")]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("`#[default]` attribute does not accept a value")]
pub(crate) struct DefaultHasArg {
    #[primary_span]
    #[suggestion(
        "try using `#[default]`",
        code = "#[default]",
        style = "hidden",
        applicability = "maybe-incorrect"
    )]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[derive(From)]` used on {$kind}")]
#[note("`#[derive(From)]` can only be used on structs with exactly one field")]
pub(crate) struct DeriveFromWrongTarget<'a> {
    #[primary_span]
    pub(crate) span: MultiSpan,
    pub(crate) kind: &'a str,
}

#[derive(Diagnostic)]
#[diag(
    "`#[derive(From)]` used on a struct with {$multiple_fields ->
        [true] multiple fields
        *[false] no fields
    }"
)]
#[note("`#[derive(From)]` can only be used on structs with exactly one field")]
pub(crate) struct DeriveFromWrongFieldCount {
    #[primary_span]
    pub(crate) span: MultiSpan,
    pub(crate) multiple_fields: bool,
}

#[derive(Diagnostic)]
#[diag("`derive` cannot be used on items with type macros")]
pub(crate) struct DeriveMacroCall {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("this trait cannot be derived for unions")]
pub(crate) struct DeriveUnion {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`env!()` takes 1 or 2 arguments")]
pub(crate) struct EnvTakesArgs {
    #[primary_span]
    pub(crate) span: Span,
}

pub(crate) struct EnvNotDefinedWithUserMessage {
    pub(crate) span: Span,
    pub(crate) msg_from_user: Symbol,
}

// Hand-written implementation to support custom user messages.
impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for EnvNotDefinedWithUserMessage {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag = Diag::new(dcx, level, self.msg_from_user.to_string());
        diag.span(self.span);
        diag
    }
}

#[derive(Diagnostic)]
pub(crate) enum EnvNotDefined<'a> {
    #[diag("environment variable `{$var}` not defined at compile time")]
    #[help(
        "Cargo sets build script variables at run time. Use `std::env::var({$var_expr})` instead"
    )]
    CargoEnvVar {
        #[primary_span]
        span: Span,
        var: Symbol,
        var_expr: &'a rustc_ast::Expr,
    },
    #[diag("environment variable `{$var}` not defined at compile time")]
    #[help("there is a similar Cargo environment variable: `{$suggested_var}`")]
    CargoEnvVarTypo {
        #[primary_span]
        span: Span,
        var: Symbol,
        suggested_var: Symbol,
    },
    #[diag("environment variable `{$var}` not defined at compile time")]
    #[help("use `std::env::var({$var_expr})` to read the variable at run time")]
    CustomEnvVar {
        #[primary_span]
        span: Span,
        var: Symbol,
        var_expr: &'a rustc_ast::Expr,
    },
}

#[derive(Diagnostic)]
#[diag("environment variable `{$var}` is not a valid Unicode string")]
pub(crate) struct EnvNotUnicode {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) var: Symbol,
}

#[derive(Diagnostic)]
#[diag("requires at least a format string argument")]
pub(crate) struct FormatRequiresString {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("duplicate argument named `{$ident}`")]
pub(crate) struct FormatDuplicateArg {
    #[primary_span]
    pub(crate) span: Span,
    #[label("previously here")]
    pub(crate) prev: Span,
    #[label("duplicate argument")]
    pub(crate) duplicate: Span,
    pub(crate) ident: Ident,
}

#[derive(Diagnostic)]
#[diag("positional arguments cannot follow named arguments")]
pub(crate) struct PositionalAfterNamed {
    #[primary_span]
    #[label("positional arguments must be before named arguments")]
    pub(crate) span: Span,
    #[label("named argument")]
    pub(crate) args: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("invalid format string: {$desc}")]
pub(crate) struct InvalidFormatString {
    #[primary_span]
    #[label("{$label1} in format string")]
    pub(crate) span: Span,
    pub(crate) desc: String,
    pub(crate) label1: String,
    #[subdiagnostic]
    pub(crate) note_: Option<InvalidFormatStringNote>,
    #[subdiagnostic]
    pub(crate) label_: Option<InvalidFormatStringLabel>,
    #[subdiagnostic]
    pub(crate) sugg_: Option<InvalidFormatStringSuggestion>,
}

#[derive(Subdiagnostic)]
#[note("{$note}")]
pub(crate) struct InvalidFormatStringNote {
    pub(crate) note: String,
}

#[derive(Subdiagnostic)]
#[label("{$label}")]
pub(crate) struct InvalidFormatStringLabel {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) label: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum InvalidFormatStringSuggestion {
    #[multipart_suggestion(
        "consider using a positional formatting argument instead",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    UsePositional {
        #[suggestion_part(code = "{len}")]
        captured: Span,
        len: String,
        #[suggestion_part(code = ", {arg}")]
        span: Span,
        arg: String,
    },
    #[suggestion("remove the `r#`", code = "", applicability = "machine-applicable")]
    RemoveRawIdent {
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        "did you mean `{$replacement}`?",
        code = "{replacement}",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    ReorderFormatParameter {
        #[primary_span]
        span: Span,
        replacement: String,
    },
    #[suggestion(
        "add a colon before the format specifier",
        code = ":?",
        applicability = "machine-applicable"
    )]
    AddMissingColon {
        #[primary_span]
        span: Span,
    },

    #[suggestion(
        "use rust debug printing macro",
        code = "{replacement}",
        style = "verbose",
        applicability = "machine-applicable"
    )]
    UseRustDebugPrintingMacro {
        #[primary_span]
        macro_span: Span,
        replacement: String,
    },
}

#[derive(Diagnostic)]
#[diag("there is no argument named `{$name}`")]
#[note("did you intend to capture a variable `{$name}` from the surrounding scope?")]
#[note(
    "to avoid ambiguity, `format_args!` cannot capture variables when the format string is expanded from a macro"
)]
pub(crate) struct FormatNoArgNamed {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag("unknown format trait `{$ty}`")]
#[note(
    "the only appropriate formatting traits are:
                                            - ``, which uses the `Display` trait
                                            - `?`, which uses the `Debug` trait
                                            - `e`, which uses the `LowerExp` trait
                                            - `E`, which uses the `UpperExp` trait
                                            - `o`, which uses the `Octal` trait
                                            - `p`, which uses the `Pointer` trait
                                            - `b`, which uses the `Binary` trait
                                            - `x`, which uses the `LowerHex` trait
                                            - `X`, which uses the `UpperHex` trait"
)]
pub(crate) struct FormatUnknownTrait<'a> {
    #[primary_span]
    pub(crate) span: Span,
    pub(crate) ty: &'a str,
    #[subdiagnostic]
    pub(crate) suggs: Vec<FormatUnknownTraitSugg>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "use the `{$trait_name}` trait",
    code = "{fmt}",
    style = "tool-only",
    applicability = "maybe-incorrect"
)]
pub(crate) struct FormatUnknownTraitSugg {
    #[primary_span]
    pub span: Span,
    pub fmt: &'static str,
    pub trait_name: &'static str,
}

#[derive(Diagnostic)]
#[diag(
    "{$named ->
        [true] named argument
        *[false] argument
    } never used"
)]
pub(crate) struct FormatUnusedArg {
    #[primary_span]
    #[label(
        "{$named ->
            [true] named argument
            *[false] argument
        } never used"
    )]
    pub(crate) span: Span,
    pub(crate) named: bool,
}

// Allow the singular form to be a subdiagnostic of the multiple-unused
// form of diagnostic.
impl Subdiagnostic for FormatUnusedArg {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("named", self.named);
        let msg = diag.eagerly_translate(inline_fluent!(
            "{$named ->
                [true] named argument
                *[false] argument
            } never used"
        ));
        diag.remove_arg("named");
        diag.span_label(self.span, msg);
    }
}

#[derive(Diagnostic)]
#[diag("multiple unused formatting arguments")]
pub(crate) struct FormatUnusedArgs {
    #[primary_span]
    pub(crate) unused: Vec<Span>,
    #[label("multiple missing formatting specifiers")]
    pub(crate) fmt: Span,
    #[subdiagnostic]
    pub(crate) unused_labels: Vec<FormatUnusedArg>,
}

#[derive(Diagnostic)]
#[diag(
    "{$n} positional {$n ->
        [one] argument
        *[more] arguments
    } in format string, but {$desc}"
)]
pub(crate) struct FormatPositionalMismatch {
    #[primary_span]
    pub(crate) span: MultiSpan,
    pub(crate) n: usize,
    pub(crate) desc: String,
    #[subdiagnostic]
    pub(crate) highlight: SingleLabelManySpans,
}

#[derive(Diagnostic)]
#[diag(
    "redundant {$n ->
        [one] argument
        *[more] arguments
    }"
)]
pub(crate) struct FormatRedundantArgs {
    #[primary_span]
    pub(crate) span: MultiSpan,
    pub(crate) n: usize,

    #[note(
        "{$n ->
            [one] the formatting specifier is referencing the binding already
            *[more] the formatting specifiers are referencing the bindings already
        }"
    )]
    pub(crate) note: MultiSpan,

    #[subdiagnostic]
    pub(crate) sugg: Option<FormatRedundantArgsSugg>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("this can be removed", applicability = "machine-applicable")]
pub(crate) struct FormatRedundantArgsSugg {
    #[suggestion_part(code = "")]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("`#[test_case]` attribute is only allowed on items")]
pub(crate) struct TestCaseNonItem {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("{$kind} functions cannot be used for tests")]
pub(crate) struct TestBadFn {
    #[primary_span]
    pub(crate) span: Span,
    #[label("`{$kind}` because of this")]
    pub(crate) cause: Span,
    pub(crate) kind: &'static str,
}

#[derive(Diagnostic)]
#[diag("explicit register arguments cannot have names")]
pub(crate) struct AsmExplicitRegisterName {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("the `{$opt1}` and `{$opt2}` options are mutually exclusive")]
pub(crate) struct AsmMutuallyExclusive {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    pub(crate) opt1: &'static str,
    pub(crate) opt2: &'static str,
}

#[derive(Diagnostic)]
#[diag("the `pure` option must be combined with either `nomem` or `readonly`")]
pub(crate) struct AsmPureCombine {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("asm with the `pure` option must have at least one output")]
pub(crate) struct AsmPureNoOutput {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("asm template modifier must be a single character")]
pub(crate) struct AsmModifierInvalid {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("this attribute is not supported on assembly")]
pub(crate) struct AsmAttributeNotSupported {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("duplicate argument named `{$name}`")]
pub(crate) struct AsmDuplicateArg {
    #[primary_span]
    #[label("duplicate argument")]
    pub(crate) span: Span,
    #[label("previously here")]
    pub(crate) prev: Span,
    pub(crate) name: Symbol,
}

#[derive(Diagnostic)]
#[diag("positional arguments cannot follow named arguments or explicit register arguments")]
pub(crate) struct AsmPositionalAfter {
    #[primary_span]
    #[label("positional argument")]
    pub(crate) span: Span,
    #[label("named argument")]
    pub(crate) named: Vec<Span>,
    #[label("explicit register argument")]
    pub(crate) explicit: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("asm outputs are not allowed with the `noreturn` option")]
pub(crate) struct AsmNoReturn {
    #[primary_span]
    pub(crate) outputs_sp: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("there is no argument named `{$name}`")]
pub(crate) struct AsmNoMatchedArgumentName {
    pub(crate) name: String,
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("asm labels are not allowed with the `may_unwind` option")]
pub(crate) struct AsmMayUnwind {
    #[primary_span]
    pub(crate) labels_sp: Vec<Span>,
}

pub(crate) struct AsmClobberNoReg {
    pub(crate) spans: Vec<Span>,
    pub(crate) clobbers: Vec<Span>,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for AsmClobberNoReg {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        // eager translation as `span_labels` takes `AsRef<str>`
        let lbl1 = dcx.eagerly_translate_to_string(inline_fluent!("clobber_abi"), [].into_iter());
        let lbl2 =
            dcx.eagerly_translate_to_string(inline_fluent!("generic outputs"), [].into_iter());
        Diag::new(
            dcx,
            level,
            inline_fluent!("asm with `clobber_abi` must specify explicit registers for outputs"),
        )
        .with_span(self.spans.clone())
        .with_span_labels(self.clobbers, &lbl1)
        .with_span_labels(self.spans, &lbl2)
    }
}

#[derive(Diagnostic)]
#[diag("the `{$symbol}` option was already provided")]
pub(crate) struct AsmOptAlreadyprovided {
    #[primary_span]
    #[label("this option was already provided")]
    pub(crate) span: Span,
    pub(crate) symbol: Symbol,
    #[suggestion(
        "remove this option",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub(crate) span_with_comma: Span,
}

#[derive(Diagnostic)]
#[diag("the `{$symbol}` option cannot be used with `{$macro_name}!`")]
pub(crate) struct AsmUnsupportedOption {
    #[primary_span]
    #[label("the `{$symbol}` option is not meaningful for global-scoped inline assembly")]
    pub(crate) span: Span,
    pub(crate) symbol: Symbol,
    #[suggestion(
        "remove this option",
        code = "",
        applicability = "machine-applicable",
        style = "tool-only"
    )]
    pub(crate) span_with_comma: Span,
    pub(crate) macro_name: &'static str,
}

#[derive(Diagnostic)]
#[diag("`clobber_abi` cannot be used with `{$macro_name}!`")]
pub(crate) struct AsmUnsupportedClobberAbi {
    #[primary_span]
    pub(crate) spans: Vec<Span>,
    pub(crate) macro_name: &'static str,
}

#[derive(Diagnostic)]
#[diag("`test_runner` argument must be a path")]
pub(crate) struct TestRunnerInvalid {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("`#![test_runner(..)]` accepts exactly 1 argument")]
pub(crate) struct TestRunnerNargs {
    #[primary_span]
    pub(crate) span: Span,
}

#[derive(Diagnostic)]
#[diag("expected token: `,`")]
pub(crate) struct ExpectedCommaInList {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("{$name} takes 1 argument")]
pub(crate) struct OnlyOneArgument<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag("{$name} takes no arguments")]
pub(crate) struct TakesNoArguments<'a> {
    #[primary_span]
    pub span: Span,
    pub name: &'a str,
}

#[derive(Diagnostic)]
#[diag("the `#[{$path}]` attribute is only usable with crates of the `proc-macro` crate type")]
pub(crate) struct AttributeOnlyUsableWithCrateType<'a> {
    #[primary_span]
    pub span: Span,
    pub path: &'a str,
}

#[derive(Diagnostic)]
#[diag("expected item, found `{$token}`")]
pub(crate) struct ExpectedItem<'a> {
    #[primary_span]
    pub span: Span,
    pub token: &'a str,
}

#[derive(Diagnostic)]
#[diag("cannot use `#[unsafe(naked)]` with testing attributes", code = E0736)]
pub(crate) struct NakedFunctionTestingAttribute {
    #[primary_span]
    #[label("`#[unsafe(naked)]` is incompatible with testing attributes")]
    pub naked_span: Span,
    #[label("function marked with testing attribute here")]
    pub testing_span: Span,
}

#[derive(Diagnostic)]
#[diag("the `#[pointee]` attribute may only be used on generic parameters")]
pub(crate) struct NonGenericPointee {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "expected operand, {$is_inline_asm ->
        [false] options
        *[true] clobber_abi, options
    }, or additional template string"
)]
pub(crate) struct AsmExpectedOther {
    #[primary_span]
    #[label(
        "expected operand, {$is_inline_asm ->
            [false] options
            *[true] clobber_abi, options
        }, or additional template string"
    )]
    pub(crate) span: Span,
    pub(crate) is_inline_asm: bool,
}

#[derive(Diagnostic)]
#[diag("none of the predicates in this `cfg_select` evaluated to true")]
pub(crate) struct CfgSelectNoMatches {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[eii_declaration(...)]` is only valid on macros")]
pub(crate) struct EiiExternTargetExpectedMacro {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[eii_declaration(...)]` expects a list of one or two elements")]
pub(crate) struct EiiExternTargetExpectedList {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("expected this argument to be \"unsafe\"")]
pub(crate) struct EiiExternTargetExpectedUnsafe {
    #[primary_span]
    #[note("the second argument is optional")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` is only valid on functions")]
pub(crate) struct EiiSharedMacroExpectedFunction {
    #[primary_span]
    pub span: Span,
    pub name: String,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` can only be used on functions inside a module")]
pub(crate) struct EiiSharedMacroInStatementPosition {
    #[primary_span]
    pub span: Span,
    pub name: String,
    #[label("`#[{$name}]` is used on this item, which is part of another item's local scope")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` can only be specified once")]
pub(crate) struct EiiOnlyOnce {
    #[primary_span]
    pub span: Span,
    #[note("specified again here")]
    pub first_span: Span,
    pub name: String,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` expected no arguments or a single argument: `#[{$name}(default)]`")]
pub(crate) struct EiiMacroExpectedMaxOneArgument {
    #[primary_span]
    pub span: Span,
    pub name: String,
}
