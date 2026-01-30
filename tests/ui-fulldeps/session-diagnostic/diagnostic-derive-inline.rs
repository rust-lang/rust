//@ check-fail
// Tests error conditions for specifying diagnostics using #[derive(Diagnostic)]
// This test specifically tests diagnostic derives involving the inline fluent syntax.

//@ normalize-stderr: "the following other types implement trait `IntoDiagArg`:(?:.*\n){0,9}\s+and \d+ others" -> "normalized in stderr"
//@ normalize-stderr: "(COMPILER_DIR/.*\.rs):[0-9]+:[0-9]+" -> "$1:LL:CC"

// The proc_macro2 crate handles spans differently when on beta/stable release rather than nightly,
// changing the output of this test. Since Diagnostic is strictly internal to the compiler
// the test is just ignored on stable and beta:
//@ ignore-stage1
//@ ignore-beta
//@ ignore-stable

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_span;
use rustc_span::symbol::Ident;
use rustc_span::Span;

extern crate rustc_fluent_macro;
extern crate rustc_macros;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};

extern crate rustc_middle;
use rustc_middle::ty::Ty;

extern crate rustc_errors;
use rustc_errors::{Applicability, DiagMessage, ErrCode, MultiSpan, SubdiagMessage};

extern crate rustc_session;

extern crate core;

// E0123 and E0456 are no longer used, so we define our own constants here just for this test.
const E0123: ErrCode = ErrCode::from_u32(0123);
const E0456: ErrCode = ErrCode::from_u32(0456);

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct Hello {}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
//~^ ERROR unsupported type attribute for diagnostic derive enum
enum DiagnosticOnEnum {
    Foo,
    //~^ ERROR diagnostic slug not specified
    Bar,
    //~^ ERROR diagnostic slug not specified
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[diag = "E0123"]
//~^ ERROR expected parentheses: #[diag(...)]
struct WrongStructAttrStyle {}

#[derive(Diagnostic)]
#[nonsense("this is an example message", code = E0123)]
//~^ ERROR `#[nonsense(...)]` is not a valid attribute
//~^^ ERROR diagnostic slug not specified
//~^^^ ERROR cannot find attribute `nonsense` in this scope
struct InvalidStructAttr {}

#[derive(Diagnostic)]
#[diag(code = E0123)]
//~^ ERROR diagnostic slug not specified
struct InvalidLitNestedAttr {}

#[derive(Diagnostic)]
#[diag(nonsense("foo"), code = E0123, slug = "foo")]
//~^ ERROR derive(Diagnostic): diagnostic slug not specified
struct InvalidNestedStructAttr1 {}

#[derive(Diagnostic)]
#[diag(nonsense = "...", code = E0123, slug = "foo")]
//~^ ERROR diagnostic slug not specified
struct InvalidNestedStructAttr2 {}

#[derive(Diagnostic)]
#[diag(nonsense = 4, code = E0123, slug = "foo")]
//~^ ERROR diagnostic slug not specified
struct InvalidNestedStructAttr3 {}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123, slug = "foo")]
//~^ ERROR unknown argument
struct InvalidNestedStructAttr4 {}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct WrongPlaceField {
    #[suggestion = "bar"]
    //~^ ERROR `#[suggestion = ...]` is not a valid attribute
    sp: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[diag("this is an example message", code = E0456)]
//~^ ERROR specified multiple times
struct DiagSpecifiedTwice {}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123, code = E0456)]
//~^ ERROR specified multiple times
struct CodeSpecifiedTwice {}

#[derive(Diagnostic)]
#[diag("this is an example message", no_crate::example, code = E0123)]
//~^ ERROR diagnostic slug must be the first argument
struct SlugSpecifiedTwice {}

#[derive(Diagnostic)]
struct KindNotProvided {} //~ ERROR diagnostic slug not specified

#[derive(Diagnostic)]
#[diag(code = E0123)]
//~^ ERROR diagnostic slug not specified
struct SlugNotProvided {}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct CodeNotProvided {}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct MessageWrongType {
    #[primary_span]
    //~^ ERROR `#[primary_span]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    foo: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct InvalidPathFieldAttr {
    #[nonsense]
    //~^ ERROR `#[nonsense]` is not a valid attribute
    //~^^ ERROR cannot find attribute `nonsense` in this scope
    foo: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithField {
    name: String,
    #[label("with a label")]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithMessageAppliedToField {
    #[label("with a label")]
    //~^ ERROR the `#[label(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    name: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithNonexistentField {
    #[suggestion("with a suggestion", code = "{name}")]
    //~^ ERROR `name` doesn't refer to a field on this type
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
//~^ ERROR invalid format string: expected `}`
#[diag("this is an example message", code = E0123)]
struct ErrorMissingClosingBrace {
    #[suggestion("with a suggestion", code = "{name")]
    suggestion: (Span, Applicability),
    name: String,
    val: usize,
}

#[derive(Diagnostic)]
//~^ ERROR invalid format string: unmatched `}`
#[diag("this is an example message", code = E0123)]
struct ErrorMissingOpeningBrace {
    #[suggestion("with a suggestion", code = "name}")]
    suggestion: (Span, Applicability),
    name: String,
    val: usize,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct LabelOnSpan {
    #[label("with a label")]
    sp: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct LabelOnNonSpan {
    #[label("with a label")]
    //~^ ERROR the `#[label(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    id: u32,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct Suggest {
    #[suggestion("with a suggestion", code = "This is the suggested code")]
    #[suggestion("with a suggestion", code = "This is the suggested code", style = "normal")]
    #[suggestion("with a suggestion", code = "This is the suggested code", style = "short")]
    #[suggestion("with a suggestion", code = "This is the suggested code", style = "hidden")]
    #[suggestion("with a suggestion", code = "This is the suggested code", style = "verbose")]
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithoutCode {
    #[suggestion("with a suggestion")]
    //~^ ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithBadKey {
    #[suggestion("with a suggestion", nonsense = "bar")]
    //~^ ERROR invalid nested attribute
    //~| ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithShorthandMsg {
    #[suggestion("with a suggestion", msg = "bar")]
    //~^ ERROR invalid nested attribute
    //~| ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithoutMsg {
    #[suggestion("with a suggestion", code = "bar")]
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithTypesSwapped {
    #[suggestion("with a suggestion", code = "This is suggested code")]
    suggestion: (Applicability, Span),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithWrongTypeApplicabilityOnly {
    #[suggestion("with a suggestion", code = "This is suggested code")]
    //~^ ERROR wrong field type for suggestion
    suggestion: Applicability,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithSpanOnly {
    #[suggestion("with a suggestion", code = "This is suggested code")]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithDuplicateSpanAndApplicability {
    #[suggestion("with a suggestion", code = "This is suggested code")]
    suggestion: (Span, Span, Applicability),
    //~^ ERROR specified multiple times
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct SuggestWithDuplicateApplicabilityAndSpan {
    #[suggestion("with a suggestion", code = "This is suggested code")]
    suggestion: (Applicability, Applicability, Span),
    //~^ ERROR specified multiple times
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct WrongKindOfAnnotation {
    #[label = "bar"]
    //~^ ERROR `#[label = ...]` is not a valid attribute
    z: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct OptionsInErrors {
    #[label("with a label")]
    label: Option<Span>,
    #[suggestion("with a suggestion", code = "...")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct MoveOutOfBorrowError<'tcx> {
    name: Ident,
    ty: Ty<'tcx>,
    #[primary_span]
    #[label("with a label")]
    span: Span,
    #[label("with a label")]
    other_span: Span,
    #[suggestion("with a suggestion", code = "{name}.clone()")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithLifetime<'a> {
    #[label("with a label")]
    span: Span,
    name: &'a str,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ArgFieldWithoutSkip {
    #[primary_span]
    span: Span,
    other: Hello,
    //~^ ERROR the trait bound `Hello: IntoDiagArg` is not satisfied
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ArgFieldWithSkip {
    #[primary_span]
    span: Span,
    // `Hello` does not implement `IntoDiagArg` so this would result in an error if
    // not for `#[skip_arg]`.
    #[skip_arg]
    other: Hello,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithSpannedNote {
    #[note("with a note")]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[note("with a note")]
struct ErrorWithNote {
    val: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithSpannedHelpCustom {
    #[help("with a help")]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[help("with a help")]
struct ErrorWithHelp {
    val: String,
}

#[derive(Diagnostic)]
#[help("with a help")]
#[diag("this is an example message", code = E0123)]
struct ErrorWithHelpWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[note("with a note")]
#[diag("this is an example message", code = E0123)]
struct ErrorWithNoteWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ApplicabilityInBoth {
    #[suggestion("with a suggestion", code = "...", applicability = "maybe-incorrect")]
    //~^ ERROR specified multiple times
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct InvalidApplicability {
    #[suggestion("with a suggestion", code = "...", applicability = "batman")]
    //~^ ERROR invalid applicability
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ValidApplicability {
    #[suggestion("with a suggestion", code = "...", applicability = "maybe-incorrect")]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct NoApplicability {
    #[suggestion("with a suggestion", code = "...")]
    suggestion: Span,
}

#[derive(Subdiagnostic)]
#[note("this is an example message")]
struct Note;

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct Subdiagnostic {
    #[subdiagnostic]
    note: Note,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct VecField {
    #[primary_span]
    #[label("with a label")]
    spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct UnitField {
    #[primary_span]
    spans: Span,
    #[help("with a help")]
    bar: (),
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct OptUnitField {
    #[primary_span]
    spans: Span,
    #[help("with a help")]
    foo: Option<()>,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct BoolField {
    #[primary_span]
    spans: Span,
    #[help("with a help")]
    foo: bool,
    #[help("with a help")]
    //~^ ERROR the `#[help(...)]` attribute can only be applied to fields of type
    // only allow plain 'bool' fields
    bar: Option<bool>,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct LabelWithTrailingPath {
    #[label("with a label", foo)]
    //~^ ERROR a diagnostic slug must be the first argument to the attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct LabelWithTrailingNameValue {
    #[label("with a label", foo = "...")]
    //~^ ERROR no nested attribute expected here
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct LabelWithTrailingList {
    #[label("with a label", foo("..."))]
    //~^ ERROR no nested attribute expected here
    span: Span,
}

#[derive(LintDiagnostic)]
#[diag("this is an example message")]
struct LintsGood {}

#[derive(LintDiagnostic)]
#[diag("this is an example message")]
struct PrimarySpanOnLint {
    #[primary_span]
    //~^ ERROR `#[primary_span]` is not a valid attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct ErrorWithMultiSpan {
    #[primary_span]
    span: MultiSpan,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[warning("with a warning")]
struct ErrorWithWarn {
    val: String,
}

#[derive(Diagnostic)]
#[error("this is an example message", code = E0123)]
//~^ ERROR `#[error(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `error` in this scope
struct ErrorAttribute {}

#[derive(Diagnostic)]
#[warn_("this is an example message", code = E0123)]
//~^ ERROR `#[warn_(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `warn_` in this scope
struct WarnAttribute {}

#[derive(Diagnostic)]
#[lint("this is an example message", code = E0123)]
//~^ ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `lint` in this scope
struct LintAttributeOnSessionDiag {}

#[derive(LintDiagnostic)]
#[lint("this is an example message", code = E0123)]
//~^ ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `lint` in this scope
struct LintAttributeOnLintDiag {}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct DuplicatedSuggestionCode {
    #[suggestion("with a suggestion", code = "...", code = ",,,")]
    //~^ ERROR specified multiple times
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct InvalidTypeInSuggestionTuple {
    #[suggestion("with a suggestion", code = "...")]
    suggestion: (Span, usize),
    //~^ ERROR wrong types for suggestion
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct MissingApplicabilityInSuggestionTuple {
    #[suggestion("with a suggestion", code = "...")]
    suggestion: (Span,),
    //~^ ERROR wrong types for suggestion
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct MissingCodeInSuggestion {
    #[suggestion("with a suggestion")]
    //~^ ERROR suggestion without `code = "..."`
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[multipart_suggestion("with a suggestion")]
//~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
//~| ERROR cannot find attribute `multipart_suggestion` in this scope
#[multipart_suggestion()]
//~^ ERROR cannot find attribute `multipart_suggestion` in this scope
//~| ERROR `#[multipart_suggestion(...)]` is not a valid attribute
struct MultipartSuggestion {
    #[multipart_suggestion("with a suggestion")]
    //~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
    //~| ERROR cannot find attribute `multipart_suggestion` in this scope
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[suggestion("with a suggestion", code = "...")]
//~^ ERROR `#[suggestion(...)]` is not a valid attribute
struct SuggestionOnStruct {
    #[primary_span]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
#[label]
//~^ ERROR `#[label]` is not a valid attribute
struct LabelOnStruct {
    #[primary_span]
    suggestion: Span,
}

#[derive(Diagnostic)]
enum ExampleEnum {
    #[diag("this is an example message")]
    Foo {
        #[primary_span]
        sp: Span,
        #[note("with a note")]
        note_sp: Span,
    },
    #[diag("this is an example message")]
    Bar {
        #[primary_span]
        sp: Span,
    },
    #[diag("this is an example message")]
    Baz,
}

#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct RawIdentDiagnosticArg {
    pub r#type: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticBad {
    #[subdiagnostic(bad)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticBadStr {
    #[subdiagnostic = "bad"]
    //~^ ERROR `#[subdiagnostic = ...]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticBadTwice {
    #[subdiagnostic(bad, bad)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticBadLitStr {
    #[subdiagnostic("bad")]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(LintDiagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticEagerLint {
    #[subdiagnostic(eager)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticEagerFormerlyCorrect {
    #[subdiagnostic(eager)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

// Check that formatting of `correct` in suggestion doesn't move the binding for that field, making
// the `arg` call a compile error; and that isn't worked around by moving the `arg` call
// after the `span_suggestion` call - which breaks eager translation.

#[derive(Subdiagnostic)]
#[suggestion("example message", applicability = "machine-applicable", code = "{correct}")]
pub(crate) struct SubdiagnosticWithSuggestion {
    #[primary_span]
    span: Span,
    invalid: String,
    correct: String,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SubdiagnosticEagerSuggestion {
    #[subdiagnostic(eager)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    sub: SubdiagnosticWithSuggestion,
}

/// with a doc comment on the type..
#[derive(Diagnostic)]
#[diag("this is an example message", code = E0123)]
struct WithDocComment {
    /// ..and the field
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionsGood {
    #[suggestion("with a suggestion", code("foo", "bar"))]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionsSingleItem {
    #[suggestion("with a suggestion", code("foo"))]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionsNoItem {
    #[suggestion("with a suggestion", code())]
    //~^ ERROR expected at least one string literal for `code(...)`
    sub: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionsInvalidItem {
    #[suggestion("with a suggestion", code(foo))]
    //~^ ERROR `code(...)` must contain only string literals
    //~| ERROR unexpected token, expected `)`
    sub: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionsInvalidLiteral {
    #[suggestion("with a suggestion", code = 3)]
    //~^ ERROR expected string literal
    sub: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionStyleGood {
    #[suggestion("with a suggestion", code = "", style = "hidden")]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag("this is an example message")]
struct SuggestionOnVec {
    #[suggestion("with a suggestion", code = "")]
    //~^ ERROR `#[suggestion(...)]` is not a valid attribute
    sub: Vec<Span>,
}
