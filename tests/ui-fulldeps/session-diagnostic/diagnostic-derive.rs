// check-fail
// Tests error conditions for specifying diagnostics using #[derive(Diagnostic)]

// normalize-stderr-test "the following other types implement trait `IntoDiagnosticArg`:(?:.*\n){0,9}\s+and \d+ others" -> "normalized in stderr"
// normalize-stderr-test "diagnostic_builder\.rs:[0-9]+:[0-9]+" -> "diagnostic_builder.rs:LL:CC"
// The proc_macro2 crate handles spans differently when on beta/stable release rather than nightly,
// changing the output of this test. Since Diagnostic is strictly internal to the compiler
// the test is just ignored on stable and beta:
// ignore-beta
// ignore-stable

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_span;
use rustc_span::symbol::Ident;
use rustc_span::Span;

extern crate rustc_macros;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};

extern crate rustc_middle;
use rustc_middle::ty::Ty;

extern crate rustc_errors;
use rustc_errors::{Applicability, MultiSpan};

extern crate rustc_session;

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct Hello {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct HelloWarn {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
//~^ ERROR unsupported type attribute for diagnostic derive enum
enum DiagnosticOnEnum {
    Foo,
    //~^ ERROR diagnostic slug not specified
    Bar,
    //~^ ERROR diagnostic slug not specified
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[diag = "E0123"]
//~^ ERROR `#[diag = ...]` is not a valid attribute
struct WrongStructAttrStyle {}

#[derive(Diagnostic)]
#[nonsense(compiletest_example, code = "E0123")]
//~^ ERROR `#[nonsense(...)]` is not a valid attribute
//~^^ ERROR diagnostic slug not specified
//~^^^ ERROR cannot find attribute `nonsense` in this scope
struct InvalidStructAttr {}

#[derive(Diagnostic)]
#[diag("E0123")]
//~^ ERROR `#[diag("...")]` is not a valid attribute
//~^^ ERROR diagnostic slug not specified
struct InvalidLitNestedAttr {}

#[derive(Diagnostic)]
#[diag(nonsense, code = "E0123")]
//~^ ERROR cannot find value `nonsense` in module `rustc_errors::fluent`
struct InvalidNestedStructAttr {}

#[derive(Diagnostic)]
#[diag(nonsense("foo"), code = "E0123", slug = "foo")]
//~^ ERROR `#[diag(nonsense(...))]` is not a valid attribute
//~^^ ERROR diagnostic slug not specified
struct InvalidNestedStructAttr1 {}

#[derive(Diagnostic)]
#[diag(nonsense = "...", code = "E0123", slug = "foo")]
//~^ ERROR `#[diag(nonsense = ...)]` is not a valid attribute
//~| ERROR `#[diag(slug = ...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
struct InvalidNestedStructAttr2 {}

#[derive(Diagnostic)]
#[diag(nonsense = 4, code = "E0123", slug = "foo")]
//~^ ERROR `#[diag(nonsense = ...)]` is not a valid attribute
//~| ERROR `#[diag(slug = ...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
struct InvalidNestedStructAttr3 {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123", slug = "foo")]
//~^ ERROR `#[diag(slug = ...)]` is not a valid attribute
struct InvalidNestedStructAttr4 {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct WrongPlaceField {
    #[suggestion = "bar"]
    //~^ ERROR `#[suggestion = ...]` is not a valid attribute
    sp: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[diag(compiletest_example, code = "E0456")]
//~^ ERROR specified multiple times
//~^^ ERROR specified multiple times
struct DiagSpecifiedTwice {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0456", code = "E0457")]
//~^ ERROR specified multiple times
struct CodeSpecifiedTwice {}

#[derive(Diagnostic)]
#[diag(compiletest_example, compiletest_example, code = "E0456")]
//~^ ERROR `#[diag(compiletest_example)]` is not a valid attribute
struct SlugSpecifiedTwice {}

#[derive(Diagnostic)]
struct KindNotProvided {} //~ ERROR diagnostic slug not specified

#[derive(Diagnostic)]
#[diag(code = "E0456")]
//~^ ERROR diagnostic slug not specified
struct SlugNotProvided {}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct CodeNotProvided {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct MessageWrongType {
    #[primary_span]
    //~^ ERROR `#[primary_span]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    foo: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct InvalidPathFieldAttr {
    #[nonsense]
    //~^ ERROR `#[nonsense]` is not a valid attribute
    //~^^ ERROR cannot find attribute `nonsense` in this scope
    foo: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithField {
    name: String,
    #[label(label)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithMessageAppliedToField {
    #[label(label)]
    //~^ ERROR the `#[label(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    name: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithNonexistentField {
    #[suggestion(suggestion, code = "{name}")]
    //~^ ERROR `name` doesn't refer to a field on this type
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
//~^ ERROR invalid format string: expected `'}'`
#[diag(compiletest_example, code = "E0123")]
struct ErrorMissingClosingBrace {
    #[suggestion(suggestion, code = "{name")]
    suggestion: (Span, Applicability),
    name: String,
    val: usize,
}

#[derive(Diagnostic)]
//~^ ERROR invalid format string: unmatched `}`
#[diag(compiletest_example, code = "E0123")]
struct ErrorMissingOpeningBrace {
    #[suggestion(suggestion, code = "name}")]
    suggestion: (Span, Applicability),
    name: String,
    val: usize,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct LabelOnSpan {
    #[label(label)]
    sp: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct LabelOnNonSpan {
    #[label(label)]
    //~^ ERROR the `#[label(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    id: u32,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct Suggest {
    #[suggestion(suggestion, code = "This is the suggested code")]
    #[suggestion(suggestion, code = "This is the suggested code", style = "normal")]
    #[suggestion(suggestion, code = "This is the suggested code", style = "short")]
    #[suggestion(suggestion, code = "This is the suggested code", style = "hidden")]
    #[suggestion(suggestion, code = "This is the suggested code", style = "verbose")]
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithoutCode {
    #[suggestion(suggestion)]
    //~^ ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithBadKey {
    #[suggestion(nonsense = "bar")]
    //~^ ERROR `#[suggestion(nonsense = ...)]` is not a valid attribute
    //~| ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithShorthandMsg {
    #[suggestion(msg = "bar")]
    //~^ ERROR `#[suggestion(msg = ...)]` is not a valid attribute
    //~| ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithoutMsg {
    #[suggestion(code = "bar")]
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithTypesSwapped {
    #[suggestion(suggestion, code = "This is suggested code")]
    suggestion: (Applicability, Span),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithWrongTypeApplicabilityOnly {
    #[suggestion(suggestion, code = "This is suggested code")]
    //~^ ERROR wrong field type for suggestion
    suggestion: Applicability,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithSpanOnly {
    #[suggestion(suggestion, code = "This is suggested code")]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithDuplicateSpanAndApplicability {
    #[suggestion(suggestion, code = "This is suggested code")]
    suggestion: (Span, Span, Applicability),
    //~^ ERROR specified multiple times
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct SuggestWithDuplicateApplicabilityAndSpan {
    #[suggestion(suggestion, code = "This is suggested code")]
    suggestion: (Applicability, Applicability, Span),
    //~^ ERROR specified multiple times
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct WrongKindOfAnnotation {
    #[label = "bar"]
    //~^ ERROR `#[label = ...]` is not a valid attribute
    z: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct OptionsInErrors {
    #[label(label)]
    label: Option<Span>,
    #[suggestion(suggestion, code = "...")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0456")]
struct MoveOutOfBorrowError<'tcx> {
    name: Ident,
    ty: Ty<'tcx>,
    #[primary_span]
    #[label(label)]
    span: Span,
    #[label(label)]
    other_span: Span,
    #[suggestion(suggestion, code = "{name}.clone()")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithLifetime<'a> {
    #[label(label)]
    span: Span,
    name: &'a str,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithDefaultLabelAttr<'a> {
    #[label]
    span: Span,
    name: &'a str,
}

#[derive(Diagnostic)]
//~^ ERROR the trait bound `Hello: IntoDiagnosticArg` is not satisfied
#[diag(compiletest_example, code = "E0123")]
struct ArgFieldWithoutSkip {
    #[primary_span]
    span: Span,
    other: Hello,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ArgFieldWithSkip {
    #[primary_span]
    span: Span,
    // `Hello` does not implement `IntoDiagnosticArg` so this would result in an error if
    // not for `#[skip_arg]`.
    #[skip_arg]
    other: Hello,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithSpannedNote {
    #[note]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithSpannedNoteCustom {
    #[note(note)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[note]
struct ErrorWithNote {
    val: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[note(note)]
struct ErrorWithNoteCustom {
    val: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithSpannedHelp {
    #[help]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithSpannedHelpCustom {
    #[help(help)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[help]
struct ErrorWithHelp {
    val: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[help(help)]
struct ErrorWithHelpCustom {
    val: String,
}

#[derive(Diagnostic)]
#[help]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithHelpWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[help(help)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithHelpCustomWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[note]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithNoteWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[note(note)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithNoteCustomWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ApplicabilityInBoth {
    #[suggestion(suggestion, code = "...", applicability = "maybe-incorrect")]
    //~^ ERROR specified multiple times
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct InvalidApplicability {
    #[suggestion(suggestion, code = "...", applicability = "batman")]
    //~^ ERROR invalid applicability
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ValidApplicability {
    #[suggestion(suggestion, code = "...", applicability = "maybe-incorrect")]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct NoApplicability {
    #[suggestion(suggestion, code = "...")]
    suggestion: Span,
}

#[derive(Subdiagnostic)]
#[note(parse_add_paren)]
struct Note;

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct Subdiagnostic {
    #[subdiagnostic]
    note: Note,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct VecField {
    #[primary_span]
    #[label]
    spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct UnitField {
    #[primary_span]
    spans: Span,
    #[help]
    foo: (),
    #[help(help)]
    bar: (),
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct OptUnitField {
    #[primary_span]
    spans: Span,
    #[help]
    foo: Option<()>,
    #[help(help)]
    bar: Option<()>,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct LabelWithTrailingPath {
    #[label(label, foo)]
    //~^ ERROR `#[label(foo)]` is not a valid attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct LabelWithTrailingNameValue {
    #[label(label, foo = "...")]
    //~^ ERROR `#[label(foo = ...)]` is not a valid attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct LabelWithTrailingList {
    #[label(label, foo("..."))]
    //~^ ERROR `#[label(foo(...))]` is not a valid attribute
    span: Span,
}

#[derive(LintDiagnostic)]
#[diag(compiletest_example)]
struct LintsGood {}

#[derive(LintDiagnostic)]
#[diag(compiletest_example)]
struct PrimarySpanOnLint {
    #[primary_span]
    //~^ ERROR `#[primary_span]` is not a valid attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct ErrorWithMultiSpan {
    #[primary_span]
    span: MultiSpan,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[warning]
struct ErrorWithWarn {
    val: String,
}

#[derive(Diagnostic)]
#[error(compiletest_example, code = "E0123")]
//~^ ERROR `#[error(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `error` in this scope
struct ErrorAttribute {}

#[derive(Diagnostic)]
#[warn_(compiletest_example, code = "E0123")]
//~^ ERROR `#[warn_(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `warn_` in this scope
struct WarnAttribute {}

#[derive(Diagnostic)]
#[lint(compiletest_example, code = "E0123")]
//~^ ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `lint` in this scope
struct LintAttributeOnSessionDiag {}

#[derive(LintDiagnostic)]
#[lint(compiletest_example, code = "E0123")]
//~^ ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `lint` in this scope
struct LintAttributeOnLintDiag {}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct DuplicatedSuggestionCode {
    #[suggestion(suggestion, code = "...", code = ",,,")]
    //~^ ERROR specified multiple times
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct InvalidTypeInSuggestionTuple {
    #[suggestion(suggestion, code = "...")]
    suggestion: (Span, usize),
    //~^ ERROR wrong types for suggestion
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct MissingApplicabilityInSuggestionTuple {
    #[suggestion(suggestion, code = "...")]
    suggestion: (Span,),
    //~^ ERROR wrong types for suggestion
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct MissingCodeInSuggestion {
    #[suggestion(suggestion)]
    //~^ ERROR suggestion without `code = "..."`
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[multipart_suggestion(suggestion)]
//~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
//~| ERROR cannot find attribute `multipart_suggestion` in this scope
#[multipart_suggestion()]
//~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
//~| ERROR cannot find attribute `multipart_suggestion` in this scope
struct MultipartSuggestion {
    #[multipart_suggestion(suggestion)]
    //~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
    //~| ERROR cannot find attribute `multipart_suggestion` in this scope
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[suggestion(suggestion, code = "...")]
//~^ ERROR `#[suggestion(...)]` is not a valid attribute
struct SuggestionOnStruct {
    #[primary_span]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
#[label]
//~^ ERROR `#[label]` is not a valid attribute
struct LabelOnStruct {
    #[primary_span]
    suggestion: Span,
}

#[derive(Diagnostic)]
enum ExampleEnum {
    #[diag(compiletest_example)]
    Foo {
        #[primary_span]
        sp: Span,
        #[note]
        note_sp: Span,
    },
    #[diag(compiletest_example)]
    Bar {
        #[primary_span]
        sp: Span,
    },
    #[diag(compiletest_example)]
    Baz,
}

#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct RawIdentDiagnosticArg {
    pub r#type: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticBad {
    #[subdiagnostic(bad)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticBadStr {
    #[subdiagnostic = "bad"]
    //~^ ERROR `#[subdiagnostic = ...]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticBadTwice {
    #[subdiagnostic(bad, bad)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticBadLitStr {
    #[subdiagnostic("bad")]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(LintDiagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticEagerLint {
    #[subdiagnostic(eager)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticEagerCorrect {
    #[subdiagnostic(eager)]
    note: Note,
}

// Check that formatting of `correct` in suggestion doesn't move the binding for that field, making
// the `set_arg` call a compile error; and that isn't worked around by moving the `set_arg` call
// after the `span_suggestion` call - which breaks eager translation.

#[derive(Subdiagnostic)]
#[suggestion(use_instead, applicability = "machine-applicable", code = "{correct}")]
pub(crate) struct SubdiagnosticWithSuggestion {
    #[primary_span]
    span: Span,
    invalid: String,
    correct: String,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SubdiagnosticEagerSuggestion {
    #[subdiagnostic(eager)]
    sub: SubdiagnosticWithSuggestion,
}

/// with a doc comment on the type..
#[derive(Diagnostic)]
#[diag(compiletest_example, code = "E0123")]
struct WithDocComment {
    /// ..and the field
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionsGood {
    #[suggestion(code("foo", "bar"))]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionsSingleItem {
    #[suggestion(code("foo"))]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionsNoItem {
    #[suggestion(code())]
    //~^ ERROR expected at least one string literal for `code(...)`
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionsInvalidItem {
    #[suggestion(code(foo))]
    //~^ ERROR `code(...)` must contain only string literals
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionsInvalidLiteral {
    #[suggestion(code = 3)]
    //~^ ERROR `code = "..."`/`code(...)` must contain only string literals
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionStyleGood {
    #[suggestion(code = "", style = "hidden")]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(compiletest_example)]
struct SuggestionOnVec {
    #[suggestion(suggestion, code = "")]
    //~^ ERROR `#[suggestion(...)]` is not a valid attribute
    sub: Vec<Span>,
}
