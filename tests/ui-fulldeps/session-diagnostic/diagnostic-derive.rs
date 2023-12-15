// check-fail
// Tests error conditions for specifying diagnostics using #[derive(Diagnostic)]

// normalize-stderr-test "the following other types implement trait `IntoDiagnosticArg`:(?:.*\n){0,9}\s+and \d+ others" -> "normalized in stderr"
// normalize-stderr-test "diagnostic_builder\.rs:[0-9]+:[0-9]+" -> "diagnostic_builder.rs:LL:CC"
// The proc_macro2 crate handles spans differently when on beta/stable release rather than nightly,
// changing the output of this test. Since Diagnostic is strictly internal to the compiler
// the test is just ignored on stable and beta:
// ignore-stage1
// ignore-beta
// ignore-stable

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
use rustc_errors::{Applicability, DiagnosticMessage, MultiSpan, SubdiagnosticMessage};

extern crate rustc_session;

rustc_fluent_macro::fluent_messages! { "./example.ftl" }

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct Hello {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct HelloWarn {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
//~^ ERROR unsupported type attribute for diagnostic derive enum
enum DiagnosticOnEnum {
    Foo,
    //~^ ERROR diagnostic slug not specified
    Bar,
    //~^ ERROR diagnostic slug not specified
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[diag = "E0123"]
//~^ ERROR failed to resolve: maybe a missing crate `core`
#[must_use]
struct WrongStructAttrStyle {}

#[derive(Diagnostic)]
#[nonsense(no_crate_example, code = "E0123")]
//~^ ERROR `#[nonsense(...)]` is not a valid attribute
//~^^ ERROR diagnostic slug not specified
//~^^^ ERROR cannot find attribute `nonsense` in this scope
#[must_use]
struct InvalidStructAttr {}

#[derive(Diagnostic)]
#[diag("E0123")]
//~^ ERROR diagnostic slug not specified
#[must_use]
struct InvalidLitNestedAttr {}

#[derive(Diagnostic)]
#[diag(nonsense, code = "E0123")]
//~^ ERROR cannot find value `nonsense` in module `crate::fluent_generated`
#[must_use]
struct InvalidNestedStructAttr {}

#[derive(Diagnostic)]
#[diag(nonsense("foo"), code = "E0123", slug = "foo")]
//~^ ERROR diagnostic slug must be the first argument
//~| ERROR diagnostic slug not specified
#[must_use]
struct InvalidNestedStructAttr1 {}

#[derive(Diagnostic)]
#[diag(nonsense = "...", code = "E0123", slug = "foo")]
//~^ ERROR unknown argument
//~| ERROR diagnostic slug not specified
#[must_use]
struct InvalidNestedStructAttr2 {}

#[derive(Diagnostic)]
#[diag(nonsense = 4, code = "E0123", slug = "foo")]
//~^ ERROR unknown argument
//~| ERROR diagnostic slug not specified
#[must_use]
struct InvalidNestedStructAttr3 {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123", slug = "foo")]
//~^ ERROR unknown argument
#[must_use]
struct InvalidNestedStructAttr4 {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct WrongPlaceField {
    #[suggestion = "bar"]
    //~^ ERROR `#[suggestion = ...]` is not a valid attribute
    sp: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[diag(no_crate_example, code = "E0456")]
//~^ ERROR specified multiple times
//~^^ ERROR specified multiple times
#[must_use]
struct DiagSpecifiedTwice {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0456", code = "E0457")]
//~^ ERROR specified multiple times
#[must_use]
struct CodeSpecifiedTwice {}

#[derive(Diagnostic)]
#[diag(no_crate_example, no_crate::example, code = "E0456")]
//~^ ERROR diagnostic slug must be the first argument
#[must_use]
struct SlugSpecifiedTwice {}

#[derive(Diagnostic)]
#[must_use] //~ ERROR diagnostic slug not specified
struct KindNotProvided {}

#[derive(Diagnostic)]
#[diag(code = "E0456")]
//~^ ERROR diagnostic slug not specified
#[must_use]
struct SlugNotProvided {}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct CodeNotProvided {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct MessageWrongType {
    #[primary_span]
    //~^ ERROR `#[primary_span]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    foo: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct InvalidPathFieldAttr {
    #[nonsense]
    //~^ ERROR `#[nonsense]` is not a valid attribute
    //~^^ ERROR cannot find attribute `nonsense` in this scope
    foo: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithField {
    name: String,
    #[label(no_crate_label)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithMessageAppliedToField {
    #[label(no_crate_label)]
    //~^ ERROR the `#[label(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    name: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithNonexistentField {
    #[suggestion(no_crate_suggestion, code = "{name}")]
    //~^ ERROR `name` doesn't refer to a field on this type
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
//~^ ERROR invalid format string: expected `'}'`
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorMissingClosingBrace {
    #[suggestion(no_crate_suggestion, code = "{name")]
    suggestion: (Span, Applicability),
    name: String,
    val: usize,
}

#[derive(Diagnostic)]
//~^ ERROR invalid format string: unmatched `}`
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorMissingOpeningBrace {
    #[suggestion(no_crate_suggestion, code = "name}")]
    suggestion: (Span, Applicability),
    name: String,
    val: usize,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct LabelOnSpan {
    #[label(no_crate_label)]
    sp: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct LabelOnNonSpan {
    #[label(no_crate_label)]
    //~^ ERROR the `#[label(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    id: u32,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct Suggest {
    #[suggestion(no_crate_suggestion, code = "This is the suggested code")]
    #[suggestion(no_crate_suggestion, code = "This is the suggested code", style = "normal")]
    #[suggestion(no_crate_suggestion, code = "This is the suggested code", style = "short")]
    #[suggestion(no_crate_suggestion, code = "This is the suggested code", style = "hidden")]
    #[suggestion(no_crate_suggestion, code = "This is the suggested code", style = "verbose")]
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithoutCode {
    #[suggestion(no_crate_suggestion)]
    //~^ ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithBadKey {
    #[suggestion(nonsense = "bar")]
    //~^ ERROR invalid nested attribute
    //~| ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithShorthandMsg {
    #[suggestion(msg = "bar")]
    //~^ ERROR invalid nested attribute
    //~| ERROR suggestion without `code = "..."`
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithoutMsg {
    #[suggestion(code = "bar")]
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithTypesSwapped {
    #[suggestion(no_crate_suggestion, code = "This is suggested code")]
    suggestion: (Applicability, Span),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithWrongTypeApplicabilityOnly {
    #[suggestion(no_crate_suggestion, code = "This is suggested code")]
    //~^ ERROR wrong field type for suggestion
    suggestion: Applicability,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithSpanOnly {
    #[suggestion(no_crate_suggestion, code = "This is suggested code")]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithDuplicateSpanAndApplicability {
    #[suggestion(no_crate_suggestion, code = "This is suggested code")]
    suggestion: (Span, Span, Applicability),
    //~^ ERROR specified multiple times
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct SuggestWithDuplicateApplicabilityAndSpan {
    #[suggestion(no_crate_suggestion, code = "This is suggested code")]
    suggestion: (Applicability, Applicability, Span),
    //~^ ERROR specified multiple times
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct WrongKindOfAnnotation {
    #[label = "bar"]
    //~^ ERROR `#[label = ...]` is not a valid attribute
    z: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct OptionsInErrors {
    #[label(no_crate_label)]
    label: Option<Span>,
    #[suggestion(no_crate_suggestion, code = "...")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0456")]
#[must_use]
struct MoveOutOfBorrowError<'tcx> {
    name: Ident,
    ty: Ty<'tcx>,
    #[primary_span]
    #[label(no_crate_label)]
    span: Span,
    #[label(no_crate_label)]
    other_span: Span,
    #[suggestion(no_crate_suggestion, code = "{name}.clone()")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithLifetime<'a> {
    #[label(no_crate_label)]
    span: Span,
    name: &'a str,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithDefaultLabelAttr<'a> {
    #[label]
    span: Span,
    name: &'a str,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ArgFieldWithoutSkip {
    #[primary_span]
    span: Span,
    other: Hello,
    //~^ ERROR the trait bound `Hello: IntoDiagnosticArg` is not satisfied
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ArgFieldWithSkip {
    #[primary_span]
    span: Span,
    // `Hello` does not implement `IntoDiagnosticArg` so this would result in an error if
    // not for `#[skip_arg]`.
    #[skip_arg]
    other: Hello,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithSpannedNote {
    #[note]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithSpannedNoteCustom {
    #[note(no_crate_note)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[note]
#[must_use]
struct ErrorWithNote {
    val: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[note(no_crate_note)]
#[must_use]
struct ErrorWithNoteCustom {
    val: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithSpannedHelp {
    #[help]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithSpannedHelpCustom {
    #[help(no_crate_help)]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[help]
#[must_use]
struct ErrorWithHelp {
    val: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[help(no_crate_help)]
#[must_use]
struct ErrorWithHelpCustom {
    val: String,
}

#[derive(Diagnostic)]
#[help]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithHelpWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[help(no_crate_help)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithHelpCustomWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[note]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithNoteWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[note(no_crate_note)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithNoteCustomWrongOrder {
    val: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ApplicabilityInBoth {
    #[suggestion(no_crate_suggestion, code = "...", applicability = "maybe-incorrect")]
    //~^ ERROR specified multiple times
    suggestion: (Span, Applicability),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct InvalidApplicability {
    #[suggestion(no_crate_suggestion, code = "...", applicability = "batman")]
    //~^ ERROR invalid applicability
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ValidApplicability {
    #[suggestion(no_crate_suggestion, code = "...", applicability = "maybe-incorrect")]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct NoApplicability {
    #[suggestion(no_crate_suggestion, code = "...")]
    suggestion: Span,
}

#[derive(Subdiagnostic)]
#[note(no_crate_example)]
struct Note;

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct Subdiagnostic {
    #[subdiagnostic]
    note: Note,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct VecField {
    #[primary_span]
    #[label]
    spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct UnitField {
    #[primary_span]
    spans: Span,
    #[help]
    foo: (),
    #[help(no_crate_help)]
    bar: (),
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct OptUnitField {
    #[primary_span]
    spans: Span,
    #[help]
    foo: Option<()>,
    #[help(no_crate_help)]
    bar: Option<()>,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct BoolField {
    #[primary_span]
    spans: Span,
    #[help]
    foo: bool,
    #[help(no_crate_help)]
    //~^ ERROR the `#[help(...)]` attribute can only be applied to fields of type
    // only allow plain 'bool' fields
    bar: Option<bool>,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct LabelWithTrailingPath {
    #[label(no_crate_label, foo)]
    //~^ ERROR a diagnostic slug must be the first argument to the attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct LabelWithTrailingNameValue {
    #[label(no_crate_label, foo = "...")]
    //~^ ERROR only `no_span` is a valid nested attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct LabelWithTrailingList {
    #[label(no_crate_label, foo("..."))]
    //~^ ERROR only `no_span` is a valid nested attribute
    span: Span,
}

#[derive(LintDiagnostic)]
#[diag(no_crate_example)]
struct LintsGood {}

#[derive(LintDiagnostic)]
#[diag(no_crate_example)]
struct PrimarySpanOnLint {
    #[primary_span]
    //~^ ERROR `#[primary_span]` is not a valid attribute
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct ErrorWithMultiSpan {
    #[primary_span]
    span: MultiSpan,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[warning]
#[must_use]
struct ErrorWithWarn {
    val: String,
}

#[derive(Diagnostic)]
#[error(no_crate_example, code = "E0123")]
//~^ ERROR `#[error(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `error` in this scope
#[must_use]
struct ErrorAttribute {}

#[derive(Diagnostic)]
#[warn_(no_crate_example, code = "E0123")]
//~^ ERROR `#[warn_(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `warn_` in this scope
#[must_use]
struct WarnAttribute {}

#[derive(Diagnostic)]
#[lint(no_crate_example, code = "E0123")]
//~^ ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `lint` in this scope
#[must_use]
struct LintAttributeOnSessionDiag {}

#[derive(LintDiagnostic)]
#[lint(no_crate_example, code = "E0123")]
//~^ ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR `#[lint(...)]` is not a valid attribute
//~| ERROR diagnostic slug not specified
//~| ERROR cannot find attribute `lint` in this scope
struct LintAttributeOnLintDiag {}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct DuplicatedSuggestionCode {
    #[suggestion(no_crate_suggestion, code = "...", code = ",,,")]
    //~^ ERROR specified multiple times
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct InvalidTypeInSuggestionTuple {
    #[suggestion(no_crate_suggestion, code = "...")]
    suggestion: (Span, usize),
    //~^ ERROR wrong types for suggestion
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct MissingApplicabilityInSuggestionTuple {
    #[suggestion(no_crate_suggestion, code = "...")]
    suggestion: (Span,),
    //~^ ERROR wrong types for suggestion
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct MissingCodeInSuggestion {
    #[suggestion(no_crate_suggestion)]
    //~^ ERROR suggestion without `code = "..."`
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[multipart_suggestion(no_crate_suggestion)]
//~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
//~| ERROR cannot find attribute `multipart_suggestion` in this scope
#[multipart_suggestion()]
//~^ ERROR cannot find attribute `multipart_suggestion` in this scope
//~| ERROR `#[multipart_suggestion(...)]` is not a valid attribute
#[must_use]
struct MultipartSuggestion {
    #[multipart_suggestion(no_crate_suggestion)]
    //~^ ERROR `#[multipart_suggestion(...)]` is not a valid attribute
    //~| ERROR cannot find attribute `multipart_suggestion` in this scope
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[suggestion(no_crate_suggestion, code = "...")]
//~^ ERROR `#[suggestion(...)]` is not a valid attribute
#[must_use]
struct SuggestionOnStruct {
    #[primary_span]
    suggestion: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[label]
//~^ ERROR `#[label]` is not a valid attribute
#[must_use]
struct LabelOnStruct {
    #[primary_span]
    suggestion: Span,
}

#[derive(Diagnostic)]
enum ExampleEnum {
    #[diag(no_crate_example)]
    Foo {
        #[primary_span]
        sp: Span,
        #[note]
        note_sp: Span,
    },
    #[diag(no_crate_example)]
    Bar {
        #[primary_span]
        sp: Span,
    },
    #[diag(no_crate_example)]
    Baz,
}

#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct RawIdentDiagnosticArg {
    pub r#type: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SubdiagnosticBad {
    #[subdiagnostic(bad)]
    //~^ ERROR `eager` is the only supported nested attribute for `subdiagnostic`
    note: Note,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SubdiagnosticBadStr {
    #[subdiagnostic = "bad"]
    //~^ ERROR `#[subdiagnostic = ...]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SubdiagnosticBadTwice {
    #[subdiagnostic(bad, bad)]
    //~^ ERROR `eager` is the only supported nested attribute for `subdiagnostic`
    note: Note,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SubdiagnosticBadLitStr {
    #[subdiagnostic("bad")]
    //~^ ERROR `eager` is the only supported nested attribute for `subdiagnostic`
    note: Note,
}

#[derive(LintDiagnostic)]
#[diag(no_crate_example)]
struct SubdiagnosticEagerLint {
    #[subdiagnostic(eager)]
    //~^ ERROR `#[subdiagnostic(...)]` is not a valid attribute
    note: Note,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SubdiagnosticEagerCorrect {
    #[subdiagnostic(eager)]
    note: Note,
}

// Check that formatting of `correct` in suggestion doesn't move the binding for that field, making
// the `set_arg` call a compile error; and that isn't worked around by moving the `set_arg` call
// after the `span_suggestion` call - which breaks eager translation.

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, applicability = "machine-applicable", code = "{correct}")]
pub(crate) struct SubdiagnosticWithSuggestion {
    #[primary_span]
    span: Span,
    invalid: String,
    correct: String,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SubdiagnosticEagerSuggestion {
    #[subdiagnostic(eager)]
    sub: SubdiagnosticWithSuggestion,
}

/// with a doc comment on the type..
#[derive(Diagnostic)]
#[diag(no_crate_example, code = "E0123")]
#[must_use]
struct WithDocComment {
    /// ..and the field
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SuggestionsGood {
    #[suggestion(code("foo", "bar"))]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SuggestionsSingleItem {
    #[suggestion(code("foo"))]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SuggestionsNoItem {
    #[suggestion(code())]
    //~^ ERROR expected at least one string literal for `code(...)`
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SuggestionsInvalidItem {
    #[suggestion(code(foo))]
    //~^ ERROR `code(...)` must contain only string literals
    //~| ERROR failed to resolve: maybe a missing crate `core`
    sub: Span,
}

#[derive(Diagnostic)] //~ ERROR cannot find value `__code_34` in this scope
#[diag(no_crate_example)]
#[must_use]
struct SuggestionsInvalidLiteral {
    #[suggestion(code = 3)]
    //~^ ERROR failed to resolve: maybe a missing crate `core`
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SuggestionStyleGood {
    #[suggestion(code = "", style = "hidden")]
    sub: Span,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)]
#[must_use]
struct SuggestionOnVec {
    #[suggestion(no_crate_suggestion, code = "")]
    //~^ ERROR `#[suggestion(...)]` is not a valid attribute
    sub: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag(no_crate_example)] //~ ERROR You must mark diagnostic structs with `#[must_use]`
struct MissingMustUseAttr {}
