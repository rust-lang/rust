// check-fail
// Tests error conditions for specifying diagnostics using #[derive(SessionDiagnostic)]

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_span;
use rustc_span::Span;
use rustc_span::symbol::Ident;

extern crate rustc_macros;
use rustc_macros::SessionDiagnostic;

extern crate rustc_middle;
use rustc_middle::ty::Ty;

extern crate rustc_errors;
use rustc_errors::Applicability;

extern crate rustc_session;

#[derive(SessionDiagnostic)]
#[message = "Hello, world!"]
#[error = "E0123"]
struct Hello {}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
//~^ ERROR `#[derive(SessionDiagnostic)]` can only be used on structs
enum SessionDiagnosticOnEnum {
    Foo,
    Bar,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[label = "This is in the wrong place"]
//~^ ERROR `#[label = ...]` is not a valid SessionDiagnostic struct attribute
struct WrongPlace {}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct WrongPlaceField {
    #[suggestion = "this is the wrong kind of attribute"]
//~^ ERROR `#[suggestion = ...]` is not a valid SessionDiagnostic field attribute
    sp: Span,
}

#[derive(SessionDiagnostic)]
#[message = "Hello, world!"]
#[error = "E0123"]
#[error = "E0456"] //~ ERROR `error` specified multiple times
struct ErrorSpecifiedTwice {}

#[derive(SessionDiagnostic)]
#[message = "Hello, world!"]
#[error = "E0123"]
#[lint = "some_useful_lint"] //~ ERROR `lint` specified when `error` was already specified
struct LintSpecifiedAfterError {}

#[derive(SessionDiagnostic)]
#[message = "Some lint message"]
#[error = "E0123"]
struct LintButHasErrorCode {}

#[derive(SessionDiagnostic)]
struct ErrorCodeNotProvided {} //~ ERROR `code` not specified

// FIXME: Uncomment when emitting lints is supported.
/*
#[derive(SessionDiagnostic)]
#[message = "Hello, world!"]
#[lint = "clashing_extern_declarations"]
#[lint = "improper_ctypes"] // FIXME: ERROR `lint` specified multiple times
struct LintSpecifiedTwice {}

#[derive(SessionDiagnostic)]
#[lint = "Some lint message"]
#[message = "Some error message"]
#[error = "E0123"] // ERROR `error` specified when `lint` was already specified
struct ErrorSpecifiedAfterLint {}
*/

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct ErrorWithField {
    name: String,
    #[message = "This error has a field, and references {name}"]
    span: Span
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct ErrorWithMessageAppliedToField {
    #[message = "this message is applied to a String field"]
    //~^ ERROR the `#[message = "..."]` attribute can only be applied to fields of type Span
    name: String,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[message = "This error has a field, and references {name}"]
//~^ ERROR `name` doesn't refer to a field on this type
struct ErrorWithNonexistentField {
    span: Span
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[message = "This is missing a closing brace: {name"]
//~^ ERROR invalid format string: expected `'}'`
struct ErrorMissingClosingBrace {
    name: String,
    span: Span
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[message = "This is missing an opening brace: name}"]
//~^ ERROR invalid format string: unmatched `}`
struct ErrorMissingOpeningBrace {
    name: String,
    span: Span
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[message = "Something something"]
struct LabelOnSpan {
    #[label = "See here"]
    sp: Span
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[message = "Something something"]
struct LabelOnNonSpan {
    #[label = "See here"]
    //~^ ERROR The `#[label = ...]` attribute can only be applied to fields of type Span
    id: u32,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct Suggest {
    #[suggestion(message = "This is a suggestion", code = "This is the suggested code")]
    #[suggestion_short(message = "This is a suggestion", code = "This is the suggested code")]
    #[suggestion_hidden(message = "This is a suggestion", code = "This is the suggested code")]
    #[suggestion_verbose(message = "This is a suggestion", code = "This is the suggested code")]
    suggestion: (Span, Applicability),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithoutCode {
    #[suggestion(message = "This is a suggestion")]
    suggestion: (Span, Applicability),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithBadKey {
    #[suggestion(nonsense = "This is nonsense")]
    //~^ ERROR `nonsense` is not a valid key for `#[suggestion(...)]`
    suggestion: (Span, Applicability),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithShorthandMsg {
    #[suggestion(msg = "This is a suggestion")]
    //~^ ERROR `msg` is not a valid key for `#[suggestion(...)]`
    suggestion: (Span, Applicability),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithoutMsg {
    #[suggestion(code = "This is suggested code")]
    //~^ ERROR missing suggestion message
    suggestion: (Span, Applicability),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithTypesSwapped {
    #[suggestion(message = "This is a message", code = "This is suggested code")]
    suggestion: (Applicability, Span),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithWrongTypeApplicabilityOnly {
    #[suggestion(message = "This is a message", code = "This is suggested code")]
    //~^ ERROR wrong field type for suggestion
    suggestion: Applicability,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithSpanOnly{
    #[suggestion(message = "This is a message", code = "This is suggested code")]
    suggestion: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithDuplicateSpanAndApplicability {
    #[suggestion(message = "This is a message", code = "This is suggested code")]
    //~^ ERROR type of field annotated with `#[suggestion(...)]` contains more than one Span
    suggestion: (Span, Span, Applicability),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct SuggestWithDuplicateApplicabilityAndSpan {
    #[suggestion(message = "This is a message", code = "This is suggested code")]
    //~^ ERROR type of field annotated with `#[suggestion(...)]` contains more than one
    suggestion: (Applicability, Applicability, Span),
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct WrongKindOfAnnotation {
    #[label("wrong kind of annotation for label")]
    //~^ ERROR invalid annotation list `#[label(...)]`
    z: Span,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
#[message = "Something something else"]
struct OptionsInErrors {
    #[label = "Label message"]
    label: Option<Span>,
    #[suggestion(message = "suggestion message")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(SessionDiagnostic)]
#[error = "E0456"]
struct MoveOutOfBorrowError<'tcx> {
    name: Ident,
    ty: Ty<'tcx>,
    #[message = "cannot move {ty} out of borrow"]
    #[label = "cannot move out of borrow"]
    span: Span,
    #[label = "`{ty}` first borrowed here"]
    other_span: Span,
    #[suggestion(message = "consider cloning here", code = "{name}.clone()")]
    opt_sugg: Option<(Span, Applicability)>,
}

#[derive(SessionDiagnostic)]
#[error = "E0123"]
struct ErrorWithLifetime<'a> {
    #[message = "Some message that references {name}"]
    span: Span,
    name: &'a str,
}
