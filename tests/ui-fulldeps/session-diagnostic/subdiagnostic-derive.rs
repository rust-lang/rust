// check-fail
// Tests error conditions for specifying subdiagnostics using #[derive(Subdiagnostic)]

// The proc_macro2 crate handles spans differently when on beta/stable release rather than nightly,
// changing the output of this test. Since Subdiagnostic is strictly internal to the compiler
// the test is just ignored on stable and beta:
// ignore-stage1
// ignore-beta
// ignore-stable

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_errors;
extern crate rustc_fluent_macro;
extern crate rustc_macros;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::{Applicability, DiagnosticMessage, SubdiagnosticMessage};
use rustc_fluent_macro::fluent_messages;
use rustc_macros::Subdiagnostic;
use rustc_span::Span;

fluent_messages! { "./example.ftl" }

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct A {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
enum B {
    #[label(no_crate_example)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
    #[label(no_crate_example)]
    B {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
//~^ ERROR label without `#[primary_span]` field
struct C {
    var: String,
}

#[derive(Subdiagnostic)]
#[label]
//~^ ERROR diagnostic slug must be first argument
struct D {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[foo]
//~^ ERROR `#[foo]` is not a valid attribute
//~^^ ERROR cannot find attribute `foo` in this scope
struct E {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label = "..."]
//~^ ERROR `#[label = ...]` is not a valid attribute
struct F {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label(bug = "...")]
//~^ ERROR only `no_span` is a valid nested attribute
//~| ERROR diagnostic slug must be first argument
struct G {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label("...")]
//~^ ERROR unexpected literal in nested attribute, expected ident
struct H {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label(slug = 4)]
//~^ ERROR only `no_span` is a valid nested attribute
//~| ERROR diagnostic slug must be first argument
struct J {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label(slug("..."))]
//~^ ERROR only `no_span` is a valid nested attribute
//~| ERROR diagnostic slug must be first argument
struct K {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label(slug)]
//~^ ERROR cannot find value `slug` in module `crate::fluent_generated`
//~^^ NOTE not found in `crate::fluent_generated`
struct L {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label()]
//~^ ERROR unexpected end of input, unexpected token in nested attribute, expected ident
struct M {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example, code = "...")]
//~^ ERROR only `no_span` is a valid nested attribute
struct N {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example, applicability = "machine-applicable")]
//~^ ERROR only `no_span` is a valid nested attribute
struct O {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[foo]
//~^ ERROR cannot find attribute `foo` in this scope
//~^^ ERROR unsupported type attribute for subdiagnostic enum
enum P {
    #[label(no_crate_example)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum Q {
    #[bar]
    //~^ ERROR `#[bar]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum R {
    #[bar = "..."]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum S {
    #[bar = 4]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum T {
    #[bar("...")]
    //~^ ERROR `#[bar(...)]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum U {
    #[label(code = "...")]
    //~^ ERROR diagnostic slug must be first argument of a `#[label(...)]` attribute
    //~| ERROR only `no_span` is a valid nested attribute
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum V {
    #[label(no_crate_example)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
    B {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
//~^ ERROR label without `#[primary_span]` field
struct W {
    #[primary_span]
    //~^ ERROR the `#[primary_span]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    span: String,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct X {
    #[primary_span]
    span: Span,
    #[applicability]
    //~^ ERROR `#[applicability]` is only valid on suggestions
    applicability: Applicability,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct Y {
    #[primary_span]
    span: Span,
    #[bar]
    //~^ ERROR `#[bar]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct Z {
    #[primary_span]
    span: Span,
    #[bar = "..."]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct AA {
    #[primary_span]
    span: Span,
    #[bar("...")]
    //~^ ERROR `#[bar(...)]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct AB {
    #[primary_span]
    span: Span,
    #[skip_arg]
    z: Z,
}

#[derive(Subdiagnostic)]
union AC {
    //~^ ERROR unexpected unsupported untagged union
    span: u32,
    b: u64,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
#[label(no_crate_example)]
struct AD {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example, no_crate::example)]
//~^ ERROR a diagnostic slug must be the first argument to the attribute
struct AE {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct AF {
    #[primary_span]
    //~^ NOTE previously specified here
    span_a: Span,
    #[primary_span]
    //~^ ERROR specified multiple times
    span_b: Span,
}

#[derive(Subdiagnostic)]
struct AG {
    //~^ ERROR subdiagnostic kind not specified
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...")]
struct AH {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
    var: String,
}

#[derive(Subdiagnostic)]
enum AI {
    #[suggestion(no_crate_example, code = "...")]
    A {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        var: String,
    },
    #[suggestion(no_crate_example, code = "...")]
    B {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        var: String,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...", code = "...")]
//~^ ERROR specified multiple times
//~^^ NOTE previously specified here
struct AJ {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...")]
struct AK {
    #[primary_span]
    span: Span,
    #[applicability]
    //~^ NOTE previously specified here
    applicability_a: Applicability,
    #[applicability]
    //~^ ERROR specified multiple times
    applicability_b: Applicability,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...")]
struct AL {
    #[primary_span]
    span: Span,
    #[applicability]
    //~^ ERROR the `#[applicability]` attribute can only be applied to fields of type `Applicability`
    applicability: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...")]
struct AM {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example)]
//~^ ERROR suggestion without `code = "..."`
struct AN {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...", applicability = "foo")]
//~^ ERROR invalid applicability
struct AO {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[help(no_crate_example)]
struct AP {
    var: String,
}

#[derive(Subdiagnostic)]
#[note(no_crate_example)]
struct AQ;

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...")]
//~^ ERROR suggestion without `#[primary_span]` field
struct AR {
    var: String,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...", applicability = "machine-applicable")]
struct AS {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[label]
//~^ ERROR unsupported type attribute for subdiagnostic enum
enum AT {
    #[label(no_crate_example)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "{var}", applicability = "machine-applicable")]
struct AU {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "{var}", applicability = "machine-applicable")]
//~^ ERROR `var` doesn't refer to a field on this type
struct AV {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
enum AW {
    #[suggestion(no_crate_example, code = "{var}", applicability = "machine-applicable")]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
}

#[derive(Subdiagnostic)]
enum AX {
    #[suggestion(no_crate_example, code = "{var}", applicability = "machine-applicable")]
    //~^ ERROR `var` doesn't refer to a field on this type
    A {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
#[warning(no_crate_example)]
struct AY {}

#[derive(Subdiagnostic)]
#[warning(no_crate_example)]
struct AZ {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "...")]
//~^ ERROR suggestion without `#[primary_span]` field
struct BA {
    #[suggestion_part]
    //~^ ERROR `#[suggestion_part]` is not a valid attribute
    span: Span,
    #[suggestion_part(code = "...")]
    //~^ ERROR `#[suggestion_part(...)]` is not a valid attribute
    span2: Span,
    #[applicability]
    applicability: Applicability,
    var: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, code = "...", applicability = "machine-applicable")]
//~^ ERROR multipart suggestion without any `#[suggestion_part(...)]` fields
//~| ERROR invalid nested attribute
struct BBa {
    var: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, applicability = "machine-applicable")]
struct BBb {
    #[suggestion_part]
    //~^ ERROR `#[suggestion_part(...)]` attribute without `code = "..."`
    span1: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, applicability = "machine-applicable")]
struct BBc {
    #[suggestion_part()]
    //~^ ERROR unexpected end of input, unexpected token in nested attribute, expected ident
    span1: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
//~^ ERROR multipart suggestion without any `#[suggestion_part(...)]` fields
struct BC {
    #[primary_span]
    //~^ ERROR `#[primary_span]` is not a valid attribute
    span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
struct BD {
    #[suggestion_part]
    //~^ ERROR `#[suggestion_part(...)]` attribute without `code = "..."`
    span1: Span,
    #[suggestion_part()]
    //~^ ERROR unexpected end of input, unexpected token in nested attribute, expected ident
    span2: Span,
    #[suggestion_part(foo = "bar")]
    //~^ ERROR `code` is the only valid nested attribute
    //~| ERROR expected `,`
    span4: Span,
    #[suggestion_part(code = "...")]
    //~^ ERROR the `#[suggestion_part(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    s1: String,
    #[suggestion_part()]
    //~^ ERROR the `#[suggestion_part(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    s2: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, applicability = "machine-applicable")]
struct BE {
    #[suggestion_part(code = "...", code = ",,,")]
    //~^ ERROR specified multiple times
    //~| NOTE previously specified here
    span: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, applicability = "machine-applicable")]
struct BF {
    #[suggestion_part(code = "(")]
    first: Span,
    #[suggestion_part(code = ")")]
    second: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
struct BG {
    #[applicability]
    appl: Applicability,
    #[suggestion_part(code = "(")]
    first: Span,
    #[suggestion_part(code = ")")]
    second: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, applicability = "machine-applicable")]
struct BH {
    #[applicability]
    //~^ ERROR `#[applicability]` has no effect
    appl: Applicability,
    #[suggestion_part(code = "(")]
    first: Span,
    #[suggestion_part(code = ")")]
    second: Span,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example, applicability = "machine-applicable")]
struct BI {
    #[suggestion_part(code = "")]
    spans: Vec<Span>,
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct BJ {
    #[primary_span]
    span: Span,
    r#type: String,
}

/// with a doc comment on the type..
#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct BK {
    /// ..and the field
    #[primary_span]
    span: Span,
}

/// with a doc comment on the type..
#[derive(Subdiagnostic)]
enum BL {
    /// ..and the variant..
    #[label(no_crate_example)]
    Foo {
        /// ..and the field
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
struct BM {
    #[suggestion_part(code("foo"))]
    //~^ ERROR expected exactly one string literal for `code = ...`
    //~| ERROR unexpected token
    span: Span,
    r#type: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
struct BN {
    #[suggestion_part(code("foo", "bar"))]
    //~^ ERROR expected exactly one string literal for `code = ...`
    //~| ERROR unexpected token
    span: Span,
    r#type: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
struct BO {
    #[suggestion_part(code(3))]
    //~^ ERROR expected exactly one string literal for `code = ...`
    //~| ERROR unexpected token
    span: Span,
    r#type: String,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(no_crate_example)]
struct BP {
    #[suggestion_part(code())]
    //~^ ERROR expected exactly one string literal for `code = ...`
    span: Span,
    r#type: String,
}

#[derive(Subdiagnostic)]
//~^ ERROR cannot find value `__code_29` in this scope
//~| NOTE in this expansion
//~| NOTE not found in this scope
#[multipart_suggestion(no_crate_example)]
struct BQ {
    #[suggestion_part(code = 3)]
    //~^ ERROR expected string literal
    span: Span,
    r#type: String,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "")]
struct SuggestionStyleDefault {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = "short")]
struct SuggestionStyleShort {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = "hidden")]
struct SuggestionStyleHidden {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = "verbose")]
struct SuggestionStyleVerbose {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = "tool-only")]
struct SuggestionStyleToolOnly {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = "hidden", style = "normal")]
//~^ ERROR specified multiple times
//~| NOTE previously specified here
struct SuggestionStyleTwice {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion_hidden(no_crate_example, code = "")]
//~^ ERROR #[suggestion_hidden(...)]` is not a valid attribute
struct SuggestionStyleOldSyntax {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion_hidden(no_crate_example, code = "", style = "normal")]
//~^ ERROR #[suggestion_hidden(...)]` is not a valid attribute
struct SuggestionStyleOldAndNewSyntax {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = "foo")]
//~^ ERROR invalid suggestion style
struct SuggestionStyleInvalid1 {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style = 42)]
//~^ ERROR expected `= "xxx"`
struct SuggestionStyleInvalid2 {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style)]
//~^ ERROR a diagnostic slug must be the first argument to the attribute
struct SuggestionStyleInvalid3 {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "", style("foo"))]
//~^ ERROR expected `= "xxx"`
//~| ERROr expected `,`
struct SuggestionStyleInvalid4 {
    #[primary_span]
    sub: Span,
}

#[derive(Subdiagnostic)]
#[suggestion(no_crate_example, code = "")]
//~^ ERROR suggestion without `#[primary_span]` field
struct PrimarySpanOnVec {
    #[primary_span]
    //~^ ERROR `#[primary_span]` is not a valid attribute
    //~| NOTE there must be exactly one primary span
    sub: Vec<Span>,
}
