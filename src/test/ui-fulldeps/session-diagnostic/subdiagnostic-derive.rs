// check-fail
// Tests error conditions for specifying subdiagnostics using #[derive(SessionSubdiagnostic)]

// The proc_macro2 crate handles spans differently when on beta/stable release rather than nightly,
// changing the output of this test. Since SessionSubdiagnostic is strictly internal to the compiler
// the test is just ignored on stable and beta:
// ignore-beta
// ignore-stable

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_errors;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_macros;

use rustc_errors::Applicability;
use rustc_span::Span;
use rustc_macros::SessionSubdiagnostic;

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct A {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
enum B {
    #[label(parser::add_paren)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
    #[label(parser::add_paren)]
    B {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
//~^ ERROR label without `#[primary_span]` field
struct C {
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label]
//~^ ERROR `#[label]` is not a valid attribute
struct D {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[foo]
//~^ ERROR `#[foo]` is not a valid attribute
//~^^ ERROR cannot find attribute `foo` in this scope
struct E {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label = "..."]
//~^ ERROR `#[label = ...]` is not a valid attribute
struct F {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(bug = "...")]
//~^ ERROR `#[label(bug = ...)]` is not a valid attribute
struct G {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label("...")]
//~^ ERROR `#[label("...")]` is not a valid attribute
struct H {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = 4)]
//~^ ERROR `#[label(slug = ...)]` is not a valid attribute
struct J {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug("..."))]
//~^ ERROR `#[label(slug(...))]` is not a valid attribute
struct K {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug)]
//~^ ERROR cannot find value `slug` in module `rustc_errors::fluent`
//~^^ NOTE not found in `rustc_errors::fluent`
struct L {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label()]
//~^ ERROR diagnostic slug must be first argument of a `#[label(...)]` attribute
struct M {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren, code = "...")]
//~^ ERROR `code` is not a valid nested attribute of a `label` attribute
struct N {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren, applicability = "machine-applicable")]
//~^ ERROR `applicability` is not a valid nested attribute of a `label` attribute
struct O {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[foo]
//~^ ERROR cannot find attribute `foo` in this scope
//~^^ ERROR unsupported type attribute for subdiagnostic enum
enum P {
    #[label(parser::add_paren)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum Q {
    #[bar]
    //~^ ERROR `#[bar]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum R {
    #[bar = "..."]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum S {
    #[bar = 4]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum T {
    #[bar("...")]
    //~^ ERROR `#[bar(...)]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum U {
    #[label(code = "...")]
    //~^ ERROR diagnostic slug must be first argument of a `#[label(...)]` attribute
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum V {
    #[label(parser::add_paren)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
    B {
    //~^ ERROR subdiagnostic kind not specified
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
//~^ ERROR label without `#[primary_span]` field
struct W {
    #[primary_span]
    //~^ ERROR the `#[primary_span]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    span: String,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct X {
    #[primary_span]
    span: Span,
    #[applicability]
    //~^ ERROR `#[applicability]` is only valid on suggestions
    applicability: Applicability,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct Y {
    #[primary_span]
    span: Span,
    #[bar]
    //~^ ERROR `#[bar]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct Z {
    #[primary_span]
    span: Span,
    #[bar = "..."]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct AA {
    #[primary_span]
    span: Span,
    #[bar("...")]
    //~^ ERROR `#[bar(...)]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct AB {
    #[primary_span]
    span: Span,
    #[skip_arg]
    z: Z
}

#[derive(SessionSubdiagnostic)]
union AC {
//~^ ERROR unexpected unsupported untagged union
    span: u32,
    b: u64
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
//~^ NOTE previously specified here
#[label(parser::add_paren)]
//~^ ERROR specified multiple times
struct AD {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren, parser::add_paren)]
//~^ ERROR `#[label(parser::add_paren)]` is not a valid attribute
struct AE {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[label(parser::add_paren)]
struct AF {
    #[primary_span]
    //~^ NOTE previously specified here
    span_a: Span,
    #[primary_span]
    //~^ ERROR specified multiple times
    span_b: Span,
}

#[derive(SessionSubdiagnostic)]
struct AG {
    //~^ ERROR subdiagnostic kind not specified
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...")]
struct AH {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
    var: String,
}

#[derive(SessionSubdiagnostic)]
enum AI {
    #[suggestion(parser::add_paren, code = "...")]
    A {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        var: String,
    },
    #[suggestion(parser::add_paren, code = "...")]
    B {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...", code = "...")]
//~^ ERROR specified multiple times
//~^^ NOTE previously specified here
struct AJ {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...")]
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

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...")]
struct AL {
    #[primary_span]
    span: Span,
    #[applicability]
    //~^ ERROR the `#[applicability]` attribute can only be applied to fields of type `Applicability`
    applicability: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...")]
struct AM {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren)]
//~^ ERROR suggestion without `code = "..."`
struct AN {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code ="...", applicability = "foo")]
//~^ ERROR invalid applicability
struct AO {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[help(parser::add_paren)]
struct AP {
    var: String
}

#[derive(SessionSubdiagnostic)]
#[note(parser::add_paren)]
struct AQ;

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...")]
//~^ ERROR suggestion without `#[primary_span]` field
struct AR {
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code ="...", applicability = "machine-applicable")]
struct AS {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[label]
//~^ ERROR unsupported type attribute for subdiagnostic enum
enum AT {
    #[label(parser::add_paren)]
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code ="{var}", applicability = "machine-applicable")]
struct AU {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code ="{var}", applicability = "machine-applicable")]
//~^ ERROR `var` doesn't refer to a field on this type
struct AV {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
enum AW {
    #[suggestion(parser::add_paren, code ="{var}", applicability = "machine-applicable")]
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum AX {
    #[suggestion(parser::add_paren, code ="{var}", applicability = "machine-applicable")]
//~^ ERROR `var` doesn't refer to a field on this type
    A {
        #[primary_span]
        span: Span,
    }
}

#[derive(SessionSubdiagnostic)]
#[warning(parser::add_paren)]
struct AY {}

#[derive(SessionSubdiagnostic)]
#[warning(parser::add_paren)]
struct AZ {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(parser::add_paren, code = "...")]
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

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, code = "...", applicability = "machine-applicable")]
//~^ ERROR multipart suggestion without any `#[suggestion_part(...)]` fields
//~| ERROR `code` is not a valid nested attribute of a `multipart_suggestion` attribute
struct BBa {
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, applicability = "machine-applicable")]
struct BBb {
    #[suggestion_part]
    //~^ ERROR `#[suggestion_part(...)]` attribute without `code = "..."`
    span1: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, applicability = "machine-applicable")]
struct BBc {
    #[suggestion_part()]
    //~^ ERROR `#[suggestion_part(...)]` attribute without `code = "..."`
    span1: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren)]
//~^ ERROR multipart suggestion without any `#[suggestion_part(...)]` fields
struct BC {
    #[primary_span]
    //~^ ERROR `#[primary_span]` is not a valid attribute
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren)]
struct BD {
    #[suggestion_part]
    //~^ ERROR `#[suggestion_part(...)]` attribute without `code = "..."`
    span1: Span,
    #[suggestion_part()]
    //~^ ERROR `#[suggestion_part(...)]` attribute without `code = "..."`
    span2: Span,
    #[suggestion_part(foo = "bar")]
    //~^ ERROR `#[suggestion_part(foo = ...)]` is not a valid attribute
    span4: Span,
    #[suggestion_part(code = "...")]
    //~^ ERROR the `#[suggestion_part(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    s1: String,
    #[suggestion_part()]
    //~^ ERROR the `#[suggestion_part(...)]` attribute can only be applied to fields of type `Span` or `MultiSpan`
    s2: String,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, applicability = "machine-applicable")]
struct BE {
    #[suggestion_part(code = "...", code = ",,,")]
    //~^ ERROR specified multiple times
    //~| NOTE previously specified here
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, applicability = "machine-applicable")]
struct BF {
    #[suggestion_part(code = "(")]
    first: Span,
    #[suggestion_part(code = ")")]
    second: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren)]
struct BG {
    #[applicability]
    appl: Applicability,
    #[suggestion_part(code = "(")]
    first: Span,
    #[suggestion_part(code = ")")]
    second: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, applicability = "machine-applicable")]
//~^ NOTE previously specified here
struct BH {
    #[applicability]
    //~^ ERROR specified multiple times
    appl: Applicability,
    #[suggestion_part(code = "(")]
    first: Span,
    #[suggestion_part(code = ")")]
    second: Span,
}

#[derive(SessionSubdiagnostic)]
#[multipart_suggestion(parser::add_paren, applicability = "machine-applicable")]
struct BI {
    #[suggestion_part(code = "")]
    spans: Vec<Span>,
}
