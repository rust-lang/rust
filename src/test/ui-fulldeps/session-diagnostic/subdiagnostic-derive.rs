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
#[label(slug = "label-a")]
struct A {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
enum B {
    #[label(slug = "label-b-a")]
    A {
        #[primary_span]
        span: Span,
        var: String,
    },
    #[label(slug = "label-b-b")]
    B {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "label-c")]
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
//~^ ERROR `#[label(slug)]` is not a valid attribute
struct L {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label()]
//~^ ERROR `slug` must be set in a `#[label(...)]` attribute
struct M {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[label(code = "...")]
//~^ ERROR `code` is not a valid nested attribute of a `label` attribute
struct N {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[foo]
//~^ ERROR cannot find attribute `foo` in this scope
//~^^ ERROR unsupported type attribute for subdiagnostic enum
enum O {
    #[label(slug = "...")]
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum P {
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
enum Q {
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
enum R {
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
enum S {
    #[bar("...")]
//~^ ERROR `#[bar("...")]` is not a valid attribute
//~^^ ERROR cannot find attribute `bar` in this scope
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum T {
    #[label(code = "...")]
//~^ ERROR `code` is not a valid nested attribute of a `label`
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum U {
    #[label(slug = "label-u")]
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
#[label(slug = "...")]
//~^ ERROR label without `#[primary_span]` field
struct V {
    #[primary_span]
    //~^ ERROR the `#[primary_span]` attribute can only be applied to fields of type `Span`
    span: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "...")]
struct W {
    #[primary_span]
    span: Span,
    #[applicability]
    //~^ ERROR `#[applicability]` is only valid on suggestions
    applicability: Applicability,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "...")]
struct X {
    #[primary_span]
    span: Span,
    #[bar]
    //~^ ERROR `#[bar]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "...")]
struct Y {
    #[primary_span]
    span: Span,
    #[bar = "..."]
    //~^ ERROR `#[bar = ...]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "...")]
struct Z {
    #[primary_span]
    span: Span,
    #[bar("...")]
    //~^ ERROR `#[bar(...)]` is not a valid attribute
    //~^^ ERROR cannot find attribute `bar` in this scope
    bar: String,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "label-aa")]
struct AA {
    #[primary_span]
    span: Span,
    #[skip_arg]
    z: Z
}

#[derive(SessionSubdiagnostic)]
union AB {
//~^ ERROR unexpected unsupported untagged union
    span: u32,
    b: u64
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "label-ac-1")]
//~^ NOTE previously specified here
//~^^ NOTE previously specified here
#[label(slug = "label-ac-2")]
//~^ ERROR specified multiple times
//~^^ ERROR specified multiple times
struct AC {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "label-ad-1", slug = "label-ad-2")]
//~^ ERROR specified multiple times
//~^^ NOTE previously specified here
struct AD {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[label(slug = "label-ad-1")]
struct AE {
    #[primary_span]
//~^ NOTE previously specified here
    span_a: Span,
    #[primary_span]
//~^ ERROR specified multiple times
    span_b: Span,
}

#[derive(SessionSubdiagnostic)]
struct AF {
//~^ ERROR subdiagnostic kind not specified
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "suggestion-af", code = "...")]
struct AG {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
    var: String,
}

#[derive(SessionSubdiagnostic)]
enum AH {
    #[suggestion(slug = "suggestion-ag-a", code = "...")]
    A {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        var: String,
    },
    #[suggestion(slug = "suggestion-ag-b", code = "...")]
    B {
        #[primary_span]
        span: Span,
        #[applicability]
        applicability: Applicability,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code = "...", code = "...")]
//~^ ERROR specified multiple times
//~^^ NOTE previously specified here
struct AI {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code = "...")]
struct AJ {
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
#[suggestion(slug = "...", code = "...")]
//~^ ERROR suggestion without `applicability`
struct AK {
    #[primary_span]
    span: Span,
    #[applicability]
//~^ ERROR the `#[applicability]` attribute can only be applied to fields of type `Applicability`
    applicability: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code = "...")]
//~^ ERROR suggestion without `applicability`
struct AL {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...")]
//~^ ERROR suggestion without `code = "..."`
struct AM {
    #[primary_span]
    span: Span,
    #[applicability]
    applicability: Applicability,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code ="...", applicability = "foo")]
//~^ ERROR invalid applicability
struct AN {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[help(slug = "label-am")]
struct AO {
    var: String
}

#[derive(SessionSubdiagnostic)]
#[note(slug = "label-an")]
struct AP;

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code = "...")]
//~^ ERROR suggestion without `applicability`
//~^^ ERROR suggestion without `#[primary_span]` field
struct AQ {
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code ="...", applicability = "machine-applicable")]
struct AR {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[label]
//~^ ERROR unsupported type attribute for subdiagnostic enum
enum AS {
    #[label(slug = "...")]
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code ="{var}", applicability = "machine-applicable")]
struct AT {
    #[primary_span]
    span: Span,
    var: String,
}

#[derive(SessionSubdiagnostic)]
#[suggestion(slug = "...", code ="{var}", applicability = "machine-applicable")]
//~^ ERROR `var` doesn't refer to a field on this type
struct AU {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
enum AV {
    #[suggestion(slug = "...", code ="{var}", applicability = "machine-applicable")]
    A {
        #[primary_span]
        span: Span,
        var: String,
    }
}

#[derive(SessionSubdiagnostic)]
enum AW {
    #[suggestion(slug = "...", code ="{var}", applicability = "machine-applicable")]
//~^ ERROR `var` doesn't refer to a field on this type
    A {
        #[primary_span]
        span: Span,
    }
}
