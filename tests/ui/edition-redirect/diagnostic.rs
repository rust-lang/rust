//@ edition: 2018
//@ aux-build: macro-source.rs
//@ check-fail

extern crate macro_source;

// A macro exported at the crate root should still be suggested when it is
// incorrectly imported through a module, even when the root binding has an
// edition redirect.
use macro_source::nested::redirected_macro;
//~^ ERROR unresolved import `macro_source::nested::redirected_macro`
//~| HELP a macro with this name exists at the root of the crate
//~| SUGGESTION macro_source::redirected_macro
//~| HELP consider importing this trait
//~| SUGGESTION use macro_source::Candidate;

// A missing import from a module that also contains redirected names should
// produce the usual unresolved-import diagnostic.
use macro_source::NoSuchImport;
//~^ ERROR unresolved import `macro_source::NoSuchImport`

// In edition 2018, `Candidate` redirects to a trait, so it should be suggested
// as an import for a missing unqualified trait. The default `Candidate` is a
// struct.
fn import_candidate<T: Candidate>() {}
//~^ ERROR cannot find trait `Candidate` in this scope

// A misspelled qualified trait name should likewise suggest the trait selected
// in edition 2018.
fn typo_candidate<T: macro_source::Canddate>() {}
//~^ ERROR cannot find trait `Canddate` in crate `macro_source`
//~| HELP a trait with a similar name exists
//~| SUGGESTION Candidate

// Doc aliases from the edition-selected target should be available in typo
// suggestions. The default `AliasCarrier` does not have this alias.
fn doc_alias(_: macro_source::OldAlias) {}
//~^ ERROR cannot find type `OldAlias` in crate `macro_source`
//~| HELP has a name defined in the doc alias attribute as `OldAlias`
//~| SUGGESTION AliasCarrier

// An enum reached through a redirected module should still produce a suggestion
// using a variant from the selected module.
fn enum_variant() -> macro_source::diagnostic_module::DiagnosticEnum {
    macro_source::diagnostic_module::DiagnosticEnum(0)
    //~^ ERROR cannot find function, tuple struct or tuple variant `DiagnosticEnum`
    //~| HELP try to construct the enum's variant
    //~| SUGGESTION macro_source::diagnostic_module::DiagnosticEnum::Variant
}

fn main() {}
