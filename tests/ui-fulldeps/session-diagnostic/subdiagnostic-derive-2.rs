//@ check-fail
// Tests error conditions for specifying subdiagnostics using #[derive(Subdiagnostic)].
// This test is split off from the main `subdiagnostic-derive`,
// because this error is generated post-expansion.

// The proc_macro2 crate handles spans differently when on beta/stable release rather than nightly,
// changing the output of this test. Since Subdiagnostic is strictly internal to the compiler
// the test is just ignored on stable and beta:
//@ ignore-stage1
//@ ignore-beta
//@ ignore-stable

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_errors;
extern crate rustc_fluent_macro;
extern crate rustc_macros;
extern crate rustc_session;
extern crate rustc_span;
extern crate core;

use rustc_errors::{Applicability, DiagMessage, SubdiagMessage};
use rustc_macros::Subdiagnostic;
use rustc_span::Span;

rustc_fluent_macro::fluent_messages! { "./example.ftl" }

#[derive(Subdiagnostic)]
#[label(slug)]
//~^ ERROR cannot find value `slug` in module `crate::fluent_generated`
//~^^ NOTE not found in `crate::fluent_generated`
struct L {
    #[primary_span]
    span: Span,
    var: String,
}
