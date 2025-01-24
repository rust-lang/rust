//@ check-fail
// Tests that a doc comment will not preclude a field from being considered a diagnostic argument
//@ normalize-stderr: "the following other types implement trait `IntoDiagArg`:(?:.*\n){0,9}\s+and \d+ others" -> "normalized in stderr"
//@ normalize-stderr: "(COMPILER_DIR/.*\.rs):[0-9]+:[0-9]+" -> "$1:LL:CC"

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

use rustc_errors::{Applicability, DiagMessage, SubdiagMessage};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;

rustc_fluent_macro::fluent_messages! { "./example.ftl" }

struct NotIntoDiagArg;

#[derive(Diagnostic)]
#[diag(no_crate_example)]
struct Test {
    #[primary_span]
    span: Span,
    /// A doc comment
    arg: NotIntoDiagArg,
    //~^ ERROR the trait bound `NotIntoDiagArg: IntoDiagArg` is not satisfied
}

#[derive(Subdiagnostic)]
#[label(no_crate_example)]
struct SubTest {
    #[primary_span]
    span: Span,
    /// A doc comment
    arg: NotIntoDiagArg,
    //~^ ERROR the trait bound `NotIntoDiagArg: IntoDiagArg` is not satisfied
}
