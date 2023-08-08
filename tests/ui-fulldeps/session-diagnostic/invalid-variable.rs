// run-fail
// compile-flags: --test
// test that messages referencing non-existent fields cause test failures

#![feature(rustc_private)]
#![crate_type = "lib"]

extern crate rustc_driver;
extern crate rustc_fluent_macro;
extern crate rustc_macros;
extern crate rustc_errors;
use rustc_fluent_macro::fluent_messages;
use rustc_macros::Diagnostic;
use rustc_errors::{SubdiagnosticMessage, DiagnosticMessage};
extern crate rustc_session;

fluent_messages! { "./example.ftl" }

#[derive(Diagnostic)]
#[diag(no_crate_bad_reference)]
struct BadRef;
