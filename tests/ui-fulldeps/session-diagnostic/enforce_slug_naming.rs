//@ rustc-env:CARGO_CRATE_NAME=rustc_dummy

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
#[diag(compiletest_example, code = E0123)]
//~^ ERROR diagnostic slug and crate name do not match
struct Hello {}
