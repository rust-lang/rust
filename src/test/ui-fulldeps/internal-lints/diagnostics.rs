// compile-flags: -Z unstable-options

#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(rustc_private)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

extern crate rustc_errors;
extern crate rustc_macros;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::{AddSubdiagnostic, Diagnostic, DiagnosticBuilder, ErrorGuaranteed, fluent};
use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_session::{parse::ParseSess, SessionDiagnostic};
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(parser::expect_path)]
struct DeriveSessionDiagnostic {
    #[primary_span]
    span: Span,
}

#[derive(SessionSubdiagnostic)]
#[note(parser::add_paren)]
struct Note {
    #[primary_span]
    span: Span,
}

pub struct UntranslatableInSessionDiagnostic;

impl<'a> SessionDiagnostic<'a, ErrorGuaranteed> for UntranslatableInSessionDiagnostic {
    fn into_diagnostic(self, sess: &'a ParseSess) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        sess.struct_err("untranslatable diagnostic")
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInSessionDiagnostic;

impl<'a> SessionDiagnostic<'a, ErrorGuaranteed> for TranslatableInSessionDiagnostic {
    fn into_diagnostic(self, sess: &'a ParseSess) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        sess.struct_err(fluent::parser::expect_path)
    }
}

pub struct UntranslatableInAddSubdiagnostic;

impl AddSubdiagnostic for UntranslatableInAddSubdiagnostic {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        diag.note("untranslatable diagnostic");
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInAddSubdiagnostic;

impl AddSubdiagnostic for TranslatableInAddSubdiagnostic {
    fn add_to_diagnostic(self, diag: &mut Diagnostic) {
        diag.note(fluent::typeck::note);
    }
}

pub fn make_diagnostics<'a>(sess: &'a ParseSess) {
    let _diag = sess.struct_err(fluent::parser::expect_path);
    //~^ ERROR diagnostics should only be created in `SessionDiagnostic`/`AddSubdiagnostic` impls

    let _diag = sess.struct_err("untranslatable diagnostic");
    //~^ ERROR diagnostics should only be created in `SessionDiagnostic`/`AddSubdiagnostic` impls
    //~^^ ERROR diagnostics should be created using translatable messages
}

// Check that `rustc_lint_diagnostics`-annotated functions aren't themselves linted.

#[rustc_lint_diagnostics]
pub fn skipped_because_of_annotation<'a>(sess: &'a ParseSess) {
    let _diag = sess.struct_err("untranslatable diagnostic"); // okay!
}
