// compile-flags: -Z unstable-options

#![crate_type = "lib"]
#![feature(rustc_attrs)]
#![feature(rustc_private)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

extern crate rustc_errors;
extern crate rustc_fluent_macro;
extern crate rustc_macros;
extern crate rustc_session;
extern crate rustc_span;

use rustc_errors::{
    AddToDiagnostic, Diagnostic, DiagnosticBuilder, DiagnosticMessage, EmissionGuarantee, DiagCtxt,
    IntoDiagnostic, Level, SubdiagnosticMessageOp,
};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::Span;

rustc_fluent_macro::fluent_messages! { "./diagnostics.ftl" }

#[derive(Diagnostic)]
#[diag(no_crate_example)]
struct DeriveDiagnostic {
    #[primary_span]
    span: Span,
}

#[derive(Subdiagnostic)]
#[note(no_crate_example)]
struct Note {
    #[primary_span]
    span: Span,
}

pub struct UntranslatableInIntoDiagnostic;

impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for UntranslatableInIntoDiagnostic {
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a, G> {
        DiagnosticBuilder::new(dcx, level, "untranslatable diagnostic")
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInIntoDiagnostic;

impl<'a, G: EmissionGuarantee> IntoDiagnostic<'a, G> for TranslatableInIntoDiagnostic {
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a, G> {
        DiagnosticBuilder::new(dcx, level, crate::fluent_generated::no_crate_example)
    }
}

pub struct UntranslatableInAddToDiagnostic;

impl AddToDiagnostic for UntranslatableInAddToDiagnostic {
    fn add_to_diagnostic_with<F: SubdiagnosticMessageOp>(self, diag: &mut Diagnostic, _: F)
    {
        diag.note("untranslatable diagnostic");
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInAddToDiagnostic;

impl AddToDiagnostic for TranslatableInAddToDiagnostic {
    fn add_to_diagnostic_with<F: SubdiagnosticMessageOp>(self, diag: &mut Diagnostic, _: F) {
        diag.note(crate::fluent_generated::no_crate_note);
    }
}

pub fn make_diagnostics<'a>(dcx: &'a DiagCtxt) {
    let _diag = dcx.struct_err(crate::fluent_generated::no_crate_example);
    //~^ ERROR diagnostics should only be created in `IntoDiagnostic`/`AddToDiagnostic` impls

    let _diag = dcx.struct_err("untranslatable diagnostic");
    //~^ ERROR diagnostics should only be created in `IntoDiagnostic`/`AddToDiagnostic` impls
    //~^^ ERROR diagnostics should be created using translatable messages
}

// Check that `rustc_lint_diagnostics`-annotated functions aren't themselves linted.

#[rustc_lint_diagnostics]
pub fn skipped_because_of_annotation<'a>(dcx: &'a DiagCtxt) {
    let _diag = dcx.struct_err("untranslatable diagnostic"); // okay!
}
