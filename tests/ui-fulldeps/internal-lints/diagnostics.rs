//@ compile-flags: -Z unstable-options
//@ ignore-stage1

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
    Diag, DiagCtxtHandle, DiagInner, DiagMessage, Diagnostic, EmissionGuarantee, Level,
    LintDiagnostic, SubdiagMessage, Subdiagnostic,
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

pub struct UntranslatableInDiagnostic;

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for UntranslatableInDiagnostic {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        Diag::new(dcx, level, "untranslatable diagnostic")
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInDiagnostic;

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for TranslatableInDiagnostic {
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        Diag::new(dcx, level, crate::fluent_generated::no_crate_example)
    }
}

pub struct UntranslatableInAddtoDiag;

impl Subdiagnostic for UntranslatableInAddtoDiag {
    fn add_to_diag<G: EmissionGuarantee>(
        self,
        diag: &mut Diag<'_, G>,
    ) {
        diag.note("untranslatable diagnostic");
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInAddtoDiag;

impl Subdiagnostic for TranslatableInAddtoDiag {
    fn add_to_diag<G: EmissionGuarantee>(
        self,
        diag: &mut Diag<'_, G>,
    ) {
        diag.note(crate::fluent_generated::no_crate_note);
    }
}

pub struct UntranslatableInLintDiagnostic;

impl<'a> LintDiagnostic<'a, ()> for UntranslatableInLintDiagnostic {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.note("untranslatable diagnostic");
        //~^ ERROR diagnostics should be created using translatable messages
    }
}

pub struct TranslatableInLintDiagnostic;

impl<'a> LintDiagnostic<'a, ()> for TranslatableInLintDiagnostic {
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, ()>) {
        diag.note(crate::fluent_generated::no_crate_note);
    }
}

pub fn make_diagnostics<'a>(dcx: DiagCtxtHandle<'a>) {
    let _diag = dcx.struct_err(crate::fluent_generated::no_crate_example);
    //~^ ERROR diagnostics should only be created in `Diagnostic`/`Subdiagnostic`/`LintDiagnostic` impls

    let _diag = dcx.struct_err("untranslatable diagnostic");
    //~^ ERROR diagnostics should only be created in `Diagnostic`/`Subdiagnostic`/`LintDiagnostic` impls
    //~^^ ERROR diagnostics should be created using translatable messages
}

// Check that `rustc_lint_diagnostics`-annotated functions aren't themselves linted for
// `diagnostic_outside_of_impl`.
#[rustc_lint_diagnostics]
pub fn skipped_because_of_annotation<'a>(dcx: DiagCtxtHandle<'a>) {
    #[allow(rustc::untranslatable_diagnostic)]
    let _diag = dcx.struct_err("untranslatable diagnostic"); // okay!
}

// Check that multiple translatable params are allowed in a single function (at one point they
// weren't).
fn f(_x: impl Into<DiagMessage>, _y: impl Into<SubdiagMessage>) {}
fn g() {
    f(crate::fluent_generated::no_crate_example, crate::fluent_generated::no_crate_example);
    f("untranslatable diagnostic", crate::fluent_generated::no_crate_example);
    //~^ ERROR diagnostics should be created using translatable messages
    f(crate::fluent_generated::no_crate_example, "untranslatable diagnostic");
    //~^ ERROR diagnostics should be created using translatable messages
    f("untranslatable diagnostic", "untranslatable diagnostic");
    //~^ ERROR diagnostics should be created using translatable messages
    //~^^ ERROR diagnostics should be created using translatable messages
}
