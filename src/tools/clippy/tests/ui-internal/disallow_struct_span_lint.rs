#![feature(rustc_private)]

extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;

use rustc_errors::{DiagnosticMessage, MultiSpan};
use rustc_hir::hir_id::HirId;
use rustc_lint::{Lint, LintContext};
use rustc_middle::ty::TyCtxt;

pub fn a(cx: impl LintContext, lint: &'static Lint, span: impl Into<MultiSpan>, msg: impl Into<DiagnosticMessage>) {
    cx.struct_span_lint(lint, span, msg, |b| b);
}

pub fn b(
    tcx: TyCtxt<'_>,
    lint: &'static Lint,
    hir_id: HirId,
    span: impl Into<MultiSpan>,
    msg: impl Into<DiagnosticMessage>,
) {
    tcx.struct_span_lint_hir(lint, hir_id, span, msg, |b| b);
}

fn main() {}
