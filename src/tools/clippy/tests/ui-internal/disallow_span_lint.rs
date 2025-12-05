#![feature(rustc_private)]
#![deny(clippy::disallowed_methods)]

extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;

use rustc_errors::{DiagMessage, MultiSpan};
use rustc_hir::hir_id::HirId;
use rustc_lint::{Lint, LintContext};
use rustc_middle::ty::TyCtxt;

pub fn a(cx: impl LintContext, lint: &'static Lint, span: impl Into<MultiSpan>, msg: impl Into<DiagMessage>) {
    cx.span_lint(lint, span, |lint| {
        //~^ disallowed_methods
        lint.primary_message(msg);
    });
}

pub fn b(tcx: TyCtxt<'_>, lint: &'static Lint, hir_id: HirId, span: impl Into<MultiSpan>, msg: impl Into<DiagMessage>) {
    tcx.node_span_lint(lint, hir_id, span, |lint| {
        //~^ disallowed_methods
        lint.primary_message(msg);
    });
}

fn main() {}
