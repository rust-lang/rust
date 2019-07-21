#![deny(clippy::internal)]
#![feature(rustc_private)]

#[macro_use]
extern crate rustc;
use rustc::hir::Expr;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};

declare_lint! {
    pub TEST_LINT,
    Warn,
    ""
}

declare_lint_pass!(Pass => [TEST_LINT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, _cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        let _ = expr.span.ctxt().outer_expn().expn_info();
    }
}

fn main() {}
