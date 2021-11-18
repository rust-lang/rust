#![deny(clippy::internal)]
#![allow(clippy::missing_clippy_version_attribute)]
#![feature(rustc_private)]

extern crate clippy_utils;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;

#[macro_use]
extern crate rustc_session;
use clippy_utils::{paths, ty::match_type};
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;

declare_lint! {
    pub TEST_LINT,
    Warn,
    ""
}

declare_lint_pass!(Pass => [TEST_LINT]);

static OPTION: [&str; 3] = ["core", "option", "Option"];

impl<'tcx> LateLintPass<'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr) {
        let ty = cx.typeck_results().expr_ty(expr);

        let _ = match_type(cx, ty, &OPTION);
        let _ = match_type(cx, ty, &["core", "result", "Result"]);

        let rc_path = &["alloc", "rc", "Rc"];
        let _ = clippy_utils::ty::match_type(cx, ty, rc_path);
    }
}

fn main() {}
