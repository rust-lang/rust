#![deny(clippy::missing_msrv_attr_impl)]
#![allow(clippy::missing_clippy_version_attribute)]
#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
use clippy_utils::extract_msrv_attr;
use clippy_utils::msrvs::MsrvStack;
use rustc_hir::Expr;
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass};

declare_lint! {
    pub TEST_LINT,
    Warn,
    ""
}

struct Pass {
    msrv: MsrvStack,
}

impl_lint_pass!(Pass => [TEST_LINT]);

impl EarlyLintPass for Pass {
    //~^ missing_msrv_attr_impl
    fn check_expr(&mut self, _: &EarlyContext<'_>, _: &rustc_ast::Expr) {}
}

fn main() {}
