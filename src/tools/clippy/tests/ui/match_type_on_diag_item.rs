#![deny(clippy::internal)]
#![feature(rustc_private)]

extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;
#[macro_use]
extern crate rustc_session;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;

mod paths {
    pub const VEC: [&str; 3] = ["alloc", "vec", "Vec"];
}

mod utils {
    use super::*;

    pub fn match_type(_cx: &LateContext<'_>, _ty: Ty<'_>, _path: &[&str]) -> bool {
        false
    }
}

use utils::match_type;

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

        let _ = match_type(cx, ty, &paths::VEC);
        let _ = match_type(cx, ty, &OPTION);
        let _ = match_type(cx, ty, &["core", "result", "Result"]);

        let rc_path = &["alloc", "rc", "Rc"];
        let _ = utils::match_type(cx, ty, rc_path);
    }
}

fn main() {}
