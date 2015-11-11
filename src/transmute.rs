use rustc::lint::*;
use rustc_front::hir::*;
use utils;

declare_lint! {
    pub USELESS_TRANSMUTE,
    Warn,
    "transmutes that have the same to and from types"
}

pub struct UselessTransmute;

impl LintPass for UselessTransmute {
    fn get_lints(&self) -> LintArray {
        lint_array!(USELESS_TRANSMUTE)
    }
}

impl LateLintPass for UselessTransmute {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprCall(ref path_expr, ref args) = e.node {
            if let ExprPath(None, _) = path_expr.node {
                let def_id = cx.tcx.def_map.borrow()[&path_expr.id].def_id();

                if utils::match_def_path(cx, def_id, &["core", "intrinsics", "transmute"]) {
                    let from_ty = cx.tcx.expr_ty(&args[0]);
                    let to_ty = cx.tcx.expr_ty(e);

                    if from_ty == to_ty {
                        cx.span_lint(USELESS_TRANSMUTE, e.span,
                                     &format!("transmute from a type (`{}`) to itself", from_ty));
                    }
                }
            }
        }
    }
}
