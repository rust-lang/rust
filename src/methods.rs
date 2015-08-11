use syntax::ast::*;
use rustc::lint::{Context, LintPass, LintArray};
use rustc::middle::ty;

use utils::{span_lint, match_def_path, walk_ptrs_ty};

#[derive(Copy,Clone)]
pub struct MethodsPass;

declare_lint!(pub OPTION_UNWRAP_USED, Warn,
              "Warn on using unwrap() on an Option value");
declare_lint!(pub RESULT_UNWRAP_USED, Allow,
              "Warn on using unwrap() on a Result value");

impl LintPass for MethodsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(OPTION_UNWRAP_USED, RESULT_UNWRAP_USED)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMethodCall(ref ident, _, ref args) = expr.node {
            if ident.node.name == "unwrap" {
                if let ty::TyEnum(did, _) = walk_ptrs_ty(cx.tcx.expr_ty(&*args[0])).sty {
                    if match_def_path(cx, did.did, &["core", "option", "Option"]) {
                        span_lint(cx, OPTION_UNWRAP_USED, expr.span,
                                  "used unwrap() on an Option value. If you don't want \
                                   to handle the None case gracefully, consider using
                                   expect() to provide a better panic message.");
                    }
                    else if match_def_path(cx, did.did, &["core", "result", "Result"]) {
                        span_lint(cx, RESULT_UNWRAP_USED, expr.span,
                                  "used unwrap() on a Result value. Graceful handling \
                                   of Err values is preferred.");
                    }
                }
            }
        }
    }
}
