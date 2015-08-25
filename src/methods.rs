use syntax::ast::*;
use rustc::lint::*;
use rustc::middle::ty;

use utils::{span_lint, match_type, walk_ptrs_ty};
use utils::{OPTION_PATH, RESULT_PATH, STRING_PATH};

#[derive(Copy,Clone)]
pub struct MethodsPass;

declare_lint!(pub OPTION_UNWRAP_USED, Allow,
              "using `Option.unwrap()`, which should at least get a better message using `expect()`");
declare_lint!(pub RESULT_UNWRAP_USED, Allow,
              "using `Result.unwrap()`, which might be better handled");
declare_lint!(pub STR_TO_STRING, Warn,
              "using `to_string()` on a str, which should be `to_owned()`");
declare_lint!(pub STRING_TO_STRING, Warn,
              "calling `String.to_string()` which is a no-op");

impl LintPass for MethodsPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(OPTION_UNWRAP_USED, RESULT_UNWRAP_USED, STR_TO_STRING, STRING_TO_STRING)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMethodCall(ref ident, _, ref args) = expr.node {
            let obj_ty = walk_ptrs_ty(cx.tcx.expr_ty(&args[0]));
            if ident.node.name == "unwrap" {
                if match_type(cx, obj_ty, &OPTION_PATH) {
                    span_lint(cx, OPTION_UNWRAP_USED, expr.span,
                              "used unwrap() on an Option value. If you don't want \
                               to handle the None case gracefully, consider using \
                               expect() to provide a better panic message");
                } else if match_type(cx, obj_ty, &RESULT_PATH) {
                    span_lint(cx, RESULT_UNWRAP_USED, expr.span,
                              "used unwrap() on a Result value. Graceful handling \
                               of Err values is preferred");
                }
            }
            else if ident.node.name == "to_string" {
                if obj_ty.sty == ty::TyStr {
                    span_lint(cx, STR_TO_STRING, expr.span, "`str.to_owned()` is faster");
                } else if match_type(cx, obj_ty, &STRING_PATH) {
                    span_lint(cx, STRING_TO_STRING, expr.span, "`String.to_string()` is a no-op; use \
                                                                `clone()` to make a copy");
                }
            }
        }
    }
}
