use rustc::hir;
use rustc::lint::*;
use rustc::ty;
use utils::{in_macro, match_type, method_chain_args, snippet, span_lint_and_then};
use utils::paths;

#[derive(Clone)]
pub struct Pass;

/// **What it does:** Checks for usage of `Option.map(f)` where f is a nil
/// function
///
/// **Why is this bad?** Readability, this can be written more clearly with
/// an if statement
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let x : Option<&str> = do_stuff();
/// x.map(log_err_msg);
/// ```
/// The correct use would be:
/// ```rust
/// let x : Option<&str> = do_stuff();
/// if let Some(msg) = x {
///     log_err_msg(msg)
/// }
/// ```
declare_lint! {
    pub OPTION_MAP_NIL_FN,
    Allow,
    "using `Option.map(f)`, where f is a nil function"
}


impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(OPTION_MAP_NIL_FN)
    }
}

fn is_nil_function(cx: &LateContext, expr: &hir::Expr) -> bool {
    let ty = cx.tables.expr_ty(expr);

    if let ty::TyFnDef(_, _, bare) = ty.sty {
        if let Some(fn_type) = cx.tcx.no_late_bound_regions(&bare.sig) {
            return fn_type.output().is_nil();
        }
    }
    false
}

fn lint_map_nil_fn(cx: &LateContext, stmt: &hir::Stmt, expr: &hir::Expr, map_args: &[hir::Expr]) {
    let var_arg = &map_args[0];
    let fn_arg = &map_args[1];

    if !match_type(cx, cx.tables.expr_ty(var_arg), &paths::OPTION) {
        return;
    }

    let suggestion = if is_nil_function(cx, fn_arg) {
        format!("if let Some(...) = {0} {{ {1}(...) }}",
                snippet(cx, var_arg.span, "_"),
                snippet(cx, fn_arg.span, "_"))
    } else {
        return;
    };

    span_lint_and_then(cx,
                       OPTION_MAP_NIL_FN,
                       expr.span,
                       "called `map(f)` on an Option value where `f` is a nil function",
                       |db| { db.span_suggestion(stmt.span, "try this", suggestion); });
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_stmt(&mut self, cx: &LateContext, stmt: &hir::Stmt) {
        if in_macro(cx, stmt.span) {
            return;
        }

        if let hir::StmtSemi(ref expr, _) = stmt.node {
            if let hir::ExprMethodCall(_, _, _) = expr.node {
                if let Some(arglists) = method_chain_args(expr, &["map"]) {
                    lint_map_nil_fn(cx, stmt, expr, arglists[0]);
                }
            }
        }
    }
}
