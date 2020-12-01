use crate::utils::{
    is_adjusted, is_type_diagnostic_item, match_path, match_trait_method, match_var, paths, remove_blocks,
    span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Body, Expr, ExprKind, Pat, PatKind, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// **What it does:** Checks for instances of `map(f)` where `f` is the identity function.
    ///
    /// **Why is this bad?** It can be written more concisely without the call to `map`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// let x = [1, 2, 3];
    /// let y: Vec<_> = x.iter().map(|x| x).map(|x| 2*x).collect();
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = [1, 2, 3];
    /// let y: Vec<_> = x.iter().map(|x| 2*x).collect();
    /// ```
    pub MAP_IDENTITY,
    complexity,
    "using iterator.map(|x| x)"
}

declare_lint_pass!(MapIdentity => [MAP_IDENTITY]);

impl<'tcx> LateLintPass<'tcx> for MapIdentity {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        if_chain! {
            if let Some([caller, func]) = get_map_argument(cx, expr);
            if is_expr_identity_function(cx, func);
            then {
                span_lint_and_sugg(
                    cx,
                    MAP_IDENTITY,
                    expr.span.trim_start(caller.span).unwrap(),
                    "unnecessary map of the identity function",
                    "remove the call to `map`",
                    String::new(),
                    Applicability::MachineApplicable
                )
            }
        }
    }
}

/// Returns the arguments passed into map() if the expression is a method call to
/// map(). Otherwise, returns None.
fn get_map_argument<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<&'a [Expr<'a>]> {
    if_chain! {
        if let ExprKind::MethodCall(ref method, _, ref args, _) = expr.kind;
        if args.len() == 2 && method.ident.as_str() == "map";
        let caller_ty = cx.typeck_results().expr_ty(&args[0]);
        if match_trait_method(cx, expr, &paths::ITERATOR)
            || is_type_diagnostic_item(cx, caller_ty, sym::result_type)
            || is_type_diagnostic_item(cx, caller_ty, sym::option_type);
        then {
            Some(args)
        } else {
            None
        }
    }
}

/// Checks if an expression represents the identity function
/// Only examines closures and `std::convert::identity`
fn is_expr_identity_function(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Closure(_, _, body_id, _, _) => is_body_identity_function(cx, cx.tcx.hir().body(body_id)),
        ExprKind::Path(QPath::Resolved(_, ref path)) => match_path(path, &paths::STD_CONVERT_IDENTITY),
        _ => false,
    }
}

/// Checks if a function's body represents the identity function
/// Looks for bodies of the form `|x| x`, `|x| return x`, `|x| { return x }` or `|x| {
/// return x; }`
fn is_body_identity_function(cx: &LateContext<'_>, func: &Body<'_>) -> bool {
    let params = func.params;
    let body = remove_blocks(&func.value);

    // if there's less/more than one parameter, then it is not the identity function
    if params.len() != 1 {
        return false;
    }

    match body.kind {
        ExprKind::Path(QPath::Resolved(None, _)) => match_expr_param(cx, body, params[0].pat),
        ExprKind::Ret(Some(ref ret_val)) => match_expr_param(cx, ret_val, params[0].pat),
        ExprKind::Block(ref block, _) => {
            if_chain! {
                if block.stmts.len() == 1;
                if let StmtKind::Semi(ref expr) | StmtKind::Expr(ref expr) = block.stmts[0].kind;
                if let ExprKind::Ret(Some(ref ret_val)) = expr.kind;
                then {
                    match_expr_param(cx, ret_val, params[0].pat)
                } else {
                    false
                }
            }
        },
        _ => false,
    }
}

/// Returns true iff an expression returns the same thing as a parameter's pattern
fn match_expr_param(cx: &LateContext<'_>, expr: &Expr<'_>, pat: &Pat<'_>) -> bool {
    if let PatKind::Binding(_, _, ident, _) = pat.kind {
        match_var(expr, ident.name) && !(cx.typeck_results().hir_owner == expr.hir_id.owner && is_adjusted(cx, expr))
    } else {
        false
    }
}
