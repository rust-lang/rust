use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_expr_identity_function, is_trait_method};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
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
        if let ExprKind::MethodCall(method, _, args, _) = expr.kind;
        if args.len() == 2 && method.ident.name == sym::map;
        let caller_ty = cx.typeck_results().expr_ty(&args[0]);
        if is_trait_method(cx, expr, sym::Iterator)
            || is_type_diagnostic_item(cx, caller_ty, sym::result_type)
            || is_type_diagnostic_item(cx, caller_ty, sym::option_type);
        then {
            Some(args)
        } else {
            None
        }
    }
}
