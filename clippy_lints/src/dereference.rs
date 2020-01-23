use rustc_hir::{Expr, ExprKind, QPath};
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, declare_lint_pass};
use crate::utils::{in_macro, span_lint_and_sugg};
use if_chain::if_chain;

declare_clippy_lint! {
    /// **What it does:** Checks for explicit `deref()` or `deref_mut()` method calls.
    ///
    /// **Why is this bad?** Derefencing by `&*x` or `&mut *x` is clearer and more concise,
    /// when not part of a method chain.
    ///
    /// **Example:**
    /// ```rust
    /// let b = a.deref();
    /// let c = a.deref_mut();
    /// ```
    /// Could be written as:
    /// ```rust
    /// let b = &*a;
    /// let c = &mut *a;
    /// ```
    /// 
    /// This lint excludes
    /// ```rust
    /// let e = d.unwrap().deref();
    /// ```
    pub EXPLICIT_DEREF_METHOD,
    pedantic,
    "Explicit use of deref or deref_mut method while not in a method chain."
}

declare_lint_pass!(Dereferencing => [
    EXPLICIT_DEREF_METHOD
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Dereferencing {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if in_macro(expr.span) {
            return;
        }

        if_chain! {
            // if this is a method call
            if let ExprKind::MethodCall(ref method_name, _, ref args) = &expr.kind;
            // on a Path (i.e. a variable/name, not another method)
            if let ExprKind::Path(QPath::Resolved(None, path)) = &args[0].kind;
            then {
                let name = method_name.ident.as_str();
                // alter help slightly to account for _mut
                match &*name {
                    "deref" => {
                        span_lint_and_sugg(
                            cx,
                            EXPLICIT_DEREF_METHOD,
                            expr.span,
                            "explicit deref method call",
                            "try this",
                            format!("&*{}", path),
                            Applicability::MachineApplicable
                        );
                    },
                    "deref_mut" => {
                        span_lint_and_sugg(
                            cx,
                            EXPLICIT_DEREF_METHOD,
                            expr.span,
                            "explicit deref_mut method call",
                            "try this",
                            format!("&mut *{}", path),
                            Applicability::MachineApplicable
                        );
                    },
                    _ => ()
                };
            }
        }
    }
}
