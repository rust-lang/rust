use crate::rustc::hir::{Expr, ExprKind, QPath};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::utils::{in_macro, span_lint_and_sugg};
use if_chain::if_chain;

/// **What it does:** Checks for explicit deref() or deref_mut() method calls.
///
/// **Why is this bad?** Derefencing by &*x or &mut *x is clearer and more concise,
/// when not part of a method chain.
///
/// **Example:**
/// ```rust
/// let b = a.deref();
/// let c = a.deref_mut();
///
/// // excludes
/// let e = d.unwrap().deref();
/// ```
declare_clippy_lint! {
    pub EXPLICIT_DEREF_METHOD,
    pedantic,
    "Explicit use of deref or deref_mut method while not in a method chain."
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(EXPLICIT_DEREF_METHOD)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, expr: &Expr) {
        if in_macro(expr.span) {
            return;
        }

        if_chain! {
            // if this is a method call
            if let ExprKind::MethodCall(ref method_name, _, ref args) = &expr.node;
            // on a Path (i.e. a variable/name, not another method)
            if let ExprKind::Path(QPath::Resolved(None, path)) = &args[0].node;
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
                        );
                    },
                    _ => ()
                };
            }
        }
    }
}
