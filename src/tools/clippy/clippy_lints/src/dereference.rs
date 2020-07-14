use crate::utils::{get_parent_expr, implements_trait, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::util::parser::{ExprPrecedence, PREC_POSTFIX, PREC_PREFIX};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for explicit `deref()` or `deref_mut()` method calls.
    ///
    /// **Why is this bad?** Derefencing by `&*x` or `&mut *x` is clearer and more concise,
    /// when not part of a method chain.
    ///
    /// **Example:**
    /// ```rust
    /// use std::ops::Deref;
    /// let a: &mut String = &mut String::from("foo");
    /// let b: &str = a.deref();
    /// ```
    /// Could be written as:
    /// ```rust
    /// let a: &mut String = &mut String::from("foo");
    /// let b = &*a;
    /// ```
    ///
    /// This lint excludes
    /// ```rust,ignore
    /// let _ = d.unwrap().deref();
    /// ```
    pub EXPLICIT_DEREF_METHODS,
    pedantic,
    "Explicit use of deref or deref_mut method while not in a method chain."
}

declare_lint_pass!(Dereferencing => [
    EXPLICIT_DEREF_METHODS
]);

impl<'tcx> LateLintPass<'tcx> for Dereferencing {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if !expr.span.from_expansion();
            if let ExprKind::MethodCall(ref method_name, _, ref args, _) = &expr.kind;
            if args.len() == 1;

            then {
                if let Some(parent_expr) = get_parent_expr(cx, expr) {
                    // Check if we have the whole call chain here
                    if let ExprKind::MethodCall(..) = parent_expr.kind {
                        return;
                    }
                    // Check for Expr that we don't want to be linted
                    let precedence = parent_expr.precedence();
                    match precedence {
                        // Lint a Call is ok though
                        ExprPrecedence::Call | ExprPrecedence::AddrOf => (),
                        _ => {
                            if precedence.order() >= PREC_PREFIX && precedence.order() <= PREC_POSTFIX {
                                return;
                            }
                        }
                    }
                }
                let name = method_name.ident.as_str();
                lint_deref(cx, &*name, &args[0], args[0].span, expr.span);
            }
        }
    }
}

fn lint_deref(cx: &LateContext<'_>, method_name: &str, call_expr: &Expr<'_>, var_span: Span, expr_span: Span) {
    match method_name {
        "deref" => {
            let impls_deref_trait = cx.tcx.lang_items().deref_trait().map_or(false, |id| {
                implements_trait(cx, cx.tables().expr_ty(&call_expr), id, &[])
            });
            if impls_deref_trait {
                span_lint_and_sugg(
                    cx,
                    EXPLICIT_DEREF_METHODS,
                    expr_span,
                    "explicit deref method call",
                    "try this",
                    format!("&*{}", &snippet(cx, var_span, "..")),
                    Applicability::MachineApplicable,
                );
            }
        },
        "deref_mut" => {
            let impls_deref_mut_trait = cx.tcx.lang_items().deref_mut_trait().map_or(false, |id| {
                implements_trait(cx, cx.tables().expr_ty(&call_expr), id, &[])
            });
            if impls_deref_mut_trait {
                span_lint_and_sugg(
                    cx,
                    EXPLICIT_DEREF_METHODS,
                    expr_span,
                    "explicit deref_mut method call",
                    "try this",
                    format!("&mut *{}", &snippet(cx, var_span, "..")),
                    Applicability::MachineApplicable,
                );
            }
        },
        _ => (),
    }
}
