use crate::utils::{get_parent_expr, method_calls, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{Expr, ExprKind, QPath, StmtKind};
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
    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx hir::Stmt<'_>) {
        if_chain! {
            if let StmtKind::Local(ref local) = stmt.kind;
            if let Some(ref init) = local.init;

            then {
                match init.kind {
                    ExprKind::Call(ref _method, args) => {
                        for arg in args {
                            if_chain! {
                                // Caller must call only one other function (deref or deref_mut)
                                // otherwise it can lead to error prone suggestions (ex: &*a.len())
                                let (method_names, arg_list, _) = method_calls(arg, 2);
                                if method_names.len() == 1;
                                // Caller must be a variable
                                let variables = arg_list[0];
                                if variables.len() == 1;
                                if let ExprKind::Path(QPath::Resolved(None, _)) = variables[0].kind;

                                then {
                                    let name = method_names[0].as_str();
                                    lint_deref(cx, &*name, variables[0].span, arg.span);
                                }
                            }
                        }
                    }
                    ExprKind::MethodCall(ref method_name, _, ref args) => {
                        if init.span.from_expansion() {
                            return;
                        }
                        if_chain! {
                            if args.len() == 1;
                            if let ExprKind::Path(QPath::Resolved(None, _)) = args[0].kind;
                            // Caller must call only one other function (deref or deref_mut)
                            // otherwise it can lead to error prone suggestions (ex: &*a.len())
                            let (method_names, arg_list, _) = method_calls(init, 2);
                            if method_names.len() == 1;
                            // Caller must be a variable
                            let variables = arg_list[0];
                            if variables.len() == 1;
                            if let ExprKind::Path(QPath::Resolved(None, _)) = variables[0].kind;

                            then {
                                let name = method_name.ident.as_str();
                                lint_deref(cx, &*name, args[0].span, init.span);
                            }
                        }
                    }
                    _ => ()
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(ref method_name, _, ref args) = &expr.kind;
            if args.len() == 1;
            if let Some(parent) = get_parent_expr(cx, &expr);

            then {
                // Call and MethodCall exprs are better reported using statements
                match parent.kind {
                    ExprKind::Call(_, _) => return,
                    ExprKind::MethodCall(_, _, _) => return,
                    _ => {
                        let name = method_name.ident.as_str();
                        lint_deref(cx, &*name, args[0].span, expr.span);
                    }
                }
            }
        }
    }
}

fn lint_deref(cx: &LateContext<'_, '_>, fn_name: &str, var_span: Span, expr_span: Span) {
    match fn_name {
        "deref" => {
            span_lint_and_sugg(
                cx,
                EXPLICIT_DEREF_METHOD,
                expr_span,
                "explicit deref method call",
                "try this",
                format!("&*{}", &snippet(cx, var_span, "..")),
                Applicability::MachineApplicable,
            );
        },
        "deref_mut" => {
            span_lint_and_sugg(
                cx,
                EXPLICIT_DEREF_METHOD,
                expr_span,
                "explicit deref_mut method call",
                "try this",
                format!("&mut *{}", &snippet(cx, var_span, "..")),
                Applicability::MachineApplicable,
            );
        },
        _ => (),
    }
}
