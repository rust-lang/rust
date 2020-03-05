use crate::utils::{
    match_type, method_calls, method_chain_args, paths, snippet, snippet_with_applicability, span_lint_and_sugg,
};
use if_chain::if_chain;
use rustc::ty;
use rustc_errors::Applicability;
use rustc_hir::{print, Expr, ExprKind, MatchSource, PatKind, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `Mutex::lock` calls in `if let` expression
    /// with lock calls in any of the else blocks.
    ///
    /// **Why is this bad?** The Mutex lock remains held for the whole
    /// `if let ... else` block and deadlocks.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # use std::sync::Mutex;
    /// let mutex = Mutex::new(10);
    /// if let Ok(thing) = mutex.lock() {
    ///     do_thing();
    /// } else {
    ///     mutex.lock();
    /// }
    /// ```
    pub IF_LET_MUTEX,
    correctness,
    "locking a `Mutex` in an `if let` block can cause deadlocks"
}

declare_lint_pass!(IfLetMutex => [IF_LET_MUTEX]);

impl LateLintPass<'_, '_> for IfLetMutex {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, ex: &'_ Expr<'_>) {
        if_chain! {
            if let ExprKind::Match(ref op, ref arms, MatchSource::IfLetDesugar {
                contains_else_clause: true,
            }) = ex.kind; // if let ... {} else {}
            if let ExprKind::MethodCall(_, _, ref args) = op.kind;
            let ty = cx.tables.expr_ty(&args[0]);
            if let ty::Adt(_, subst) = ty.kind;
            if match_type(cx, ty, &paths::MUTEX); // make sure receiver is Mutex
            if method_chain_names(op, 10).iter().any(|s| s == "lock"); // and lock is called

            let mut suggestion = String::from(&format!("if let _ = {} {{\n", snippet(cx, op.span, "_")));
            
            if arms.iter().any(|arm| if_chain! {
                if let ExprKind::Block(ref block, _l) = arm.body.kind;
                if block.stmts.iter().any(|stmt| match stmt.kind {
                    StmtKind::Local(l) => if_chain! {
                        if let Some(ex) = l.init;
                        if let ExprKind::MethodCall(_, _, ref args) = op.kind;
                        if method_chain_names(ex, 10).iter().any(|s| s == "lock"); // and lock is called
                        then {
                            let ty = cx.tables.expr_ty(&args[0]);
                            // // make sure receiver is Result<MutexGuard<...>>
                            match_type(cx, ty, &paths::RESULT)
                        } else {
                            suggestion.push_str(&format!("    {}\n", snippet(cx, l.span, "_")));
                            false
                        }
                    },
                    StmtKind::Expr(e) => if_chain! {
                        if let ExprKind::MethodCall(_, _, ref args) = e.kind;
                        if method_chain_names(e, 10).iter().any(|s| s == "lock"); // and lock is called
                        then {
                            let ty = cx.tables.expr_ty(&args[0]);
                            // // make sure receiver is Result<MutexGuard<...>>
                            match_type(cx, ty, &paths::RESULT)
                        } else {
                            suggestion.push_str(&format!("    {}\n", snippet(cx, e.span, "_")));
                            false
                        }
                    },
                    StmtKind::Semi(e) => if_chain! {
                        if let ExprKind::MethodCall(_, _, ref args) = e.kind;
                        if method_chain_names(e, 10).iter().any(|s| s == "lock"); // and lock is called
                        then {
                            let ty = cx.tables.expr_ty(&args[0]);
                            // // make sure receiver is Result<MutexGuard<...>>
                            match_type(cx, ty, &paths::RESULT)
                        } else {
                            suggestion.push_str(&format!("    {}\n", snippet(cx, e.span, "_")));
                            false
                        }
                    },
                    _ => { suggestion.push_str(&format!("     {}\n", snippet(cx, stmt.span, "_"))); false },
                });
                then {
                    true
                } else {
                    suggestion.push_str(&format!("else {}\n", snippet(cx, arm.span, "_")));
                    false
                }
            });
            then {
                println!("{}", suggestion);
                span_lint_and_sugg(
                    cx,
                    IF_LET_MUTEX,
                    ex.span,
                    "using a `Mutex` in inner scope of `.lock` call",
                    "try",
                    format!("{:?}", "hello"),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

fn method_chain_names<'tcx>(expr: &'tcx Expr<'tcx>, max_depth: usize) -> Vec<String> {
    let mut method_names = Vec::with_capacity(max_depth);

    let mut current = expr;
    for _ in 0..max_depth {
        if let ExprKind::MethodCall(path, span, args) = &current.kind {
            if args.iter().any(|e| e.span.from_expansion()) {
                break;
            }
            method_names.push(path.ident.to_string());
            println!("{:?}", method_names);
            current = &args[0];
        } else {
            break;
        }
    }

    method_names
}
