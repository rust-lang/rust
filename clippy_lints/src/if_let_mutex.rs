use crate::utils::{match_type, paths, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir::{Arm, Expr, ExprKind, MatchSource, Stmt, StmtKind};
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
    /// ```rust,ignore
    /// if let Ok(thing) = mutex.lock() {
    ///     do_thing();
    /// } else {
    ///     mutex.lock();
    /// }
    /// ```
    /// Should be written
    /// ```rust,ignore
    /// let locked = mutex.lock();
    /// if let Ok(thing) = locked {
    ///     do_thing(thing);
    /// } else {
    ///     use_locked(locked);
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
            if match_type(cx, ty, &paths::MUTEX); // make sure receiver is Mutex
            if method_chain_names(op, 10).iter().any(|s| s == "lock"); // and lock is called

            if arms.iter().any(|arm| matching_arm(arm, op, ex, cx));
            then {
                span_lint_and_help(
                    cx,
                    IF_LET_MUTEX,
                    ex.span,
                    "calling `Mutex::lock` inside the scope of another `Mutex::lock` causes a deadlock",
                    "move the lock call outside of the `if let ...` expression",
                );
            }
        }
    }
}

fn matching_arm(arm: &Arm<'_>, op: &Expr<'_>, ex: &Expr<'_>, cx: &LateContext<'_, '_>) -> bool {
    if_chain! {
        if let ExprKind::Block(ref block, _l) = arm.body.kind;
        if block.stmts.iter().any(|stmt| matching_stmt(stmt, op, ex, cx));
        then {
            true
        } else {
            false
        }
    }
}

fn matching_stmt(stmt: &Stmt<'_>, op: &Expr<'_>, ex: &Expr<'_>, cx: &LateContext<'_, '_>) -> bool {
    match stmt.kind {
        StmtKind::Local(l) => if_chain! {
            if let Some(ex) = l.init;
            if let ExprKind::MethodCall(_, _, _) = op.kind;
            if method_chain_names(ex, 10).iter().any(|s| s == "lock"); // and lock is called
            then {
                match_type_method_chain(cx, ex, 5)
            } else {
                false
            }
        },
        StmtKind::Expr(e) => if_chain! {
            if let ExprKind::MethodCall(_, _, _) = e.kind;
            if method_chain_names(e, 10).iter().any(|s| s == "lock"); // and lock is called
            then {
                match_type_method_chain(cx, ex, 5)
            } else {
                false
            }
        },
        StmtKind::Semi(e) => if_chain! {
            if let ExprKind::MethodCall(_, _, _) = e.kind;
            if method_chain_names(e, 10).iter().any(|s| s == "lock"); // and lock is called
            then {
                match_type_method_chain(cx, ex, 5)
            } else {
                false
            }
        },
        _ => false,
    }
}

/// Return the names of `max_depth` number of methods called in the chain.
fn method_chain_names<'tcx>(expr: &'tcx Expr<'tcx>, max_depth: usize) -> Vec<String> {
    let mut method_names = Vec::with_capacity(max_depth);
    let mut current = expr;
    for _ in 0..max_depth {
        if let ExprKind::MethodCall(path, _, args) = &current.kind {
            if args.iter().any(|e| e.span.from_expansion()) {
                break;
            }
            method_names.push(path.ident.to_string());
            current = &args[0];
        } else {
            break;
        }
    }
    method_names
}

/// Check that lock is called on a `Mutex`.
fn match_type_method_chain<'tcx>(cx: &LateContext<'_, '_>, expr: &'tcx Expr<'tcx>, max_depth: usize) -> bool {
    let mut current = expr;
    for _ in 0..max_depth {
        if let ExprKind::MethodCall(_, _, args) = &current.kind {
            let ty = cx.tables.expr_ty(&args[0]);
            if match_type(cx, ty, &paths::MUTEX) {
                return true;
            }
            current = &args[0];
        }
    }
    false
}
