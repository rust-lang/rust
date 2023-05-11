use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::SpanlessEq;
use if_chain::if_chain;
use rustc_errors::Diagnostic;
use rustc_hir::intravisit::{self as visit, Visitor};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Mutex::lock` calls in `if let` expression
    /// with lock calls in any of the else blocks.
    ///
    /// ### Why is this bad?
    /// The Mutex lock remains held for the whole
    /// `if let ... else` block and deadlocks.
    ///
    /// ### Example
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
    #[clippy::version = "1.45.0"]
    pub IF_LET_MUTEX,
    correctness,
    "locking a `Mutex` in an `if let` block can cause deadlocks"
}

declare_lint_pass!(IfLetMutex => [IF_LET_MUTEX]);

impl<'tcx> LateLintPass<'tcx> for IfLetMutex {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let mut arm_visit = ArmVisitor { found_mutex: None, cx };
        let mut op_visit = OppVisitor { found_mutex: None, cx };
        if let Some(higher::IfLet {
            let_expr,
            if_then,
            if_else: Some(if_else),
            ..
        }) = higher::IfLet::hir(cx, expr)
        {
            op_visit.visit_expr(let_expr);
            if let Some(op_mutex) = op_visit.found_mutex {
                arm_visit.visit_expr(if_then);
                arm_visit.visit_expr(if_else);

                if let Some(arm_mutex) = arm_visit.found_mutex_if_same_as(op_mutex) {
                    let diag = |diag: &mut Diagnostic| {
                        diag.span_label(
                            op_mutex.span,
                            "this Mutex will remain locked for the entire `if let`-block...",
                        );
                        diag.span_label(
                            arm_mutex.span,
                            "... and is tried to lock again here, which will always deadlock.",
                        );
                        diag.help("move the lock call outside of the `if let ...` expression");
                    };
                    span_lint_and_then(
                        cx,
                        IF_LET_MUTEX,
                        expr.span,
                        "calling `Mutex::lock` inside the scope of another `Mutex::lock` causes a deadlock",
                        diag,
                    );
                }
            }
        }
    }
}

/// Checks if `Mutex::lock` is called in the `if let` expr.
pub struct OppVisitor<'a, 'tcx> {
    found_mutex: Option<&'tcx Expr<'tcx>>,
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for OppVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if let Some(mutex) = is_mutex_lock_call(self.cx, expr) {
            self.found_mutex = Some(mutex);
            return;
        }
        visit::walk_expr(self, expr);
    }
}

/// Checks if `Mutex::lock` is called in any of the branches.
pub struct ArmVisitor<'a, 'tcx> {
    found_mutex: Option<&'tcx Expr<'tcx>>,
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for ArmVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if let Some(mutex) = is_mutex_lock_call(self.cx, expr) {
            self.found_mutex = Some(mutex);
            return;
        }
        visit::walk_expr(self, expr);
    }
}

impl<'tcx, 'l> ArmVisitor<'tcx, 'l> {
    fn found_mutex_if_same_as(&self, op_mutex: &Expr<'_>) -> Option<&Expr<'_>> {
        self.found_mutex.and_then(|arm_mutex| {
            SpanlessEq::new(self.cx)
                .eq_expr(op_mutex, arm_mutex)
                .then_some(arm_mutex)
        })
    }
}

fn is_mutex_lock_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if_chain! {
        if let ExprKind::MethodCall(path, self_arg, ..) = &expr.kind;
        if path.ident.as_str() == "lock";
        let ty = cx.typeck_results().expr_ty(self_arg).peel_refs();
        if is_type_diagnostic_item(cx, ty, sym::Mutex);
        then {
            Some(self_arg)
        } else {
            None
        }
    }
}
