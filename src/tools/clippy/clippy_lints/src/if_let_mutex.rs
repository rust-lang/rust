use crate::utils::{is_type_diagnostic_item, span_lint_and_help, SpanlessEq};
use if_chain::if_chain;
use rustc_hir::intravisit::{self as visit, NestedVisitorMap, Visitor};
use rustc_hir::{Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
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

impl<'tcx> LateLintPass<'tcx> for IfLetMutex {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, ex: &'tcx Expr<'tcx>) {
        let mut arm_visit = ArmVisitor {
            mutex_lock_called: false,
            found_mutex: None,
            cx,
        };
        let mut op_visit = OppVisitor {
            mutex_lock_called: false,
            found_mutex: None,
            cx,
        };
        if let ExprKind::Match(
            ref op,
            ref arms,
            MatchSource::IfLetDesugar {
                contains_else_clause: true,
            },
        ) = ex.kind
        {
            op_visit.visit_expr(op);
            if op_visit.mutex_lock_called {
                for arm in *arms {
                    arm_visit.visit_arm(arm);
                }

                if arm_visit.mutex_lock_called && arm_visit.same_mutex(cx, op_visit.found_mutex.unwrap()) {
                    span_lint_and_help(
                        cx,
                        IF_LET_MUTEX,
                        ex.span,
                        "calling `Mutex::lock` inside the scope of another `Mutex::lock` causes a deadlock",
                        None,
                        "move the lock call outside of the `if let ...` expression",
                    );
                }
            }
        }
    }
}

/// Checks if `Mutex::lock` is called in the `if let _ = expr.
pub struct OppVisitor<'a, 'tcx> {
    mutex_lock_called: bool,
    found_mutex: Option<&'tcx Expr<'tcx>>,
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for OppVisitor<'_, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let Some(mutex) = is_mutex_lock_call(self.cx, expr);
            then {
                self.found_mutex = Some(mutex);
                self.mutex_lock_called = true;
                return;
            }
        }
        visit::walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Checks if `Mutex::lock` is called in any of the branches.
pub struct ArmVisitor<'a, 'tcx> {
    mutex_lock_called: bool,
    found_mutex: Option<&'tcx Expr<'tcx>>,
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for ArmVisitor<'_, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if_chain! {
            if let Some(mutex) = is_mutex_lock_call(self.cx, expr);
            then {
                self.found_mutex = Some(mutex);
                self.mutex_lock_called = true;
                return;
            }
        }
        visit::walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

impl<'tcx, 'l> ArmVisitor<'tcx, 'l> {
    fn same_mutex(&self, cx: &LateContext<'_>, op_mutex: &Expr<'_>) -> bool {
        self.found_mutex
            .map_or(false, |arm_mutex| SpanlessEq::new(cx).eq_expr(op_mutex, arm_mutex))
    }
}

fn is_mutex_lock_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if_chain! {
        if let ExprKind::MethodCall(path, _span, args, _) = &expr.kind;
        if path.ident.as_str() == "lock";
        let ty = cx.typeck_results().expr_ty(&args[0]);
        if is_type_diagnostic_item(cx, ty, sym!(mutex_type));
        then {
            Some(&args[0])
        } else {
            None
        }
    }
}
