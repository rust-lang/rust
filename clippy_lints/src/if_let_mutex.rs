use crate::utils::{match_type, paths, span_lint_and_help};
use rustc::hir::map::Map;
use rustc_hir::intravisit::{self as visit, NestedVisitorMap, Visitor};
use rustc_hir::{Arm, Expr, ExprKind, MatchSource, StmtKind};
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
        let mut arm_visit = ArmVisitor {
            arm_mutex: false,
            arm_lock: false,
            cx,
        };
        let mut op_visit = IfLetMutexVisitor {
            op_mutex: false,
            op_lock: false,
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
            if op_visit.op_mutex && op_visit.op_lock {
                for arm in *arms {
                    arm_visit.visit_arm(arm);
                }

                if arm_visit.arm_mutex && arm_visit.arm_lock {
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
}

/// Checks if `Mutex::lock` is called in the `if let _ = expr.
pub struct IfLetMutexVisitor<'tcx, 'l> {
    pub op_mutex: bool,
    pub op_lock: bool,
    pub cx: &'tcx LateContext<'tcx, 'l>,
}

impl<'tcx, 'l> Visitor<'tcx> for IfLetMutexVisitor<'tcx, 'l> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(path, _span, args) = &expr.kind {
            if path.ident.to_string() == "lock" {
                self.op_lock = true;
            }
            let ty = self.cx.tables.expr_ty(&args[0]);
            if match_type(self.cx, ty, &paths::MUTEX) {
                self.op_mutex = true;
            }
        }
        visit::walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

/// Checks if `Mutex::lock` is called in any of the branches.
pub struct ArmVisitor<'tcx, 'l> {
    pub arm_mutex: bool,
    pub arm_lock: bool,
    pub cx: &'tcx LateContext<'tcx, 'l>,
}

impl<'tcx, 'l> Visitor<'tcx> for ArmVisitor<'tcx, 'l> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(path, _span, args) = &expr.kind {
            if path.ident.to_string() == "lock" {
                self.arm_lock = true;
            }
            let ty = self.cx.tables.expr_ty(&args[0]);
            if match_type(self.cx, ty, &paths::MUTEX) {
                self.arm_mutex = true;
            }
        }
        visit::walk_expr(self, expr);
    }

    fn visit_arm(&mut self, arm: &'tcx Arm<'_>) {
        if let ExprKind::Block(ref block, _l) = arm.body.kind {
            for stmt in block.stmts {
                match stmt.kind {
                    StmtKind::Local(loc) => {
                        if let Some(expr) = loc.init {
                            self.visit_expr(expr)
                        }
                    },
                    StmtKind::Expr(expr) => self.visit_expr(expr),
                    StmtKind::Semi(expr) => self.visit_expr(expr),
                    // we don't care about `Item`
                    _ => {},
                }
            }
        };
        visit::walk_arm(self, arm);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
