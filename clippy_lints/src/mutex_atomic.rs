//! Checks for uses of mutex where an atomic value could be used
//!
//! This lint is **warn** by default

use crate::utils::{match_type, paths, span_lint};
use rustc::hir::Expr;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty::{self, Ty};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast;

declare_clippy_lint! {
    /// **What it does:** Checks for usages of `Mutex<X>` where an atomic will do.
    ///
    /// **Why is this bad?** Using a mutex just to make access to a plain bool or
    /// reference sequential is shooting flies with cannons.
    /// `std::sync::atomic::AtomicBool` and `std::sync::atomic::AtomicPtr` are leaner and
    /// faster.
    ///
    /// **Known problems:** This lint cannot detect if the mutex is actually used
    /// for waiting before a critical section.
    ///
    /// **Example:**
    /// ```rust
    /// # use std::sync::Mutex;
    /// # let y = 1;
    /// let x = Mutex::new(&y);
    /// ```
    pub MUTEX_ATOMIC,
    perf,
    "using a mutex where an atomic value could be used instead"
}

declare_clippy_lint! {
    /// **What it does:** Checks for usages of `Mutex<X>` where `X` is an integral
    /// type.
    ///
    /// **Why is this bad?** Using a mutex just to make access to a plain integer
    /// sequential is
    /// shooting flies with cannons. `std::sync::atomic::AtomicUsize` is leaner and faster.
    ///
    /// **Known problems:** This lint cannot detect if the mutex is actually used
    /// for waiting before a critical section.
    ///
    /// **Example:**
    /// ```rust
    /// let x = Mutex::new(0usize);
    /// ```
    pub MUTEX_INTEGER,
    nursery,
    "using a mutex for an integer type"
}

declare_lint_pass!(Mutex => [MUTEX_ATOMIC, MUTEX_INTEGER]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Mutex {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        let ty = cx.tables.expr_ty(expr);
        if let ty::Adt(_, subst) = ty.sty {
            if match_type(cx, ty, &paths::MUTEX) {
                let mutex_param = subst.type_at(0);
                if let Some(atomic_name) = get_atomic_name(mutex_param) {
                    let msg = format!(
                        "Consider using an {} instead of a Mutex here. If you just want the locking \
                         behaviour and not the internal type, consider using Mutex<()>.",
                        atomic_name
                    );
                    match mutex_param.sty {
                        ty::Uint(t) if t != ast::UintTy::Usize => span_lint(cx, MUTEX_INTEGER, expr.span, &msg),
                        ty::Int(t) if t != ast::IntTy::Isize => span_lint(cx, MUTEX_INTEGER, expr.span, &msg),
                        _ => span_lint(cx, MUTEX_ATOMIC, expr.span, &msg),
                    };
                }
            }
        }
    }
}

fn get_atomic_name(ty: Ty<'_>) -> Option<(&'static str)> {
    match ty.sty {
        ty::Bool => Some("AtomicBool"),
        ty::Uint(_) => Some("AtomicUsize"),
        ty::Int(_) => Some("AtomicIsize"),
        ty::RawPtr(_) => Some("AtomicPtr"),
        _ => None,
    }
}
