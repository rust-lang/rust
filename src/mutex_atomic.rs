//! Checks for uses of Mutex where an atomic value could be used
//!
//! This lint is **warn** by default

use rustc::lint::{LintPass, LintArray, LateLintPass, LateContext};
use rustc_front::hir::Expr;

use syntax::ast;
use rustc::middle::ty;
use rustc::middle::subst::ParamSpace;

use utils::{span_lint, MUTEX_PATH, match_type};

declare_lint! {
    pub MUTEX_ATOMIC,
    Warn,
    "using a Mutex where an atomic value could be used instead"
}

declare_lint! {
    pub MUTEX_INTEGER,
    Allow,
    "using a Mutex for an integer type"
}

impl LintPass for MutexAtomic {
    fn get_lints(&self) -> LintArray {
        lint_array!(MUTEX_ATOMIC, MUTEX_INTEGER)
    }
}

pub struct MutexAtomic;

impl LateLintPass for MutexAtomic {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        let ty = cx.tcx.expr_ty(expr);
        if let ty::TyStruct(_, subst) = ty.sty {
            if match_type(cx, ty, &MUTEX_PATH) {
                let mutex_param = &subst.types.get(ParamSpace::TypeSpace, 0).sty;
                if let Some(atomic_name) = get_atomic_name(mutex_param) {
                    let msg = format!("Consider using an {} instead of a \
                                       Mutex here. If you just want the \
                                       locking behaviour and not the internal \
                                       type, consider using Mutex<()>.",
                                      atomic_name);
                    match *mutex_param {
                        ty::TyUint(t) if t != ast::TyUs =>
                            span_lint(cx, MUTEX_INTEGER, expr.span, &msg),
                        ty::TyInt(t) if t != ast::TyIs =>
                            span_lint(cx, MUTEX_INTEGER, expr.span, &msg),
                        _ => span_lint(cx, MUTEX_ATOMIC, expr.span, &msg)
                    }
                }
            }
        }
    }
}

fn get_atomic_name(ty: &ty::TypeVariants) -> Option<(&'static str)> {
    match *ty {
        ty::TyBool => Some("AtomicBool"),
        ty::TyUint(_) => Some("AtomicUsize"),
        ty::TyInt(_) => Some("AtomicIsize"),
        ty::TyRawPtr(_) => Some("AtomicPtr"),
        _ => None
    }
}
