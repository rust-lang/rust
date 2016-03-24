use rustc::lint::*;
use rustc_front::hir::*;
use rustc::middle::ty::TyS;
use rustc::middle::ty::TypeVariants::TyRawPtr;
use utils;

/// **What it does:** This lint checks for transmutes to the original type of the object.
///
/// **Why is this bad?** Readability. The code tricks people into thinking that the original value was of some other type.
///
/// **Known problems:** None.
///
/// **Example:** `core::intrinsics::transmute(t)` where the result type is the same as `t`'s.
declare_lint! {
    pub USELESS_TRANSMUTE,
    Warn,
    "transmutes that have the same to and from types"
}

/// **What it does:*** This lint checks for transmutes between a type T and *T.
///
/// **Why is this bad?** It's easy to mistakenly transmute between a type and a pointer to that type.
///
/// **Known problems:** None.
///
/// **Example:** `core::intrinsics::transmute(t)` where the result type is the same as `*t` or `&t`'s.
declare_lint! {
    pub CROSSPOINTER_TRANSMUTE,
    Warn,
    "transmutes that have to or from types that are a pointer to the other"
}

pub struct UselessTransmute;

impl LintPass for UselessTransmute {
    fn get_lints(&self) -> LintArray {
        lint_array!(USELESS_TRANSMUTE)
    }
}

impl LateLintPass for UselessTransmute {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprCall(ref path_expr, ref args) = e.node {
            if let ExprPath(None, _) = path_expr.node {
                let def_id = cx.tcx.def_map.borrow()[&path_expr.id].def_id();

                if utils::match_def_path(cx, def_id, &["core", "intrinsics", "transmute"]) {
                    let from_ty = cx.tcx.expr_ty(&args[0]);
                    let to_ty = cx.tcx.expr_ty(e);

                    if from_ty == to_ty {
                        cx.span_lint(USELESS_TRANSMUTE,
                                     e.span,
                                     &format!("transmute from a type (`{}`) to itself", from_ty));
                    }
                }
            }
        }
    }
}

pub struct CrosspointerTransmute;

impl LintPass for CrosspointerTransmute {
    fn get_lints(&self) -> LintArray {
        lint_array!(CROSSPOINTER_TRANSMUTE)
    }
}

fn is_ptr_to(from: &TyS, to: &TyS) -> bool {
    if let TyRawPtr(from_ptr) = from.sty {
        from_ptr.ty == to
    } else {
        false
    }
}

impl LateLintPass for CrosspointerTransmute {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprCall(ref path_expr, ref args) = e.node {
            if let ExprPath(None, _) = path_expr.node {
                let def_id = cx.tcx.def_map.borrow()[&path_expr.id].def_id();

                if utils::match_def_path(cx, def_id, &["core", "intrinsics", "transmute"]) {
                    let from_ty = cx.tcx.expr_ty(&args[0]);
                    let to_ty = cx.tcx.expr_ty(e);

                    if is_ptr_to(to_ty, from_ty) {
                        cx.span_lint(CROSSPOINTER_TRANSMUTE,
                                     e.span,
                                     &format!("transmute from a type (`{}`) to a pointer to that type (`{}`)", from_ty, to_ty));
                    }

                    if is_ptr_to(from_ty, to_ty) {
                        cx.span_lint(CROSSPOINTER_TRANSMUTE,
                                     e.span,
                                     &format!("transmute from a type (`{}`) to the type that it points to (`{}`)", from_ty, to_ty));
                    }
                }
            }
        }
    }
}
