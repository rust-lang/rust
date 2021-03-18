use crate::utils::{match_def_path, match_qpath, paths, span_lint};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::UNINIT_ASSUMED_INIT;

/// lint for `MaybeUninit::uninit().assume_init()` (we already have the latter)
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, outer: &hir::Expr<'_>) {
    if_chain! {
        if let hir::ExprKind::Call(ref callee, ref args) = expr.kind;
        if args.is_empty();
        if let hir::ExprKind::Path(ref path) = callee.kind;
        if match_qpath(path, &paths::MEM_MAYBEUNINIT_UNINIT);
        if !is_maybe_uninit_ty_valid(cx, cx.typeck_results().expr_ty_adjusted(outer));
        then {
            span_lint(
                cx,
                UNINIT_ASSUMED_INIT,
                outer.span,
                "this call for this type may be undefined behavior"
            );
        }
    }
}

fn is_maybe_uninit_ty_valid(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Array(ref component, _) => is_maybe_uninit_ty_valid(cx, component),
        ty::Tuple(ref types) => types.types().all(|ty| is_maybe_uninit_ty_valid(cx, ty)),
        ty::Adt(ref adt, _) => match_def_path(cx, adt.did, &paths::MEM_MAYBEUNINIT),
        _ => false,
    }
}
