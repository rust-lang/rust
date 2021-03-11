use crate::utils::span_lint;
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::ZST_OFFSET;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    if_chain! {
        if args.len() == 2;
        if let ty::RawPtr(ty::TypeAndMut { ref ty, .. }) = cx.typeck_results().expr_ty(&args[0]).kind();
        if let Ok(layout) = cx.tcx.layout_of(cx.param_env.and(ty));
        if layout.is_zst();
        then {
            span_lint(cx, ZST_OFFSET, expr.span, "offset calculation on zero-sized value");
        }
    }
}
