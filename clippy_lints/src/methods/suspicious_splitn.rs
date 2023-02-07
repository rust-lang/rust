use clippy_utils::diagnostics::span_lint_and_note;
use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::SUSPICIOUS_SPLITN;

pub(super) fn check(cx: &LateContext<'_>, method_name: &str, expr: &Expr<'_>, self_arg: &Expr<'_>, count: u128) {
    if_chain! {
        if count <= 1;
        if let Some(call_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(call_id);
        if cx.tcx.impl_trait_ref(impl_id).is_none();
        let self_ty = cx.tcx.bound_type_of(impl_id).subst_identity();
        if self_ty.is_slice() || self_ty.is_str();
        then {
            // Ignore empty slice and string literals when used with a literal count.
            if matches!(self_arg.kind, ExprKind::Array([]))
                || matches!(self_arg.kind, ExprKind::Lit(Spanned { node: LitKind::Str(s, _), .. }) if s.is_empty())
            {
                return;
            }

            let (msg, note_msg) = if count == 0 {
                (format!("`{method_name}` called with `0` splits"),
                "the resulting iterator will always return `None`")
            } else {
                (format!("`{method_name}` called with `1` split"),
                if self_ty.is_slice() {
                    "the resulting iterator will always return the entire slice followed by `None`"
                } else {
                    "the resulting iterator will always return the entire string followed by `None`"
                })
            };

            span_lint_and_note(
                cx,
                SUSPICIOUS_SPLITN,
                expr.span,
                &msg,
                None,
                note_msg,
            );
        }
    }
}
