use clippy_utils::consts::constant_simple;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline, snippet_opt};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::usage::contains_return_break_continue_macro;
use clippy_utils::{is_res_lang_ctor, path_to_local_id, sugg};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::LangItem::{OptionNone, ResultErr};
use rustc_hir::{Arm, Expr, PatKind};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::MANUAL_UNWRAP_OR;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>, scrutinee: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>]) {
    let ty = cx.typeck_results().expr_ty(scrutinee);
    if let Some(ty_name) = if is_type_diagnostic_item(cx, ty, sym::Option) {
        Some("Option")
    } else if is_type_diagnostic_item(cx, ty, sym::Result) {
        Some("Result")
    } else {
        None
    } && let Some(or_arm) = applicable_or_arm(cx, arms)
        && let Some(or_body_snippet) = snippet_opt(cx, or_arm.body.span)
        && let Some(indent) = indent_of(cx, expr.span)
        && constant_simple(cx, cx.typeck_results(), or_arm.body).is_some()
    {
        let reindented_or_body = reindent_multiline(or_body_snippet.into(), true, Some(indent));

        let mut app = Applicability::MachineApplicable;
        let suggestion = sugg::Sugg::hir_with_context(cx, scrutinee, expr.span.ctxt(), "..", &mut app).maybe_par();
        span_lint_and_sugg(
            cx,
            MANUAL_UNWRAP_OR,
            expr.span,
            &format!("this pattern reimplements `{ty_name}::unwrap_or`"),
            "replace with",
            format!("{suggestion}.unwrap_or({reindented_or_body})",),
            app,
        );
    }
}

fn applicable_or_arm<'a>(cx: &LateContext<'_>, arms: &'a [Arm<'a>]) -> Option<&'a Arm<'a>> {
    if arms.len() == 2
        && arms.iter().all(|arm| arm.guard.is_none())
        && let Some((idx, or_arm)) = arms.iter().enumerate().find(|(_, arm)| match arm.pat.kind {
            PatKind::Path(ref qpath) => is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), OptionNone),
            PatKind::TupleStruct(ref qpath, [pat], _) => {
                matches!(pat.kind, PatKind::Wild)
                    && is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), ResultErr)
            },
            _ => false,
        })
        && let unwrap_arm = &arms[1 - idx]
        && let PatKind::TupleStruct(ref qpath, [unwrap_pat], _) = unwrap_arm.pat.kind
        && let Res::Def(DefKind::Ctor(..), ctor_id) = cx.qpath_res(qpath, unwrap_arm.pat.hir_id)
        && let Some(variant_id) = cx.tcx.opt_parent(ctor_id)
        && (cx.tcx.lang_items().option_some_variant() == Some(variant_id)
            || cx.tcx.lang_items().result_ok_variant() == Some(variant_id))
        && let PatKind::Binding(_, binding_hir_id, ..) = unwrap_pat.kind
        && path_to_local_id(unwrap_arm.body, binding_hir_id)
        && cx.typeck_results().expr_adjustments(unwrap_arm.body).is_empty()
        && !contains_return_break_continue_macro(or_arm.body)
    {
        Some(or_arm)
    } else {
        None
    }
}
