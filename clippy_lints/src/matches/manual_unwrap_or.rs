use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SpanRangeExt, indent_of, reindent_multiline};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::usage::contains_return_break_continue_macro;
use clippy_utils::{is_res_lang_ctor, path_to_local_id, peel_blocks, sugg};
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, ResultErr};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Arm, Expr, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::sym;

use super::MANUAL_UNWRAP_OR;

pub(super) fn check_match<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    arms: &'tcx [Arm<'_>],
) {
    let ty = cx.typeck_results().expr_ty(scrutinee);
    if let Some((or_arm, unwrap_arm)) = applicable_or_arm(cx, arms) {
        check_and_lint(cx, expr, unwrap_arm.pat, scrutinee, unwrap_arm.body, or_arm.body, ty);
    }
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    then_expr: &'tcx Expr<'_>,
    else_expr: &'tcx Expr<'_>,
) {
    let ty = cx.typeck_results().expr_ty(let_expr);
    let then_ty = cx.typeck_results().expr_ty(then_expr);
    // The signature is `fn unwrap_or<T>(self: Option<T>, default: T) -> T`.
    // When `expr_adjustments(then_expr).is_empty()`, `T` should equate to `default`'s type.
    // Otherwise, type error will occur.
    if cx.typeck_results().expr_adjustments(then_expr).is_empty()
        && let rustc_middle::ty::Adt(_did, args) = ty.kind()
        && let Some(some_ty) = args.first().and_then(|arg| arg.as_type())
        && some_ty != then_ty
    {
        return;
    }
    check_and_lint(cx, expr, let_pat, let_expr, then_expr, peel_blocks(else_expr), ty);
}

fn check_and_lint<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    then_expr: &'tcx Expr<'_>,
    else_expr: &'tcx Expr<'_>,
    ty: Ty<'tcx>,
) {
    if let PatKind::TupleStruct(ref qpath, [unwrap_pat], _) = let_pat.kind
        && let Res::Def(DefKind::Ctor(..), ctor_id) = cx.qpath_res(qpath, let_pat.hir_id)
        && let Some(variant_id) = cx.tcx.opt_parent(ctor_id)
        && (cx.tcx.lang_items().option_some_variant() == Some(variant_id)
            || cx.tcx.lang_items().result_ok_variant() == Some(variant_id))
        && let PatKind::Binding(_, binding_hir_id, ..) = unwrap_pat.kind
        && path_to_local_id(peel_blocks(then_expr), binding_hir_id)
        && cx.typeck_results().expr_adjustments(then_expr).is_empty()
        && let Some(ty_name) = find_type_name(cx, ty)
        && let Some(or_body_snippet) = else_expr.span.get_source_text(cx)
        && let Some(indent) = indent_of(cx, expr.span)
        && ConstEvalCtxt::new(cx).eval_simple(else_expr).is_some()
    {
        lint(cx, expr, let_expr, ty_name, &or_body_snippet, indent);
    }
}

fn find_type_name<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<&'static str> {
    if is_type_diagnostic_item(cx, ty, sym::Option) {
        Some("Option")
    } else if is_type_diagnostic_item(cx, ty, sym::Result) {
        Some("Result")
    } else {
        None
    }
}

fn applicable_or_arm<'a>(cx: &LateContext<'_>, arms: &'a [Arm<'a>]) -> Option<(&'a Arm<'a>, &'a Arm<'a>)> {
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
        && !contains_return_break_continue_macro(or_arm.body)
    {
        Some((or_arm, unwrap_arm))
    } else {
        None
    }
}

fn lint<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    scrutinee: &'tcx Expr<'_>,
    ty_name: &str,
    or_body_snippet: &str,
    indent: usize,
) {
    let reindented_or_body = reindent_multiline(or_body_snippet.into(), true, Some(indent));

    let mut app = Applicability::MachineApplicable;
    let suggestion = sugg::Sugg::hir_with_context(cx, scrutinee, expr.span.ctxt(), "..", &mut app).maybe_par();
    span_lint_and_sugg(
        cx,
        MANUAL_UNWRAP_OR,
        expr.span,
        format!("this pattern reimplements `{ty_name}::unwrap_or`"),
        "replace with",
        format!("{suggestion}.unwrap_or({reindented_or_body})",),
        app,
    );
}
