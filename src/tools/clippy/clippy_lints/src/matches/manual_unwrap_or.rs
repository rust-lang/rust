use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::source::{SpanRangeExt as _, indent_of, reindent_multiline};
use rustc_ast::{BindingMode, ByRef};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Arm, Expr, ExprKind, HirId, LangItem, Pat, PatExpr, PatExprKind, PatKind, QPath};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::{GenericArgKind, Ty};
use rustc_span::sym;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{expr_type_is_certain, get_type_diagnostic_name, implements_trait};
use clippy_utils::{is_default_equivalent, is_lint_allowed, path_res, peel_blocks, span_contains_comment};

use super::{MANUAL_UNWRAP_OR, MANUAL_UNWRAP_OR_DEFAULT};

fn get_some(cx: &LateContext<'_>, pat: &Pat<'_>) -> Option<HirId> {
    if let PatKind::TupleStruct(QPath::Resolved(_, path), &[pat], _) = pat.kind
        && let PatKind::Binding(BindingMode(ByRef::No, _), pat_id, _, _) = pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        // Since it comes from a pattern binding, we need to get the parent to actually match
        // against it.
        && let Some(def_id) = cx.tcx.opt_parent(def_id)
        && let Some(lang_item) = cx.tcx.lang_items().from_def_id(def_id)
        && matches!(lang_item, LangItem::OptionSome | LangItem::ResultOk)
    {
        Some(pat_id)
    } else {
        None
    }
}

fn get_none<'tcx>(cx: &LateContext<'_>, arm: &Arm<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let PatKind::Expr(PatExpr { kind: PatExprKind::Path(QPath::Resolved(_, path)), .. }) = arm.pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        // Since it comes from a pattern binding, we need to get the parent to actually match
        // against it.
        && let Some(def_id) = cx.tcx.opt_parent(def_id)
        && cx.tcx.lang_items().get(LangItem::OptionNone) == Some(def_id)
    {
        Some(arm.body)
    } else if let PatKind::TupleStruct(QPath::Resolved(_, path), _, _)= arm.pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        // Since it comes from a pattern binding, we need to get the parent to actually match
        // against it.
        && let Some(def_id) = cx.tcx.opt_parent(def_id)
        && cx.tcx.lang_items().get(LangItem::ResultErr) == Some(def_id)
    {
        Some(arm.body)
    } else if let PatKind::Wild = arm.pat.kind {
        // We consider that the `Some` check will filter it out if it's not right.
        Some(arm.body)
    } else {
        None
    }
}

fn get_some_and_none_bodies<'tcx>(
    cx: &LateContext<'tcx>,
    arm1: &'tcx Arm<'tcx>,
    arm2: &'tcx Arm<'tcx>,
) -> Option<((&'tcx Expr<'tcx>, HirId), &'tcx Expr<'tcx>)> {
    if let Some(binding_id) = get_some(cx, arm1.pat)
        && let Some(body_none) = get_none(cx, arm2)
    {
        Some(((arm1.body, binding_id), body_none))
    } else if let Some(binding_id) = get_some(cx, arm2.pat)
        && let Some(body_none) = get_none(cx, arm1)
    {
        Some(((arm2.body, binding_id), body_none))
    } else {
        None
    }
}

fn handle(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    expr_name: &'static str,
    condition: &Expr<'_>,
    body_some: &Expr<'_>,
    body_none: &Expr<'_>,
    binding_id: HirId,
) {
    // Only deal with situations where both alternatives return the same non-adjusted type.
    if cx.typeck_results().expr_ty(body_some) != cx.typeck_results().expr_ty(body_none) {
        return;
    }

    let expr_type = cx.typeck_results().expr_ty(expr);
    // We check that the `Some(x) => x` doesn't do anything apart "returning" the value in `Some`.
    if let ExprKind::Path(QPath::Resolved(_, path)) = peel_blocks(body_some).kind
        && let Res::Local(local_id) = path.res
        && local_id == binding_id
    {
        // Machine applicable only if there are no comments present
        let mut applicability = if span_contains_comment(cx.sess().source_map(), expr.span) {
            Applicability::MaybeIncorrect
        } else {
            Applicability::MachineApplicable
        };
        let receiver = Sugg::hir_with_applicability(cx, condition, "_", &mut applicability).maybe_paren();

        // We now check the `None` arm is calling a method equivalent to `Default::default`.
        if !is_lint_allowed(cx, MANUAL_UNWRAP_OR_DEFAULT, expr.hir_id)
            // We check if the return type of the expression implements Default.
            && let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
            && implements_trait(cx, expr_type, default_trait_id, &[])
            // We check if the initial condition implements Default.
            && let Some(condition_ty) = cx.typeck_results().expr_ty(condition).walk().nth(1)
            && let GenericArgKind::Type(condition_ty) = condition_ty.kind()
            && implements_trait(cx, condition_ty, default_trait_id, &[])
            && is_default_equivalent(cx, peel_blocks(body_none))
        {
            // We now check if the condition is a None variant, in which case we need to specify the type
            if path_res(cx, condition)
                .opt_def_id()
                .is_some_and(|id| Some(cx.tcx.parent(id)) == cx.tcx.lang_items().option_none_variant())
            {
                return span_lint_and_sugg(
                    cx,
                    MANUAL_UNWRAP_OR_DEFAULT,
                    expr.span,
                    format!("{expr_name} can be simplified with `.unwrap_or_default()`"),
                    "replace it with",
                    format!("{receiver}::<{expr_type}>.unwrap_or_default()"),
                    applicability,
                );
            }

            // We check if the expression type is still uncertain, in which case we ask the user to specify it
            if !expr_type_is_certain(cx, condition) {
                return span_lint_and_sugg(
                    cx,
                    MANUAL_UNWRAP_OR_DEFAULT,
                    expr.span,
                    format!("{expr_name} can be simplified with `.unwrap_or_default()`"),
                    format!("ascribe the type {expr_type} and replace your expression with"),
                    format!("{receiver}.unwrap_or_default()"),
                    Applicability::Unspecified,
                );
            }

            span_lint_and_sugg(
                cx,
                MANUAL_UNWRAP_OR_DEFAULT,
                expr.span,
                format!("{expr_name} can be simplified with `.unwrap_or_default()`"),
                "replace it with",
                format!("{receiver}.unwrap_or_default()"),
                applicability,
            );
        } else if let Some(ty_name) = find_type_name(cx, cx.typeck_results().expr_ty(condition))
            && cx.typeck_results().expr_adjustments(body_some).is_empty()
            && let Some(or_body_snippet) = peel_blocks(body_none).span.get_source_text(cx)
            && let Some(indent) = indent_of(cx, expr.span)
            && ConstEvalCtxt::new(cx).eval_simple(body_none).is_some()
        {
            let reindented_or_body = reindent_multiline(&or_body_snippet, true, Some(indent));
            let mut app = Applicability::MachineApplicable;
            let suggestion = Sugg::hir_with_context(cx, condition, expr.span.ctxt(), "..", &mut app).maybe_paren();
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
    }
}

fn find_type_name<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<&'static str> {
    match get_type_diagnostic_name(cx, ty)? {
        sym::Option => Some("Option"),
        sym::Result => Some("Result"),
        _ => None,
    }
}

pub fn check_match<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    scrutinee: &'tcx Expr<'tcx>,
    arms: &'tcx [Arm<'tcx>],
) {
    if let [arm1, arm2] = arms
        // Make sure there are no guards to keep things simple
        && arm1.guard.is_none()
        && arm2.guard.is_none()
        // Get the some and none bodies and the binding id of the some arm
        && let Some(((body_some, binding_id), body_none)) = get_some_and_none_bodies(cx, arm1, arm2)
    {
        handle(cx, expr, "match", scrutinee, body_some, body_none, binding_id);
    }
}

pub fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    pat: &'tcx Pat<'tcx>,
    scrutinee: &'tcx Expr<'tcx>,
    then_expr: &'tcx Expr<'tcx>,
    else_expr: &'tcx Expr<'tcx>,
) {
    if let Some(binding_id) = get_some(cx, pat) {
        handle(
            cx,
            expr,
            "if let",
            scrutinee,
            peel_blocks(then_expr),
            peel_blocks(else_expr),
            binding_id,
        );
    }
}
