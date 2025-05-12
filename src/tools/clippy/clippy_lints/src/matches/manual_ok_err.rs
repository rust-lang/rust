use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::option_arg_ty;
use clippy_utils::{get_parent_expr, is_res_lang_ctor, path_res, peel_blocks, span_contains_comment};
use rustc_ast::BindingMode;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome, ResultErr};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Arm, Expr, ExprKind, Pat, PatExpr, PatExprKind, PatKind, Path, QPath};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::Ty;
use rustc_span::symbol::Ident;

use super::MANUAL_OK_ERR;

pub(crate) fn check_if_let(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    let_pat: &Pat<'_>,
    let_expr: &Expr<'_>,
    if_then: &Expr<'_>,
    else_expr: &Expr<'_>,
) {
    if let Some(inner_expr_ty) = option_arg_ty(cx, cx.typeck_results().expr_ty(expr))
        && let Some((is_ok, ident)) = is_ok_or_err(cx, let_pat)
        && is_some_ident(cx, if_then, ident, inner_expr_ty)
        && is_none(cx, else_expr)
    {
        apply_lint(cx, expr, let_expr, is_ok);
    }
}

pub(crate) fn check_match(cx: &LateContext<'_>, expr: &Expr<'_>, scrutinee: &Expr<'_>, arms: &[Arm<'_>]) {
    if let Some(inner_expr_ty) = option_arg_ty(cx, cx.typeck_results().expr_ty(expr))
        && arms.len() == 2
        && arms.iter().all(|arm| arm.guard.is_none())
        && let Some((idx, is_ok)) = arms.iter().enumerate().find_map(|(arm_idx, arm)| {
            // Check if the arm is a `Ok(x) => x` or `Err(x) => x` alternative.
            // In this case, return its index and whether it uses `Ok` or `Err`.
             if let Some((is_ok, ident)) = is_ok_or_err(cx, arm.pat)
                && is_some_ident(cx, arm.body, ident, inner_expr_ty)
            {
                Some((arm_idx, is_ok))
            } else {
                None
            }
        })
        // Accept wildcard only as the second arm
        && is_variant_or_wildcard(cx, arms[1-idx].pat, idx == 0, is_ok)
        // Check that the body of the non `Ok`/`Err` arm is `None`
        && is_none(cx, arms[1 - idx].body)
    {
        apply_lint(cx, expr, scrutinee, is_ok);
    }
}

/// Check that `pat` applied to a `Result` only matches `Ok(_)`, `Err(_)`, not a subset or a
/// superset of it. If `can_be_wild` is `true`, wildcards are also accepted. In the case of
/// a non-wildcard, `must_match_err` indicates whether the `Err` or the `Ok` variant should be
/// accepted.
fn is_variant_or_wildcard(cx: &LateContext<'_>, pat: &Pat<'_>, can_be_wild: bool, must_match_err: bool) -> bool {
    match pat.kind {
        PatKind::Wild
        | PatKind::Expr(PatExpr {
            kind: PatExprKind::Path(_),
            ..
        })
        | PatKind::Binding(_, _, _, None)
            if can_be_wild =>
        {
            true
        },
        PatKind::TupleStruct(qpath, ..) => {
            is_res_lang_ctor(cx, cx.qpath_res(&qpath, pat.hir_id), ResultErr) == must_match_err
        },
        PatKind::Binding(_, _, _, Some(pat)) | PatKind::Ref(pat, _) => {
            is_variant_or_wildcard(cx, pat, can_be_wild, must_match_err)
        },
        _ => false,
    }
}

/// Return `Some((true, IDENT))` if `pat` contains `Ok(IDENT)`, `Some((false, IDENT))` if it
/// contains `Err(IDENT)`, `None` otherwise.
fn is_ok_or_err<'hir>(cx: &LateContext<'_>, pat: &Pat<'hir>) -> Option<(bool, &'hir Ident)> {
    if let PatKind::TupleStruct(qpath, [arg], _) = &pat.kind
        && let PatKind::Binding(BindingMode::NONE, _, ident, None) = &arg.kind
        && let res = cx.qpath_res(qpath, pat.hir_id)
        && let Res::Def(DefKind::Ctor(..), id) = res
        && let id @ Some(_) = cx.tcx.opt_parent(id)
    {
        let lang_items = cx.tcx.lang_items();
        if id == lang_items.result_ok_variant() {
            return Some((true, ident));
        } else if id == lang_items.result_err_variant() {
            return Some((false, ident));
        }
    }
    None
}

/// Check if `expr` contains `Some(ident)`, possibly as a block
fn is_some_ident<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, ident: &Ident, ty: Ty<'tcx>) -> bool {
    if let ExprKind::Call(body_callee, [body_arg]) = peel_blocks(expr).kind
        && is_res_lang_ctor(cx, path_res(cx, body_callee), OptionSome)
        && cx.typeck_results().expr_ty(body_arg) == ty
        && let ExprKind::Path(QPath::Resolved(
            _,
            Path {
                segments: [segment], ..
            },
        )) = body_arg.kind
    {
        segment.ident.name == ident.name
    } else {
        false
    }
}

/// Check if `expr` is `None`, possibly as a block
fn is_none(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    is_res_lang_ctor(cx, path_res(cx, peel_blocks(expr)), OptionNone)
}

/// Suggest replacing `expr` by `scrutinee.METHOD()`, where `METHOD` is either `ok` or
/// `err`, depending on `is_ok`.
fn apply_lint(cx: &LateContext<'_>, expr: &Expr<'_>, scrutinee: &Expr<'_>, is_ok: bool) {
    let method = if is_ok { "ok" } else { "err" };
    let mut app = if span_contains_comment(cx.sess().source_map(), expr.span) {
        Applicability::MaybeIncorrect
    } else {
        Applicability::MachineApplicable
    };
    let scrut = Sugg::hir_with_applicability(cx, scrutinee, "..", &mut app).maybe_paren();
    let sugg = format!("{scrut}.{method}()");
    // If the expression being expanded is the `if …` part of an `else if …`, it must be blockified.
    let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr)
        && let ExprKind::If(_, _, Some(else_part)) = parent_expr.kind
        && else_part.hir_id == expr.hir_id
    {
        reindent_multiline(&format!("{{\n    {sugg}\n}}"), true, indent_of(cx, parent_expr.span))
    } else {
        sugg
    };
    span_lint_and_sugg(
        cx,
        MANUAL_OK_ERR,
        expr.span,
        format!("manual implementation of `{method}`"),
        "replace with",
        sugg,
        app,
    );
}
