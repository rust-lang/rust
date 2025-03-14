use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Arm, Expr, ExprKind, HirId, LangItem, MatchSource, Pat, PatExpr, PatExprKind, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::GenericArgKind;
use rustc_session::declare_lint_pass;
use rustc_span::sym;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::IfLetOrMatch;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{expr_type_is_certain, implements_trait};
use clippy_utils::{is_default_equivalent, is_in_const_context, path_res, peel_blocks, span_contains_comment};

declare_clippy_lint! {
    /// ### What it does
    /// Checks if a `match` or `if let` expression can be simplified using
    /// `.unwrap_or_default()`.
    ///
    /// ### Why is this bad?
    /// It can be done in one call with `.unwrap_or_default()`.
    ///
    /// ### Example
    /// ```no_run
    /// let x: Option<String> = Some(String::new());
    /// let y: String = match x {
    ///     Some(v) => v,
    ///     None => String::new(),
    /// };
    ///
    /// let x: Option<Vec<String>> = Some(Vec::new());
    /// let y: Vec<String> = if let Some(v) = x {
    ///     v
    /// } else {
    ///     Vec::new()
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: Option<String> = Some(String::new());
    /// let y: String = x.unwrap_or_default();
    ///
    /// let x: Option<Vec<String>> = Some(Vec::new());
    /// let y: Vec<String> = x.unwrap_or_default();
    /// ```
    #[clippy::version = "1.79.0"]
    pub MANUAL_UNWRAP_OR_DEFAULT,
    suspicious,
    "check if a `match` or `if let` can be simplified with `unwrap_or_default`"
}

declare_lint_pass!(ManualUnwrapOrDefault => [MANUAL_UNWRAP_OR_DEFAULT]);

fn get_some<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'tcx>) -> Option<HirId> {
    if let PatKind::TupleStruct(QPath::Resolved(_, path), &[pat], _) = pat.kind
        && let PatKind::Binding(_, pat_id, _, _) = pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        // Since it comes from a pattern binding, we need to get the parent to actually match
        // against it.
        && let Some(def_id) = cx.tcx.opt_parent(def_id)
        && (cx.tcx.lang_items().get(LangItem::OptionSome) == Some(def_id)
        || cx.tcx.lang_items().get(LangItem::ResultOk) == Some(def_id))
    {
        Some(pat_id)
    } else {
        None
    }
}

fn get_none<'tcx>(cx: &LateContext<'tcx>, arm: &Arm<'tcx>) -> Option<&'tcx Expr<'tcx>> {
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

#[allow(clippy::needless_pass_by_value)]
fn handle<'tcx>(cx: &LateContext<'tcx>, if_let_or_match: IfLetOrMatch<'tcx>, expr: &'tcx Expr<'tcx>) {
    // Get expr_name ("if let" or "match" depending on kind of expression),  the condition, the body for
    // the some arm, the body for the none arm and the binding id of the some arm
    let (expr_name, condition, body_some, body_none, binding_id) = match if_let_or_match {
        IfLetOrMatch::Match(condition, [arm1, arm2], MatchSource::Normal | MatchSource::ForLoopDesugar)
            // Make sure there are no guards to keep things simple
            if arm1.guard.is_none()
                && arm2.guard.is_none()
                // Get the some and none bodies and the binding id of the some arm
                && let Some(((body_some, binding_id), body_none)) = get_some_and_none_bodies(cx, arm1, arm2) =>
        {
            ("match", condition, body_some, body_none, binding_id)
        },
        IfLetOrMatch::IfLet(condition, pat, if_expr, Some(else_expr), _)
            if let Some(binding_id) = get_some(cx, pat) =>
        {
            ("if let", condition, if_expr, else_expr, binding_id)
        },
        _ => {
            // All other cases (match with number of arms != 2, if let without else, etc.)
            return;
        },
    };

    // We check if the return type of the expression implements Default.
    let expr_type = cx.typeck_results().expr_ty(expr);
    if let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
        && implements_trait(cx, expr_type, default_trait_id, &[])
        // We check if the initial condition implements Default.
        && let Some(condition_ty) = cx.typeck_results().expr_ty(condition).walk().nth(1)
        && let GenericArgKind::Type(condition_ty) = condition_ty.unpack()
        && implements_trait(cx, condition_ty, default_trait_id, &[])
        // We check that the `Some(x) => x` doesn't do anything apart "returning" the value in `Some`.
        && let ExprKind::Path(QPath::Resolved(_, path)) = peel_blocks(body_some).kind
        && let Res::Local(local_id) = path.res
        && local_id == binding_id
        // We now check the `None` arm is calling a method equivalent to `Default::default`.
        && let body_none = peel_blocks(body_none)
        && is_default_equivalent(cx, body_none)
        && let Some(receiver) = Sugg::hir_opt(cx, condition).map(Sugg::maybe_par)
    {
        // Machine applicable only if there are no comments present
        let applicability = if span_contains_comment(cx.sess().source_map(), expr.span) {
            Applicability::MaybeIncorrect
        } else {
            Applicability::MachineApplicable
        };

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
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualUnwrapOrDefault {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some(if_let_or_match) = IfLetOrMatch::parse(cx, expr)
            && !expr.span.from_expansion()
            && !is_in_const_context(cx)
        {
            handle(cx, if_let_or_match, expr);
        }
    }
}
