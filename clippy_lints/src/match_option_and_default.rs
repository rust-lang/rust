use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Arm, Expr, ExprKind, HirId, LangItem, MatchSource, Pat, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_default_equivalent;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;

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
    #[clippy::version = "1.78.0"]
    pub MATCH_OPTION_AND_DEFAULT,
    suspicious,
    "check if a `match` or `if let` can be simplified with `unwrap_or_default`"
}

declare_lint_pass!(MatchOptionAndDefault => [MATCH_OPTION_AND_DEFAULT]);

fn get_some<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'tcx>) -> Option<HirId> {
    if let PatKind::TupleStruct(QPath::Resolved(_, path), &[pat], _) = pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        // Since it comes from a pattern binding, we need to get the parent to actually match
        // against it.
        && let Some(def_id) = cx.tcx.opt_parent(def_id)
        && cx.tcx.lang_items().get(LangItem::OptionSome) == Some(def_id)
    {
        let mut bindings = Vec::new();
        pat.each_binding(|_, id, _, _| bindings.push(id));
        if let &[id] = bindings.as_slice() {
            Some(id)
        } else {
            None
        }
    } else {
        None
    }
}

fn get_none<'tcx>(cx: &LateContext<'tcx>, arm: &Arm<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let PatKind::Path(QPath::Resolved(_, path)) = arm.pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        // Since it comes from a pattern binding, we need to get the parent to actually match
        // against it.
        && let Some(def_id) = cx.tcx.opt_parent(def_id)
        && cx.tcx.lang_items().get(LangItem::OptionNone) == Some(def_id)
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

fn handle_match<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    let ExprKind::Match(match_expr, [arm1, arm2], MatchSource::Normal | MatchSource::ForLoopDesugar) = expr.kind else {
        return false;
    };
    // We don't want conditions on the arms to simplify things.
    if arm1.guard.is_none()
        && arm2.guard.is_none()
        // We check that the returned type implements the `Default` trait.
        && let match_ty = cx.typeck_results().expr_ty(expr)
        && let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
        && implements_trait(cx, match_ty, default_trait_id, &[])
        // We now get the bodies for both the `Some` and `None` arms.
        && let Some(((body_some, binding_id), body_none)) = get_some_and_none_bodies(cx, arm1, arm2)
        // We check that the `Some(x) => x` doesn't do anything apart "returning" the value in `Some`.
        && let ExprKind::Path(QPath::Resolved(_, path)) = body_some.peel_blocks().kind
        && let Res::Local(local_id) = path.res
        && local_id == binding_id
        // We now check the `None` arm is calling a method equivalent to `Default::default`.
        && let body_none = body_none.peel_blocks()
        && let ExprKind::Call(_, &[]) = body_none.kind
        && is_default_equivalent(cx, body_none)
        && let Some(match_expr_snippet) = snippet_opt(cx, match_expr.span)
    {
        span_lint_and_sugg(
            cx,
            MATCH_OPTION_AND_DEFAULT,
            expr.span,
            "match can be simplified with `.unwrap_or_default()`",
            "replace it with",
            format!("{match_expr_snippet}.unwrap_or_default()"),
            Applicability::MachineApplicable,
        );
    }
    true
}

fn handle_if_let<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    if let ExprKind::If(cond, if_block, Some(else_expr)) = expr.kind
        && let ExprKind::Let(let_) = cond.kind
        && let ExprKind::Block(_, _) = else_expr.kind
        // We check that the returned type implements the `Default` trait.
        && let match_ty = cx.typeck_results().expr_ty(expr)
        && let Some(default_trait_id) = cx.tcx.get_diagnostic_item(sym::Default)
        && implements_trait(cx, match_ty, default_trait_id, &[])
        && let Some(binding_id) = get_some(cx, let_.pat)
        // We check that the `Some(x) => x` doesn't do anything apart "returning" the value in `Some`.
        && let ExprKind::Path(QPath::Resolved(_, path)) = if_block.peel_blocks().kind
        && let Res::Local(local_id) = path.res
        && local_id == binding_id
        // We now check the `None` arm is calling a method equivalent to `Default::default`.
        && let body_else = else_expr.peel_blocks()
        && let ExprKind::Call(_, &[]) = body_else.kind
        && is_default_equivalent(cx, body_else)
        && let Some(if_let_expr_snippet) = snippet_opt(cx, let_.init.span)
    {
        span_lint_and_sugg(
            cx,
            MATCH_OPTION_AND_DEFAULT,
            expr.span,
            "if let can be simplified with `.unwrap_or_default()`",
            "replace it with",
            format!("{if_let_expr_snippet}.unwrap_or_default()"),
            Applicability::MachineApplicable,
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for MatchOptionAndDefault {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() {
            return;
        }
        if !handle_match(cx, expr) {
            handle_if_let(cx, expr);
        }
    }
}
