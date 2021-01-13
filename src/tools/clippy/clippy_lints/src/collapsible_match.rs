use crate::utils::visitors::LocalUsedVisitor;
use crate::utils::{span_lint_and_then, SpanlessEq};
use if_chain::if_chain;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{Arm, Expr, ExprKind, Guard, HirId, Pat, PatKind, QPath, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{DefIdTree, TyCtxt};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{MultiSpan, Span};

declare_clippy_lint! {
    /// **What it does:** Finds nested `match` or `if let` expressions where the patterns may be "collapsed" together
    /// without adding any branches.
    ///
    /// Note that this lint is not intended to find _all_ cases where nested match patterns can be merged, but only
    /// cases where merging would most likely make the code more readable.
    ///
    /// **Why is this bad?** It is unnecessarily verbose and complex.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn func(opt: Option<Result<u64, String>>) {
    ///     let n = match opt {
    ///         Some(n) => match n {
    ///             Ok(n) => n,
    ///             _ => return,
    ///         }
    ///         None => return,
    ///     };
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn func(opt: Option<Result<u64, String>>) {
    ///     let n = match opt {
    ///         Some(Ok(n)) => n,
    ///         _ => return,
    ///     };
    /// }
    /// ```
    pub COLLAPSIBLE_MATCH,
    style,
    "Nested `match` or `if let` expressions where the patterns may be \"collapsed\" together."
}

declare_lint_pass!(CollapsibleMatch => [COLLAPSIBLE_MATCH]);

impl<'tcx> LateLintPass<'tcx> for CollapsibleMatch {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::Match(_expr, arms, _source) = expr.kind {
            if let Some(wild_arm) = arms.iter().rfind(|arm| arm_is_wild_like(arm, cx.tcx)) {
                for arm in arms {
                    check_arm(arm, wild_arm, cx);
                }
            }
        }
    }
}

fn check_arm(arm: &Arm<'_>, wild_outer_arm: &Arm<'_>, cx: &LateContext<'_>) {
    if_chain! {
        let expr = strip_singleton_blocks(arm.body);
        if let ExprKind::Match(expr_in, arms_inner, _) = expr.kind;
        // the outer arm pattern and the inner match
        if expr_in.span.ctxt() == arm.pat.span.ctxt();
        // there must be no more than two arms in the inner match for this lint
        if arms_inner.len() == 2;
        // no if guards on the inner match
        if arms_inner.iter().all(|arm| arm.guard.is_none());
        // match expression must be a local binding
        // match <local> { .. }
        if let ExprKind::Path(QPath::Resolved(None, path)) = expr_in.kind;
        if let Res::Local(binding_id) = path.res;
        // one of the branches must be "wild-like"
        if let Some(wild_inner_arm_idx) = arms_inner.iter().rposition(|arm_inner| arm_is_wild_like(arm_inner, cx.tcx));
        let (wild_inner_arm, non_wild_inner_arm) =
            (&arms_inner[wild_inner_arm_idx], &arms_inner[1 - wild_inner_arm_idx]);
        if !pat_contains_or(non_wild_inner_arm.pat);
        // the binding must come from the pattern of the containing match arm
        // ..<local>.. => match <local> { .. }
        if let Some(binding_span) = find_pat_binding(arm.pat, binding_id);
        // the "wild-like" branches must be equal
        if SpanlessEq::new(cx).eq_expr(wild_inner_arm.body, wild_outer_arm.body);
        // the binding must not be used in the if guard
        if !matches!(arm.guard, Some(Guard::If(guard)) if LocalUsedVisitor::new(binding_id).check_expr(guard));
        // ...or anywhere in the inner match
        if !arms_inner.iter().any(|arm| LocalUsedVisitor::new(binding_id).check_arm(arm));
        then {
            span_lint_and_then(
                cx,
                COLLAPSIBLE_MATCH,
                expr.span,
                "Unnecessary nested match",
                |diag| {
                    let mut help_span = MultiSpan::from_spans(vec![binding_span, non_wild_inner_arm.pat.span]);
                    help_span.push_span_label(binding_span, "Replace this binding".into());
                    help_span.push_span_label(non_wild_inner_arm.pat.span, "with this pattern".into());
                    diag.span_help(help_span, "The outer pattern can be modified to include the inner pattern.");
                },
            );
        }
    }
}

fn strip_singleton_blocks<'hir>(mut expr: &'hir Expr<'hir>) -> &'hir Expr<'hir> {
    while let ExprKind::Block(block, _) = expr.kind {
        match (block.stmts, block.expr) {
            ([stmt], None) => match stmt.kind {
                StmtKind::Expr(e) | StmtKind::Semi(e) => expr = e,
                _ => break,
            },
            ([], Some(e)) => expr = e,
            _ => break,
        }
    }
    expr
}

/// A "wild-like" pattern is wild ("_") or `None`.
/// For this lint to apply, both the outer and inner match expressions
/// must have "wild-like" branches that can be combined.
fn arm_is_wild_like(arm: &Arm<'_>, tcx: TyCtxt<'_>) -> bool {
    if arm.guard.is_some() {
        return false;
    }
    match arm.pat.kind {
        PatKind::Binding(..) | PatKind::Wild => true,
        PatKind::Path(QPath::Resolved(None, path)) if is_none_ctor(path.res, tcx) => true,
        _ => false,
    }
}

fn find_pat_binding(pat: &Pat<'_>, hir_id: HirId) -> Option<Span> {
    let mut span = None;
    pat.walk_short(|p| match &p.kind {
        // ignore OR patterns
        PatKind::Or(_) => false,
        PatKind::Binding(_bm, _, _ident, _) => {
            let found = p.hir_id == hir_id;
            if found {
                span = Some(p.span);
            }
            !found
        },
        _ => true,
    });
    span
}

fn pat_contains_or(pat: &Pat<'_>) -> bool {
    let mut result = false;
    pat.walk(|p| {
        let is_or = matches!(p.kind, PatKind::Or(_));
        result |= is_or;
        !is_or
    });
    result
}

fn is_none_ctor(res: Res, tcx: TyCtxt<'_>) -> bool {
    if let Some(none_id) = tcx.lang_items().option_none_variant() {
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), id) = res {
            if let Some(variant_id) = tcx.parent(id) {
                return variant_id == none_id;
            }
        }
    }
    false
}
