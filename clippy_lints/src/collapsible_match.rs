use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::IfLetOrMatch;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{is_lang_ctor, is_unit_expr, path_to_local, peel_ref_operators, SpanlessEq};
use if_chain::if_chain;
use rustc_hir::LangItem::OptionNone;
use rustc_hir::{Arm, Expr, ExprKind, Guard, HirId, Pat, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{MultiSpan, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Finds nested `match` or `if let` expressions where the patterns may be "collapsed" together
    /// without adding any branches.
    ///
    /// Note that this lint is not intended to find _all_ cases where nested match patterns can be merged, but only
    /// cases where merging would most likely make the code more readable.
    ///
    /// ### Why is this bad?
    /// It is unnecessarily verbose and complex.
    ///
    /// ### Example
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
        match IfLetOrMatch::parse(cx, expr) {
            Some(IfLetOrMatch::Match(_, arms, _)) => {
                if let Some(els_arm) = arms.iter().rfind(|arm| arm_is_wild_like(cx, arm)) {
                    for arm in arms {
                        check_arm(cx, true, arm.pat, arm.body, arm.guard.as_ref(), Some(els_arm.body));
                    }
                }
            },
            Some(IfLetOrMatch::IfLet(_, pat, body, els)) => {
                check_arm(cx, false, pat, body, None, els);
            },
            None => {},
        }
    }
}

fn check_arm<'tcx>(
    cx: &LateContext<'tcx>,
    outer_is_match: bool,
    outer_pat: &'tcx Pat<'tcx>,
    outer_then_body: &'tcx Expr<'tcx>,
    outer_guard: Option<&'tcx Guard<'tcx>>,
    outer_else_body: Option<&'tcx Expr<'tcx>>,
) {
    let inner_expr = strip_singleton_blocks(outer_then_body);
    if_chain! {
        if let Some(inner) = IfLetOrMatch::parse(cx, inner_expr);
        if let Some((inner_scrutinee, inner_then_pat, inner_else_body)) = match inner {
            IfLetOrMatch::IfLet(scrutinee, pat, _, els) => Some((scrutinee, pat, els)),
            IfLetOrMatch::Match(scrutinee, arms, ..) => if_chain! {
                // if there are more than two arms, collapsing would be non-trivial
                if arms.len() == 2 && arms.iter().all(|a| a.guard.is_none());
                // one of the arms must be "wild-like"
                if let Some(wild_idx) = arms.iter().rposition(|a| arm_is_wild_like(cx, a));
                then {
                    let (then, els) = (&arms[1 - wild_idx], &arms[wild_idx]);
                    Some((scrutinee, then.pat, Some(els.body)))
                } else {
                    None
                }
            },
        };
        if outer_pat.span.ctxt() == inner_scrutinee.span.ctxt();
        // match expression must be a local binding
        // match <local> { .. }
        if let Some(binding_id) = path_to_local(peel_ref_operators(cx, inner_scrutinee));
        if !pat_contains_or(inner_then_pat);
        // the binding must come from the pattern of the containing match arm
        // ..<local>.. => match <local> { .. }
        if let Some(binding_span) = find_pat_binding(outer_pat, binding_id);
        // the "else" branches must be equal
        if match (outer_else_body, inner_else_body) {
            (None, None) => true,
            (None, Some(e)) | (Some(e), None) => is_unit_expr(e),
            (Some(a), Some(b)) => SpanlessEq::new(cx).eq_expr(a, b),
        };
        // the binding must not be used in the if guard
        if outer_guard.map_or(true, |(Guard::If(e) | Guard::IfLet(_, e))| !is_local_used(cx, *e, binding_id));
        // ...or anywhere in the inner expression
        if match inner {
            IfLetOrMatch::IfLet(_, _, body, els) => {
                !is_local_used(cx, body, binding_id) && els.map_or(true, |e| !is_local_used(cx, e, binding_id))
            },
            IfLetOrMatch::Match(_, arms, ..) => !arms.iter().any(|arm| is_local_used(cx, arm, binding_id)),
        };
        then {
            let msg = format!(
                "this `{}` can be collapsed into the outer `{}`",
                if matches!(inner, IfLetOrMatch::Match(..)) { "match" } else { "if let" },
                if outer_is_match { "match" } else { "if let" },
            );
            span_lint_and_then(
                cx,
                COLLAPSIBLE_MATCH,
                inner_expr.span,
                &msg,
                |diag| {
                    let mut help_span = MultiSpan::from_spans(vec![binding_span, inner_then_pat.span]);
                    help_span.push_span_label(binding_span, "replace this binding".into());
                    help_span.push_span_label(inner_then_pat.span, "with this pattern".into());
                    diag.span_help(help_span, "the outer pattern can be modified to include the inner pattern");
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

/// A "wild-like" arm has a wild (`_`) or `None` pattern and no guard. Such arms can be "collapsed"
/// into a single wild arm without any significant loss in semantics or readability.
fn arm_is_wild_like(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    if arm.guard.is_some() {
        return false;
    }
    match arm.pat.kind {
        PatKind::Binding(..) | PatKind::Wild => true,
        PatKind::Path(ref qpath) => is_lang_ctor(cx, qpath, OptionNone),
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
