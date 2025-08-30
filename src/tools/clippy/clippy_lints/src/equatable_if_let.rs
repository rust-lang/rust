use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::Ty;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for pattern matchings that can be expressed using equality.
    ///
    /// ### Why is this bad?
    ///
    /// * It reads better and has less cognitive load because equality won't cause binding.
    /// * It is a [Yoda condition](https://en.wikipedia.org/wiki/Yoda_conditions). Yoda conditions are widely
    /// criticized for increasing the cognitive load of reading the code.
    /// * Equality is a simple bool expression and can be merged with `&&` and `||` and
    /// reuse if blocks
    ///
    /// ### Example
    /// ```rust,ignore
    /// if let Some(2) = x {
    ///     do_thing();
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// if x == Some(2) {
    ///     do_thing();
    /// }
    /// ```
    #[clippy::version = "1.57.0"]
    pub EQUATABLE_IF_LET,
    nursery,
    "using pattern matching instead of equality"
}

declare_lint_pass!(PatternEquality => [EQUATABLE_IF_LET]);

/// detects if pattern matches just one thing
fn unary_pattern(pat: &Pat<'_>) -> bool {
    fn array_rec(pats: &[Pat<'_>]) -> bool {
        pats.iter().all(unary_pattern)
    }
    match &pat.kind {
        PatKind::Missing => unreachable!(),
        PatKind::Slice(_, _, _)
        | PatKind::Range(_, _, _)
        | PatKind::Binding(..)
        | PatKind::Wild
        | PatKind::Never
        | PatKind::Or(_)
        | PatKind::Err(_) => false,
        PatKind::Struct(_, a, etc) => etc.is_none() && a.iter().all(|x| unary_pattern(x.pat)),
        PatKind::Tuple(a, etc) | PatKind::TupleStruct(_, a, etc) => etc.as_opt_usize().is_none() && array_rec(a),
        PatKind::Ref(x, _) | PatKind::Box(x) | PatKind::Deref(x) | PatKind::Guard(x, _) => unary_pattern(x),
        PatKind::Expr(_) => true,
    }
}

fn is_structural_partial_eq<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, other: Ty<'tcx>) -> bool {
    if let Some(def_id) = cx.tcx.lang_items().eq_trait() {
        implements_trait(cx, ty, def_id, &[other.into()])
    } else {
        false
    }
}

/// Check if the pattern has any type mismatch that would prevent it from being used in an equality
/// check. This can happen if the expr has a reference type and the corresponding pattern is a
/// literal.
fn contains_type_mismatch(cx: &LateContext<'_>, pat: &Pat<'_>) -> bool {
    let mut result = false;
    pat.walk(|p| {
        if result {
            return false;
        }

        if p.span.in_external_macro(cx.sess().source_map()) {
            return true;
        }

        let adjust_pat = match p.kind {
            PatKind::Or([p, ..]) => p,
            _ => p,
        };

        if let Some(adjustments) = cx.typeck_results().pat_adjustments().get(adjust_pat.hir_id)
            && adjustments.first().is_some_and(|first| first.source.is_ref())
        {
            result = true;
            return false;
        }

        true
    });

    result
}

impl<'tcx> LateLintPass<'tcx> for PatternEquality {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Let(let_expr) = expr.kind
            && unary_pattern(let_expr.pat)
            && !expr.span.in_external_macro(cx.sess().source_map())
        {
            let exp_ty = cx.typeck_results().expr_ty(let_expr.init);
            let pat_ty = cx.typeck_results().pat_ty(let_expr.pat);
            let mut applicability = Applicability::MachineApplicable;

            if is_structural_partial_eq(cx, exp_ty, pat_ty) && !contains_type_mismatch(cx, let_expr.pat) {
                let pat_str = match let_expr.pat.kind {
                    PatKind::Struct(..) => format!(
                        "({})",
                        snippet_with_context(cx, let_expr.pat.span, expr.span.ctxt(), "..", &mut applicability).0,
                    ),
                    _ => snippet_with_context(cx, let_expr.pat.span, expr.span.ctxt(), "..", &mut applicability)
                        .0
                        .to_string(),
                };
                span_lint_and_sugg(
                    cx,
                    EQUATABLE_IF_LET,
                    expr.span,
                    "this pattern matching can be expressed using equality",
                    "try",
                    format!(
                        "{} == {pat_str}",
                        snippet_with_context(cx, let_expr.init.span, expr.span.ctxt(), "..", &mut applicability).0,
                    ),
                    applicability,
                );
            } else {
                span_lint_and_sugg(
                    cx,
                    EQUATABLE_IF_LET,
                    expr.span,
                    "this pattern matching can be expressed using `matches!`",
                    "try",
                    format!(
                        "matches!({}, {})",
                        snippet_with_context(cx, let_expr.init.span, expr.span.ctxt(), "..", &mut applicability).0,
                        snippet_with_context(cx, let_expr.pat.span, expr.span.ctxt(), "..", &mut applicability).0,
                    ),
                    applicability,
                );
            }
        }
    }
}
