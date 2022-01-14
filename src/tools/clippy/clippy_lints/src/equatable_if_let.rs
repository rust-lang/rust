use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::implements_trait;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Pat, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
    /// Should be written
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
        PatKind::Slice(_, _, _) | PatKind::Range(_, _, _) | PatKind::Binding(..) | PatKind::Wild | PatKind::Or(_) => {
            false
        },
        PatKind::Struct(_, a, etc) => !etc && a.iter().all(|x| unary_pattern(x.pat)),
        PatKind::Tuple(a, etc) | PatKind::TupleStruct(_, a, etc) => !etc.is_some() && array_rec(a),
        PatKind::Ref(x, _) | PatKind::Box(x) => unary_pattern(x),
        PatKind::Path(_) | PatKind::Lit(_) => true,
    }
}

fn is_structural_partial_eq<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, other: Ty<'tcx>) -> bool {
    if let Some(def_id) = cx.tcx.lang_items().eq_trait() {
        implements_trait(cx, ty, def_id, &[other.into()])
    } else {
        false
    }
}

impl<'tcx> LateLintPass<'tcx> for PatternEquality {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if_chain! {
            if let ExprKind::Let(let_expr) = expr.kind;
            if unary_pattern(let_expr.pat);
            let exp_ty = cx.typeck_results().expr_ty(let_expr.init);
            let pat_ty = cx.typeck_results().pat_ty(let_expr.pat);
            if is_structural_partial_eq(cx, exp_ty, pat_ty);
            then {

                let mut applicability = Applicability::MachineApplicable;
                let pat_str = match let_expr.pat.kind {
                    PatKind::Struct(..) => format!(
                        "({})",
                        snippet_with_context(cx, let_expr.pat.span, expr.span.ctxt(), "..", &mut applicability).0,
                    ),
                    _ => snippet_with_context(cx, let_expr.pat.span, expr.span.ctxt(), "..", &mut applicability).0.to_string(),
                };
                span_lint_and_sugg(
                    cx,
                    EQUATABLE_IF_LET,
                    expr.span,
                    "this pattern matching can be expressed using equality",
                    "try",
                    format!(
                        "{} == {}",
                        snippet_with_context(cx, let_expr.init.span, expr.span.ctxt(), "..", &mut applicability).0,
                        pat_str,
                    ),
                    applicability,
                );
            }
        }
    }
}
