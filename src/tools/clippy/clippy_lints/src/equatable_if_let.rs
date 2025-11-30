use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_const_context;
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
fn is_unary_pattern(pat: &Pat<'_>) -> bool {
    fn array_rec(pats: &[Pat<'_>]) -> bool {
        pats.iter().all(is_unary_pattern)
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
        PatKind::Struct(_, a, etc) => etc.is_none() && a.iter().all(|x| is_unary_pattern(x.pat)),
        PatKind::Tuple(a, etc) | PatKind::TupleStruct(_, a, etc) => etc.as_opt_usize().is_none() && array_rec(a),
        PatKind::Ref(x, _, _) | PatKind::Box(x) | PatKind::Deref(x) | PatKind::Guard(x, _) => is_unary_pattern(x),
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
            && is_unary_pattern(let_expr.pat)
            && !expr.span.in_external_macro(cx.sess().source_map())
            && !let_expr.pat.span.from_expansion()
            && !let_expr.init.span.from_expansion()
        {
            let exp_ty = cx.typeck_results().expr_ty(let_expr.init);
            let pat_ty = cx.typeck_results().pat_ty(let_expr.pat);

            let mut app = Applicability::MachineApplicable;
            let ctxt = expr.span.ctxt();

            if is_structural_partial_eq(cx, exp_ty, pat_ty)
                && !contains_type_mismatch(cx, let_expr.pat)
                // Calls to trait methods (`PartialEq::eq` in this case) aren't stable yet. We could _technically_
                // try looking at whether:
                // 1) features `const_trait_impl` and `const_cmp` are enabled
                // 2) implementation of `PartialEq<Rhs=PatTy> for ExpTy` has `fn eq` that is `const`
                //
                // but that didn't quite work out (see #15482), so we just reject outright in this case
                && !is_in_const_context(cx)
            {
                span_lint_and_then(
                    cx,
                    EQUATABLE_IF_LET,
                    expr.span,
                    "this pattern matching can be expressed using equality",
                    |diag| {
                        let pat_str = {
                            let str = snippet_with_context(cx, let_expr.pat.span, ctxt, "..", &mut app).0;
                            if let PatKind::Struct(..) = let_expr.pat.kind {
                                format!("({str})").into()
                            } else {
                                str
                            }
                        };

                        let sugg = format!(
                            "{} == {pat_str}",
                            snippet_with_context(cx, let_expr.init.span, ctxt, "..", &mut app).0,
                        );
                        diag.span_suggestion(expr.span, "try", sugg, app);
                    },
                );
            } else {
                span_lint_and_then(
                    cx,
                    EQUATABLE_IF_LET,
                    expr.span,
                    "this pattern matching can be expressed using `matches!`",
                    |diag| {
                        let sugg = format!(
                            "matches!({}, {})",
                            snippet_with_context(cx, let_expr.init.span, ctxt, "..", &mut app).0,
                            snippet_with_context(cx, let_expr.pat.span, ctxt, "..", &mut app).0,
                        );
                        diag.span_suggestion(expr.span, "try", sugg, app);
                    },
                );
            }
        }
    }
}
