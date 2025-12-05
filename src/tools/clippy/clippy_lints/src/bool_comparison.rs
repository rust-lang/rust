use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_expn_of, peel_blocks, sym};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions of the form `x == true`,
    /// `x != true` and order comparisons such as `x < true` (or vice versa) and
    /// suggest using the variable directly.
    ///
    /// ### Why is this bad?
    /// Unnecessary code.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if x == true {}
    /// if y == false {}
    /// ```
    /// use `x` directly:
    /// ```rust,ignore
    /// if x {}
    /// if !y {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BOOL_COMPARISON,
    complexity,
    "comparing a variable to a boolean, e.g., `if x == true` or `if x != true`"
}

declare_lint_pass!(BoolComparison => [BOOL_COMPARISON]);

impl<'tcx> LateLintPass<'tcx> for BoolComparison {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if let ExprKind::Binary(Spanned { node, .. }, left_side, right_side) = e.kind
            && is_expn_of(left_side.span, sym::cfg).is_none()
            && is_expn_of(right_side.span, sym::cfg).is_none()
            && cx.typeck_results().expr_ty(left_side).is_bool()
            && cx.typeck_results().expr_ty(right_side).is_bool()
        {
            let ignore_case = None::<(fn(_) -> _, &str)>;
            let ignore_no_literal = None::<(fn(_, _) -> _, &str)>;
            match node {
                BinOpKind::Eq => {
                    let true_case = Some((|h| h, "equality checks against true are unnecessary"));
                    let false_case = Some((
                        |h: Sugg<'tcx>| !h,
                        "equality checks against false can be replaced by a negation",
                    ));
                    check_comparison(cx, e, true_case, false_case, true_case, false_case, ignore_no_literal);
                },
                BinOpKind::Ne => {
                    let true_case = Some((
                        |h: Sugg<'tcx>| !h,
                        "inequality checks against true can be replaced by a negation",
                    ));
                    let false_case = Some((|h| h, "inequality checks against false are unnecessary"));
                    check_comparison(cx, e, true_case, false_case, true_case, false_case, ignore_no_literal);
                },
                BinOpKind::Lt => check_comparison(
                    cx,
                    e,
                    ignore_case,
                    Some((|h| h, "greater than checks against false are unnecessary")),
                    Some((
                        |h: Sugg<'tcx>| !h,
                        "less than comparison against true can be replaced by a negation",
                    )),
                    ignore_case,
                    Some((
                        |l: Sugg<'tcx>, r: Sugg<'tcx>| (!l).bit_and(&r),
                        "order comparisons between booleans can be simplified",
                    )),
                ),
                BinOpKind::Gt => check_comparison(
                    cx,
                    e,
                    Some((
                        |h: Sugg<'tcx>| !h,
                        "less than comparison against true can be replaced by a negation",
                    )),
                    ignore_case,
                    ignore_case,
                    Some((|h| h, "greater than checks against false are unnecessary")),
                    Some((
                        |l: Sugg<'tcx>, r: Sugg<'tcx>| l.bit_and(&(!r)),
                        "order comparisons between booleans can be simplified",
                    )),
                ),
                _ => (),
            }
        }
    }
}

fn check_comparison<'a, 'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    left_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    left_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    right_true: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    right_false: Option<(impl FnOnce(Sugg<'a>) -> Sugg<'a>, &'static str)>,
    no_literal: Option<(impl FnOnce(Sugg<'a>, Sugg<'a>) -> Sugg<'a>, &'static str)>,
) {
    if let ExprKind::Binary(_, left_side, right_side) = e.kind {
        let mut applicability = Applicability::MachineApplicable;
        // Eliminate parentheses in `e` by using the lo pos of lhs and hi pos of rhs,
        // calling `source_callsite` make sure macros are handled correctly, see issue #9907
        let binop_span = left_side.span.source_callsite().to(right_side.span.source_callsite());

        match (fetch_bool_expr(left_side), fetch_bool_expr(right_side)) {
            (Some(true), None) => left_true.map_or((), |(h, m)| {
                suggest_bool_comparison(cx, binop_span, right_side, applicability, m, h);
            }),
            (None, Some(true)) => right_true.map_or((), |(h, m)| {
                suggest_bool_comparison(cx, binop_span, left_side, applicability, m, h);
            }),
            (Some(false), None) => left_false.map_or((), |(h, m)| {
                suggest_bool_comparison(cx, binop_span, right_side, applicability, m, h);
            }),
            (None, Some(false)) => right_false.map_or((), |(h, m)| {
                suggest_bool_comparison(cx, binop_span, left_side, applicability, m, h);
            }),
            (None, None) => no_literal.map_or((), |(h, m)| {
                let left_side = Sugg::hir_with_context(cx, left_side, binop_span.ctxt(), "..", &mut applicability);
                let right_side = Sugg::hir_with_context(cx, right_side, binop_span.ctxt(), "..", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    BOOL_COMPARISON,
                    binop_span,
                    m,
                    "try",
                    h(left_side, right_side).into_string(),
                    applicability,
                );
            }),
            _ => (),
        }
    }
}

fn suggest_bool_comparison<'a, 'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    expr: &Expr<'_>,
    mut app: Applicability,
    message: &'static str,
    conv_hint: impl FnOnce(Sugg<'a>) -> Sugg<'a>,
) {
    let hint = Sugg::hir_with_context(cx, expr, span.ctxt(), "..", &mut app);
    span_lint_and_sugg(
        cx,
        BOOL_COMPARISON,
        span,
        message,
        "try",
        conv_hint(hint).into_string(),
        app,
    );
}

fn fetch_bool_expr(expr: &Expr<'_>) -> Option<bool> {
    if let ExprKind::Lit(lit_ptr) = peel_blocks(expr).kind
        && let LitKind::Bool(value) = lit_ptr.node
    {
        return Some(value);
    }
    None
}
