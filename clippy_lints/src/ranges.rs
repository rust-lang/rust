use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::RangeLimits;
use syntax::source_map::Spanned;

use crate::utils::sugg::Sugg;
use crate::utils::{get_trait_def_id, higher, implements_trait, SpanlessEq};
use crate::utils::{is_integer_literal, paths, snippet, snippet_opt, span_lint, span_lint_and_then};

declare_clippy_lint! {
    /// **What it does:** Checks for calling `.step_by(0)` on iterators,
    /// which never terminates.
    ///
    /// **Why is this bad?** This very much looks like an oversight, since with
    /// `loop { .. }` there is an obvious better way to endlessly loop.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// for x in (5..5).step_by(0) {
    ///     ..
    /// }
    /// ```
    pub ITERATOR_STEP_BY_ZERO,
    correctness,
    "using `Iterator::step_by(0)`, which produces an infinite iterator"
}

declare_clippy_lint! {
    /// **What it does:** Checks for zipping a collection with the range of
    /// `0.._.len()`.
    ///
    /// **Why is this bad?** The code is better expressed with `.enumerate()`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// x.iter().zip(0..x.len())
    /// ```
    pub RANGE_ZIP_WITH_LEN,
    complexity,
    "zipping iterator with a range when `enumerate()` would do"
}

declare_clippy_lint! {
    /// **What it does:** Checks for exclusive ranges where 1 is added to the
    /// upper bound, e.g., `x..(y+1)`.
    ///
    /// **Why is this bad?** The code is more readable with an inclusive range
    /// like `x..=y`.
    ///
    /// **Known problems:** Will add unnecessary pair of parentheses when the
    /// expression is not wrapped in a pair but starts with a opening parenthesis
    /// and ends with a closing one.
    /// I.e., `let _ = (f()+1)..(f()+1)` results in `let _ = ((f()+1)..=f())`.
    ///
    /// **Example:**
    /// ```rust
    /// for x..(y+1) { .. }
    /// ```
    pub RANGE_PLUS_ONE,
    complexity,
    "`x..(y+1)` reads better as `x..=y`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for inclusive ranges where 1 is subtracted from
    /// the upper bound, e.g., `x..=(y-1)`.
    ///
    /// **Why is this bad?** The code is more readable with an exclusive range
    /// like `x..y`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// for x..=(y-1) { .. }
    /// ```
    pub RANGE_MINUS_ONE,
    complexity,
    "`x..=(y-1)` reads better as `x..y`"
}

declare_lint_pass!(Ranges => [
    ITERATOR_STEP_BY_ZERO,
    RANGE_ZIP_WITH_LEN,
    RANGE_PLUS_ONE,
    RANGE_MINUS_ONE
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Ranges {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::MethodCall(ref path, _, ref args) = expr.node {
            let name = path.ident.as_str();

            // Range with step_by(0).
            if name == "step_by" && args.len() == 2 && has_step_by(cx, &args[0]) {
                use crate::consts::{constant, Constant};
                if let Some((Constant::Int(0), _)) = constant(cx, cx.tables, &args[1]) {
                    span_lint(
                        cx,
                        ITERATOR_STEP_BY_ZERO,
                        expr.span,
                        "Iterator::step_by(0) will panic at runtime",
                    );
                }
            } else if name == "zip" && args.len() == 2 {
                let iter = &args[0].node;
                let zip_arg = &args[1];
                if_chain! {
                    // `.iter()` call
                    if let ExprKind::MethodCall(ref iter_path, _, ref iter_args ) = *iter;
                    if iter_path.ident.name == sym!(iter);
                    // range expression in `.zip()` call: `0..x.len()`
                    if let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::range(cx, zip_arg);
                    if is_integer_literal(start, 0);
                    // `.len()` call
                    if let ExprKind::MethodCall(ref len_path, _, ref len_args) = end.node;
                    if len_path.ident.name == sym!(len) && len_args.len() == 1;
                    // `.iter()` and `.len()` called on same `Path`
                    if let ExprKind::Path(QPath::Resolved(_, ref iter_path)) = iter_args[0].node;
                    if let ExprKind::Path(QPath::Resolved(_, ref len_path)) = len_args[0].node;
                    if SpanlessEq::new(cx).eq_path_segments(&iter_path.segments, &len_path.segments);
                     then {
                         span_lint(cx,
                                   RANGE_ZIP_WITH_LEN,
                                   expr.span,
                                   &format!("It is more idiomatic to use {}.iter().enumerate()",
                                            snippet(cx, iter_args[0].span, "_")));
                    }
                }
            }
        }

        // exclusive range plus one: `x..(y+1)`
        if_chain! {
            if let Some(higher::Range {
                start,
                end: Some(end),
                limits: RangeLimits::HalfOpen
            }) = higher::range(cx, expr);
            if let Some(y) = y_plus_one(end);
            then {
                let span = expr.span
                    .ctxt()
                    .outer_expn_info()
                    .map_or(expr.span, |info| info.call_site);
                span_lint_and_then(
                    cx,
                    RANGE_PLUS_ONE,
                    span,
                    "an inclusive range would be more readable",
                    |db| {
                        let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").to_string());
                        let end = Sugg::hir(cx, y, "y");
                        if let Some(is_wrapped) = &snippet_opt(cx, span) {
                            if is_wrapped.starts_with('(') && is_wrapped.ends_with(')') {
                                db.span_suggestion(
                                    span,
                                    "use",
                                    format!("({}..={})", start, end),
                                    Applicability::MaybeIncorrect,
                                );
                            } else {
                                db.span_suggestion(
                                    span,
                                    "use",
                                    format!("{}..={}", start, end),
                                    Applicability::MachineApplicable, // snippet
                                );
                            }
                        }
                    },
                );
            }
        }

        // inclusive range minus one: `x..=(y-1)`
        if_chain! {
            if let Some(higher::Range { start, end: Some(end), limits: RangeLimits::Closed }) = higher::range(cx, expr);
            if let Some(y) = y_minus_one(end);
            then {
                span_lint_and_then(
                    cx,
                    RANGE_MINUS_ONE,
                    expr.span,
                    "an exclusive range would be more readable",
                    |db| {
                        let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").to_string());
                        let end = Sugg::hir(cx, y, "y");
                        db.span_suggestion(
                            expr.span,
                            "use",
                            format!("{}..{}", start, end),
                            Applicability::MachineApplicable, // snippet
                        );
                    },
                );
            }
        }
    }
}

fn has_step_by(cx: &LateContext<'_, '_>, expr: &Expr) -> bool {
    // No need for `walk_ptrs_ty` here because `step_by` moves `self`, so it
    // can't be called on a borrowed range.
    let ty = cx.tables.expr_ty_adjusted(expr);

    get_trait_def_id(cx, &paths::ITERATOR).map_or(false, |iterator_trait| implements_trait(cx, ty, iterator_trait, &[]))
}

fn y_plus_one(expr: &Expr) -> Option<&Expr> {
    match expr.node {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            ref lhs,
            ref rhs,
        ) => {
            if is_integer_literal(lhs, 1) {
                Some(rhs)
            } else if is_integer_literal(rhs, 1) {
                Some(lhs)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn y_minus_one(expr: &Expr) -> Option<&Expr> {
    match expr.node {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Sub, ..
            },
            ref lhs,
            ref rhs,
        ) if is_integer_literal(rhs, 1) => Some(lhs),
        _ => None,
    }
}
