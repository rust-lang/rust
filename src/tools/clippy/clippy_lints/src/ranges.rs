use if_chain::if_chain;
use rustc_ast::ast::RangeLimits;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

use crate::utils::sugg::Sugg;
use crate::utils::{higher, SpanlessEq};
use crate::utils::{is_integer_const, snippet, snippet_opt, span_lint, span_lint_and_then};

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
    /// # let x = vec![1];
    /// x.iter().zip(0..x.len());
    /// ```
    /// Could be written as
    /// ```rust
    /// # let x = vec![1];
    /// x.iter().enumerate();
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
    /// Also in many cases, inclusive ranges are still slower to run than
    /// exclusive ranges, because they essentially add an extra branch that
    /// LLVM may fail to hoist out of the loop.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// for x..(y+1) { .. }
    /// ```
    /// Could be written as
    /// ```rust,ignore
    /// for x..=y { .. }
    /// ```
    pub RANGE_PLUS_ONE,
    pedantic,
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
    /// ```rust,ignore
    /// for x..=(y-1) { .. }
    /// ```
    /// Could be written as
    /// ```rust,ignore
    /// for x..y { .. }
    /// ```
    pub RANGE_MINUS_ONE,
    complexity,
    "`x..=(y-1)` reads better as `x..y`"
}

declare_lint_pass!(Ranges => [
    RANGE_ZIP_WITH_LEN,
    RANGE_PLUS_ONE,
    RANGE_MINUS_ONE
]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Ranges {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(ref path, _, ref args) = expr.kind {
            let name = path.ident.as_str();
            if name == "zip" && args.len() == 2 {
                let iter = &args[0].kind;
                let zip_arg = &args[1];
                if_chain! {
                    // `.iter()` call
                    if let ExprKind::MethodCall(ref iter_path, _, ref iter_args ) = *iter;
                    if iter_path.ident.name == sym!(iter);
                    // range expression in `.zip()` call: `0..x.len()`
                    if let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::range(cx, zip_arg);
                    if is_integer_const(cx, start, 0);
                    // `.len()` call
                    if let ExprKind::MethodCall(ref len_path, _, ref len_args) = end.kind;
                    if len_path.ident.name == sym!(len) && len_args.len() == 1;
                    // `.iter()` and `.len()` called on same `Path`
                    if let ExprKind::Path(QPath::Resolved(_, ref iter_path)) = iter_args[0].kind;
                    if let ExprKind::Path(QPath::Resolved(_, ref len_path)) = len_args[0].kind;
                    if SpanlessEq::new(cx).eq_path_segments(&iter_path.segments, &len_path.segments);
                     then {
                         span_lint(cx,
                                   RANGE_ZIP_WITH_LEN,
                                   expr.span,
                                   &format!("It is more idiomatic to use `{}.iter().enumerate()`",
                                            snippet(cx, iter_args[0].span, "_")));
                    }
                }
            }
        }

        check_exclusive_range_plus_one(cx, expr);
        check_inclusive_range_minus_one(cx, expr);
    }
}

// exclusive range plus one: `x..(y+1)`
fn check_exclusive_range_plus_one(cx: &LateContext<'_, '_>, expr: &Expr<'_>) {
    if_chain! {
        if let Some(higher::Range {
            start,
            end: Some(end),
            limits: RangeLimits::HalfOpen
        }) = higher::range(cx, expr);
        if let Some(y) = y_plus_one(cx, end);
        then {
            let span = if expr.span.from_expansion() {
                expr.span
                    .ctxt()
                    .outer_expn_data()
                    .call_site
            } else {
                expr.span
            };
            span_lint_and_then(
                cx,
                RANGE_PLUS_ONE,
                span,
                "an inclusive range would be more readable",
                |diag| {
                    let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").to_string());
                    let end = Sugg::hir(cx, y, "y");
                    if let Some(is_wrapped) = &snippet_opt(cx, span) {
                        if is_wrapped.starts_with('(') && is_wrapped.ends_with(')') {
                            diag.span_suggestion(
                                span,
                                "use",
                                format!("({}..={})", start, end),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            diag.span_suggestion(
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
}

// inclusive range minus one: `x..=(y-1)`
fn check_inclusive_range_minus_one(cx: &LateContext<'_, '_>, expr: &Expr<'_>) {
    if_chain! {
        if let Some(higher::Range { start, end: Some(end), limits: RangeLimits::Closed }) = higher::range(cx, expr);
        if let Some(y) = y_minus_one(cx, end);
        then {
            span_lint_and_then(
                cx,
                RANGE_MINUS_ONE,
                expr.span,
                "an exclusive range would be more readable",
                |diag| {
                    let start = start.map_or(String::new(), |x| Sugg::hir(cx, x, "x").to_string());
                    let end = Sugg::hir(cx, y, "y");
                    diag.span_suggestion(
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

fn y_plus_one<'t>(cx: &LateContext<'_, '_>, expr: &'t Expr<'_>) -> Option<&'t Expr<'t>> {
    match expr.kind {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Add, ..
            },
            ref lhs,
            ref rhs,
        ) => {
            if is_integer_const(cx, lhs, 1) {
                Some(rhs)
            } else if is_integer_const(cx, rhs, 1) {
                Some(lhs)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn y_minus_one<'t>(cx: &LateContext<'_, '_>, expr: &'t Expr<'_>) -> Option<&'t Expr<'t>> {
    match expr.kind {
        ExprKind::Binary(
            Spanned {
                node: BinOpKind::Sub, ..
            },
            ref lhs,
            ref rhs,
        ) if is_integer_const(cx, rhs, 1) => Some(lhs),
        _ => None,
    }
}
