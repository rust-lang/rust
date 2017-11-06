use rustc::lint::*;
use rustc::hir::*;
use syntax::ast::RangeLimits;
use syntax::codemap::Spanned;
use utils::{is_integer_literal, paths, snippet, span_lint, span_lint_and_then};
use utils::{get_trait_def_id, higher, implements_trait};
use utils::sugg::Sugg;

/// **What it does:** Checks for calling `.step_by(0)` on iterators,
/// which never terminates.
///
/// **Why is this bad?** This very much looks like an oversight, since with
/// `loop { .. }` there is an obvious better way to endlessly loop.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for x in (5..5).step_by(0) { .. }
/// ```
declare_lint! {
    pub ITERATOR_STEP_BY_ZERO,
    Warn,
    "using `Iterator::step_by(0)`, which produces an infinite iterator"
}

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
declare_lint! {
    pub RANGE_ZIP_WITH_LEN,
    Warn,
    "zipping iterator with a range when `enumerate()` would do"
}

/// **What it does:** Checks for exclusive ranges where 1 is added to the
/// upper bound, e.g. `x..(y+1)`.
///
/// **Why is this bad?** The code is more readable with an inclusive range
/// like `x..=y`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for x..(y+1) { .. }
/// ```
declare_lint! {
    pub RANGE_PLUS_ONE,
    Allow,
    "`x..(y+1)` reads better as `x..=y`"
}

/// **What it does:** Checks for inclusive ranges where 1 is subtracted from
/// the upper bound, e.g. `x..=(y-1)`.
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
declare_lint! {
    pub RANGE_MINUS_ONE,
    Warn,
    "`x..=(y-1)` reads better as `x..y`"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(ITERATOR_STEP_BY_ZERO, RANGE_ZIP_WITH_LEN, RANGE_PLUS_ONE, RANGE_MINUS_ONE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprMethodCall(ref path, _, ref args) = expr.node {
            let name = path.name.as_str();

            // Range with step_by(0).
            if name == "step_by" && args.len() == 2 && has_step_by(cx, &args[0]) {
                use consts::{constant, Constant};
                use rustc_const_math::ConstInt::Usize;
                if let Some((Constant::Int(Usize(us)), _)) = constant(cx, &args[1]) {
                    if us.as_u64() == 0 {
                        span_lint(
                            cx,
                            ITERATOR_STEP_BY_ZERO,
                            expr.span,
                            "Iterator::step_by(0) will panic at runtime",
                        );
                    }
                }
            } else if name == "zip" && args.len() == 2 {
                let iter = &args[0].node;
                let zip_arg = &args[1];
                if_chain! {
                    // .iter() call
                    if let ExprMethodCall(ref iter_path, _, ref iter_args ) = *iter;
                    if iter_path.name == "iter";
                    // range expression in .zip() call: 0..x.len()
                    if let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::range(zip_arg);
                    if is_integer_literal(start, 0);
                    // .len() call
                    if let ExprMethodCall(ref len_path, _, ref len_args) = end.node;
                    if len_path.name == "len" && len_args.len() == 1;
                    // .iter() and .len() called on same Path
                    if let ExprPath(QPath::Resolved(_, ref iter_path)) = iter_args[0].node;
                    if let ExprPath(QPath::Resolved(_, ref len_path)) = len_args[0].node;
                    if iter_path.segments == len_path.segments;
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

        // exclusive range plus one: x..(y+1)
        if_chain! {
            if let Some(higher::Range { start, end: Some(end), limits: RangeLimits::HalfOpen }) = higher::range(expr);
            if let Some(y) = y_plus_one(end);
            then {
                span_lint_and_then(
                    cx,
                    RANGE_PLUS_ONE,
                    expr.span,
                    "an inclusive range would be more readable",
                    |db| {
                        let start = start.map_or("".to_owned(), |x| Sugg::hir(cx, x, "x").to_string());
                        let end = Sugg::hir(cx, y, "y");
                        db.span_suggestion(expr.span,
                                           "use",
                                           format!("{}..={}", start, end));
                    },
                );
            }
        }

        // inclusive range minus one: x..=(y-1)
        if_chain! {
            if let Some(higher::Range { start, end: Some(end), limits: RangeLimits::Closed }) = higher::range(expr);
            if let Some(y) = y_minus_one(end);
            then {
                span_lint_and_then(
                    cx,
                    RANGE_MINUS_ONE,
                    expr.span,
                    "an exclusive range would be more readable",
                    |db| {
                        let start = start.map_or("".to_owned(), |x| Sugg::hir(cx, x, "x").to_string());
                        let end = Sugg::hir(cx, y, "y");
                        db.span_suggestion(expr.span,
                                           "use",
                                           format!("{}..{}", start, end));
                    },
                );
            }
        }
    }
}

fn has_step_by(cx: &LateContext, expr: &Expr) -> bool {
    // No need for walk_ptrs_ty here because step_by moves self, so it
    // can't be called on a borrowed range.
    let ty = cx.tables.expr_ty_adjusted(expr);

    get_trait_def_id(cx, &paths::ITERATOR).map_or(false, |iterator_trait| implements_trait(cx, ty, iterator_trait, &[]))
}

fn y_plus_one(expr: &Expr) -> Option<&Expr> {
    match expr.node {
        ExprBinary(Spanned { node: BiAdd, .. }, ref lhs, ref rhs) => if is_integer_literal(lhs, 1) {
            Some(rhs)
        } else if is_integer_literal(rhs, 1) {
            Some(lhs)
        } else {
            None
        },
        _ => None,
    }
}

fn y_minus_one(expr: &Expr) -> Option<&Expr> {
    match expr.node {
        ExprBinary(Spanned { node: BiSub, .. }, ref lhs, ref rhs) if is_integer_literal(rhs, 1) => Some(lhs),
        _ => None,
    }
}
