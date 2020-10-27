use crate::utils::{eq_expr_value, in_macro, search_same, SpanlessEq, SpanlessHash};
use crate::utils::{get_parent_expr, higher, if_sequence, span_lint_and_note};
use rustc_hir::{Block, Expr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for consecutive `if`s with the same condition.
    ///
    /// **Why is this bad?** This is probably a copy & paste error.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    /// ```ignore
    /// if a == b {
    ///     …
    /// } else if a == b {
    ///     …
    /// }
    /// ```
    ///
    /// Note that this lint ignores all conditions with a function call as it could
    /// have side effects:
    ///
    /// ```ignore
    /// if foo() {
    ///     …
    /// } else if foo() { // not linted
    ///     …
    /// }
    /// ```
    pub IFS_SAME_COND,
    correctness,
    "consecutive `if`s with the same condition"
}

declare_clippy_lint! {
    /// **What it does:** Checks for consecutive `if`s with the same function call.
    ///
    /// **Why is this bad?** This is probably a copy & paste error.
    /// Despite the fact that function can have side effects and `if` works as
    /// intended, such an approach is implicit and can be considered a "code smell".
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    /// ```ignore
    /// if foo() == bar {
    ///     …
    /// } else if foo() == bar {
    ///     …
    /// }
    /// ```
    ///
    /// This probably should be:
    /// ```ignore
    /// if foo() == bar {
    ///     …
    /// } else if foo() == baz {
    ///     …
    /// }
    /// ```
    ///
    /// or if the original code was not a typo and called function mutates a state,
    /// consider move the mutation out of the `if` condition to avoid similarity to
    /// a copy & paste error:
    ///
    /// ```ignore
    /// let first = foo();
    /// if first == bar {
    ///     …
    /// } else {
    ///     let second = foo();
    ///     if second == bar {
    ///     …
    ///     }
    /// }
    /// ```
    pub SAME_FUNCTIONS_IN_IF_CONDITION,
    pedantic,
    "consecutive `if`s with the same function call"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `if/else` with the same body as the *then* part
    /// and the *else* part.
    ///
    /// **Why is this bad?** This is probably a copy & paste error.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    /// ```ignore
    /// let foo = if … {
    ///     42
    /// } else {
    ///     42
    /// };
    /// ```
    pub IF_SAME_THEN_ELSE,
    correctness,
    "`if` with the same `then` and `else` blocks"
}

declare_lint_pass!(CopyAndPaste => [IFS_SAME_COND, SAME_FUNCTIONS_IN_IF_CONDITION, IF_SAME_THEN_ELSE]);

impl<'tcx> LateLintPass<'tcx> for CopyAndPaste {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion() {
            // skip ifs directly in else, it will be checked in the parent if
            if let Some(expr) = get_parent_expr(cx, expr) {
                if let Some((_, _, Some(ref else_expr))) = higher::if_block(&expr) {
                    if else_expr.hir_id == expr.hir_id {
                        return;
                    }
                }
            }

            let (conds, blocks) = if_sequence(expr);
            lint_same_then_else(cx, &blocks);
            lint_same_cond(cx, &conds);
            lint_same_fns_in_if_cond(cx, &conds);
        }
    }
}

/// Implementation of `IF_SAME_THEN_ELSE`.
fn lint_same_then_else(cx: &LateContext<'_>, blocks: &[&Block<'_>]) {
    let eq: &dyn Fn(&&Block<'_>, &&Block<'_>) -> bool =
        &|&lhs, &rhs| -> bool { SpanlessEq::new(cx).eq_block(lhs, rhs) };

    if let Some((i, j)) = search_same_sequenced(blocks, eq) {
        span_lint_and_note(
            cx,
            IF_SAME_THEN_ELSE,
            j.span,
            "this `if` has identical blocks",
            Some(i.span),
            "same as this",
        );
    }
}

/// Implementation of `IFS_SAME_COND`.
fn lint_same_cond(cx: &LateContext<'_>, conds: &[&Expr<'_>]) {
    let hash: &dyn Fn(&&Expr<'_>) -> u64 = &|expr| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_expr(expr);
        h.finish()
    };

    let eq: &dyn Fn(&&Expr<'_>, &&Expr<'_>) -> bool = &|&lhs, &rhs| -> bool { eq_expr_value(cx, lhs, rhs) };

    for (i, j) in search_same(conds, hash, eq) {
        span_lint_and_note(
            cx,
            IFS_SAME_COND,
            j.span,
            "this `if` has the same condition as a previous `if`",
            Some(i.span),
            "same as this",
        );
    }
}

/// Implementation of `SAME_FUNCTIONS_IN_IF_CONDITION`.
fn lint_same_fns_in_if_cond(cx: &LateContext<'_>, conds: &[&Expr<'_>]) {
    let hash: &dyn Fn(&&Expr<'_>) -> u64 = &|expr| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_expr(expr);
        h.finish()
    };

    let eq: &dyn Fn(&&Expr<'_>, &&Expr<'_>) -> bool = &|&lhs, &rhs| -> bool {
        // Do not lint if any expr originates from a macro
        if in_macro(lhs.span) || in_macro(rhs.span) {
            return false;
        }
        // Do not spawn warning if `IFS_SAME_COND` already produced it.
        if eq_expr_value(cx, lhs, rhs) {
            return false;
        }
        SpanlessEq::new(cx).eq_expr(lhs, rhs)
    };

    for (i, j) in search_same(conds, hash, eq) {
        span_lint_and_note(
            cx,
            SAME_FUNCTIONS_IN_IF_CONDITION,
            j.span,
            "this `if` has the same function call as a previous `if`",
            Some(i.span),
            "same as this",
        );
    }
}

fn search_same_sequenced<T, Eq>(exprs: &[T], eq: Eq) -> Option<(&T, &T)>
where
    Eq: Fn(&T, &T) -> bool,
{
    for win in exprs.windows(2) {
        if eq(&win[0], &win[1]) {
            return Some((&win[0], &win[1]));
        }
    }
    None
}
