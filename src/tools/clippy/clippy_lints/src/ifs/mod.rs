use clippy_config::Conf;
use clippy_utils::ty::InteriorMut;
use clippy_utils::{if_sequence, is_else_clause, is_lint_allowed};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;

mod branches_sharing_code;
mod if_same_then_else;
mod ifs_same_cond;
mod same_functions_in_if_cond;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for consecutive `if`s with the same condition.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error.
    ///
    /// ### Example
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
    #[clippy::version = "pre 1.29.0"]
    pub IFS_SAME_COND,
    correctness,
    "consecutive `if`s with the same condition"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for consecutive `if`s with the same function call.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error.
    /// Despite the fact that function can have side effects and `if` works as
    /// intended, such an approach is implicit and can be considered a "code smell".
    ///
    /// ### Example
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
    #[clippy::version = "1.41.0"]
    pub SAME_FUNCTIONS_IN_IF_CONDITION,
    pedantic,
    "consecutive `if`s with the same function call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `if/else` with the same body as the *then* part
    /// and the *else* part.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error.
    ///
    /// ### Example
    /// ```ignore
    /// let foo = if … {
    ///     42
    /// } else {
    ///     42
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IF_SAME_THEN_ELSE,
    style,
    "`if` with the same `then` and `else` blocks"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if the `if` and `else` block contain shared code that can be
    /// moved out of the blocks.
    ///
    /// ### Why is this bad?
    /// Duplicate code is less maintainable.
    ///
    /// ### Example
    /// ```ignore
    /// let foo = if … {
    ///     println!("Hello World");
    ///     13
    /// } else {
    ///     println!("Hello World");
    ///     42
    /// };
    /// ```
    ///
    /// Use instead:
    /// ```ignore
    /// println!("Hello World");
    /// let foo = if … {
    ///     13
    /// } else {
    ///     42
    /// };
    /// ```
    #[clippy::version = "1.53.0"]
    pub BRANCHES_SHARING_CODE,
    nursery,
    "`if` statement with shared code in all blocks"
}

pub struct CopyAndPaste<'tcx> {
    interior_mut: InteriorMut<'tcx>,
}

impl<'tcx> CopyAndPaste<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, conf: &'static Conf) -> Self {
        Self {
            interior_mut: InteriorMut::new(tcx, &conf.ignore_interior_mutability),
        }
    }
}

impl_lint_pass!(CopyAndPaste<'_> => [
    IFS_SAME_COND,
    SAME_FUNCTIONS_IN_IF_CONDITION,
    IF_SAME_THEN_ELSE,
    BRANCHES_SHARING_CODE
]);

impl<'tcx> LateLintPass<'tcx> for CopyAndPaste<'tcx> {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion() && matches!(expr.kind, ExprKind::If(..)) && !is_else_clause(cx.tcx, expr) {
            let (conds, blocks) = if_sequence(expr);
            ifs_same_cond::check(cx, &conds, &mut self.interior_mut);
            same_functions_in_if_cond::check(cx, &conds);
            let all_same =
                !is_lint_allowed(cx, IF_SAME_THEN_ELSE, expr.hir_id) && if_same_then_else::check(cx, &conds, &blocks);
            if !all_same && conds.len() != blocks.len() {
                branches_sharing_code::check(cx, &conds, &blocks, expr);
            }
        }
    }
}
