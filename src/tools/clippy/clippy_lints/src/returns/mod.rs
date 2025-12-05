use rustc_hir::intravisit::FnKind;
use rustc_hir::{Block, Body, FnDecl, Stmt};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

mod let_and_return;
mod needless_return;
mod needless_return_with_question_mark;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `let`-bindings, which are subsequently
    /// returned.
    ///
    /// ### Why is this bad?
    /// It is just extraneous code. Remove it to make your code
    /// more rusty.
    ///
    /// ### Known problems
    /// In the case of some temporaries, e.g. locks, eliding the variable binding could lead
    /// to deadlocks. See [this issue](https://github.com/rust-lang/rust/issues/37612).
    /// This could become relevant if the code is later changed to use the code that would have been
    /// bound without first assigning it to a let-binding.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo() -> String {
    ///     let x = String::new();
    ///     x
    /// }
    /// ```
    /// instead, use
    /// ```no_run
    /// fn foo() -> String {
    ///     String::new()
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LET_AND_RETURN,
    style,
    "creating a let-binding and then immediately returning it like `let x = expr; x` at the end of a block"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for return statements at the end of a block.
    ///
    /// ### Why is this bad?
    /// Removing the `return` and semicolon will make the code
    /// more rusty.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo(x: usize) -> usize {
    ///     return x;
    /// }
    /// ```
    /// simplify to
    /// ```no_run
    /// fn foo(x: usize) -> usize {
    ///     x
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_RETURN,
    // This lint requires some special handling in `check_final_expr` for `#[expect]`.
    // This handling needs to be updated if the group gets changed. This should also
    // be caught by tests.
    style,
    "using a return statement like `return expr;` where an expression would suffice"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for return statements on `Err` paired with the `?` operator.
    ///
    /// ### Why is this bad?
    /// The `return` is unnecessary.
    ///
    /// Returns may be used to add attributes to the return expression. Return
    /// statements with attributes are therefore be accepted by this lint.
    ///
    /// ### Example
    /// ```rust,ignore
    /// fn foo(x: usize) -> Result<(), Box<dyn Error>> {
    ///     if x == 0 {
    ///         return Err(...)?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    /// simplify to
    /// ```rust,ignore
    /// fn foo(x: usize) -> Result<(), Box<dyn Error>> {
    ///     if x == 0 {
    ///         Err(...)?;
    ///     }
    ///     Ok(())
    /// }
    /// ```
    /// if paired with `try_err`, use instead:
    /// ```rust,ignore
    /// fn foo(x: usize) -> Result<(), Box<dyn Error>> {
    ///     if x == 0 {
    ///         return Err(...);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub NEEDLESS_RETURN_WITH_QUESTION_MARK,
    style,
    "using a return statement like `return Err(expr)?;` where removing it would suffice"
}

declare_lint_pass!(Return => [LET_AND_RETURN, NEEDLESS_RETURN, NEEDLESS_RETURN_WITH_QUESTION_MARK]);

impl<'tcx> LateLintPass<'tcx> for Return {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        needless_return_with_question_mark::check_stmt(cx, stmt);
    }

    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        let_and_return::check_block(cx, block);
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        sp: Span,
        _: LocalDefId,
    ) {
        needless_return::check_fn(cx, kind, body, sp);
    }
}
