use clippy_utils::diagnostics::{multispan_sugg_with_applicability, span_lint_and_then};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Suggests moving the semicolon after a block to the inside of the block, after its last
    /// expression.
    ///
    /// ### Why is this bad?
    ///
    /// For consistency it's best to have the semicolon inside/outside the block. Either way is fine
    /// and this lint suggests inside the block.
    /// Take a look at `semicolon_outside_block` for the other alternative.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x) };
    /// ```
    /// Use instead:
    /// ```rust
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x); }
    /// ```
    #[clippy::version = "1.68.0"]
    pub SEMICOLON_INSIDE_BLOCK,
    restriction,
    "add a semicolon inside the block"
}
declare_clippy_lint! {
    /// ### What it does
    ///
    /// Suggests moving the semicolon from a block's final expression outside of the block.
    ///
    /// ### Why is this bad?
    ///
    /// For consistency it's best to have the semicolon inside/outside the block. Either way is fine
    /// and this lint suggests outside the block.
    /// Take a look at `semicolon_inside_block` for the other alternative.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x); }
    /// ```
    /// Use instead:
    /// ```rust
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x) };
    /// ```
    #[clippy::version = "1.68.0"]
    pub SEMICOLON_OUTSIDE_BLOCK,
    restriction,
    "add a semicolon outside the block"
}
impl_lint_pass!(SemicolonBlock => [SEMICOLON_INSIDE_BLOCK, SEMICOLON_OUTSIDE_BLOCK]);

#[derive(Copy, Clone)]
pub struct SemicolonBlock {
    semicolon_inside_block_ignore_singleline: bool,
    semicolon_outside_block_ignore_multiline: bool,
}

impl SemicolonBlock {
    pub fn new(semicolon_inside_block_ignore_singleline: bool, semicolon_outside_block_ignore_multiline: bool) -> Self {
        Self {
            semicolon_inside_block_ignore_singleline,
            semicolon_outside_block_ignore_multiline,
        }
    }

    fn semicolon_inside_block(self, cx: &LateContext<'_>, block: &Block<'_>, tail: &Expr<'_>, semi_span: Span) {
        let insert_span = tail.span.source_callsite().shrink_to_hi();
        let remove_span = semi_span.with_lo(block.span.hi());

        if self.semicolon_inside_block_ignore_singleline && get_line(cx, remove_span) == get_line(cx, insert_span) {
            return;
        }

        span_lint_and_then(
            cx,
            SEMICOLON_INSIDE_BLOCK,
            semi_span,
            "consider moving the `;` inside the block for consistent formatting",
            |diag| {
                multispan_sugg_with_applicability(
                    diag,
                    "put the `;` here",
                    Applicability::MachineApplicable,
                    [(remove_span, String::new()), (insert_span, ";".to_owned())],
                );
            },
        );
    }

    fn semicolon_outside_block(
        self,
        cx: &LateContext<'_>,
        block: &Block<'_>,
        tail_stmt_expr: &Expr<'_>,
        semi_span: Span,
    ) {
        let insert_span = block.span.with_lo(block.span.hi());
        // account for macro calls
        let semi_span = cx.sess().source_map().stmt_span(semi_span, block.span);
        let remove_span = semi_span.with_lo(tail_stmt_expr.span.source_callsite().hi());

        if self.semicolon_outside_block_ignore_multiline && get_line(cx, remove_span) != get_line(cx, insert_span) {
            return;
        }

        span_lint_and_then(
            cx,
            SEMICOLON_OUTSIDE_BLOCK,
            block.span,
            "consider moving the `;` outside the block for consistent formatting",
            |diag| {
                multispan_sugg_with_applicability(
                    diag,
                    "put the `;` here",
                    Applicability::MachineApplicable,
                    [(remove_span, String::new()), (insert_span, ";".to_owned())],
                );
            },
        );
    }
}

impl LateLintPass<'_> for SemicolonBlock {
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        match stmt.kind {
            StmtKind::Expr(Expr {
                kind: ExprKind::Block(block, _),
                ..
            }) if !block.span.from_expansion() => {
                let Block {
                    expr: None,
                    stmts: [.., stmt],
                    ..
                } = block
                else {
                    return;
                };
                let &Stmt {
                    kind: StmtKind::Semi(expr),
                    span,
                    ..
                } = stmt
                else {
                    return;
                };
                self.semicolon_outside_block(cx, block, expr, span);
            },
            StmtKind::Semi(Expr {
                kind: ExprKind::Block(block @ Block { expr: Some(tail), .. }, _),
                ..
            }) if !block.span.from_expansion() => {
                self.semicolon_inside_block(cx, block, tail, stmt.span);
            },
            _ => (),
        }
    }
}

fn get_line(cx: &LateContext<'_>, span: Span) -> Option<usize> {
    if let Ok(line) = cx.sess().source_map().lookup_line(span.lo()) {
        return Some(line.line);
    }

    None
}
