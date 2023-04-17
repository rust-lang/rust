use clippy_utils::diagnostics::{multispan_sugg_with_applicability, span_lint_and_then};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
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
declare_clippy_lint! {
    /// ### What it does
    ///
    /// Suggests moving the semicolon from a block's final expression outside of
    /// the block if it's singleline, and inside the block if it's multiline.
    ///
    /// ### Why is this bad?
    ///
    /// Some may prefer if the  semicolon is outside if a block is only one
    /// expression, as this allows rustfmt to make it singleline. In the case that
    /// it isn't, it should be inside.
    /// Take a look at both `semicolon_inside_block` and `semicolon_outside_block` for alternatives.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x); }
    ///
    /// unsafe {
    ///     let x = 1;
    ///     f(x)
    /// };
    /// ```
    /// Use instead:
    /// ```rust
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x) };
    ///
    /// unsafe {
    ///     let x = 1;
    ///     f(x);
    /// }
    /// ```
    #[clippy::version = "1.68.0"]
    pub SEMICOLON_OUTSIDE_BLOCK_IF_SINGLELINE,
    restriction,
    "add a semicolon inside the block if it's singleline, otherwise outside"
}
declare_lint_pass!(SemicolonBlock => [SEMICOLON_INSIDE_BLOCK, SEMICOLON_OUTSIDE_BLOCK, SEMICOLON_OUTSIDE_BLOCK_IF_SINGLELINE]);

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
                } = block else { return };
                let &Stmt {
                    kind: StmtKind::Semi(expr),
                    span,
                    ..
                } = stmt else { return };
                semicolon_outside_block(cx, block, expr, span);
            },
            StmtKind::Semi(Expr {
                kind: ExprKind::Block(block @ Block { expr: Some(tail), .. }, _),
                ..
            }) if !block.span.from_expansion() => semicolon_inside_block(cx, block, tail, stmt.span),
            _ => (),
        }
    }
}

fn semicolon_inside_block(cx: &LateContext<'_>, block: &Block<'_>, tail: &Expr<'_>, semi_span: Span) {
    let insert_span = tail.span.source_callsite().shrink_to_hi();
    let remove_span = semi_span.with_lo(block.span.hi());

    check_semicolon_outside_block_if_singleline(cx, block, remove_span, insert_span, true, "inside");

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

fn semicolon_outside_block(cx: &LateContext<'_>, block: &Block<'_>, tail_stmt_expr: &Expr<'_>, semi_span: Span) {
    let insert_span = block.span.with_lo(block.span.hi());
    // account for macro calls
    let semi_span = cx.sess().source_map().stmt_span(semi_span, block.span);
    let remove_span = semi_span.with_lo(tail_stmt_expr.span.source_callsite().hi());

    check_semicolon_outside_block_if_singleline(cx, block, remove_span, insert_span, false, "outside");

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

fn check_semicolon_outside_block_if_singleline(
    cx: &LateContext<'_>,
    block: &Block<'_>,
    remove_span: Span,
    insert_span: Span,
    inequality: bool,
    ty: &str,
) {
    let remove_line = cx
        .sess()
        .source_map()
        .lookup_line(remove_span.lo())
        .expect("failed to get `remove_span`'s line")
        .line;
    let insert_line = cx
        .sess()
        .source_map()
        .lookup_line(insert_span.lo())
        .expect("failed to get `insert_span`'s line")
        .line;

    let eq = if inequality {
        remove_line != insert_line
    } else {
        remove_line == insert_line
    };

    if eq {
        span_lint_and_then(
            cx,
            SEMICOLON_OUTSIDE_BLOCK_IF_SINGLELINE,
            block.span,
            &format!("consider moving the `;` {ty} the block for consistent formatting"),
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
