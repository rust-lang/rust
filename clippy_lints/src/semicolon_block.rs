use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::{get_parent_expr_for_hir, get_parent_node};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Node, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// For () returning expressions, check that the semicolon is inside the block.
    ///
    /// ### Why is this bad?
    ///
    /// For consistency it's best to have the semicolon inside/outside the block. Either way is fine and this lint suggests inside the block.
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
    #[clippy::version = "1.66.0"]
    pub SEMICOLON_INSIDE_BLOCK,
    restriction,
    "add a semicolon inside the block"
}
declare_clippy_lint! {
    /// ### What it does
    ///
    /// For () returning expressions, check that the semicolon is outside the block.
    ///
    /// ### Why is this bad?
    ///
    /// For consistency it's best to have the semicolon inside/outside the block. Either way is fine and this lint suggests outside the block.
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
    #[clippy::version = "1.66.0"]
    pub SEMICOLON_OUTSIDE_BLOCK,
    restriction,
    "add a semicolon outside the block"
}
declare_lint_pass!(SemicolonBlock => [SEMICOLON_INSIDE_BLOCK, SEMICOLON_OUTSIDE_BLOCK]);

impl LateLintPass<'_> for SemicolonBlock {
    fn check_block(&mut self, cx: &LateContext<'_>, block: &Block<'_>) {
        semicolon_inside_block(cx, block);
        semicolon_outside_block(cx, block);
    }
}

fn semicolon_inside_block(cx: &LateContext<'_>, block: &Block<'_>) {
    if !block.span.from_expansion()
        && let Some(tail) = block.expr
        && let Some(block_expr @ Expr { kind: ExprKind::Block(_, _), ..}) = get_parent_expr_for_hir(cx, block.hir_id)
        && let Some(Node::Stmt(Stmt { kind: StmtKind::Semi(_), span, .. })) = get_parent_node(cx.tcx, block_expr.hir_id)
    {
        let expr_snip = snippet_with_macro_callsite(cx, tail.span, "..");

        let mut suggestion: String = snippet_with_macro_callsite(cx, block.span, "..").to_string();

        if let Some((expr_offset, _)) = suggestion.rmatch_indices(&*expr_snip).next() {
            suggestion.insert(expr_offset + expr_snip.len(), ';');
        } else {
            return;
        }

        span_lint_and_sugg(
            cx,
            SEMICOLON_INSIDE_BLOCK,
            *span,
            "consider moving the `;` inside the block for consistent formatting",
            "put the `;` here",
            suggestion,
            Applicability::MaybeIncorrect,
        );
    }
}

fn semicolon_outside_block(cx: &LateContext<'_>, block: &Block<'_>) {
    if !block.span.from_expansion()
        && block.expr.is_none()
        && let [.., Stmt { kind: StmtKind::Semi(expr), .. }] = block.stmts
        && let Some(block_expr @ Expr { kind: ExprKind::Block(_, _), ..}) = get_parent_expr_for_hir(cx,block.hir_id)
        && let Some(Node::Stmt(Stmt { kind: StmtKind::Expr(_), .. })) = get_parent_node(cx.tcx, block_expr.hir_id)
    {
        let expr_snip = snippet_with_macro_callsite(cx, expr.span, "..");

        let mut suggestion: String = snippet_with_macro_callsite(cx, block.span, "..").to_string();

        if let Some((expr_offset, _)) = suggestion.rmatch_indices(&*expr_snip).next()
            && let Some(semi_offset) = suggestion[expr_offset + expr_snip.len()..].find(';')
        {
            suggestion.remove(expr_offset +  expr_snip.len() + semi_offset);
        } else {
            return;
        }

        suggestion.push(';');

        span_lint_and_sugg(
            cx,
            SEMICOLON_OUTSIDE_BLOCK,
            block.span,
            "consider moving the `;` outside the block for consistent formatting",
            "put the `;` outside the block",
            suggestion,
            Applicability::MaybeIncorrect,
        );
    }
}
