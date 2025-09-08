use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Suggests moving the semicolon after a block to the inside of the block, after its last
    /// expression.
    ///
    /// ### Why restrict this?
    /// For consistency it's best to have the semicolon inside/outside the block. Either way is fine
    /// and this lint suggests inside the block.
    /// Take a look at `semicolon_outside_block` for the other alternative.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x) };
    /// ```
    /// Use instead:
    /// ```no_run
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
    /// ### Why restrict this?
    /// For consistency it's best to have the semicolon inside/outside the block. Either way is fine
    /// and this lint suggests outside the block.
    /// Take a look at `semicolon_inside_block` for the other alternative.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// # fn f(_: u32) {}
    /// # let x = 0;
    /// unsafe { f(x); }
    /// ```
    /// Use instead:
    /// ```no_run
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

pub struct SemicolonBlock {
    semicolon_inside_block_ignore_singleline: bool,
    semicolon_outside_block_ignore_multiline: bool,
}

impl SemicolonBlock {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            semicolon_inside_block_ignore_singleline: conf.semicolon_inside_block_ignore_singleline,
            semicolon_outside_block_ignore_multiline: conf.semicolon_outside_block_ignore_multiline,
        }
    }

    fn semicolon_inside_block(&self, cx: &LateContext<'_>, block: &Block<'_>, tail: &Expr<'_>, semi_span: Span) {
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
                diag.multipart_suggestion(
                    "put the `;` here",
                    vec![(remove_span, String::new()), (insert_span, ";".to_owned())],
                    Applicability::MachineApplicable,
                );
            },
        );
    }

    fn semicolon_outside_block(&self, cx: &LateContext<'_>, block: &Block<'_>, tail_stmt_expr: &Expr<'_>) {
        let insert_span = block.span.shrink_to_hi();

        // For macro call semicolon statements (`mac!();`), the statement's span does not actually
        // include the semicolon itself, so use `mac_call_stmt_semi_span`, which finds the semicolon
        // based on a source snippet.
        // (Does not use `stmt_span` as that requires `.from_expansion()` to return true,
        // which is not the case for e.g. `line!();` and `asm!();`)
        let Some(remove_span) = cx
            .sess()
            .source_map()
            .mac_call_stmt_semi_span(tail_stmt_expr.span.source_callsite())
        else {
            return;
        };

        if self.semicolon_outside_block_ignore_multiline && get_line(cx, remove_span) != get_line(cx, insert_span) {
            return;
        }

        span_lint_and_then(
            cx,
            SEMICOLON_OUTSIDE_BLOCK,
            block.span,
            "consider moving the `;` outside the block for consistent formatting",
            |diag| {
                diag.multipart_suggestion(
                    "put the `;` here",
                    vec![(remove_span, String::new()), (insert_span, ";".to_owned())],
                    Applicability::MachineApplicable,
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
            }) if !block.span.from_expansion() && stmt.span.contains(block.span) => {
                if block.expr.is_none()
                    && let [.., stmt] = block.stmts
                    && let StmtKind::Semi(expr) = stmt.kind
                {
                    self.semicolon_outside_block(cx, block, expr);
                }
            },
            StmtKind::Semi(Expr {
                kind: ExprKind::Block(block, _),
                ..
            }) if !block.span.from_expansion() => {
                let attrs = cx.tcx.hir_attrs(stmt.hir_id);
                if !attrs.is_empty() && !cx.tcx.features().stmt_expr_attributes() {
                    return;
                }

                if let Some(tail) = block.expr {
                    self.semicolon_inside_block(cx, block, tail, stmt.span);
                }
            },
            _ => (),
        }
    }
}

fn get_line(cx: &LateContext<'_>, span: Span) -> Option<usize> {
    cx.sess().source_map().lookup_line(span.lo()).ok().map(|line| line.line)
}
