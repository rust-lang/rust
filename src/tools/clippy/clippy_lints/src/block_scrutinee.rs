use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{indent_of, snippet};
use rustc_errors::Applicability;
use rustc_hir::{BlockCheckMode, Expr, ExprKind, LoopSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::edition::Edition;

declare_clippy_lint! {
    /// ### What it does
    /// Warns when a match, if let, or while let scrutinee is wrapped in a block.
    /// This lint only triggers on the 2021 edition and older.
    ///
    /// ### Why is this bad?
    /// It is unusual to write `{ expr }` when you could just have written
    /// `expr`, and it is unlikely that anyone would write that for any reason
    /// other than wanting temporaries in `expr` to be dropped before executing
    /// the body of the `match`/`if let`/`while` statement. However, prior to
    /// the 2024 edition, wrapping the scrutinee in a block did not drop
    /// temporaries before the body executes.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if let Some(x) = { my_function() } { .. }
    /// ```
    #[clippy::version = "1.98.0"]
    pub BLOCK_SCRUTINEE,
    suspicious,
    "warns when the scrutinee is wrapped in a block in older editions"
}

declare_lint_pass!(BlockScrutinee => [BLOCK_SCRUTINEE]);

impl<'tcx> LateLintPass<'tcx> for BlockScrutinee {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if cx.tcx.sess.edition() >= Edition::Edition2024 {
            return;
        }

        let (scrutinee, keyword_fallback) = match expr.kind {
            ExprKind::Match(scrutinee, _, _) => (scrutinee, "`match`"),
            ExprKind::Let(let_expr) => (let_expr.init, "`if let` / `while let`"),
            _ => return,
        };

        if scrutinee.span.from_expansion() || expr.span.from_expansion() {
            return;
        }

        if let ExprKind::Block(block, _) = scrutinee.kind
            && matches!(block.rules, BlockCheckMode::DefaultBlock)
            && block.stmts.is_empty()
            && let Some(inner_expr) = block.expr
        {
            let inner_snippet = snippet(cx, inner_expr.span, "..");

            let main_msg = "this scrutinee is wrapped in a block";

            span_lint_and_then(cx, BLOCK_SCRUTINEE, scrutinee.span, main_msg, |diag| {
                let mut keyword = keyword_fallback;
                let mut outer_span = expr.span;

                if let ExprKind::Let(_) = expr.kind {
                    keyword = "`if let`";
                    for (_, node) in cx.tcx.hir_parent_iter(expr.hir_id) {
                        if let rustc_hir::Node::Expr(e) = node {
                            if let ExprKind::If(..) = e.kind {
                                if keyword == "`if let`" {
                                    outer_span = e.span;
                                }
                            } else if let ExprKind::Loop(_, _, LoopSource::While, _) = e.kind {
                                keyword = "`while let`";
                                outer_span = e.span;
                                break;
                            }
                        } else if matches!(node, rustc_hir::Node::Item(..) | rustc_hir::Node::ImplItem(..)) {
                            break;
                        }
                    }
                }

                diag.note(format!(
                    "temporary values in this block-wrapped scrutinee will be dropped after the body of the {keyword} statement"
                ));

                diag.note("starting with the 2024 edition, temporaries within a block's final expression are dropped immediately at the end of the block");

                let suggestion_msg = format!("to drop temporaries after the surrounding {keyword}, remove the block");

                diag.span_suggestion(
                    scrutinee.span,
                    suggestion_msg,
                    inner_snippet.to_string(),
                    Applicability::MaybeIncorrect,
                );

                let indent = indent_of(cx, outer_span).unwrap_or(0);
                let pad = " ".repeat(indent);

                diag.multipart_suggestion(
                    "to drop temporaries early, move them to a separate local binding (or update your `Cargo.toml` to the 2024 edition)",
                    vec![
                        (outer_span.shrink_to_lo(), format!("let res = {inner_snippet};\n{pad}")),
                        (scrutinee.span, "res".to_string()),
                    ],
                    Applicability::MaybeIncorrect,
                );
            });
        }
    }
}
