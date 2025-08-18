use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline, snippet};
use rustc_ast::ast::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_ast::visit::{Visitor, walk_expr};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `else` blocks that can be removed without changing semantics.
    ///
    /// ### Why is this bad?
    /// The `else` block adds unnecessary indentation and verbosity.
    ///
    /// ### Known problems
    /// Some may prefer to keep the `else` block for clarity.
    ///
    /// ### Example
    /// ```no_run
    /// fn my_func(count: u32) {
    ///     if count == 0 {
    ///         print!("Nothing to do");
    ///         return;
    ///     } else {
    ///         print!("Moving on...");
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn my_func(count: u32) {
    ///     if count == 0 {
    ///         print!("Nothing to do");
    ///         return;
    ///     }
    ///     print!("Moving on...");
    /// }
    /// ```
    #[clippy::version = "1.50.0"]
    pub REDUNDANT_ELSE,
    pedantic,
    "`else` branch that can be removed without changing semantics"
}

declare_lint_pass!(RedundantElse => [REDUNDANT_ELSE]);

impl EarlyLintPass for RedundantElse {
    fn check_stmt(&mut self, cx: &EarlyContext<'_>, stmt: &Stmt) {
        if stmt.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        // Only look at expressions that are a whole statement
        let expr: &Expr = match &stmt.kind {
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr,
            _ => return,
        };
        // if else
        let (mut then, mut els): (&Block, &Expr) = match &expr.kind {
            ExprKind::If(_, then, Some(els)) => (then, els),
            _ => return,
        };
        loop {
            if !BreakVisitor::default().check_block(then) {
                // then block does not always break
                return;
            }
            match &els.kind {
                // else if else
                ExprKind::If(_, next_then, Some(next_els)) => {
                    then = next_then;
                    els = next_els;
                },
                // else if without else
                ExprKind::If(..) => return,
                // done
                _ => break,
            }
        }

        let mut app = Applicability::MachineApplicable;
        if let ExprKind::Block(block, _) = &els.kind {
            for stmt in &block.stmts {
                // If the `else` block contains a local binding or a macro invocation, Clippy shouldn't auto-fix it
                if matches!(&stmt.kind, StmtKind::Let(_) | StmtKind::MacCall(_)) {
                    app = Applicability::Unspecified;
                    break;
                }
            }
        }

        // FIXME: The indentation of the suggestion would be the same as the one of the macro invocation in this implementation, see https://github.com/rust-lang/rust-clippy/pull/13936#issuecomment-2569548202
        span_lint_and_sugg(
            cx,
            REDUNDANT_ELSE,
            els.span.with_lo(then.span.hi()),
            "redundant else block",
            "remove the `else` block and move the contents out",
            make_sugg(cx, els.span, "..", Some(expr.span)),
            app,
        );
    }
}

/// Call `check` functions to check if an expression always breaks control flow
#[derive(Default)]
struct BreakVisitor {
    is_break: bool,
}

impl<'ast> Visitor<'ast> for BreakVisitor {
    fn visit_block(&mut self, block: &'ast Block) {
        self.is_break = match block.stmts.as_slice() {
            [.., last] => self.check_stmt(last),
            _ => false,
        };
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.is_break = match expr.kind {
            ExprKind::Break(..) | ExprKind::Continue(..) | ExprKind::Ret(..) => true,
            ExprKind::Match(_, ref arms, _) => arms.iter().all(|arm|
                arm.body.is_none() || arm.body.as_deref().is_some_and(|body| self.check_expr(body))
            ),
            ExprKind::If(_, ref then, Some(ref els)) => self.check_block(then) && self.check_expr(els),
            ExprKind::If(_, _, None)
            // ignore loops for simplicity
            | ExprKind::While(..) | ExprKind::ForLoop { .. } | ExprKind::Loop(..) => false,
            _ => {
                walk_expr(self, expr);
                return;
            },
        };
    }
}

impl BreakVisitor {
    fn check<T>(&mut self, item: T, visit: fn(&mut Self, T)) -> bool {
        visit(self, item);
        std::mem::replace(&mut self.is_break, false)
    }

    fn check_block(&mut self, block: &Block) -> bool {
        self.check(block, Self::visit_block)
    }

    fn check_expr(&mut self, expr: &Expr) -> bool {
        self.check(expr, Self::visit_expr)
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> bool {
        self.check(stmt, Self::visit_stmt)
    }
}

// Extract the inner contents of an `else` block str
// e.g. `{ foo(); bar(); }` -> `foo(); bar();`
fn extract_else_block(mut block: &str) -> String {
    block = block.strip_prefix("{").unwrap_or(block);
    block = block.strip_suffix("}").unwrap_or(block);
    block.trim_end().to_string()
}

fn make_sugg(cx: &EarlyContext<'_>, els_span: Span, default: &str, indent_relative_to: Option<Span>) -> String {
    let extracted = extract_else_block(&snippet(cx, els_span, default));
    let indent = indent_relative_to.and_then(|s| indent_of(cx, s));

    reindent_multiline(&extracted, false, indent)
}
