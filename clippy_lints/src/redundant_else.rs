use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::{indent_of, reindent_multiline, snippet};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TypeckResults;
use rustc_session::declare_lint_pass;
use rustc_span::{Span, SyntaxContext};

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

impl<'tcx> LateLintPass<'tcx> for RedundantElse {
    fn check_block_post(&mut self, cx: &LateContext<'tcx>, b: &Block<'_>) {
        if let Some(e) = b.expr {
            check(cx, b.span.ctxt(), e);
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, s: &Stmt<'_>) {
        if let StmtKind::Expr(e) | StmtKind::Semi(e) = s.kind {
            check(cx, s.span.ctxt(), e);
        }
    }
}

fn check(cx: &LateContext<'_>, ctxt: SyntaxContext, e: &Expr<'_>) {
    if e.span.ctxt() != ctxt || ctxt.in_external_macro(cx.tcx.sess.source_map()) {
        return;
    }
    // if else
    let (mut then, mut els) = match e.kind {
        ExprKind::If(_, then, Some(els)) => (then, els),
        _ => return,
    };
    loop {
        if then.span.ctxt() != ctxt || els.span.ctxt() != ctxt || !is_never(cx.typeck_results(), ctxt, then) {
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
        for stmt in block.stmts {
            // If the `else` block contains a local binding, Clippy shouldn't auto-fix it
            if matches!(&stmt.kind, StmtKind::Let(_)) {
                app = Applicability::Unspecified;
                break;
            }
        }
    }

    // FIXME: The indentation of the suggestion would be the same as the one of the macro invocation in this implementation, see https://github.com/rust-lang/rust-clippy/pull/13936#issuecomment-2569548202
    let sp = els.span.with_lo(then.span.hi());
    span_lint_hir_and_then(cx, REDUNDANT_ELSE, e.hir_id, sp, "redundant else block", |diag| {
        diag.span_suggestion(
            sp,
            "remove the `else` block and move the contents out",
            make_sugg(cx, els.span, "..", Some(e.span)),
            app,
        );
    });
}

fn is_never(typeck: &TypeckResults<'_>, ctxt: SyntaxContext, e: &Expr<'_>) -> bool {
    if ctxt.is_root() {
        is_never_root(typeck, e)
    } else {
        is_never_mac(ctxt, e)
    }
}

fn is_never_root(typeck: &TypeckResults<'_>, e: &Expr<'_>) -> bool {
    match e.kind {
        ExprKind::Break(..) | ExprKind::Continue(_) | ExprKind::Ret(_) | ExprKind::Become(..) => true,
        ExprKind::DropTemps(e) => is_never_root(typeck, e),
        ExprKind::Block(b, _)
            if let Some(e) = b.expr
                && !b.targeted_by_break =>
        {
            is_never_root(typeck, e)
        },
        ExprKind::Match(_, arms, _) => arms.iter().all(|a| is_never_root(typeck, a.body)),
        ExprKind::If(_, then, Some(else_)) => is_never_root(typeck, then) && is_never_root(typeck, else_),
        ExprKind::Call(..)
        | ExprKind::MethodCall(..)
        | ExprKind::Binary(..)
        | ExprKind::Unary(..)
        | ExprKind::Block(..)
        | ExprKind::Loop(..)
        | ExprKind::Path(_) => typeck.expr_ty(e).is_never(),
        _ => false,
    }
}

fn is_never_mac(ctxt: SyntaxContext, mut e: &Expr<'_>) -> bool {
    loop {
        let next = match e.kind {
            ExprKind::Break(..) | ExprKind::Continue(_) | ExprKind::Ret(_) | ExprKind::Become(..) => return true,
            ExprKind::DropTemps(e) => e,
            ExprKind::Block(b, _)
                if !b.targeted_by_break
                    && let Some(e) = match (b.expr, b.stmts) {
                        (Some(e), _) => Some(e),
                        (None, [.., s]) if let StmtKind::Expr(e) | StmtKind::Semi(e) = s.kind => Some(e),
                        _ => None,
                    }
                    && ctxt == b.span.ctxt() =>
            {
                e
            },
            ExprKind::Match(_, arms, _) => {
                return arms
                    .iter()
                    .all(|a| ctxt == a.span.ctxt() && ctxt == a.body.span.ctxt() && is_never_mac(ctxt, a.body));
            },
            ExprKind::If(_, then, Some(else_))
                if ctxt == then.span.ctxt() && ctxt == else_.span.ctxt() && is_never_mac(ctxt, then) =>
            {
                else_
            },
            _ => return false,
        };
        if ctxt != next.span.ctxt() {
            return false;
        }
        e = next;
    }
}

// Extract the inner contents of an `else` block str
// e.g. `{ foo(); bar(); }` -> `foo(); bar();`
fn extract_else_block(mut block: &str) -> String {
    block = block.strip_prefix("{").unwrap_or(block);
    block = block.strip_suffix("}").unwrap_or(block);
    block.trim_end().to_string()
}

fn make_sugg(cx: &LateContext<'_>, els_span: Span, default: &str, indent_relative_to: Option<Span>) -> String {
    let extracted = extract_else_block(&snippet(cx, els_span, default));
    let indent = indent_relative_to.and_then(|s| indent_of(cx, s));

    reindent_multiline(&extracted, false, indent)
}
