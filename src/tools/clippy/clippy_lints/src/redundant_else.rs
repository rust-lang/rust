use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::is_from_proc_macro;
use clippy_utils::source::{SpanExt, indent_of, reindent_multiline};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, MatchSource, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TypeckResults;
use rustc_session::declare_lint_pass;
use rustc_span::{ExpnKind, SyntaxContext};

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
    fn check_block_post(&mut self, cx: &LateContext<'tcx>, b: &'tcx Block<'_>) {
        if let Some(e) = b.expr {
            check(cx, b.span.ctxt(), false, e);
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'tcx>, s: &'tcx Stmt<'_>) {
        if let StmtKind::Expr(e) | StmtKind::Semi(e) = s.kind {
            check(cx, s.span.ctxt(), matches!(s.kind, StmtKind::Expr(_)), e);
        }
    }
}

fn check<'tcx>(cx: &LateContext<'tcx>, ctxt: SyntaxContext, needs_semi: bool, e: &'tcx Expr<'_>) {
    // Find the final `else` block in an `if` chain.
    let mut prev_then = None;
    let mut next = e;
    let (then, else_) = loop {
        match next.kind {
            ExprKind::If(_, then, Some(else_))
                if is_never(cx.typeck_results, ctxt, then)
                    && then.span.ctxt() == ctxt
                    && else_.span.ctxt() == ctxt =>
            {
                prev_then = Some(then);
                next = else_;
            },
            ExprKind::Block(b, _) if let Some(then) = prev_then => break (then, b),
            _ => return,
        }
    };

    if e.span.ctxt() == ctxt
        && !ctxt.in_external_macro(cx.tcx.sess.source_map())
        && !is_from_proc_macro(cx, e)
        && let Some(src) = else_.span.get_text(cx)
        && let Some(src) = src.strip_prefix('{')
        && let Some(src) = src.strip_suffix('}')
        // FIXME(@Jarcho): `indent_of` walks to the root context before getting the indent
        // which gives the wrong result here.
        && let Some(indent) = indent_of(cx, e.span)
    {
        let sp = else_.span.with_lo(then.span.hi());
        span_lint_hir_and_then(cx, REDUNDANT_ELSE, e.hir_id, sp, "redundant else block", |diag| {
            let mut sugg = reindent_multiline(src.trim_end(), false, Some(indent));
            if needs_semi && else_.expr.is_some_and(|e| expr_needs_semi(ctxt, e)) {
                sugg.push(';');
            }
            diag.span_suggestion(
                sp,
                "remove the `else` block and move the contents out",
                sugg,
                if ctxt.is_root() && else_.stmts.iter().all(|s| !matches!(s.kind, StmtKind::Let(_))) {
                    Applicability::MachineApplicable
                } else {
                    Applicability::MaybeIncorrect
                },
            );
        });
    }
}

/// Checks if an expression, viewed from the specified context, needs a trailing semicolon
/// to be parsed as a statement.
fn expr_needs_semi(ctxt: SyntaxContext, e: &Expr<'_>) -> bool {
    match e.kind {
        ExprKind::Block(..) | ExprKind::Loop(..) | ExprKind::Match(..) | ExprKind::If(..) if ctxt == e.span.ctxt() => {
            false
        },
        ExprKind::Loop(..) => {
            let expn = e.span.ctxt().outer_expn_data();
            ctxt != expn.call_site.ctxt() || !matches!(expn.kind, ExpnKind::Desugaring(_))
        },
        ExprKind::Match(_, _, MatchSource::ForLoopDesugar) => ctxt != e.span.ctxt().outer_expn_data().call_site.ctxt(),
        _ => true,
    }
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
