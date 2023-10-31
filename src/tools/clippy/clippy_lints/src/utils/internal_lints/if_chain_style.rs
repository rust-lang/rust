use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::{higher, is_else_clause, is_expn_of};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, HirId, Local, Node, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{BytePos, Span};

declare_clippy_lint! {
    /// Finds unidiomatic usage of `if_chain!`
    pub IF_CHAIN_STYLE,
    internal,
    "non-idiomatic `if_chain!` usage"
}

declare_lint_pass!(IfChainStyle => [IF_CHAIN_STYLE]);

impl<'tcx> LateLintPass<'tcx> for IfChainStyle {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        let (local, after, if_chain_span) = if_chain! {
            if let [Stmt { kind: StmtKind::Local(local), .. }, after @ ..] = block.stmts;
            if let Some(if_chain_span) = is_expn_of(block.span, "if_chain");
            then { (local, after, if_chain_span) } else { return }
        };
        if is_first_if_chain_expr(cx, block.hir_id, if_chain_span) {
            span_lint(
                cx,
                IF_CHAIN_STYLE,
                if_chain_local_span(cx, local, if_chain_span),
                "`let` expression should be above the `if_chain!`",
            );
        } else if local.span.ctxt() == block.span.ctxt() && is_if_chain_then(after, block.expr, if_chain_span) {
            span_lint(
                cx,
                IF_CHAIN_STYLE,
                if_chain_local_span(cx, local, if_chain_span),
                "`let` expression should be inside `then { .. }`",
            );
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        let (cond, then, els) = if let Some(higher::IfOrIfLet { cond, r#else, then }) = higher::IfOrIfLet::hir(expr) {
            (cond, then, r#else.is_some())
        } else {
            return;
        };
        let ExprKind::Block(then_block, _) = then.kind else {
            return;
        };
        let if_chain_span = is_expn_of(expr.span, "if_chain");
        if !els {
            check_nested_if_chains(cx, expr, then_block, if_chain_span);
        }
        let Some(if_chain_span) = if_chain_span else { return };
        // check for `if a && b;`
        if_chain! {
            if let ExprKind::Binary(op, _, _) = cond.kind;
            if op.node == BinOpKind::And;
            if cx.sess().source_map().is_multiline(cond.span);
            then {
                span_lint(cx, IF_CHAIN_STYLE, cond.span, "`if a && b;` should be `if a; if b;`");
            }
        }
        if is_first_if_chain_expr(cx, expr.hir_id, if_chain_span)
            && is_if_chain_then(then_block.stmts, then_block.expr, if_chain_span)
        {
            span_lint(cx, IF_CHAIN_STYLE, expr.span, "`if_chain!` only has one `if`");
        }
    }
}

fn check_nested_if_chains(
    cx: &LateContext<'_>,
    if_expr: &Expr<'_>,
    then_block: &Block<'_>,
    if_chain_span: Option<Span>,
) {
    #[rustfmt::skip]
    let (head, tail) = match *then_block {
        Block { stmts, expr: Some(tail), .. } => (stmts, tail),
        Block {
            stmts: &[
                ref head @ ..,
                Stmt { kind: StmtKind::Expr(tail) | StmtKind::Semi(tail), .. }
            ],
            ..
        } => (head, tail),
        _ => return,
    };
    if_chain! {
        if let Some(higher::IfOrIfLet { r#else: None, .. }) = higher::IfOrIfLet::hir(tail);
        let sm = cx.sess().source_map();
        if head
            .iter()
            .all(|stmt| matches!(stmt.kind, StmtKind::Local(..)) && !sm.is_multiline(stmt.span));
        if if_chain_span.is_some() || !is_else_clause(cx.tcx, if_expr);
        then {
        } else {
            return;
        }
    }
    let (span, msg) = match (if_chain_span, is_expn_of(tail.span, "if_chain")) {
        (None, Some(_)) => (if_expr.span, "this `if` can be part of the inner `if_chain!`"),
        (Some(_), None) => (tail.span, "this `if` can be part of the outer `if_chain!`"),
        (Some(a), Some(b)) if a != b => (b, "this `if_chain!` can be merged with the outer `if_chain!`"),
        _ => return,
    };
    span_lint_and_then(cx, IF_CHAIN_STYLE, span, msg, |diag| {
        let (span, msg) = match head {
            [] => return,
            [stmt] => (stmt.span, "this `let` statement can also be in the `if_chain!`"),
            [a, .., b] => (
                a.span.to(b.span),
                "these `let` statements can also be in the `if_chain!`",
            ),
        };
        diag.span_help(span, msg);
    });
}

fn is_first_if_chain_expr(cx: &LateContext<'_>, hir_id: HirId, if_chain_span: Span) -> bool {
    cx.tcx
        .hir()
        .parent_iter(hir_id)
        .find(|(_, node)| {
            #[rustfmt::skip]
            !matches!(node, Node::Expr(Expr { kind: ExprKind::Block(..), .. }) | Node::Stmt(_))
        })
        .map_or(false, |(id, _)| {
            is_expn_of(cx.tcx.hir().span(id), "if_chain") != Some(if_chain_span)
        })
}

/// Checks a trailing slice of statements and expression of a `Block` to see if they are part
/// of the `then {..}` portion of an `if_chain!`
fn is_if_chain_then(stmts: &[Stmt<'_>], expr: Option<&Expr<'_>>, if_chain_span: Span) -> bool {
    let span = if let [stmt, ..] = stmts {
        stmt.span
    } else if let Some(expr) = expr {
        expr.span
    } else {
        // empty `then {}`
        return true;
    };
    is_expn_of(span, "if_chain").map_or(true, |span| span != if_chain_span)
}

/// Creates a `Span` for `let x = ..;` in an `if_chain!` call.
fn if_chain_local_span(cx: &LateContext<'_>, local: &Local<'_>, if_chain_span: Span) -> Span {
    let mut span = local.pat.span;
    if let Some(init) = local.init {
        span = span.to(init.span);
    }
    span.adjust(if_chain_span.ctxt().outer_expn());
    let sm = cx.sess().source_map();
    let span = sm.span_extend_to_prev_str(span, "let", false, true).unwrap_or(span);
    let span = sm.span_extend_to_next_char(span, ';', false);
    Span::new(
        span.lo() - BytePos(3),
        span.hi() + BytePos(1),
        span.ctxt(),
        span.parent(),
    )
}
