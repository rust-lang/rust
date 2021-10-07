use super::utils::make_iterator_snippet;
use super::NEVER_LOOP;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::ForLoop;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, HirId, InlineAsmOperand, LoopSource, Node, Pat, Stmt, StmtKind};
use rustc_lint::LateContext;
use std::iter::{once, Iterator};

pub(super) fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let ExprKind::Loop(block, _, source, _) = expr.kind {
        match never_loop_block(block, expr.hir_id) {
            NeverLoopResult::AlwaysBreak => {
                span_lint_and_then(cx, NEVER_LOOP, expr.span, "this loop never actually loops", |diag| {
                    if_chain! {
                        if source == LoopSource::ForLoop;
                        if let Some((_, Node::Expr(parent_match))) = cx.tcx.hir().parent_iter(expr.hir_id).nth(1);
                        if let Some(ForLoop { arg: iterator, pat, span: for_span, .. }) = ForLoop::hir(parent_match);
                        then {
                            // Suggests using an `if let` instead. This is `Unspecified` because the
                            // loop may (probably) contain `break` statements which would be invalid
                            // in an `if let`.
                            diag.span_suggestion_verbose(
                                for_span.with_hi(iterator.span.hi()),
                                "if you need the first element of the iterator, try writing",
                                for_to_if_let_sugg(cx, iterator, pat),
                                Applicability::Unspecified,
                            );
                        }
                    };
                });
            },
            NeverLoopResult::MayContinueMainLoop | NeverLoopResult::Otherwise => (),
        }
    }
}

enum NeverLoopResult {
    // A break/return always get triggered but not necessarily for the main loop.
    AlwaysBreak,
    // A continue may occur for the main loop.
    MayContinueMainLoop,
    Otherwise,
}

#[must_use]
fn absorb_break(arg: &NeverLoopResult) -> NeverLoopResult {
    match *arg {
        NeverLoopResult::AlwaysBreak | NeverLoopResult::Otherwise => NeverLoopResult::Otherwise,
        NeverLoopResult::MayContinueMainLoop => NeverLoopResult::MayContinueMainLoop,
    }
}

// Combine two results for parts that are called in order.
#[must_use]
fn combine_seq(first: NeverLoopResult, second: NeverLoopResult) -> NeverLoopResult {
    match first {
        NeverLoopResult::AlwaysBreak | NeverLoopResult::MayContinueMainLoop => first,
        NeverLoopResult::Otherwise => second,
    }
}

// Combine two results where both parts are called but not necessarily in order.
#[must_use]
fn combine_both(left: NeverLoopResult, right: NeverLoopResult) -> NeverLoopResult {
    match (left, right) {
        (NeverLoopResult::MayContinueMainLoop, _) | (_, NeverLoopResult::MayContinueMainLoop) => {
            NeverLoopResult::MayContinueMainLoop
        },
        (NeverLoopResult::AlwaysBreak, _) | (_, NeverLoopResult::AlwaysBreak) => NeverLoopResult::AlwaysBreak,
        (NeverLoopResult::Otherwise, NeverLoopResult::Otherwise) => NeverLoopResult::Otherwise,
    }
}

// Combine two results where only one of the part may have been executed.
#[must_use]
fn combine_branches(b1: NeverLoopResult, b2: NeverLoopResult) -> NeverLoopResult {
    match (b1, b2) {
        (NeverLoopResult::AlwaysBreak, NeverLoopResult::AlwaysBreak) => NeverLoopResult::AlwaysBreak,
        (NeverLoopResult::MayContinueMainLoop, _) | (_, NeverLoopResult::MayContinueMainLoop) => {
            NeverLoopResult::MayContinueMainLoop
        },
        (NeverLoopResult::Otherwise, _) | (_, NeverLoopResult::Otherwise) => NeverLoopResult::Otherwise,
    }
}

fn never_loop_block(block: &Block<'_>, main_loop_id: HirId) -> NeverLoopResult {
    let stmts = block.stmts.iter().map(stmt_to_expr);
    let expr = once(block.expr);
    let mut iter = stmts.chain(expr).flatten();
    never_loop_expr_seq(&mut iter, main_loop_id)
}

fn never_loop_expr_seq<'a, T: Iterator<Item = &'a Expr<'a>>>(es: &mut T, main_loop_id: HirId) -> NeverLoopResult {
    es.map(|e| never_loop_expr(e, main_loop_id))
        .fold(NeverLoopResult::Otherwise, combine_seq)
}

fn stmt_to_expr<'tcx>(stmt: &Stmt<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match stmt.kind {
        StmtKind::Semi(e, ..) | StmtKind::Expr(e, ..) => Some(e),
        StmtKind::Local(local) => local.init,
        StmtKind::Item(..) => None,
    }
}

fn never_loop_expr(expr: &Expr<'_>, main_loop_id: HirId) -> NeverLoopResult {
    match expr.kind {
        ExprKind::Box(e)
        | ExprKind::Unary(_, e)
        | ExprKind::Cast(e, _)
        | ExprKind::Type(e, _)
        | ExprKind::Let(_, e, _)
        | ExprKind::Field(e, _)
        | ExprKind::AddrOf(_, _, e)
        | ExprKind::Struct(_, _, Some(e))
        | ExprKind::Repeat(e, _)
        | ExprKind::DropTemps(e) => never_loop_expr(e, main_loop_id),
        ExprKind::Array(es) | ExprKind::MethodCall(_, _, es, _) | ExprKind::Tup(es) => {
            never_loop_expr_all(&mut es.iter(), main_loop_id)
        },
        ExprKind::Call(e, es) => never_loop_expr_all(&mut once(e).chain(es.iter()), main_loop_id),
        ExprKind::Binary(_, e1, e2)
        | ExprKind::Assign(e1, e2, _)
        | ExprKind::AssignOp(_, e1, e2)
        | ExprKind::Index(e1, e2) => never_loop_expr_all(&mut [e1, e2].iter().copied(), main_loop_id),
        ExprKind::Loop(b, _, _, _) => {
            // Break can come from the inner loop so remove them.
            absorb_break(&never_loop_block(b, main_loop_id))
        },
        ExprKind::If(e, e2, e3) => {
            let e1 = never_loop_expr(e, main_loop_id);
            let e2 = never_loop_expr(e2, main_loop_id);
            let e3 = e3
                .as_ref()
                .map_or(NeverLoopResult::Otherwise, |e| never_loop_expr(e, main_loop_id));
            combine_seq(e1, combine_branches(e2, e3))
        },
        ExprKind::Match(e, arms, _) => {
            let e = never_loop_expr(e, main_loop_id);
            if arms.is_empty() {
                e
            } else {
                let arms = never_loop_expr_branch(&mut arms.iter().map(|a| &*a.body), main_loop_id);
                combine_seq(e, arms)
            }
        },
        ExprKind::Block(b, _) => never_loop_block(b, main_loop_id),
        ExprKind::Continue(d) => {
            let id = d
                .target_id
                .expect("target ID can only be missing in the presence of compilation errors");
            if id == main_loop_id {
                NeverLoopResult::MayContinueMainLoop
            } else {
                NeverLoopResult::AlwaysBreak
            }
        },
        ExprKind::Break(_, e) | ExprKind::Ret(e) => e.as_ref().map_or(NeverLoopResult::AlwaysBreak, |e| {
            combine_seq(never_loop_expr(e, main_loop_id), NeverLoopResult::AlwaysBreak)
        }),
        ExprKind::InlineAsm(asm) => asm
            .operands
            .iter()
            .map(|(o, _)| match o {
                InlineAsmOperand::In { expr, .. }
                | InlineAsmOperand::InOut { expr, .. }
                | InlineAsmOperand::Sym { expr } => never_loop_expr(expr, main_loop_id),
                InlineAsmOperand::Out { expr, .. } => never_loop_expr_all(&mut expr.iter(), main_loop_id),
                InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    never_loop_expr_all(&mut once(in_expr).chain(out_expr.iter()), main_loop_id)
                },
                InlineAsmOperand::Const { .. } => NeverLoopResult::Otherwise,
            })
            .fold(NeverLoopResult::Otherwise, combine_both),
        ExprKind::Struct(_, _, None)
        | ExprKind::Yield(_, _)
        | ExprKind::Closure(_, _, _, _, _)
        | ExprKind::LlvmInlineAsm(_)
        | ExprKind::Path(_)
        | ExprKind::ConstBlock(_)
        | ExprKind::Lit(_)
        | ExprKind::Err => NeverLoopResult::Otherwise,
    }
}

fn never_loop_expr_all<'a, T: Iterator<Item = &'a Expr<'a>>>(es: &mut T, main_loop_id: HirId) -> NeverLoopResult {
    es.map(|e| never_loop_expr(e, main_loop_id))
        .fold(NeverLoopResult::Otherwise, combine_both)
}

fn never_loop_expr_branch<'a, T: Iterator<Item = &'a Expr<'a>>>(e: &mut T, main_loop_id: HirId) -> NeverLoopResult {
    e.map(|e| never_loop_expr(e, main_loop_id))
        .fold(NeverLoopResult::AlwaysBreak, combine_branches)
}

fn for_to_if_let_sugg(cx: &LateContext<'_>, iterator: &Expr<'_>, pat: &Pat<'_>) -> String {
    let pat_snippet = snippet(cx, pat.span, "_");
    let iter_snippet = make_iterator_snippet(cx, iterator, &mut Applicability::Unspecified);

    format!(
        "if let Some({pat}) = {iter}.next()",
        pat = pat_snippet,
        iter = iter_snippet
    )
}
