use super::utils::make_iterator_snippet;
use super::NEVER_LOOP;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::ForLoop;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::{Block, Destination, Expr, ExprKind, HirId, InlineAsmOperand, Pat, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_span::Span;
use std::iter::{once, Iterator};

pub(super) fn check(
    cx: &LateContext<'_>,
    block: &Block<'_>,
    loop_id: HirId,
    span: Span,
    for_loop: Option<&ForLoop<'_>>,
) {
    match never_loop_block(block, &mut Vec::new(), loop_id) {
        NeverLoopResult::AlwaysBreak => {
            span_lint_and_then(cx, NEVER_LOOP, span, "this loop never actually loops", |diag| {
                if let Some(ForLoop {
                    arg: iterator,
                    pat,
                    span: for_span,
                    ..
                }) = for_loop
                {
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
            });
        },
        NeverLoopResult::MayContinueMainLoop | NeverLoopResult::Otherwise => (),
    }
}

#[derive(Copy, Clone)]
enum NeverLoopResult {
    // A break/return always get triggered but not necessarily for the main loop.
    AlwaysBreak,
    // A continue may occur for the main loop.
    MayContinueMainLoop,
    Otherwise,
}

#[must_use]
fn absorb_break(arg: NeverLoopResult) -> NeverLoopResult {
    match arg {
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

fn never_loop_block(block: &Block<'_>, ignore_ids: &mut Vec<HirId>, main_loop_id: HirId) -> NeverLoopResult {
    let iter = block
        .stmts
        .iter()
        .filter_map(stmt_to_expr)
        .chain(block.expr.map(|expr| (expr, None)));

    iter.map(|(e, els)| {
        let e = never_loop_expr(e, ignore_ids, main_loop_id);
        // els is an else block in a let...else binding
        els.map_or(e, |els| {
            combine_branches(e, never_loop_block(els, ignore_ids, main_loop_id))
        })
    })
    .fold(NeverLoopResult::Otherwise, combine_seq)
}

fn stmt_to_expr<'tcx>(stmt: &Stmt<'tcx>) -> Option<(&'tcx Expr<'tcx>, Option<&'tcx Block<'tcx>>)> {
    match stmt.kind {
        StmtKind::Semi(e) | StmtKind::Expr(e) => Some((e, None)),
        // add the let...else expression (if present)
        StmtKind::Local(local) => local.init.map(|init| (init, local.els)),
        StmtKind::Item(..) => None,
    }
}

#[allow(clippy::too_many_lines)]
fn never_loop_expr(expr: &Expr<'_>, ignore_ids: &mut Vec<HirId>, main_loop_id: HirId) -> NeverLoopResult {
    match expr.kind {
        ExprKind::Box(e)
        | ExprKind::Unary(_, e)
        | ExprKind::Cast(e, _)
        | ExprKind::Type(e, _)
        | ExprKind::Field(e, _)
        | ExprKind::AddrOf(_, _, e)
        | ExprKind::Repeat(e, _)
        | ExprKind::DropTemps(e) => never_loop_expr(e, ignore_ids, main_loop_id),
        ExprKind::Let(let_expr) => never_loop_expr(let_expr.init, ignore_ids, main_loop_id),
        ExprKind::Array(es) | ExprKind::Tup(es) => never_loop_expr_all(&mut es.iter(), ignore_ids, main_loop_id),
        ExprKind::MethodCall(_, receiver, es, _) => never_loop_expr_all(
            &mut std::iter::once(receiver).chain(es.iter()),
            ignore_ids,
            main_loop_id,
        ),
        ExprKind::Struct(_, fields, base) => {
            let fields = never_loop_expr_all(&mut fields.iter().map(|f| f.expr), ignore_ids, main_loop_id);
            if let Some(base) = base {
                combine_both(fields, never_loop_expr(base, ignore_ids, main_loop_id))
            } else {
                fields
            }
        },
        ExprKind::Call(e, es) => never_loop_expr_all(&mut once(e).chain(es.iter()), ignore_ids, main_loop_id),
        ExprKind::Binary(_, e1, e2)
        | ExprKind::Assign(e1, e2, _)
        | ExprKind::AssignOp(_, e1, e2)
        | ExprKind::Index(e1, e2) => never_loop_expr_all(&mut [e1, e2].iter().copied(), ignore_ids, main_loop_id),
        ExprKind::Loop(b, _, _, _) => {
            // Break can come from the inner loop so remove them.
            absorb_break(never_loop_block(b, ignore_ids, main_loop_id))
        },
        ExprKind::If(e, e2, e3) => {
            let e1 = never_loop_expr(e, ignore_ids, main_loop_id);
            let e2 = never_loop_expr(e2, ignore_ids, main_loop_id);
            let e3 = e3.as_ref().map_or(NeverLoopResult::Otherwise, |e| {
                never_loop_expr(e, ignore_ids, main_loop_id)
            });
            combine_seq(e1, combine_branches(e2, e3))
        },
        ExprKind::Match(e, arms, _) => {
            let e = never_loop_expr(e, ignore_ids, main_loop_id);
            if arms.is_empty() {
                e
            } else {
                let arms = never_loop_expr_branch(&mut arms.iter().map(|a| a.body), ignore_ids, main_loop_id);
                combine_seq(e, arms)
            }
        },
        ExprKind::Block(b, l) => {
            if l.is_some() {
                ignore_ids.push(b.hir_id);
            }
            let ret = never_loop_block(b, ignore_ids, main_loop_id);
            ignore_ids.pop();
            ret
        },
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
        // checks if break targets a block instead of a loop
        ExprKind::Break(Destination { target_id: Ok(t), .. }, e) if ignore_ids.contains(&t) => e
            .map_or(NeverLoopResult::Otherwise, |e| {
                never_loop_expr(e, ignore_ids, main_loop_id)
            }),
        ExprKind::Break(_, e) | ExprKind::Ret(e) => e.as_ref().map_or(NeverLoopResult::AlwaysBreak, |e| {
            combine_seq(
                never_loop_expr(e, ignore_ids, main_loop_id),
                NeverLoopResult::AlwaysBreak,
            )
        }),
        ExprKind::InlineAsm(asm) => asm
            .operands
            .iter()
            .map(|(o, _)| match o {
                InlineAsmOperand::In { expr, .. } | InlineAsmOperand::InOut { expr, .. } => {
                    never_loop_expr(expr, ignore_ids, main_loop_id)
                },
                InlineAsmOperand::Out { expr, .. } => {
                    never_loop_expr_all(&mut expr.iter().copied(), ignore_ids, main_loop_id)
                },
                InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => never_loop_expr_all(
                    &mut once(*in_expr).chain(out_expr.iter().copied()),
                    ignore_ids,
                    main_loop_id,
                ),
                InlineAsmOperand::Const { .. }
                | InlineAsmOperand::SymFn { .. }
                | InlineAsmOperand::SymStatic { .. } => NeverLoopResult::Otherwise,
            })
            .fold(NeverLoopResult::Otherwise, combine_both),
        ExprKind::Yield(_, _)
        | ExprKind::Closure { .. }
        | ExprKind::Path(_)
        | ExprKind::ConstBlock(_)
        | ExprKind::Lit(_)
        | ExprKind::Err => NeverLoopResult::Otherwise,
    }
}

fn never_loop_expr_all<'a, T: Iterator<Item = &'a Expr<'a>>>(
    es: &mut T,
    ignore_ids: &mut Vec<HirId>,
    main_loop_id: HirId,
) -> NeverLoopResult {
    es.map(|e| never_loop_expr(e, ignore_ids, main_loop_id))
        .fold(NeverLoopResult::Otherwise, combine_both)
}

fn never_loop_expr_branch<'a, T: Iterator<Item = &'a Expr<'a>>>(
    e: &mut T,
    ignore_ids: &mut Vec<HirId>,
    main_loop_id: HirId,
) -> NeverLoopResult {
    e.map(|e| never_loop_expr(e, ignore_ids, main_loop_id))
        .fold(NeverLoopResult::AlwaysBreak, combine_branches)
}

fn for_to_if_let_sugg(cx: &LateContext<'_>, iterator: &Expr<'_>, pat: &Pat<'_>) -> String {
    let pat_snippet = snippet(cx, pat.span, "_");
    let iter_snippet = make_iterator_snippet(cx, iterator, &mut Applicability::Unspecified);

    format!("if let Some({pat_snippet}) = {iter_snippet}.next()")
}
