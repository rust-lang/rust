use super::NEVER_LOOP;
use super::utils::make_iterator_snippet;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::ForLoop;
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::source::snippet;
use clippy_utils::visitors::{Descend, for_each_expr_without_closures};
use rustc_errors::Applicability;
use rustc_hir::{Block, Destination, Expr, ExprKind, HirId, InlineAsmOperand, Pat, Stmt, StmtKind, StructTailExpr};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};
use std::iter::once;
use std::ops::ControlFlow;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    block: &Block<'tcx>,
    loop_id: HirId,
    span: Span,
    for_loop: Option<&ForLoop<'_>>,
) {
    match never_loop_block(cx, block, &mut Vec::new(), loop_id) {
        NeverLoopResult::Diverging => {
            span_lint_and_then(cx, NEVER_LOOP, span, "this loop never actually loops", |diag| {
                if let Some(ForLoop {
                    arg: iterator,
                    pat,
                    span: for_span,
                    label,
                    ..
                }) = for_loop
                {
                    // If the block contains a break or continue, or if the loop has a label, `MachineApplicable` is not
                    // appropriate.
                    let app = if !contains_any_break_or_continue(block) && label.is_none() {
                        Applicability::MachineApplicable
                    } else {
                        Applicability::Unspecified
                    };

                    diag.span_suggestion_verbose(
                        for_span.with_hi(iterator.span.hi()),
                        "if you need the first element of the iterator, try writing",
                        for_to_if_let_sugg(cx, iterator, pat),
                        app,
                    );
                }
            });
        },
        NeverLoopResult::MayContinueMainLoop | NeverLoopResult::Normal => (),
    }
}

fn contains_any_break_or_continue(block: &Block<'_>) -> bool {
    for_each_expr_without_closures(block, |e| match e.kind {
        ExprKind::Break(..) | ExprKind::Continue(..) => ControlFlow::Break(()),
        ExprKind::Loop(..) => ControlFlow::Continue(Descend::No),
        _ => ControlFlow::Continue(Descend::Yes),
    })
    .is_some()
}

/// The `never_loop` analysis keeps track of three things:
///
/// * Has any (reachable) code path hit a `continue` of the main loop?
/// * Is the current code path diverging (that is, the next expression is not reachable)
/// * For each block label `'a` inside the main loop, has any (reachable) code path encountered a
///   `break 'a`?
///
/// The first two bits of information are in this enum, and the last part is in the
/// `local_labels` variable, which contains a list of `(block_id, reachable)` pairs ordered by
/// scope.
#[derive(Copy, Clone)]
enum NeverLoopResult {
    /// A continue may occur for the main loop.
    MayContinueMainLoop,
    /// We have not encountered any main loop continue,
    /// but we are diverging (subsequent control flow is not reachable)
    Diverging,
    /// We have not encountered any main loop continue,
    /// and subsequent control flow is (possibly) reachable
    Normal,
}

#[must_use]
fn absorb_break(arg: NeverLoopResult) -> NeverLoopResult {
    match arg {
        NeverLoopResult::Diverging | NeverLoopResult::Normal => NeverLoopResult::Normal,
        NeverLoopResult::MayContinueMainLoop => NeverLoopResult::MayContinueMainLoop,
    }
}

// Combine two results for parts that are called in order.
#[must_use]
fn combine_seq(first: NeverLoopResult, second: impl FnOnce() -> NeverLoopResult) -> NeverLoopResult {
    match first {
        NeverLoopResult::Diverging | NeverLoopResult::MayContinueMainLoop => first,
        NeverLoopResult::Normal => second(),
    }
}

// Combine an iterator of results for parts that are called in order.
#[must_use]
fn combine_seq_many(iter: impl IntoIterator<Item = NeverLoopResult>) -> NeverLoopResult {
    for e in iter {
        if let NeverLoopResult::Diverging | NeverLoopResult::MayContinueMainLoop = e {
            return e;
        }
    }
    NeverLoopResult::Normal
}

// Combine two results where only one of the part may have been executed.
#[must_use]
fn combine_branches(b1: NeverLoopResult, b2: NeverLoopResult) -> NeverLoopResult {
    match (b1, b2) {
        (NeverLoopResult::MayContinueMainLoop, _) | (_, NeverLoopResult::MayContinueMainLoop) => {
            NeverLoopResult::MayContinueMainLoop
        },
        (NeverLoopResult::Normal, _) | (_, NeverLoopResult::Normal) => NeverLoopResult::Normal,
        (NeverLoopResult::Diverging, NeverLoopResult::Diverging) => NeverLoopResult::Diverging,
    }
}

fn never_loop_block<'tcx>(
    cx: &LateContext<'tcx>,
    block: &Block<'tcx>,
    local_labels: &mut Vec<(HirId, bool)>,
    main_loop_id: HirId,
) -> NeverLoopResult {
    let iter = block
        .stmts
        .iter()
        .filter_map(stmt_to_expr)
        .chain(block.expr.map(|expr| (expr, None)));
    combine_seq_many(iter.map(|(e, els)| {
        let e = never_loop_expr(cx, e, local_labels, main_loop_id);
        // els is an else block in a let...else binding
        els.map_or(e, |els| {
            combine_seq(e, || match never_loop_block(cx, els, local_labels, main_loop_id) {
                // Returning MayContinueMainLoop here means that
                // we will not evaluate the rest of the body
                NeverLoopResult::MayContinueMainLoop => NeverLoopResult::MayContinueMainLoop,
                // An else block always diverges, so the Normal case should not happen,
                // but the analysis is approximate so it might return Normal anyway.
                // Returning Normal here says that nothing more happens on the main path
                NeverLoopResult::Diverging | NeverLoopResult::Normal => NeverLoopResult::Normal,
            })
        })
    }))
}

fn stmt_to_expr<'tcx>(stmt: &Stmt<'tcx>) -> Option<(&'tcx Expr<'tcx>, Option<&'tcx Block<'tcx>>)> {
    match stmt.kind {
        StmtKind::Semi(e) | StmtKind::Expr(e) => Some((e, None)),
        // add the let...else expression (if present)
        StmtKind::Let(local) => local.init.map(|init| (init, local.els)),
        StmtKind::Item(..) => None,
    }
}

#[allow(clippy::too_many_lines)]
fn never_loop_expr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    local_labels: &mut Vec<(HirId, bool)>,
    main_loop_id: HirId,
) -> NeverLoopResult {
    let result = match expr.kind {
        ExprKind::Unary(_, e)
        | ExprKind::Cast(e, _)
        | ExprKind::Type(e, _)
        | ExprKind::Field(e, _)
        | ExprKind::AddrOf(_, _, e)
        | ExprKind::Repeat(e, _)
        | ExprKind::DropTemps(e)
        | ExprKind::UnsafeBinderCast(_, e, _) => never_loop_expr(cx, e, local_labels, main_loop_id),
        ExprKind::Let(let_expr) => never_loop_expr(cx, let_expr.init, local_labels, main_loop_id),
        ExprKind::Array(es) | ExprKind::Tup(es) => never_loop_expr_all(cx, es.iter(), local_labels, main_loop_id),
        ExprKind::Use(expr, _) => never_loop_expr(cx, expr, local_labels, main_loop_id),
        ExprKind::MethodCall(_, receiver, es, _) => {
            never_loop_expr_all(cx, once(receiver).chain(es.iter()), local_labels, main_loop_id)
        },
        ExprKind::Struct(_, fields, base) => {
            let fields = never_loop_expr_all(cx, fields.iter().map(|f| f.expr), local_labels, main_loop_id);
            if let StructTailExpr::Base(base) = base {
                combine_seq(fields, || never_loop_expr(cx, base, local_labels, main_loop_id))
            } else {
                fields
            }
        },
        ExprKind::Call(e, es) => never_loop_expr_all(cx, once(e).chain(es.iter()), local_labels, main_loop_id),
        ExprKind::Binary(_, e1, e2)
        | ExprKind::Assign(e1, e2, _)
        | ExprKind::AssignOp(_, e1, e2)
        | ExprKind::Index(e1, e2, _) => never_loop_expr_all(cx, [e1, e2].iter().copied(), local_labels, main_loop_id),
        ExprKind::Loop(b, _, _, _) => {
            // We don't attempt to track reachability after a loop,
            // just assume there may have been a break somewhere
            absorb_break(never_loop_block(cx, b, local_labels, main_loop_id))
        },
        ExprKind::If(e, e2, e3) => {
            let e1 = never_loop_expr(cx, e, local_labels, main_loop_id);
            combine_seq(e1, || {
                let e2 = never_loop_expr(cx, e2, local_labels, main_loop_id);
                let e3 = e3.as_ref().map_or(NeverLoopResult::Normal, |e| {
                    never_loop_expr(cx, e, local_labels, main_loop_id)
                });
                combine_branches(e2, e3)
            })
        },
        ExprKind::Match(e, arms, _) => {
            let e = never_loop_expr(cx, e, local_labels, main_loop_id);
            combine_seq(e, || {
                arms.iter().fold(NeverLoopResult::Diverging, |a, b| {
                    combine_branches(a, never_loop_expr(cx, b.body, local_labels, main_loop_id))
                })
            })
        },
        ExprKind::Block(b, _) => {
            if b.targeted_by_break {
                local_labels.push((b.hir_id, false));
            }
            let ret = never_loop_block(cx, b, local_labels, main_loop_id);
            let jumped_to = b.targeted_by_break && local_labels.pop().unwrap().1;
            match ret {
                NeverLoopResult::Diverging if jumped_to => NeverLoopResult::Normal,
                _ => ret,
            }
        },
        ExprKind::Continue(d) => {
            let id = d
                .target_id
                .expect("target ID can only be missing in the presence of compilation errors");
            if id == main_loop_id {
                NeverLoopResult::MayContinueMainLoop
            } else {
                NeverLoopResult::Diverging
            }
        },
        ExprKind::Break(_, e) | ExprKind::Ret(e) => {
            let first = e.as_ref().map_or(NeverLoopResult::Normal, |e| {
                never_loop_expr(cx, e, local_labels, main_loop_id)
            });
            combine_seq(first, || {
                // checks if break targets a block instead of a loop
                if let ExprKind::Break(Destination { target_id: Ok(t), .. }, _) = expr.kind
                    && let Some((_, reachable)) = local_labels.iter_mut().find(|(label, _)| *label == t)
                {
                    *reachable = true;
                }
                NeverLoopResult::Diverging
            })
        },
        ExprKind::Become(e) => combine_seq(never_loop_expr(cx, e, local_labels, main_loop_id), || {
            NeverLoopResult::Diverging
        }),
        ExprKind::InlineAsm(asm) => combine_seq_many(asm.operands.iter().map(|(o, _)| match o {
            InlineAsmOperand::In { expr, .. } | InlineAsmOperand::InOut { expr, .. } => {
                never_loop_expr(cx, expr, local_labels, main_loop_id)
            },
            InlineAsmOperand::Out { expr, .. } => {
                never_loop_expr_all(cx, expr.iter().copied(), local_labels, main_loop_id)
            },
            InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => never_loop_expr_all(
                cx,
                once(*in_expr).chain(out_expr.iter().copied()),
                local_labels,
                main_loop_id,
            ),
            InlineAsmOperand::Const { .. } | InlineAsmOperand::SymFn { .. } | InlineAsmOperand::SymStatic { .. } => {
                NeverLoopResult::Normal
            },
            InlineAsmOperand::Label { block } => never_loop_block(cx, block, local_labels, main_loop_id),
        })),
        ExprKind::OffsetOf(_, _)
        | ExprKind::Yield(_, _)
        | ExprKind::Closure { .. }
        | ExprKind::Path(_)
        | ExprKind::ConstBlock(_)
        | ExprKind::Lit(_)
        | ExprKind::Err(_) => NeverLoopResult::Normal,
    };
    let result = combine_seq(result, || {
        if cx.typeck_results().expr_ty(expr).is_never() {
            NeverLoopResult::Diverging
        } else {
            NeverLoopResult::Normal
        }
    });
    if let NeverLoopResult::Diverging = result
        && let Some(macro_call) = root_macro_call_first_node(cx, expr)
        && let Some(sym::todo_macro) = cx.tcx.get_diagnostic_name(macro_call.def_id)
    {
        // We return MayContinueMainLoop here because we treat `todo!()`
        // as potentially containing any code, including a continue of the main loop.
        // This effectively silences the lint whenever a loop contains this macro anywhere.
        NeverLoopResult::MayContinueMainLoop
    } else {
        result
    }
}

fn never_loop_expr_all<'tcx, T: Iterator<Item = &'tcx Expr<'tcx>>>(
    cx: &LateContext<'tcx>,
    es: T,
    local_labels: &mut Vec<(HirId, bool)>,
    main_loop_id: HirId,
) -> NeverLoopResult {
    combine_seq_many(es.map(|e| never_loop_expr(cx, e, local_labels, main_loop_id)))
}

fn for_to_if_let_sugg(cx: &LateContext<'_>, iterator: &Expr<'_>, pat: &Pat<'_>) -> String {
    let pat_snippet = snippet(cx, pat.span, "_");
    let iter_snippet = make_iterator_snippet(cx, iterator, &mut Applicability::Unspecified);

    format!("if let Some({pat_snippet}) = {iter_snippet}.next()")
}
