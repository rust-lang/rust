//! Analysis and instrumentation for branch coverage.

use std::mem;

use rustc_ast::UnOp;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::{self as hir, HirId};
use rustc_middle::mir;
use rustc_middle::mir::coverage::{BasicCoverageBlock, CoverageKind, PointKind};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use crate::coverage::graph::CoverageGraph;

#[derive(Debug)]
pub(crate) struct BranchSpan {
    pub(crate) span: Span,
    pub(crate) true_bcb: BasicCoverageBlock,
    pub(crate) false_bcb: BasicCoverageBlock,
}

#[derive(Default)]
struct BranchPair {
    true_bcb: Option<BasicCoverageBlock>,
    false_bcb: Option<BasicCoverageBlock>,
}

pub(crate) fn extract_branch_spans<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir_body: &mir::Body<'tcx>,
    graph: &CoverageGraph,
) -> Vec<BranchSpan> {
    let mut map = FxIndexMap::<HirId, BranchPair>::default();

    // Scan through MIR basic blocks that are part of the coverage graph, to
    // reconstruct pairs of `(true_bcb, false_bcb)` each associated with a HirId.
    for (bcb, bcb_data) in graph.iter_enumerated() {
        for &bb in &bcb_data.basic_blocks {
            for stmt in &mir_body[bb].statements {
                if let mir::StatementKind::Coverage(cov_kind) = &stmt.kind
                    && let &CoverageKind::Point { point_kind, hir_id } = cov_kind
                    && let PointKind::BranchOutcome { outcome } = point_kind
                {
                    let branch = map.entry(hir_id).or_default();
                    if outcome { branch.true_bcb = Some(bcb) } else { branch.false_bcb = Some(bcb) }
                }
            }
        }
    }

    // For each branch pair, inspect HIR to find a span to use for the branch mapping.
    map.into_iter()
        .filter_map(|(hir_id, branch_pair)| choose_branch_span(tcx, hir_id, branch_pair))
        .collect::<Vec<_>>()
}

fn choose_branch_span<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut hir_id: HirId,
    branch_pair: BranchPair,
) -> Option<BranchSpan> {
    let BranchPair { true_bcb, false_bcb } = branch_pair;

    let mut true_bcb = true_bcb?;
    let mut false_bcb = false_bcb?;

    if !matches!(tcx.hir_node(hir_id), hir::Node::Expr(_)) {
        return None;
    }

    let mut parents = tcx.hir_parent_iter(hir_id).peekable();

    // Check if `hir_id` is the scrutinee of a let-else statement.
    // ```
    // let Some(x) = scrutinee_expr else { ... }
    //               ^^^^^^^^^^^^^^ - if `hir_id` is this expression,
    //     ^^^^^^^^^^^^^^^^^^^^^^^^ - use this span
    // ```
    if let Some(&(_, parent)) = parents.peek()
        && let hir::Node::LetStmt(parent_let) = parent
        && parent_let.els.is_some()
        && let Some(init) = parent_let.init
        && init.hir_id == hir_id
    {
        // The span of the let-else statement includes the whole else block, which
        // we don't want. Instead, combine the pattern and initializer spans.
        let span = parent_let.pat.span.to(init.span);
        return Some(BranchSpan { span, true_bcb, false_bcb });
    }

    // Otherwise, `hir_id` should be a boolean condition, or the scrutinee of a let-expression.
    // We might need to traverse up to a node with a more relevant span.
    while let Some((_, parent)) = parents.next()
        && let hir::Node::Expr(parent_expr) = parent
    {
        match parent_expr.kind {
            // MIR building effectively lowers `if !cond` by treating it as `if cond`
            // and then swapping the true/false blocks. Here we undo that.
            hir::ExprKind::Unary(UnOp::Not, not_arg) if not_arg.hir_id == hir_id => {
                hir_id = parent_expr.hir_id;
                mem::swap(&mut true_bcb, &mut false_bcb);
            }
            // MIR building also discards the no-op cast in `if !(cond as bool)`.
            // Without this case, we wouldn't be able to recover the outer `!`.
            hir::ExprKind::Cast(cast_arg, _ty) if cast_arg.hir_id == hir_id => {
                hir_id = parent_expr.hir_id;
            }
            // For let-expressions, we recorded the `hir_id` of the scrutinee.
            // We want to use the span of the let-expression itself instead.
            hir::ExprKind::Let(let_expr) if let_expr.init.hir_id == hir_id => {
                hir_id = parent_expr.hir_id;
                break;
            }
            _ => break,
        }
    }

    let span = tcx.hir_span(hir_id);
    Some(BranchSpan { span, true_bcb, false_bcb })
}
