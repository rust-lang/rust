use super::MUT_RANGE_BOUND;
use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::{get_enclosing_block, higher, path_to_local};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{BindingMode, Expr, ExprKind, HirId, Node, PatKind};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_lint::LateContext;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty;
use rustc_span::Span;
use std::ops::ControlFlow;

pub(super) fn check(cx: &LateContext<'_>, arg: &Expr<'_>, body: &Expr<'_>) {
    if let Some(higher::Range {
        start: Some(start),
        end: Some(end),
        ..
    }) = higher::Range::hir(arg)
        && let (mut_id_start, mut_id_end) = (check_for_mutability(cx, start), check_for_mutability(cx, end))
        && (mut_id_start.is_some() || mut_id_end.is_some())
    {
        let (span_low, span_high) = check_for_mutation(cx, body, mut_id_start, mut_id_end);
        mut_warn_with_span(cx, span_low);
        mut_warn_with_span(cx, span_high);
    }
}

fn mut_warn_with_span(cx: &LateContext<'_>, span: Option<Span>) {
    if let Some(sp) = span {
        span_lint_and_note(
            cx,
            MUT_RANGE_BOUND,
            sp,
            "attempt to mutate range bound within loop",
            None,
            "the range of the loop is unchanged",
        );
    }
}

fn check_for_mutability(cx: &LateContext<'_>, bound: &Expr<'_>) -> Option<HirId> {
    if let Some(hir_id) = path_to_local(bound)
        && let Node::Pat(pat) = cx.tcx.hir_node(hir_id)
        && let PatKind::Binding(BindingMode::MUT, ..) = pat.kind
    {
        return Some(hir_id);
    }
    None
}

fn check_for_mutation(
    cx: &LateContext<'_>,
    body: &Expr<'_>,
    bound_id_start: Option<HirId>,
    bound_id_end: Option<HirId>,
) -> (Option<Span>, Option<Span>) {
    let mut delegate = MutatePairDelegate {
        cx,
        hir_id_low: bound_id_start,
        hir_id_high: bound_id_end,
        span_low: None,
        span_high: None,
    };
    ExprUseVisitor::for_clippy(cx, body.hir_id.owner.def_id, &mut delegate)
        .walk_expr(body)
        .into_ok();

    delegate.mutation_span()
}

struct MutatePairDelegate<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    hir_id_low: Option<HirId>,
    hir_id_high: Option<HirId>,
    span_low: Option<Span>,
    span_high: Option<Span>,
}

impl<'tcx> Delegate<'tcx> for MutatePairDelegate<'_, 'tcx> {
    fn consume(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn use_cloned(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, diag_expr_id: HirId, bk: ty::BorrowKind) {
        if bk == ty::BorrowKind::Mutable {
            if let PlaceBase::Local(id) = cmt.place.base {
                if Some(id) == self.hir_id_low && !BreakAfterExprVisitor::is_found(self.cx, diag_expr_id) {
                    self.span_low = Some(self.cx.tcx.hir_span(diag_expr_id));
                }
                if Some(id) == self.hir_id_high && !BreakAfterExprVisitor::is_found(self.cx, diag_expr_id) {
                    self.span_high = Some(self.cx.tcx.hir_span(diag_expr_id));
                }
            }
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        if let PlaceBase::Local(id) = cmt.place.base {
            if Some(id) == self.hir_id_low && !BreakAfterExprVisitor::is_found(self.cx, diag_expr_id) {
                self.span_low = Some(self.cx.tcx.hir_span(diag_expr_id));
            }
            if Some(id) == self.hir_id_high && !BreakAfterExprVisitor::is_found(self.cx, diag_expr_id) {
                self.span_high = Some(self.cx.tcx.hir_span(diag_expr_id));
            }
        }
    }

    fn fake_read(&mut self, _: &PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}

impl MutatePairDelegate<'_, '_> {
    fn mutation_span(&self) -> (Option<Span>, Option<Span>) {
        (self.span_low, self.span_high)
    }
}

struct BreakAfterExprVisitor {
    hir_id: HirId,
    past_expr: bool,
    break_after_expr: bool,
}

impl BreakAfterExprVisitor {
    pub fn is_found(cx: &LateContext<'_>, hir_id: HirId) -> bool {
        let mut visitor = BreakAfterExprVisitor {
            hir_id,
            past_expr: false,
            break_after_expr: false,
        };

        get_enclosing_block(cx, hir_id).is_some_and(|block| {
            let _ = visitor.visit_block(block);
            visitor.break_after_expr
        })
    }
}

impl<'tcx> Visitor<'tcx> for BreakAfterExprVisitor {
    type Result = ControlFlow<()>;
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> ControlFlow<()> {
        if expr.hir_id == self.hir_id {
            self.past_expr = true;
            ControlFlow::Continue(())
        } else if self.past_expr {
            if matches!(&expr.kind, ExprKind::Break(..)) {
                self.break_after_expr = true;
            }

            ControlFlow::Break(())
        } else {
            intravisit::walk_expr(self, expr)
        }
    }
}
