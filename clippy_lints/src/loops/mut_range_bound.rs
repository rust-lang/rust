use super::MUT_RANGE_BOUND;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{higher, path_to_local};
use if_chain::if_chain;
use rustc_hir::{BindingAnnotation, Expr, HirId, Node, PatKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::LateContext;
use rustc_middle::{mir::FakeReadCause, ty};
use rustc_span::source_map::Span;
use rustc_typeck::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};

pub(super) fn check(cx: &LateContext<'_>, arg: &Expr<'_>, body: &Expr<'_>) {
    if let Some(higher::Range {
        start: Some(start),
        end: Some(end),
        ..
    }) = higher::range(arg)
    {
        let mut_ids = vec![check_for_mutability(cx, start), check_for_mutability(cx, end)];
        if mut_ids[0].is_some() || mut_ids[1].is_some() {
            let (span_low, span_high) = check_for_mutation(cx, body, &mut_ids);
            mut_warn_with_span(cx, span_low);
            mut_warn_with_span(cx, span_high);
        }
    }
}

fn mut_warn_with_span(cx: &LateContext<'_>, span: Option<Span>) {
    if let Some(sp) = span {
        span_lint(
            cx,
            MUT_RANGE_BOUND,
            sp,
            "attempt to mutate range bound within loop; note that the range of the loop is unchanged",
        );
    }
}

fn check_for_mutability(cx: &LateContext<'_>, bound: &Expr<'_>) -> Option<HirId> {
    if_chain! {
        if let Some(hir_id) = path_to_local(bound);
        if let Node::Binding(pat) = cx.tcx.hir().get(hir_id);
        if let PatKind::Binding(BindingAnnotation::Mutable, ..) = pat.kind;
        then {
            return Some(hir_id);
        }
    }
    None
}

fn check_for_mutation<'tcx>(
    cx: &LateContext<'tcx>,
    body: &Expr<'_>,
    bound_ids: &[Option<HirId>],
) -> (Option<Span>, Option<Span>) {
    let mut delegate = MutatePairDelegate {
        cx,
        hir_id_low: bound_ids[0],
        hir_id_high: bound_ids[1],
        span_low: None,
        span_high: None,
    };
    cx.tcx.infer_ctxt().enter(|infcx| {
        ExprUseVisitor::new(
            &mut delegate,
            &infcx,
            body.hir_id.owner,
            cx.param_env,
            cx.typeck_results(),
        )
        .walk_expr(body);
    });
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
    fn consume(&mut self, _: &PlaceWithHirId<'tcx>, _: HirId, _: ConsumeMode) {}

    fn borrow(&mut self, cmt: &PlaceWithHirId<'tcx>, diag_expr_id: HirId, bk: ty::BorrowKind) {
        if let ty::BorrowKind::MutBorrow = bk {
            if let PlaceBase::Local(id) = cmt.place.base {
                if Some(id) == self.hir_id_low {
                    self.span_low = Some(self.cx.tcx.hir().span(diag_expr_id))
                }
                if Some(id) == self.hir_id_high {
                    self.span_high = Some(self.cx.tcx.hir().span(diag_expr_id))
                }
            }
        }
    }

    fn mutate(&mut self, cmt: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        if let PlaceBase::Local(id) = cmt.place.base {
            if Some(id) == self.hir_id_low {
                self.span_low = Some(self.cx.tcx.hir().span(diag_expr_id))
            }
            if Some(id) == self.hir_id_high {
                self.span_high = Some(self.cx.tcx.hir().span(diag_expr_id))
            }
        }
    }

    fn fake_read(&mut self, _: rustc_typeck::expr_use_visitor::Place<'tcx>, _: FakeReadCause, _: HirId) {}
}

impl MutatePairDelegate<'_, '_> {
    fn mutation_span(&self) -> (Option<Span>, Option<Span>) {
        (self.span_low, self.span_high)
    }
}
