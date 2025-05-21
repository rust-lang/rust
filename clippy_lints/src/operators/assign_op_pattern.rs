use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::implements_trait;
use clippy_utils::visitors::for_each_expr_without_closures;
use clippy_utils::{binop_traits, eq_expr_value, trait_ref_of_method};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{HirId, HirIdSet};
use rustc_hir_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};
use rustc_lint::LateContext;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::BorrowKind;

use super::ASSIGN_OP_PATTERN;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    assignee: &'tcx hir::Expr<'_>,
    e: &'tcx hir::Expr<'_>,
) {
    if let hir::ExprKind::Binary(op, l, r) = &e.kind {
        let lint = |assignee: &hir::Expr<'_>, rhs: &hir::Expr<'_>| {
            let ty = cx.typeck_results().expr_ty(assignee);
            let rty = cx.typeck_results().expr_ty(rhs);
            if let Some((_, lang_item)) = binop_traits(op.node)
                && let Some(trait_id) = cx.tcx.lang_items().get(lang_item)
                && let parent_fn = cx.tcx.hir_get_parent_item(e.hir_id)
                && trait_ref_of_method(cx, parent_fn).is_none_or(|t| t.path.res.def_id() != trait_id)
                && implements_trait(cx, ty, trait_id, &[rty.into()])
            {
                // Primitive types execute assign-ops right-to-left. Every other type is left-to-right.
                if !(ty.is_primitive() && rty.is_primitive()) {
                    // TODO: This will have false negatives as it doesn't check if the borrows are
                    // actually live at the end of their respective expressions.
                    let mut_borrows = mut_borrows_in_expr(cx, assignee);
                    let imm_borrows = imm_borrows_in_expr(cx, rhs);
                    if mut_borrows.iter().any(|id| imm_borrows.contains(id)) {
                        return;
                    }
                }
                span_lint_and_then(
                    cx,
                    ASSIGN_OP_PATTERN,
                    expr.span,
                    "manual implementation of an assign operation",
                    |diag| {
                        if let Some(snip_a) = assignee.span.get_source_text(cx)
                            && let Some(snip_r) = rhs.span.get_source_text(cx)
                        {
                            diag.span_suggestion(
                                expr.span,
                                "replace it with",
                                format!("{snip_a} {}= {snip_r}", op.node.as_str()),
                                Applicability::MachineApplicable,
                            );
                        }
                    },
                );
            }
        };

        let mut found = false;
        let found_multiple = for_each_expr_without_closures(e, |e| {
            if eq_expr_value(cx, assignee, e) {
                if found {
                    return ControlFlow::Break(());
                }
                found = true;
            }
            ControlFlow::Continue(())
        })
        .is_some();

        if found && !found_multiple {
            // a = a op b
            if eq_expr_value(cx, assignee, l) {
                lint(assignee, r);
            }
            // a = b commutative_op a
            // Limited to primitive type as these ops are know to be commutative
            if eq_expr_value(cx, assignee, r) && cx.typeck_results().expr_ty(assignee).is_primitive_ty() {
                match op.node {
                    hir::BinOpKind::Add
                    | hir::BinOpKind::Mul
                    | hir::BinOpKind::And
                    | hir::BinOpKind::Or
                    | hir::BinOpKind::BitXor
                    | hir::BinOpKind::BitAnd
                    | hir::BinOpKind::BitOr => {
                        lint(assignee, l);
                    },
                    _ => {},
                }
            }
        }
    }
}

fn imm_borrows_in_expr(cx: &LateContext<'_>, e: &hir::Expr<'_>) -> HirIdSet {
    struct S(HirIdSet);
    impl Delegate<'_> for S {
        fn borrow(&mut self, place: &PlaceWithHirId<'_>, _: HirId, kind: BorrowKind) {
            if matches!(kind, BorrowKind::Immutable | BorrowKind::UniqueImmutable) {
                self.0.insert(match place.place.base {
                    PlaceBase::Local(id) => id,
                    PlaceBase::Upvar(id) => id.var_path.hir_id,
                    _ => return,
                });
            }
        }

        fn consume(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
        fn use_cloned(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
        fn mutate(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
        fn fake_read(&mut self, _: &PlaceWithHirId<'_>, _: FakeReadCause, _: HirId) {}
        fn copy(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
    }

    let mut s = S(HirIdSet::default());
    let v = ExprUseVisitor::for_clippy(cx, e.hir_id.owner.def_id, &mut s);
    v.consume_expr(e).into_ok();
    s.0
}

fn mut_borrows_in_expr(cx: &LateContext<'_>, e: &hir::Expr<'_>) -> HirIdSet {
    struct S(HirIdSet);
    impl Delegate<'_> for S {
        fn borrow(&mut self, place: &PlaceWithHirId<'_>, _: HirId, kind: BorrowKind) {
            if matches!(kind, BorrowKind::Mutable) {
                self.0.insert(match place.place.base {
                    PlaceBase::Local(id) => id,
                    PlaceBase::Upvar(id) => id.var_path.hir_id,
                    _ => return,
                });
            }
        }

        fn consume(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
        fn use_cloned(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
        fn mutate(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
        fn fake_read(&mut self, _: &PlaceWithHirId<'_>, _: FakeReadCause, _: HirId) {}
        fn copy(&mut self, _: &PlaceWithHirId<'_>, _: HirId) {}
    }

    let mut s = S(HirIdSet::default());
    let v = ExprUseVisitor::for_clippy(cx, e.hir_id.owner.def_id, &mut s);
    v.consume_expr(e).into_ok();
    s.0
}
