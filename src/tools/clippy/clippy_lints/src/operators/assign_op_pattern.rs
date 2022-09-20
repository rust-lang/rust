use clippy_utils::binop_traits;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{eq_expr_value, trait_ref_of_method};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_lint::LateContext;
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::BorrowKind;
use rustc_trait_selection::infer::TyCtxtInferExt;
use rustc_typeck::expr_use_visitor::{Delegate, ExprUseVisitor, PlaceBase, PlaceWithHirId};

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
            if_chain! {
                if let Some((_, lang_item)) = binop_traits(op.node);
                if let Ok(trait_id) = cx.tcx.lang_items().require(lang_item);
                let parent_fn = cx.tcx.hir().get_parent_item(e.hir_id).def_id;
                if trait_ref_of_method(cx, parent_fn)
                    .map_or(true, |t| t.path.res.def_id() != trait_id);
                if implements_trait(cx, ty, trait_id, &[rty.into()]);
                then {
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
                            if let (Some(snip_a), Some(snip_r)) =
                                (snippet_opt(cx, assignee.span), snippet_opt(cx, rhs.span))
                            {
                                diag.span_suggestion(
                                    expr.span,
                                    "replace it with",
                                    format!("{} {}= {}", snip_a, op.node.as_str(), snip_r),
                                    Applicability::MachineApplicable,
                                );
                            }
                        },
                    );
                }
            }
        };

        let mut visitor = ExprVisitor {
            assignee,
            counter: 0,
            cx,
        };

        walk_expr(&mut visitor, e);

        if visitor.counter == 1 {
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

struct ExprVisitor<'a, 'tcx> {
    assignee: &'a hir::Expr<'a>,
    counter: u8,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for ExprVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if eq_expr_value(self.cx, self.assignee, expr) {
            self.counter += 1;
        }

        walk_expr(self, expr);
    }
}

fn imm_borrows_in_expr(cx: &LateContext<'_>, e: &hir::Expr<'_>) -> hir::HirIdSet {
    struct S(hir::HirIdSet);
    impl Delegate<'_> for S {
        fn borrow(&mut self, place: &PlaceWithHirId<'_>, _: hir::HirId, kind: BorrowKind) {
            if matches!(kind, BorrowKind::ImmBorrow | BorrowKind::UniqueImmBorrow) {
                self.0.insert(match place.place.base {
                    PlaceBase::Local(id) => id,
                    PlaceBase::Upvar(id) => id.var_path.hir_id,
                    _ => return,
                });
            }
        }

        fn consume(&mut self, _: &PlaceWithHirId<'_>, _: hir::HirId) {}
        fn mutate(&mut self, _: &PlaceWithHirId<'_>, _: hir::HirId) {}
        fn fake_read(&mut self, _: &PlaceWithHirId<'_>, _: FakeReadCause, _: hir::HirId) {}
        fn copy(&mut self, _: &PlaceWithHirId<'_>, _: hir::HirId) {}
    }

    let mut s = S(hir::HirIdSet::default());
    cx.tcx.infer_ctxt().enter(|infcx| {
        let mut v = ExprUseVisitor::new(
            &mut s,
            &infcx,
            cx.tcx.hir().body_owner_def_id(cx.enclosing_body.unwrap()),
            cx.param_env,
            cx.typeck_results(),
        );
        v.consume_expr(e);
    });
    s.0
}

fn mut_borrows_in_expr(cx: &LateContext<'_>, e: &hir::Expr<'_>) -> hir::HirIdSet {
    struct S(hir::HirIdSet);
    impl Delegate<'_> for S {
        fn borrow(&mut self, place: &PlaceWithHirId<'_>, _: hir::HirId, kind: BorrowKind) {
            if matches!(kind, BorrowKind::MutBorrow) {
                self.0.insert(match place.place.base {
                    PlaceBase::Local(id) => id,
                    PlaceBase::Upvar(id) => id.var_path.hir_id,
                    _ => return,
                });
            }
        }

        fn consume(&mut self, _: &PlaceWithHirId<'_>, _: hir::HirId) {}
        fn mutate(&mut self, _: &PlaceWithHirId<'_>, _: hir::HirId) {}
        fn fake_read(&mut self, _: &PlaceWithHirId<'_>, _: FakeReadCause, _: hir::HirId) {}
        fn copy(&mut self, _: &PlaceWithHirId<'_>, _: hir::HirId) {}
    }

    let mut s = S(hir::HirIdSet::default());
    cx.tcx.infer_ctxt().enter(|infcx| {
        let mut v = ExprUseVisitor::new(
            &mut s,
            &infcx,
            cx.tcx.hir().body_owner_def_id(cx.enclosing_body.unwrap()),
            cx.param_env,
            cx.typeck_results(),
        );
        v.consume_expr(e);
    });
    s.0
}
