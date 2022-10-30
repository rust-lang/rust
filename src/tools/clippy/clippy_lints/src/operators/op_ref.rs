use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::get_enclosing_block;
use clippy_utils::source::snippet;
use clippy_utils::ty::{implements_trait, is_copy};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{def::Res, def_id::DefId, BinOpKind, BorrowKind, Expr, ExprKind, GenericArg, ItemKind, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::OP_REF;

#[expect(clippy::similar_names, clippy::too_many_lines)]
pub(crate) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    let (trait_id, requires_ref) = match op {
        BinOpKind::Add => (cx.tcx.lang_items().add_trait(), false),
        BinOpKind::Sub => (cx.tcx.lang_items().sub_trait(), false),
        BinOpKind::Mul => (cx.tcx.lang_items().mul_trait(), false),
        BinOpKind::Div => (cx.tcx.lang_items().div_trait(), false),
        BinOpKind::Rem => (cx.tcx.lang_items().rem_trait(), false),
        // don't lint short circuiting ops
        BinOpKind::And | BinOpKind::Or => return,
        BinOpKind::BitXor => (cx.tcx.lang_items().bitxor_trait(), false),
        BinOpKind::BitAnd => (cx.tcx.lang_items().bitand_trait(), false),
        BinOpKind::BitOr => (cx.tcx.lang_items().bitor_trait(), false),
        BinOpKind::Shl => (cx.tcx.lang_items().shl_trait(), false),
        BinOpKind::Shr => (cx.tcx.lang_items().shr_trait(), false),
        BinOpKind::Ne | BinOpKind::Eq => (cx.tcx.lang_items().eq_trait(), true),
        BinOpKind::Lt | BinOpKind::Le | BinOpKind::Ge | BinOpKind::Gt => {
            (cx.tcx.lang_items().partial_ord_trait(), true)
        },
    };
    if let Some(trait_id) = trait_id {
        match (&left.kind, &right.kind) {
            // do not suggest to dereference literals
            (&ExprKind::Lit(..), _) | (_, &ExprKind::Lit(..)) => {},
            // &foo == &bar
            (&ExprKind::AddrOf(BorrowKind::Ref, _, l), &ExprKind::AddrOf(BorrowKind::Ref, _, r)) => {
                let lty = cx.typeck_results().expr_ty(l);
                let rty = cx.typeck_results().expr_ty(r);
                let lcpy = is_copy(cx, lty);
                let rcpy = is_copy(cx, rty);
                if let Some((self_ty, other_ty)) = in_impl(cx, e, trait_id) {
                    if (are_equal(cx, rty, self_ty) && are_equal(cx, lty, other_ty))
                        || (are_equal(cx, rty, other_ty) && are_equal(cx, lty, self_ty))
                    {
                        return; // Don't lint
                    }
                }
                // either operator autorefs or both args are copyable
                if (requires_ref || (lcpy && rcpy)) && implements_trait(cx, lty, trait_id, &[rty.into()]) {
                    span_lint_and_then(
                        cx,
                        OP_REF,
                        e.span,
                        "needlessly taken reference of both operands",
                        |diag| {
                            let lsnip = snippet(cx, l.span, "...").to_string();
                            let rsnip = snippet(cx, r.span, "...").to_string();
                            multispan_sugg(
                                diag,
                                "use the values directly",
                                vec![(left.span, lsnip), (right.span, rsnip)],
                            );
                        },
                    );
                } else if lcpy
                    && !rcpy
                    && implements_trait(cx, lty, trait_id, &[cx.typeck_results().expr_ty(right).into()])
                {
                    span_lint_and_then(
                        cx,
                        OP_REF,
                        e.span,
                        "needlessly taken reference of left operand",
                        |diag| {
                            let lsnip = snippet(cx, l.span, "...").to_string();
                            diag.span_suggestion(
                                left.span,
                                "use the left value directly",
                                lsnip,
                                Applicability::MaybeIncorrect, // FIXME #2597
                            );
                        },
                    );
                } else if !lcpy
                    && rcpy
                    && implements_trait(cx, cx.typeck_results().expr_ty(left), trait_id, &[rty.into()])
                {
                    span_lint_and_then(
                        cx,
                        OP_REF,
                        e.span,
                        "needlessly taken reference of right operand",
                        |diag| {
                            let rsnip = snippet(cx, r.span, "...").to_string();
                            diag.span_suggestion(
                                right.span,
                                "use the right value directly",
                                rsnip,
                                Applicability::MaybeIncorrect, // FIXME #2597
                            );
                        },
                    );
                }
            },
            // &foo == bar
            (&ExprKind::AddrOf(BorrowKind::Ref, _, l), _) => {
                let lty = cx.typeck_results().expr_ty(l);
                if let Some((self_ty, other_ty)) = in_impl(cx, e, trait_id) {
                    let rty = cx.typeck_results().expr_ty(right);
                    if (are_equal(cx, rty, self_ty) && are_equal(cx, lty, other_ty))
                        || (are_equal(cx, rty, other_ty) && are_equal(cx, lty, self_ty))
                    {
                        return; // Don't lint
                    }
                }
                let lcpy = is_copy(cx, lty);
                if (requires_ref || lcpy)
                    && implements_trait(cx, lty, trait_id, &[cx.typeck_results().expr_ty(right).into()])
                {
                    span_lint_and_then(
                        cx,
                        OP_REF,
                        e.span,
                        "needlessly taken reference of left operand",
                        |diag| {
                            let lsnip = snippet(cx, l.span, "...").to_string();
                            diag.span_suggestion(
                                left.span,
                                "use the left value directly",
                                lsnip,
                                Applicability::MaybeIncorrect, // FIXME #2597
                            );
                        },
                    );
                }
            },
            // foo == &bar
            (_, &ExprKind::AddrOf(BorrowKind::Ref, _, r)) => {
                let rty = cx.typeck_results().expr_ty(r);
                if let Some((self_ty, other_ty)) = in_impl(cx, e, trait_id) {
                    let lty = cx.typeck_results().expr_ty(left);
                    if (are_equal(cx, rty, self_ty) && are_equal(cx, lty, other_ty))
                        || (are_equal(cx, rty, other_ty) && are_equal(cx, lty, self_ty))
                    {
                        return; // Don't lint
                    }
                }
                let rcpy = is_copy(cx, rty);
                if (requires_ref || rcpy)
                    && implements_trait(cx, cx.typeck_results().expr_ty(left), trait_id, &[rty.into()])
                {
                    span_lint_and_then(cx, OP_REF, e.span, "taken reference of right operand", |diag| {
                        let rsnip = snippet(cx, r.span, "...").to_string();
                        diag.span_suggestion(
                            right.span,
                            "use the right value directly",
                            rsnip,
                            Applicability::MaybeIncorrect, // FIXME #2597
                        );
                    });
                }
            },
            _ => {},
        }
    }
}

fn in_impl<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    bin_op: DefId,
) -> Option<(&'tcx rustc_hir::Ty<'tcx>, &'tcx rustc_hir::Ty<'tcx>)> {
    if_chain! {
        if let Some(block) = get_enclosing_block(cx, e.hir_id);
        if let Some(impl_def_id) = cx.tcx.impl_of_method(block.hir_id.owner.to_def_id());
        let item = cx.tcx.hir().expect_item(impl_def_id.expect_local());
        if let ItemKind::Impl(item) = &item.kind;
        if let Some(of_trait) = &item.of_trait;
        if let Some(seg) = of_trait.path.segments.last();
        if let Res::Def(_, trait_id) = seg.res;
        if trait_id == bin_op;
        if let Some(generic_args) = seg.args;
        if let Some(GenericArg::Type(other_ty)) = generic_args.args.last();

        then {
            Some((item.self_ty, other_ty))
        }
        else {
            None
        }
    }
}

fn are_equal<'tcx>(cx: &LateContext<'tcx>, middle_ty: Ty<'_>, hir_ty: &rustc_hir::Ty<'_>) -> bool {
    if_chain! {
        if let ty::Adt(adt_def, _) = middle_ty.kind();
        if let Some(local_did) = adt_def.did().as_local();
        let item = cx.tcx.hir().expect_item(local_did);
        let middle_ty_id = item.def_id.to_def_id();
        if let TyKind::Path(QPath::Resolved(_, path)) = hir_ty.kind;
        if let Res::Def(_, hir_ty_id) = path.res;

        then {
            hir_ty_id == middle_ty_id
        }
        else {
            false
        }
    }
}
