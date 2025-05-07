use super::EXPLICIT_ITER_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sym;
use clippy_utils::ty::{
    implements_trait, implements_trait_with_env, is_copy, is_type_lang_item, make_normalized_projection,
    make_normalized_projection_with_regions, normalize_with_regions,
};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_middle::ty::{self, EarlyBinder, Ty};

pub(super) fn check(
    cx: &LateContext<'_>,
    self_arg: &Expr<'_>,
    call_expr: &Expr<'_>,
    msrv: Msrv,
    enforce_iter_loop_reborrow: bool,
) {
    let Some((adjust, ty)) = is_ref_iterable(cx, self_arg, call_expr, enforce_iter_loop_reborrow, msrv) else {
        return;
    };

    if let ty::Array(_, count) = *ty.peel_refs().kind() {
        if !ty.is_ref() {
            if !msrv.meets(cx, msrvs::ARRAY_INTO_ITERATOR) {
                return;
            }
        } else if count.try_to_target_usize(cx.tcx).is_none_or(|x| x > 32) && !msrv.meets(cx, msrvs::ARRAY_IMPL_ANY_LEN)
        {
            return;
        }
    }

    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, self_arg.span, "_", &mut applicability);
    span_lint_and_sugg(
        cx,
        EXPLICIT_ITER_LOOP,
        call_expr.span,
        "it is more concise to loop over references to containers instead of using explicit \
         iteration methods",
        "to write this more concisely, try",
        format!("{}{object}", adjust.display()),
        applicability,
    );
}

#[derive(Clone, Copy)]
enum AdjustKind {
    None,
    Borrow,
    BorrowMut,
    Deref,
    Reborrow,
    ReborrowMut,
}
impl AdjustKind {
    fn borrow(mutbl: Mutability) -> Self {
        match mutbl {
            Mutability::Not => Self::Borrow,
            Mutability::Mut => Self::BorrowMut,
        }
    }

    fn auto_borrow(mutbl: AutoBorrowMutability) -> Self {
        match mutbl {
            AutoBorrowMutability::Not => Self::Borrow,
            AutoBorrowMutability::Mut { .. } => Self::BorrowMut,
        }
    }

    fn reborrow(mutbl: Mutability) -> Self {
        match mutbl {
            Mutability::Not => Self::Reborrow,
            Mutability::Mut => Self::ReborrowMut,
        }
    }

    fn auto_reborrow(mutbl: AutoBorrowMutability) -> Self {
        match mutbl {
            AutoBorrowMutability::Not => Self::Reborrow,
            AutoBorrowMutability::Mut { .. } => Self::ReborrowMut,
        }
    }

    fn display(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Borrow => "&",
            Self::BorrowMut => "&mut ",
            Self::Deref => "*",
            Self::Reborrow => "&*",
            Self::ReborrowMut => "&mut *",
        }
    }
}

/// Checks if an `iter` or `iter_mut` call returns `IntoIterator::IntoIter`. Returns how the
/// argument needs to be adjusted.
#[expect(clippy::too_many_lines)]
fn is_ref_iterable<'tcx>(
    cx: &LateContext<'tcx>,
    self_arg: &Expr<'_>,
    call_expr: &Expr<'_>,
    enforce_iter_loop_reborrow: bool,
    msrv: Msrv,
) -> Option<(AdjustKind, Ty<'tcx>)> {
    let typeck = cx.typeck_results();
    if let Some(trait_id) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
        && let Some(fn_id) = typeck.type_dependent_def_id(call_expr.hir_id)
        && let sig = cx
            .tcx
            .liberate_late_bound_regions(fn_id, cx.tcx.fn_sig(fn_id).skip_binder())
        && let &[req_self_ty, req_res_ty] = &**sig.inputs_and_output
        && let typing_env = ty::TypingEnv::non_body_analysis(cx.tcx, fn_id)
        && implements_trait_with_env(cx.tcx, typing_env, req_self_ty, trait_id, Some(fn_id), &[])
        && let Some(into_iter_ty) =
            make_normalized_projection_with_regions(cx.tcx, typing_env, trait_id, sym::IntoIter, [req_self_ty])
        && let req_res_ty = normalize_with_regions(cx.tcx, typing_env, req_res_ty)
        && into_iter_ty == req_res_ty
    {
        let adjustments = typeck.expr_adjustments(self_arg);
        let self_ty = typeck.expr_ty(self_arg);
        let self_is_copy = is_copy(cx, self_ty);

        if is_type_lang_item(cx, self_ty.peel_refs(), rustc_hir::LangItem::OwnedBox)
            && !msrv.meets(cx, msrvs::BOX_INTO_ITER)
        {
            return None;
        }

        if adjustments.is_empty() && self_is_copy {
            // Exact type match, already checked earlier
            return Some((AdjustKind::None, self_ty));
        }

        let res_ty = cx
            .tcx
            .erase_regions(EarlyBinder::bind(req_res_ty).instantiate(cx.tcx, typeck.node_args(call_expr.hir_id)));
        let mutbl = if let ty::Ref(_, _, mutbl) = *req_self_ty.kind() {
            Some(mutbl)
        } else {
            None
        };

        if !adjustments.is_empty() {
            if self_is_copy {
                // Using by value won't consume anything
                if implements_trait(cx, self_ty, trait_id, &[])
                    && let Some(ty) =
                        make_normalized_projection(cx.tcx, cx.typing_env(), trait_id, sym::IntoIter, [self_ty])
                    && ty == res_ty
                {
                    return Some((AdjustKind::None, self_ty));
                }
            } else if enforce_iter_loop_reborrow
                && let ty::Ref(region, ty, Mutability::Mut) = *self_ty.kind()
                && let Some(mutbl) = mutbl
            {
                // Attempt to reborrow the mutable reference
                let self_ty = if mutbl.is_mut() {
                    self_ty
                } else {
                    Ty::new_ref(cx.tcx, region, ty, mutbl)
                };
                if implements_trait(cx, self_ty, trait_id, &[])
                    && let Some(ty) =
                        make_normalized_projection(cx.tcx, cx.typing_env(), trait_id, sym::IntoIter, [self_ty])
                    && ty == res_ty
                {
                    return Some((AdjustKind::reborrow(mutbl), self_ty));
                }
            }
        }
        if let Some(mutbl) = mutbl
            && !self_ty.is_ref()
        {
            // Attempt to borrow
            let self_ty = Ty::new_ref(cx.tcx, cx.tcx.lifetimes.re_erased, self_ty, mutbl);
            if implements_trait(cx, self_ty, trait_id, &[])
                && let Some(ty) =
                    make_normalized_projection(cx.tcx, cx.typing_env(), trait_id, sym::IntoIter, [self_ty])
                && ty == res_ty
            {
                return Some((AdjustKind::borrow(mutbl), self_ty));
            }
        }

        match adjustments {
            [] => Some((AdjustKind::None, self_ty)),
            &[
                Adjustment {
                    kind: Adjust::Deref(_), ..
                },
                Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                    target,
                },
                ..,
            ] => {
                if enforce_iter_loop_reborrow
                    && target != self_ty
                    && implements_trait(cx, target, trait_id, &[])
                    && let Some(ty) =
                        make_normalized_projection(cx.tcx, cx.typing_env(), trait_id, sym::IntoIter, [target])
                    && ty == res_ty
                {
                    Some((AdjustKind::auto_reborrow(mutbl), target))
                } else {
                    None
                }
            },
            &[
                Adjustment {
                    kind: Adjust::Deref(_),
                    target,
                },
                ..,
            ] => {
                if is_copy(cx, target)
                    && implements_trait(cx, target, trait_id, &[])
                    && let Some(ty) =
                        make_normalized_projection(cx.tcx, cx.typing_env(), trait_id, sym::IntoIter, [target])
                    && ty == res_ty
                {
                    Some((AdjustKind::Deref, target))
                } else {
                    None
                }
            },
            &[
                Adjustment {
                    kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                    target,
                },
                ..,
            ] => {
                if self_ty.is_ref()
                    && implements_trait(cx, target, trait_id, &[])
                    && let Some(ty) =
                        make_normalized_projection(cx.tcx, cx.typing_env(), trait_id, sym::IntoIter, [target])
                    && ty == res_ty
                {
                    Some((AdjustKind::auto_borrow(mutbl), target))
                } else {
                    None
                }
            },
            _ => None,
        }
    } else {
        None
    }
}
