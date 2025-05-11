use rustc_hir::{self as hir, LangItem};
use rustc_middle::ty::{self, Ty};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::edition::Edition;

use crate::lints::{ShadowedIntoIterDiag, ShadowedIntoIterDiagSub};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `array_into_iter` lint detects calling `into_iter` on arrays.
    ///
    /// ### Example
    ///
    /// ```rust,edition2018
    /// # #![allow(unused)]
    /// [1, 2, 3].into_iter().for_each(|n| { *n; });
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Since Rust 1.53, arrays implement `IntoIterator`. However, to avoid
    /// breakage, `array.into_iter()` in Rust 2015 and 2018 code will still
    /// behave as `(&array).into_iter()`, returning an iterator over
    /// references, just like in Rust 1.52 and earlier.
    /// This only applies to the method call syntax `array.into_iter()`, not to
    /// any other syntax such as `for _ in array` or `IntoIterator::into_iter(array)`.
    pub ARRAY_INTO_ITER,
    Warn,
    "detects calling `into_iter` on arrays in Rust 2015 and 2018",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2021),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2021/IntoIterator-for-arrays.html>",
    };
}

declare_lint! {
    /// The `boxed_slice_into_iter` lint detects calling `into_iter` on boxed slices.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// # #![allow(unused)]
    /// vec![1, 2, 3].into_boxed_slice().into_iter().for_each(|n| { *n; });
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Since Rust 1.80.0, boxed slices implement `IntoIterator`. However, to avoid
    /// breakage, `boxed_slice.into_iter()` in Rust 2015, 2018, and 2021 code will still
    /// behave as `(&boxed_slice).into_iter()`, returning an iterator over
    /// references, just like in Rust 1.79.0 and earlier.
    /// This only applies to the method call syntax `boxed_slice.into_iter()`, not to
    /// any other syntax such as `for _ in boxed_slice` or `IntoIterator::into_iter(boxed_slice)`.
    pub BOXED_SLICE_INTO_ITER,
    Warn,
    "detects calling `into_iter` on boxed slices in Rust 2015, 2018, and 2021",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/intoiterator-box-slice.html>"
    };
}

#[derive(Copy, Clone)]
pub(crate) struct ShadowedIntoIter;

impl_lint_pass!(ShadowedIntoIter => [ARRAY_INTO_ITER, BOXED_SLICE_INTO_ITER]);

impl<'tcx> LateLintPass<'tcx> for ShadowedIntoIter {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::MethodCall(call, receiver_arg, ..) = &expr.kind else {
            return;
        };

        // Check if the method call actually calls the libcore
        // `IntoIterator::into_iter`.
        let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) else {
            return;
        };
        if !cx.tcx.is_lang_item(method_def_id, LangItem::IntoIterIntoIter) {
            return;
        }

        // As this is a method call expression, we have at least one argument.
        let receiver_ty = cx.typeck_results().expr_ty(receiver_arg);
        let adjustments = cx.typeck_results().expr_adjustments(receiver_arg);

        let adjusted_receiver_tys: Vec<_> =
            [receiver_ty].into_iter().chain(adjustments.iter().map(|adj| adj.target)).collect();

        fn is_ref_to_array(ty: Ty<'_>) -> bool {
            if let ty::Ref(_, pointee_ty, _) = *ty.kind() { pointee_ty.is_array() } else { false }
        }
        fn is_ref_to_boxed_slice(ty: Ty<'_>) -> bool {
            if let ty::Ref(_, pointee_ty, _) = *ty.kind() {
                pointee_ty.boxed_ty().is_some_and(Ty::is_slice)
            } else {
                false
            }
        }

        let (lint, target, edition, can_suggest_ufcs) =
            if is_ref_to_array(*adjusted_receiver_tys.last().unwrap())
                && let Some(idx) = adjusted_receiver_tys
                    .iter()
                    .copied()
                    .take_while(|ty| !is_ref_to_array(*ty))
                    .position(|ty| ty.is_array())
            {
                (ARRAY_INTO_ITER, "[T; N]", "2021", idx == 0)
            } else if is_ref_to_boxed_slice(*adjusted_receiver_tys.last().unwrap())
                && let Some(idx) = adjusted_receiver_tys
                    .iter()
                    .copied()
                    .take_while(|ty| !is_ref_to_boxed_slice(*ty))
                    .position(|ty| ty.boxed_ty().is_some_and(Ty::is_slice))
            {
                (BOXED_SLICE_INTO_ITER, "Box<[T]>", "2024", idx == 0)
            } else {
                return;
            };

        // If this expression comes from the `IntoIter::into_iter` inside of a for loop,
        // we should just suggest removing the `.into_iter()` or changing it to `.iter()`
        // to disambiguate if we want to iterate by-value or by-ref.
        let sub = if let Some((_, hir::Node::Expr(parent_expr))) =
            cx.tcx.hir_parent_iter(expr.hir_id).nth(1)
            && let hir::ExprKind::Match(arg, [_], hir::MatchSource::ForLoopDesugar) =
                &parent_expr.kind
            && let hir::ExprKind::Call(path, [_]) = &arg.kind
            && let hir::ExprKind::Path(hir::QPath::LangItem(hir::LangItem::IntoIterIntoIter, ..)) =
                &path.kind
        {
            Some(ShadowedIntoIterDiagSub::RemoveIntoIter {
                span: receiver_arg.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
            })
        } else if can_suggest_ufcs {
            Some(ShadowedIntoIterDiagSub::UseExplicitIntoIter {
                start_span: expr.span.shrink_to_lo(),
                end_span: receiver_arg.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
            })
        } else {
            None
        };

        cx.emit_span_lint(
            lint,
            call.ident.span,
            ShadowedIntoIterDiag { target, edition, suggestion: call.ident.span, sub },
        );
    }
}
