use crate::{
    lints::{ArrayIntoIterDiag, ArrayIntoIterDiagSub},
    LateContext, LateLintPass, LintContext,
};
use rustc_hir as hir;
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_middle::ty::{self, Ty};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::edition::Edition;
use rustc_span::symbol::sym;
use rustc_span::Span;
use std::ops::ControlFlow;

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
    /// Since Rust 1.??, boxed slices implement `IntoIterator`. However, to avoid
    /// breakage, `boxed_slice.into_iter()` in Rust 2015, 2018, and 2021 code will still
    /// behave as `(&boxed_slice).into_iter()`, returning an iterator over
    /// references, just like in Rust 1.?? and earlier.
    /// This only applies to the method call syntax `boxed_slice.into_iter()`, not to
    /// any other syntax such as `for _ in boxed_slice` or `IntoIterator::into_iter(boxed_slice)`.
    pub BOXED_SLICE_INTO_ITER,
    Warn,
    "detects calling `into_iter` on boxed slices in Rust 2015, 2018, and 2021",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
    };
}

#[derive(Copy, Clone)]
pub struct CommonIntoIter<F, N> {
    for_expr_span: Span,
    filter: F,
    namer: N,
}

#[derive(Copy, Clone)]
pub struct ArrayIntoIter(CommonIntoIter<ArrayFilter, ArrayNamer>);
impl Default for ArrayIntoIter {
    fn default() -> ArrayIntoIter {
        ArrayIntoIter(CommonIntoIter {
            for_expr_span: Span::default(),
            filter: array_filter,
            namer: array_namer,
        })
    }
}

#[derive(Copy, Clone)]
pub struct BoxedSliceIntoIter(CommonIntoIter<BoxedSliceFilter, BoxedSliceNamer>);
impl Default for BoxedSliceIntoIter {
    fn default() -> BoxedSliceIntoIter {
        BoxedSliceIntoIter(CommonIntoIter {
            for_expr_span: Span::default(),
            filter: boxed_slice_filter,
            namer: boxed_slice_namer,
        })
    }
}

impl_lint_pass!(ArrayIntoIter => [ARRAY_INTO_ITER]);
impl_lint_pass!(BoxedSliceIntoIter => [BOXED_SLICE_INTO_ITER]);

type ArrayFilter = impl Copy + FnMut(Ty<'_>) -> ControlFlow<bool>;
type BoxedSliceFilter = impl Copy + FnMut(Ty<'_>) -> ControlFlow<bool>;
type ArrayNamer = impl Copy + FnMut(Ty<'_>) -> &'static str;
type BoxedSliceNamer = impl Copy + FnMut(Ty<'_>) -> &'static str;

fn array_filter(ty: Ty<'_>) -> ControlFlow<bool> {
    match ty.kind() {
        // If we run into a &[T; N] or &[T] first, there's nothing to warn about.
        // It'll resolve to the reference version.
        ty::Ref(_, inner_ty, _) if inner_ty.is_array() => ControlFlow::Break(false),
        ty::Ref(_, inner_ty, _) if matches!(inner_ty.kind(), ty::Slice(..)) => {
            ControlFlow::Break(false)
        }
        // Found an actual array type without matching a &[T; N] first.
        // This is the problematic case.
        ty::Array(..) => ControlFlow::Break(true),
        _ => ControlFlow::Continue(()),
    }
}

fn boxed_slice_filter(_ty: Ty<'_>) -> ControlFlow<bool> {
    todo!()
}

fn array_namer(ty: Ty<'_>) -> &'static str {
    match *ty.kind() {
        ty::Ref(_, inner_ty, _) if inner_ty.is_array() => "[T; N]",
        ty::Ref(_, inner_ty, _) if matches!(inner_ty.kind(), ty::Slice(..)) => "[T]",
        // We know the original first argument type is an array type,
        // we know that the first adjustment was an autoref coercion
        // and we know that `IntoIterator` is the trait involved. The
        // array cannot be coerced to something other than a reference
        // to an array or to a slice.
        _ => bug!("array type coerced to something other than array or slice"),
    }
}

fn boxed_slice_namer(_ty: Ty<'_>) -> &'static str {
    todo!()
}

impl<F, N> CommonIntoIter<F, N>
where
    F: FnMut(Ty<'_>) -> ControlFlow<bool>,
    N: FnMut(Ty<'_>) -> &'static str,
{
    fn check_expr<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> Option<(Span, ArrayIntoIterDiag<'tcx>)> {
        // Save the span of expressions in `for _ in expr` syntax,
        // so we can give a better suggestion for those later.
        if let hir::ExprKind::Match(arg, [_], hir::MatchSource::ForLoopDesugar) = &expr.kind {
            if let hir::ExprKind::Call(path, [arg]) = &arg.kind {
                if let hir::ExprKind::Path(hir::QPath::LangItem(
                    hir::LangItem::IntoIterIntoIter,
                    ..,
                )) = &path.kind
                {
                    self.for_expr_span = arg.span;
                }
            }
        }

        // We only care about method call expressions.
        if let hir::ExprKind::MethodCall(call, receiver_arg, ..) = &expr.kind {
            if call.ident.name != sym::into_iter {
                return None;
            }

            // Check if the method call actually calls the libcore
            // `IntoIterator::into_iter`.
            let def_id = cx.typeck_results().type_dependent_def_id(expr.hir_id).unwrap();
            match cx.tcx.trait_of_item(def_id) {
                Some(trait_id) if cx.tcx.is_diagnostic_item(sym::IntoIterator, trait_id) => {}
                _ => return None,
            };

            // As this is a method call expression, we have at least one argument.
            let receiver_ty = cx.typeck_results().expr_ty(receiver_arg);
            let adjustments = cx.typeck_results().expr_adjustments(receiver_arg);

            let Some(Adjustment { kind: Adjust::Borrow(_), target }) = adjustments.last() else {
                return None;
            };

            let types =
                std::iter::once(receiver_ty).chain(adjustments.iter().map(|adj| adj.target));

            let found_it = 'outer: {
                for ty in types {
                    match (self.filter)(ty) {
                        ControlFlow::Break(b) => break 'outer b,
                        ControlFlow::Continue(()) => (),
                    }
                }
                false
            };
            if !found_it {
                return None;
            }

            // Emit lint diagnostic.
            let target = (self.namer)(*target);
            let sub = if self.for_expr_span == expr.span {
                Some(ArrayIntoIterDiagSub::RemoveIntoIter {
                    span: receiver_arg.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
                })
            } else if receiver_ty.is_array() {
                Some(ArrayIntoIterDiagSub::UseExplicitIntoIter {
                    start_span: expr.span.shrink_to_lo(),
                    end_span: receiver_arg.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
                })
            } else {
                None
            };

            Some((call.ident.span, ArrayIntoIterDiag { target, suggestion: call.ident.span, sub }))
        } else {
            None
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ArrayIntoIter {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let Some((span, decorator)) = self.0.check_expr(cx, expr) {
            cx.emit_spanned_lint(ARRAY_INTO_ITER, span, decorator);
        }
    }
}
impl<'tcx> LateLintPass<'tcx> for BoxedSliceIntoIter {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let Some((span, decorator)) = self.0.check_expr(cx, expr) {
            cx.emit_spanned_lint(BOXED_SLICE_INTO_ITER, span, decorator);
        }
    }
}
