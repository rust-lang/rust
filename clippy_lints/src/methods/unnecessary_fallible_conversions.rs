use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::get_parent_expr;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_span::{sym, Span};

use super::UNNECESSARY_FALLIBLE_CONVERSIONS;

/// What function is being called and whether that call is written as a method call or a function
/// call
#[derive(Copy, Clone)]
#[expect(clippy::enum_variant_names)]
enum FunctionKind {
    /// `T::try_from(U)`
    TryFromFunction,
    /// `t.try_into()`
    TryIntoMethod,
    /// `U::try_into(t)`
    TryIntoFunction,
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
    node_args: ty::GenericArgsRef<'tcx>,
    kind: FunctionKind,
    primary_span: Span,
) {
    if let &[self_ty, other_ty] = node_args.as_slice()
        // useless_conversion already warns `T::try_from(T)`, so ignore it here
        && self_ty != other_ty
        && let Some(self_ty) = self_ty.as_type()
        && let Some(from_into_trait) = cx.tcx.get_diagnostic_item(match kind {
            FunctionKind::TryFromFunction => sym::From,
            FunctionKind::TryIntoMethod | FunctionKind::TryIntoFunction => sym::Into,
        })
        // If `T: TryFrom<U>` and `T: From<U>` both exist, then that means that the `TryFrom`
        // _must_ be from the blanket impl and cannot have been manually implemented
        // (else there would be conflicting impls, even with #![feature(spec)]), so we don't even need to check
        // what `<T as TryFrom<U>>::Error` is: it's always `Infallible`
        && implements_trait(cx, self_ty, from_into_trait, &[other_ty])
        && let Some(other_ty) = other_ty.as_type()
    {
        let parent_unwrap_call = get_parent_expr(cx, expr).and_then(|parent| {
            if let ExprKind::MethodCall(path, .., span) = parent.kind
                && let sym::unwrap | sym::expect = path.ident.name
            {
                Some(span)
            } else {
                None
            }
        });
        let (source_ty, target_ty, sugg, span, applicability) = match kind {
            FunctionKind::TryIntoMethod if let Some(unwrap_span) = parent_unwrap_call => {
                // Extend the span to include the unwrap/expect call:
                // `foo.try_into().expect("..")`
                //      ^^^^^^^^^^^^^^^^^^^^^^^
                //
                // `try_into().unwrap()` specifically can be trivially replaced with just `into()`,
                // so that can be machine-applicable

                (
                    self_ty,
                    other_ty,
                    "into()",
                    primary_span.with_hi(unwrap_span.hi()),
                    Applicability::MachineApplicable,
                )
            },
            FunctionKind::TryFromFunction => (
                other_ty,
                self_ty,
                "From::from",
                primary_span,
                Applicability::Unspecified,
            ),
            FunctionKind::TryIntoFunction => (
                self_ty,
                other_ty,
                "Into::into",
                primary_span,
                Applicability::Unspecified,
            ),
            FunctionKind::TryIntoMethod => (self_ty, other_ty, "into", primary_span, Applicability::Unspecified),
        };

        span_lint_and_then(
            cx,
            UNNECESSARY_FALLIBLE_CONVERSIONS,
            span,
            "use of a fallible conversion when an infallible one could be used",
            |diag| {
                with_forced_trimmed_paths!({
                    diag.note(format!("converting `{source_ty}` to `{target_ty}` cannot fail"));
                });
                diag.span_suggestion(span, "use", sugg, applicability);
            },
        );
    }
}

/// Checks method call exprs:
/// - `0i32.try_into()`
pub(super) fn check_method(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::MethodCall(path, ..) = expr.kind {
        check(
            cx,
            expr,
            cx.typeck_results().node_args(expr.hir_id),
            FunctionKind::TryIntoMethod,
            path.ident.span,
        );
    }
}

/// Checks function call exprs:
/// - `<i64 as TryFrom<_>>::try_from(0i32)`
/// - `<_ as TryInto<i64>>::try_into(0i32)`
pub(super) fn check_function(cx: &LateContext<'_>, expr: &Expr<'_>, callee: &Expr<'_>) {
    if let ExprKind::Path(ref qpath) = callee.kind
        && let Some(item_def_id) = cx.qpath_res(qpath, callee.hir_id).opt_def_id()
        && let Some(trait_def_id) = cx.tcx.trait_of_item(item_def_id)
    {
        check(
            cx,
            expr,
            cx.typeck_results().node_args(callee.hir_id),
            match cx.tcx.get_diagnostic_name(trait_def_id) {
                Some(sym::TryFrom) => FunctionKind::TryFromFunction,
                Some(sym::TryInto) => FunctionKind::TryIntoFunction,
                _ => return,
            },
            callee.span,
        );
    }
}
