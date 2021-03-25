use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_trait_method;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::sym;

use super::MAP_FLATTEN;

/// lint use of `map().flatten()` for `Iterators` and 'Options'
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, map_args: &'tcx [hir::Expr<'_>]) {
    // lint if caller of `.map().flatten()` is an Iterator
    if is_trait_method(cx, expr, sym::Iterator) {
        let map_closure_ty = cx.typeck_results().expr_ty(&map_args[1]);
        let is_map_to_option = match map_closure_ty.kind() {
            ty::Closure(_, _) | ty::FnDef(_, _) | ty::FnPtr(_) => {
                let map_closure_sig = match map_closure_ty.kind() {
                    ty::Closure(_, substs) => substs.as_closure().sig(),
                    _ => map_closure_ty.fn_sig(cx.tcx),
                };
                let map_closure_return_ty = cx.tcx.erase_late_bound_regions(map_closure_sig.output());
                is_type_diagnostic_item(cx, map_closure_return_ty, sym::option_type)
            },
            _ => false,
        };

        let method_to_use = if is_map_to_option {
            // `(...).map(...)` has type `impl Iterator<Item=Option<...>>
            "filter_map"
        } else {
            // `(...).map(...)` has type `impl Iterator<Item=impl Iterator<...>>
            "flat_map"
        };
        let func_snippet = snippet(cx, map_args[1].span, "..");
        let hint = format!(".{0}({1})", method_to_use, func_snippet);
        span_lint_and_sugg(
            cx,
            MAP_FLATTEN,
            expr.span.with_lo(map_args[0].span.hi()),
            "called `map(..).flatten()` on an `Iterator`",
            &format!("try using `{}` instead", method_to_use),
            hint,
            Applicability::MachineApplicable,
        );
    }

    // lint if caller of `.map().flatten()` is an Option
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_args[0]), sym::option_type) {
        let func_snippet = snippet(cx, map_args[1].span, "..");
        let hint = format!(".and_then({})", func_snippet);
        span_lint_and_sugg(
            cx,
            MAP_FLATTEN,
            expr.span.with_lo(map_args[0].span.hi()),
            "called `map(..).flatten()` on an `Option`",
            "try using `and_then` instead",
            hint,
            Applicability::MachineApplicable,
        );
    }
}
