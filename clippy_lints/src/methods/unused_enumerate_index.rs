use clippy_utils::diagnostics::{multispan_sugg, span_lint_hir_and_then};
use clippy_utils::paths::{CORE_ITER_ENUMERATE_METHOD, CORE_ITER_ENUMERATE_STRUCT};
use clippy_utils::source::snippet;
use clippy_utils::{expr_or_init, is_trait_method, match_def_path, pat_is_wild};
use rustc_hir::{Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::AdtDef;
use rustc_span::sym;

use crate::loops::UNUSED_ENUMERATE_INDEX;

/// Check for the `UNUSED_ENUMERATE_INDEX` lint outside of loops.
///
/// The lint is declared in `clippy_lints/src/loops/mod.rs`. There, the following pattern is
/// checked:
/// ```ignore
/// for (_, x) in some_iter.enumerate() {
///     // Index is ignored
/// }
/// ```
///
/// This `check` function checks for chained method calls constructs where we can detect that the
/// index is unused. Currently, this checks only for the following patterns:
/// ```ignore
/// some_iter.enumerate().map_function(|(_, x)| ..)
/// let x = some_iter.enumerate();
/// x.map_function(|(_, x)| ..)
/// ```
/// where `map_function` is one of `all`, `any`, `filter_map`, `find_map`, `flat_map`, `for_each` or
/// `map`.
///
/// # Preconditions
/// This function must be called not on the `enumerate` call expression itself, but on any of the
/// map functions listed above. It will ensure that `recv` is a `std::iter::Enumerate` instance and
/// that the method call is one of the `std::iter::Iterator` trait.
///
/// * `call_expr`: The map function call expression
/// * `recv`: The receiver of the call
/// * `closure_arg`: The argument to the map function call containing the closure/function to apply
pub(super) fn check(cx: &LateContext<'_>, call_expr: &Expr<'_>, recv: &Expr<'_>, closure_arg: &Expr<'_>) {
    let recv_ty = cx.typeck_results().expr_ty(recv);
    if let Some(recv_ty_defid) = recv_ty.ty_adt_def().map(AdtDef::did)
        // If we call a method on a `std::iter::Enumerate` instance
        && match_def_path(cx, recv_ty_defid, &CORE_ITER_ENUMERATE_STRUCT)
        // If we are calling a method of the `Iterator` trait
        && is_trait_method(cx, call_expr, sym::Iterator)
        // And the map argument is a closure
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let closure_body = cx.tcx.hir().body(closure.body)
        // And that closure has one argument ...
        && let [closure_param] = closure_body.params
        // .. which is a tuple of 2 elements
        && let PatKind::Tuple([index, elem], ..) = closure_param.pat.kind
        // And that the first element (the index) is either `_` or unused in the body
        && pat_is_wild(cx, &index.kind, closure_body)
        // Try to find the initializer for `recv`. This is needed in case `recv` is a local_binding. In the
        // first example below, `expr_or_init` would return `recv`.
        // ```
        // iter.enumerate().map(|(_, x)| x)
        // ^^^^^^^^^^^^^^^^ `recv`, a call to `std::iter::Iterator::enumerate`
        //
        // let binding = iter.enumerate();
        //               ^^^^^^^^^^^^^^^^ `recv_init_expr`
        // binding.map(|(_, x)| x)
        // ^^^^^^^ `recv`, not a call to `std::iter::Iterator::enumerate`
        // ```
        && let recv_init_expr = expr_or_init(cx, recv)
        // Make sure the initializer is a method call. It may be that the `Enumerate` comes from something
        // that we cannot control.
        // This would for instance happen with:
        // ```
        // external_lib::some_function_returning_enumerate().map(|(_, x)| x)
        // ```
        && let ExprKind::MethodCall(_, enumerate_recv, _, enumerate_span) = recv_init_expr.kind
        && let Some(enumerate_defid) = cx.typeck_results().type_dependent_def_id(recv_init_expr.hir_id)
        // Make sure the method call is `std::iter::Iterator::enumerate`.
        && match_def_path(cx, enumerate_defid, &CORE_ITER_ENUMERATE_METHOD)
    {
        // Suggest removing the tuple from the closure and the preceding call to `enumerate`, whose span we
        // can get from the `MethodCall`.
        span_lint_hir_and_then(
            cx,
            UNUSED_ENUMERATE_INDEX,
            recv_init_expr.hir_id,
            enumerate_span,
            "you seem to use `.enumerate()` and immediately discard the index",
            |diag| {
                multispan_sugg(
                    diag,
                    "remove the `.enumerate()` call",
                    vec![
                        (closure_param.span, snippet(cx, elem.span, "..").into_owned()),
                        (
                            enumerate_span.with_lo(enumerate_recv.span.source_callsite().hi()),
                            String::new(),
                        ),
                    ],
                );
            },
        );
    }
}
