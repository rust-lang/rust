use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::{SpanRangeExt, snippet};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{expr_or_init, is_trait_method, pat_is_wild};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, FnDecl, PatKind, TyKind};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

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
    // If we call a method on a `std::iter::Enumerate` instance
    if is_type_diagnostic_item(cx, recv_ty, sym::Enumerate)
        // If we are calling a method of the `Iterator` trait
        && is_trait_method(cx, call_expr, sym::Iterator)
        // And the map argument is a closure
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let closure_body = cx.tcx.hir_body(closure.body)
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
        && cx.tcx.is_diagnostic_item(sym::enumerate_method, enumerate_defid)
    {
        // Check if the tuple type was explicit. It may be the type system _needs_ the type of the element
        // that would be explicitly in the closure.
        let new_closure_param = match find_elem_explicit_type_span(closure.fn_decl) {
            // We have an explicit type. Get its snippet, that of the binding name, and do `binding: ty`.
            // Fallback to `..` if we fail getting either snippet.
            Some(ty_span) => elem
                .span
                .get_source_text(cx)
                .and_then(|binding_name| {
                    ty_span
                        .get_source_text(cx)
                        .map(|ty_name| format!("{binding_name}: {ty_name}"))
                })
                .unwrap_or_else(|| "..".to_string()),
            // Otherwise, we have no explicit type. We can replace with the binding name of the element.
            None => snippet(cx, elem.span, "..").into_owned(),
        };

        // Suggest removing the tuple from the closure and the preceding call to `enumerate`, whose span we
        // can get from the `MethodCall`.
        span_lint_hir_and_then(
            cx,
            UNUSED_ENUMERATE_INDEX,
            recv_init_expr.hir_id,
            enumerate_span,
            "you seem to use `.enumerate()` and immediately discard the index",
            |diag| {
                diag.multipart_suggestion(
                    "remove the `.enumerate()` call",
                    vec![
                        (closure_param.span, new_closure_param),
                        (
                            enumerate_span.with_lo(enumerate_recv.span.source_callsite().hi()),
                            String::new(),
                        ),
                    ],
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

/// Find the span of the explicit type of the element.
///
/// # Returns
/// If the tuple argument:
/// * Has no explicit type, returns `None`
/// * Has an explicit tuple type with an implicit element type (`(usize, _)`), returns `None`
/// * Has an explicit tuple type with an explicit element type (`(_, i32)`), returns the span for
///   the element type.
fn find_elem_explicit_type_span(fn_decl: &FnDecl<'_>) -> Option<Span> {
    if let [tuple_ty] = fn_decl.inputs
        && let TyKind::Tup([_idx_ty, elem_ty]) = tuple_ty.kind
        && !matches!(elem_ty.kind, TyKind::Err(..) | TyKind::Infer(()))
    {
        Some(elem_ty.span)
    } else {
        None
    }
}
