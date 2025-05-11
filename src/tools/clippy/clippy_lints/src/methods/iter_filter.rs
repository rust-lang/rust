use clippy_utils::ty::get_iterator_item_ty;
use hir::ExprKind;
use rustc_lint::{LateContext, LintContext};

use super::{ITER_FILTER_IS_OK, ITER_FILTER_IS_SOME};

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline};
use clippy_utils::{get_parent_expr, is_trait_method, peel_blocks, span_contains_comment, sym};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::QPath;
use rustc_span::Span;
use rustc_span::symbol::{Ident, Symbol};

///
/// Returns true if the expression is a method call to `method_name`
/// e.g. `a.method_name()` or `Option::method_name`.
///
/// The type-checker verifies for us that the method accepts the right kind of items
/// (e.g. `Option::is_some` accepts `Option<_>`), so we don't need to check that.
///
/// How to capture each case:
///
/// `.filter(|a| { std::option::Option::is_some(a) })`
///          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ <- this is a closure, getting unwrapped and
/// recursively checked.
/// `std::option::Option::is_some(a)`
///  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ <- this is a call. It unwraps to a path with
/// `QPath::TypeRelative`. Since this is a type relative path, we need to check the method name, the
/// type, and that the parameter of the closure is passed in the call. This part is the dual of
/// `receiver.method_name()` below.
///
/// `filter(std::option::Option::is_some);`
///          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ <- this is a type relative path, like above, we check the
/// type and the method name.
///
/// `filter(|a| a.is_some());`
///         ^^^^^^^^^^^^^^^ <- this is a method call inside a closure,
/// we check that the parameter of the closure is the receiver of the method call and don't allow
/// any other parameters.
fn is_method(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    type_symbol: Symbol,
    method_name: Symbol,
    params: &[&hir::Pat<'_>],
) -> bool {
    fn pat_is_recv(ident: Ident, param: &hir::Pat<'_>) -> bool {
        match param.kind {
            hir::PatKind::Binding(_, _, other, _) => ident == other,
            hir::PatKind::Ref(pat, _) => pat_is_recv(ident, pat),
            _ => false,
        }
    }
    match expr.kind {
        ExprKind::MethodCall(hir::PathSegment { ident, .. }, recv, ..) => {
            // compare the identifier of the receiver to the parameter
            // we are in a filter => closure has a single parameter and a single, non-block
            // expression, this means that the parameter shadows all outside variables with
            // the same name => avoid FPs. If the parameter is not the receiver, then this hits
            // outside variables => avoid FP
            if ident.name == method_name
                && let ExprKind::Path(QPath::Resolved(None, path)) = recv.kind
                && let &[seg] = path.segments
                && params.iter().any(|p| pat_is_recv(seg.ident, p))
            {
                return true;
            }
            false
        },
        // This is used to check for complete paths via `|a| std::option::Option::is_some(a)`
        // this then unwraps to a path with `QPath::TypeRelative`
        // we pass the params as they've been passed to the current call through the closure
        ExprKind::Call(expr, [param]) => {
            // this will hit the `QPath::TypeRelative` case and check that the method name is correct
            if is_method(cx, expr, type_symbol, method_name, params)
                // we then check that this is indeed passing the parameter of the closure
                && let ExprKind::Path(QPath::Resolved(None, path)) = param.kind
                && let &[seg] = path.segments
                && params.iter().any(|p| pat_is_recv(seg.ident, p))
            {
                return true;
            }
            false
        },
        ExprKind::Path(QPath::TypeRelative(ty, mname)) => {
            let ty = cx.typeck_results().node_type(ty.hir_id);
            if let Some(did) = cx.tcx.get_diagnostic_item(type_symbol)
                && ty.ty_adt_def() == cx.tcx.type_of(did).skip_binder().ty_adt_def()
            {
                return mname.ident.name == method_name;
            }
            false
        },
        ExprKind::Closure(&hir::Closure { body, .. }) => {
            let body = cx.tcx.hir_body(body);
            let closure_expr = peel_blocks(body.value);
            let params = body.params.iter().map(|param| param.pat).collect::<Vec<_>>();
            is_method(cx, closure_expr, type_symbol, method_name, params.as_slice())
        },
        _ => false,
    }
}

fn parent_is_map(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    if let Some(expr) = get_parent_expr(cx, expr)
        && let ExprKind::MethodCall(path, _, [_], _) = expr.kind
        && path.ident.name == sym::map
        && is_trait_method(cx, expr, sym::Iterator)
    {
        return true;
    }
    false
}

enum FilterType {
    IsSome,
    IsOk,
}

/// Returns the `FilterType` of the expression if it is a filter over an Iter<Option> or
/// Iter<Result> with the parent expression not being a map, and not having a comment in the span of
/// the filter. If it is not a filter over an Iter<Option> or Iter<Result> then it returns None
///
/// How this is done:
/// 1. we know that this is invoked in a method call with `filter` as the method name via `mod.rs`
/// 2. we check that we are in a trait method. Therefore we are in an `(x as
///    Iterator).filter({filter_arg})` method call.
/// 3. we check that the parent expression is not a map. This is because we don't want to lint
///    twice, and we already have a specialized lint for that.
/// 4. we check that the span of the filter does not contain a comment.
/// 5. we get the type of the `Item` in the `Iterator`, and compare against the type of Option and
///    Result.
/// 6. we finally check the contents of the filter argument to see if it is a call to `is_some` or
///    `is_ok`.
/// 7. if all of the above are true, then we return the `FilterType`
fn expression_type(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    filter_arg: &hir::Expr<'_>,
    filter_span: Span,
) -> Option<FilterType> {
    if !is_trait_method(cx, expr, sym::Iterator)
        || parent_is_map(cx, expr)
        || span_contains_comment(cx.sess().source_map(), filter_span.with_hi(expr.span.hi()))
    {
        return None;
    }
    if let ExprKind::MethodCall(_, receiver, _, _) = expr.kind
        && let receiver_ty = cx.typeck_results().expr_ty(receiver)
        && let Some(iter_item_ty) = get_iterator_item_ty(cx, receiver_ty)
    {
        if let Some(opt_defid) = cx.tcx.get_diagnostic_item(sym::Option)
            && let opt_ty = cx.tcx.type_of(opt_defid).skip_binder()
            && iter_item_ty.ty_adt_def() == opt_ty.ty_adt_def()
            && is_method(cx, filter_arg, sym::Option, sym::is_some, &[])
        {
            return Some(FilterType::IsSome);
        }

        if let Some(opt_defid) = cx.tcx.get_diagnostic_item(sym::Result)
            && let opt_ty = cx.tcx.type_of(opt_defid).skip_binder()
            && iter_item_ty.ty_adt_def() == opt_ty.ty_adt_def()
            && is_method(cx, filter_arg, sym::Result, sym::is_ok, &[])
        {
            return Some(FilterType::IsOk);
        }
    }
    None
}

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, filter_arg: &hir::Expr<'_>, filter_span: Span) {
    // we are in a filter inside an iterator
    match expression_type(cx, expr, filter_arg, filter_span) {
        None => (),
        Some(FilterType::IsOk) => span_lint_and_sugg(
            cx,
            ITER_FILTER_IS_OK,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `is_ok` on iterator over `Result`s",
            "consider using `flatten` instead",
            reindent_multiline("flatten()", true, indent_of(cx, filter_span)),
            Applicability::HasPlaceholders,
        ),
        Some(FilterType::IsSome) => span_lint_and_sugg(
            cx,
            ITER_FILTER_IS_SOME,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `is_some` on iterator over `Option`",
            "consider using `flatten` instead",
            reindent_multiline("flatten()", true, indent_of(cx, filter_span)),
            Applicability::HasPlaceholders,
        ),
    }
}
