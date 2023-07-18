use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, reindent_multiline, snippet};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_trait_method, path_to_local_id, peel_blocks, SpanlessEq};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::{Closure, Expr, ExprKind, PatKind, PathSegment, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::source_map::Span;
use rustc_span::symbol::{sym, Symbol};
use std::borrow::Cow;

use super::{MANUAL_FILTER_MAP, MANUAL_FIND_MAP, OPTION_FILTER_MAP};

fn is_method(cx: &LateContext<'_>, expr: &hir::Expr<'_>, method_name: Symbol) -> bool {
    match &expr.kind {
        hir::ExprKind::Path(QPath::TypeRelative(_, mname)) => mname.ident.name == method_name,
        hir::ExprKind::Path(QPath::Resolved(_, segments)) => {
            segments.segments.last().unwrap().ident.name == method_name
        },
        hir::ExprKind::Closure(&hir::Closure { body, .. }) => {
            let body = cx.tcx.hir().body(body);
            let closure_expr = peel_blocks(body.value);
            let arg_id = body.params[0].pat.hir_id;
            match closure_expr.kind {
                hir::ExprKind::MethodCall(hir::PathSegment { ident, .. }, receiver, ..) => {
                    if_chain! {
                        if ident.name == method_name;
                        if let hir::ExprKind::Path(path) = &receiver.kind;
                        if let Res::Local(ref local) = cx.qpath_res(path, receiver.hir_id);
                        then {
                            return arg_id == *local
                        }
                    }
                    false
                },
                _ => false,
            }
        },
        _ => false,
    }
}

fn is_option_filter_map(cx: &LateContext<'_>, filter_arg: &hir::Expr<'_>, map_arg: &hir::Expr<'_>) -> bool {
    is_method(cx, map_arg, sym::unwrap) && is_method(cx, filter_arg, sym!(is_some))
}

/// is `filter(|x| x.is_some()).map(|x| x.unwrap())`
fn is_filter_some_map_unwrap(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    filter_recv: &hir::Expr<'_>,
    filter_arg: &hir::Expr<'_>,
    map_arg: &hir::Expr<'_>,
) -> bool {
    let iterator = is_trait_method(cx, expr, sym::Iterator);
    let option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(filter_recv), sym::Option);

    (iterator || option) && is_option_filter_map(cx, filter_arg, map_arg)
}

/// lint use of `filter().map()` or `find().map()` for `Iterators`
#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    filter_recv: &hir::Expr<'_>,
    filter_arg: &hir::Expr<'_>,
    filter_span: Span,
    map_recv: &hir::Expr<'_>,
    map_arg: &hir::Expr<'_>,
    map_span: Span,
    is_find: bool,
) {
    if is_filter_some_map_unwrap(cx, expr, filter_recv, filter_arg, map_arg) {
        span_lint_and_sugg(
            cx,
            OPTION_FILTER_MAP,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `Some` followed by `unwrap`",
            "consider using `flatten` instead",
            reindent_multiline(Cow::Borrowed("flatten()"), true, indent_of(cx, map_span)).into_owned(),
            Applicability::MachineApplicable,
        );

        return;
    }

    if_chain! {
        if is_trait_method(cx, map_recv, sym::Iterator);

        // filter(|x| ...is_some())...
        if let ExprKind::Closure(&Closure { body: filter_body_id, .. }) = filter_arg.kind;
        let filter_body = cx.tcx.hir().body(filter_body_id);
        if let [filter_param] = filter_body.params;
        // optional ref pattern: `filter(|&x| ..)`
        let (filter_pat, is_filter_param_ref) = if let PatKind::Ref(ref_pat, _) = filter_param.pat.kind {
            (ref_pat, true)
        } else {
            (filter_param.pat, false)
        };
        // closure ends with is_some() or is_ok()
        if let PatKind::Binding(_, filter_param_id, _, None) = filter_pat.kind;
        if let ExprKind::MethodCall(path, filter_arg, [], _) = filter_body.value.kind;
        if let Some(opt_ty) = cx.typeck_results().expr_ty(filter_arg).peel_refs().ty_adt_def();
        if let Some(is_result) = if cx.tcx.is_diagnostic_item(sym::Option, opt_ty.did()) {
            Some(false)
        } else if cx.tcx.is_diagnostic_item(sym::Result, opt_ty.did()) {
            Some(true)
        } else {
            None
        };
        if path.ident.name.as_str() == if is_result { "is_ok" } else { "is_some" };

        // ...map(|x| ...unwrap())
        if let ExprKind::Closure(&Closure { body: map_body_id, .. }) = map_arg.kind;
        let map_body = cx.tcx.hir().body(map_body_id);
        if let [map_param] = map_body.params;
        if let PatKind::Binding(_, map_param_id, map_param_ident, None) = map_param.pat.kind;
        // closure ends with expect() or unwrap()
        if let ExprKind::MethodCall(seg, map_arg, ..) = map_body.value.kind;
        if matches!(seg.ident.name, sym::expect | sym::unwrap | sym::unwrap_or);

        // .filter(..).map(|y| f(y).copied().unwrap())
        //                     ~~~~
        let map_arg_peeled = match map_arg.kind {
            ExprKind::MethodCall(method, original_arg, [], _) if acceptable_methods(method) => {
                original_arg
            },
            _ => map_arg,
        };

        // .filter(|x| x.is_some()).map(|y| y[.acceptable_method()].unwrap())
        let simple_equal = path_to_local_id(filter_arg, filter_param_id)
            && path_to_local_id(map_arg_peeled, map_param_id);

        let eq_fallback = |a: &Expr<'_>, b: &Expr<'_>| {
            // in `filter(|x| ..)`, replace `*x` with `x`
            let a_path = if_chain! {
                if !is_filter_param_ref;
                if let ExprKind::Unary(UnOp::Deref, expr_path) = a.kind;
                then { expr_path } else { a }
            };
            // let the filter closure arg and the map closure arg be equal
            path_to_local_id(a_path, filter_param_id)
                && path_to_local_id(b, map_param_id)
                && cx.typeck_results().expr_ty_adjusted(a) == cx.typeck_results().expr_ty_adjusted(b)
        };

        if simple_equal || SpanlessEq::new(cx).expr_fallback(eq_fallback).eq_expr(filter_arg, map_arg_peeled);
        then {
            let span = filter_span.with_hi(expr.span.hi());
            let (filter_name, lint) = if is_find {
                ("find", MANUAL_FIND_MAP)
            } else {
                ("filter", MANUAL_FILTER_MAP)
            };
            let msg = format!("`{filter_name}(..).map(..)` can be simplified as `{filter_name}_map(..)`");
            let (to_opt, deref) = if is_result {
                (".ok()", String::new())
            } else {
                let derefs = cx.typeck_results()
                    .expr_adjustments(map_arg)
                    .iter()
                    .filter(|adj| matches!(adj.kind, Adjust::Deref(_)))
                    .count();

                ("", "*".repeat(derefs))
            };
            let sugg = format!(
                "{filter_name}_map(|{map_param_ident}| {deref}{}{to_opt})",
                snippet(cx, map_arg.span, ".."),
            );
            span_lint_and_sugg(cx, lint, span, &msg, "try", sugg, Applicability::MachineApplicable);
        }
    }
}

fn acceptable_methods(method: &PathSegment<'_>) -> bool {
    let methods: [Symbol; 8] = [
        sym::clone,
        sym::as_ref,
        sym!(copied),
        sym!(cloned),
        sym!(as_deref),
        sym!(as_mut),
        sym!(as_deref_mut),
        sym!(to_owned),
    ];

    methods.contains(&method.ident.name)
}
