use std::borrow::Cow;

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use clippy_utils::sugg::{Sugg, make_binop};
use clippy_utils::ty::{get_type_diagnostic_name, implements_trait};
use clippy_utils::visitors::is_local_used;
use clippy_utils::{is_from_proc_macro, path_to_local_id};
use rustc_ast::LitKind::Bool;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::UNNECESSARY_MAP_OR;

pub(super) enum Variant {
    Ok,
    Some,
}
impl Variant {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Variant::Ok => "Ok",
            Variant::Some => "Some",
        }
    }

    pub fn method_name(&self) -> &'static str {
        match self {
            Variant::Ok => "is_ok_and",
            Variant::Some => "is_some_and",
        }
    }
}

pub(super) fn check<'a>(
    cx: &LateContext<'a>,
    expr: &Expr<'a>,
    recv: &Expr<'_>,
    def: &Expr<'_>,
    map: &Expr<'_>,
    msrv: &Msrv,
) {
    let ExprKind::Lit(def_kind) = def.kind else {
        return;
    };

    let recv_ty = cx.typeck_results().expr_ty(recv);

    let Bool(def_bool) = def_kind.node else {
        return;
    };

    let variant = match get_type_diagnostic_name(cx, recv_ty) {
        Some(sym::Option) => Variant::Some,
        Some(sym::Result) => Variant::Ok,
        Some(_) | None => return,
    };

    let (sugg, method, applicability) = if let ExprKind::Closure(map_closure) = map.kind
            && let closure_body = cx.tcx.hir().body(map_closure.body)
            && let closure_body_value = closure_body.value.peel_blocks()
            && let ExprKind::Binary(op, l, r) = closure_body_value.kind
            && let Some(param) = closure_body.params.first()
            && let PatKind::Binding(_, hir_id, _, _) = param.pat.kind
            // checking that map_or is one of the following:
            // .map_or(false, |x| x == y)
            // .map_or(false, |x| y == x) - swapped comparison
            // .map_or(true, |x| x != y)
            // .map_or(true, |x| y != x) - swapped comparison
            && ((BinOpKind::Eq == op.node && !def_bool) || (BinOpKind::Ne == op.node && def_bool))
            && let non_binding_location = if path_to_local_id(l, hir_id) { r } else { l }
            && switch_to_eager_eval(cx, non_binding_location)
            // xor, because if its both then thats a strange edge case and
            // we can just ignore it, since by default clippy will error on this
            && (path_to_local_id(l, hir_id) ^ path_to_local_id(r, hir_id))
            && !is_local_used(cx, non_binding_location, hir_id)
            && let typeck_results = cx.typeck_results()
            && typeck_results.expr_ty(l) == typeck_results.expr_ty(r)
            && let Some(partial_eq) = cx.tcx.get_diagnostic_item(sym::PartialEq)
            && implements_trait(cx, recv_ty, partial_eq, &[recv_ty.into()])
    {
        let wrap = variant.variant_name();

        // we may need to add parens around the suggestion
        // in case the parent expression has additional method calls,
        // since for example `Some(5).map_or(false, |x| x == 5).then(|| 1)`
        // being converted to `Some(5) == Some(5).then(|| 1)` isnt
        // the same thing

        let inner_non_binding = Sugg::NonParen(Cow::Owned(format!(
            "{wrap}({})",
            Sugg::hir(cx, non_binding_location, "")
        )));

        let binop = make_binop(op.node, &Sugg::hir(cx, recv, ".."), &inner_non_binding)
            .maybe_par()
            .into_string();

        (binop, "a standard comparison", Applicability::MaybeIncorrect)
    } else if !def_bool
        && msrv.meets(msrvs::OPTION_RESULT_IS_VARIANT_AND)
        && let Some(recv_callsite) = snippet_opt(cx, recv.span.source_callsite())
        && let Some(span_callsite) = snippet_opt(cx, map.span.source_callsite())
    {
        let suggested_name = variant.method_name();
        (
            format!("{recv_callsite}.{suggested_name}({span_callsite})",),
            suggested_name,
            Applicability::MachineApplicable,
        )
    } else if def_bool
        && matches!(variant, Variant::Some)
        && msrv.meets(msrvs::IS_NONE_OR)
        && let Some(recv_callsite) = snippet_opt(cx, recv.span.source_callsite())
        && let Some(span_callsite) = snippet_opt(cx, map.span.source_callsite())
    {
        (
            format!("{recv_callsite}.is_none_or({span_callsite})"),
            "is_none_or",
            Applicability::MachineApplicable,
        )
    } else {
        return;
    };

    if is_from_proc_macro(cx, expr) {
        return;
    }

    span_lint_and_sugg(
        cx,
        UNNECESSARY_MAP_OR,
        expr.span,
        "this `map_or` can be simplified",
        format!("use {method} instead"),
        sugg,
        applicability,
    );
}
