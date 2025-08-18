use std::borrow::Cow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::{Sugg, make_binop};
use clippy_utils::ty::{get_type_diagnostic_name, implements_trait, is_copy};
use clippy_utils::visitors::is_local_used;
use clippy_utils::{get_parent_expr, is_from_proc_macro, path_to_local_id};
use rustc_ast::LitKind::Bool;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

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
    method_span: Span,
    msrv: Msrv,
) {
    let ExprKind::Lit(def_kind) = def.kind else {
        return;
    };

    let recv_ty = cx.typeck_results().expr_ty_adjusted(recv);

    let Bool(def_bool) = def_kind.node else {
        return;
    };

    let variant = match get_type_diagnostic_name(cx, recv_ty) {
        Some(sym::Option) => Variant::Some,
        Some(sym::Result) => Variant::Ok,
        Some(_) | None => return,
    };

    let ext_def_span = def.span.until(map.span);

    let (sugg, method, applicability) = if cx.typeck_results().expr_adjustments(recv).is_empty()
            && let ExprKind::Closure(map_closure) = map.kind
            && let closure_body = cx.tcx.hir_body(map_closure.body)
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
            // xor, because if its both then that's a strange edge case and
            // we can just ignore it, since by default clippy will error on this
            && (path_to_local_id(l, hir_id) ^ path_to_local_id(r, hir_id))
            && !is_local_used(cx, non_binding_location, hir_id)
            && let typeck_results = cx.typeck_results()
            && let l_ty = typeck_results.expr_ty(l)
            && l_ty == typeck_results.expr_ty(r)
            && let Some(partial_eq) = cx.tcx.get_diagnostic_item(sym::PartialEq)
            && implements_trait(cx, recv_ty, partial_eq, &[recv_ty.into()])
            && is_copy(cx, l_ty)
    {
        let wrap = variant.variant_name();

        // we may need to add parens around the suggestion
        // in case the parent expression has additional method calls,
        // since for example `Some(5).map_or(false, |x| x == 5).then(|| 1)`
        // being converted to `Some(5) == Some(5).then(|| 1)` isn't
        // the same thing

        let inner_non_binding = Sugg::NonParen(Cow::Owned(format!(
            "{wrap}({})",
            Sugg::hir(cx, non_binding_location, "")
        )));

        let mut app = Applicability::MachineApplicable;
        let binop = make_binop(
            op.node,
            &Sugg::hir_with_applicability(cx, recv, "..", &mut app),
            &inner_non_binding,
        );

        let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr) {
            if parent_expr.span.eq_ctxt(expr.span) {
                match parent_expr.kind {
                    ExprKind::Binary(..) | ExprKind::Unary(..) | ExprKind::Cast(..) => binop.maybe_paren(),
                    ExprKind::MethodCall(_, receiver, _, _) if receiver.hir_id == expr.hir_id => binop.maybe_paren(),
                    _ => binop,
                }
            } else {
                // if our parent expr is created by a macro, then it should be the one taking care of
                // parenthesising us if necessary
                binop
            }
        } else {
            binop
        }
        .into_string();

        (vec![(expr.span, sugg)], "a standard comparison", app)
    } else if !def_bool && msrv.meets(cx, msrvs::OPTION_RESULT_IS_VARIANT_AND) {
        let suggested_name = variant.method_name();
        (
            vec![(method_span, suggested_name.into()), (ext_def_span, String::default())],
            suggested_name,
            Applicability::MachineApplicable,
        )
    } else if def_bool && matches!(variant, Variant::Some) && msrv.meets(cx, msrvs::IS_NONE_OR) {
        (
            vec![(method_span, "is_none_or".into()), (ext_def_span, String::default())],
            "is_none_or",
            Applicability::MachineApplicable,
        )
    } else {
        return;
    };

    if is_from_proc_macro(cx, expr) {
        return;
    }

    span_lint_and_then(
        cx,
        UNNECESSARY_MAP_OR,
        expr.span,
        "this `map_or` can be simplified",
        |diag| {
            diag.multipart_suggestion_verbose(format!("use {method} instead"), sugg, applicability);
        },
    );
}
