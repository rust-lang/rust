use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArg, Ty};
use rustc_span::def_id::DefId;
use rustc_span::{Symbol, sym};

use super::CONFUSING_METHOD_TO_NUMERIC_CAST;

fn get_primitive_ty_name(ty: Ty<'_>) -> Option<&'static str> {
    match ty.kind() {
        ty::Char => Some("char"),
        ty::Int(int) => Some(int.name_str()),
        ty::Uint(uint) => Some(uint.name_str()),
        ty::Float(float) => Some(float.name_str()),
        _ => None,
    }
}

fn get_const_name_and_ty_name(
    cx: &LateContext<'_>,
    method_name: Symbol,
    method_def_id: DefId,
    generics: &[GenericArg<'_>],
) -> Option<(&'static str, &'static str)> {
    let method_name = method_name.as_str();
    let diagnostic_name = cx.tcx.get_diagnostic_name(method_def_id);

    let ty_name = if diagnostic_name.is_some_and(|diag| diag == sym::cmp_ord_min || diag == sym::cmp_ord_max) {
        // We get the type on which the `min`/`max` method of the `Ord` trait is implemented.
        if let [ty] = generics
            && let Some(ty) = ty.as_type()
        {
            get_primitive_ty_name(ty)?
        } else {
            return None;
        }
    } else if let Some(impl_id) = cx.tcx.impl_of_method(method_def_id)
        && let Some(ty_name) = get_primitive_ty_name(cx.tcx.type_of(impl_id).instantiate_identity())
        && ["min", "max", "minimum", "maximum", "min_value", "max_value"].contains(&method_name)
    {
        ty_name
    } else {
        return None;
    };

    let const_name = if method_name.starts_with("max") { "MAX" } else { "MIN" };
    Some((const_name, ty_name))
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // We allow casts from any function type to any function type.
    match cast_to.kind() {
        ty::FnDef(..) | ty::FnPtr(..) => return,
        _ => { /* continue to checks */ },
    }

    if let ty::FnDef(def_id, generics) = cast_from.kind()
        && let Some(method_name) = cx.tcx.opt_item_name(*def_id)
        && let Some((const_name, ty_name)) = get_const_name_and_ty_name(cx, method_name, *def_id, generics.as_slice())
    {
        let mut applicability = Applicability::MaybeIncorrect;
        let from_snippet = snippet_with_applicability(cx, cast_expr.span, "..", &mut applicability);

        span_lint_and_then(
            cx,
            CONFUSING_METHOD_TO_NUMERIC_CAST,
            expr.span,
            format!("casting function pointer `{from_snippet}` to `{cast_to}`"),
            |diag| {
                diag.span_suggestion_verbose(
                    expr.span,
                    "did you mean to use the associated constant?",
                    format!("{ty_name}::{const_name} as {cast_to}"),
                    applicability,
                );
            },
        );
    }
}
