use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::match_def_path;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::PRIMITIVE_METHOD_TO_NUMERIC_CAST;

fn get_primitive_ty_name(ty: Ty<'_>) -> Option<&'static str> {
    match ty.kind() {
        ty::Char => Some("char"),
        ty::Int(int) => Some(int.name_str()),
        ty::Uint(uint) => Some(uint.name_str()),
        ty::Float(float) => Some(float.name_str()),
        _ => None,
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    // We allow casts from any function type to any function type.
    match cast_to.kind() {
        ty::FnDef(..) | ty::FnPtr(..) => return,
        _ => { /* continue to checks */ },
    }

    if let ty::FnDef(def_id, generics) = cast_from.kind()
        && let Some(method_name) = cx.tcx.opt_item_name(*def_id)
        && let method_name = method_name.as_str()
        && (method_name == "min" || method_name == "max")
        // We get the type on which the `min`/`max` method of the `Ord` trait is implemented.
        && let [ty] = generics.as_slice()
        && let Some(ty) = ty.as_type()
        // We get its name in case it's a primitive with an associated MIN/MAX constant.
        && let Some(ty_name) = get_primitive_ty_name(ty)
        && match_def_path(cx, *def_id, &["core", "cmp", "Ord", method_name])
    {
        let mut applicability = Applicability::MaybeIncorrect;
        let from_snippet = snippet_with_applicability(cx, cast_expr.span, "..", &mut applicability);

        span_lint_and_then(
            cx,
            PRIMITIVE_METHOD_TO_NUMERIC_CAST,
            expr.span,
            format!("casting function pointer `{from_snippet}` to `{cast_to}`"),
            |diag| {
                diag.span_suggestion_verbose(
                    expr.span,
                    "did you mean to use the associated constant?",
                    format!("{ty_name}::{} as {cast_to}", method_name.to_ascii_uppercase()),
                    applicability,
                );
            },
        );
    }
}
