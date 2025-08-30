use clippy_utils::diagnostics::span_lint_and_then;
use rustc_hir::{Pat, PatKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::REST_PAT_IN_FULLY_BOUND_STRUCTS;

pub(crate) fn check(cx: &LateContext<'_>, pat: &Pat<'_>) {
    if !pat.span.from_expansion()
        && let PatKind::Struct(QPath::Resolved(_, path), fields, Some(_)) = pat.kind
        && let Some(def_id) = path.res.opt_def_id()
        && let ty = cx.tcx.type_of(def_id).instantiate_identity()
        && let ty::Adt(def, _) = ty.kind()
        && (def.is_struct() || def.is_union())
        && fields.len() == def.non_enum_variant().fields.len()
        && !def.non_enum_variant().is_field_list_non_exhaustive()
    {
        #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
        span_lint_and_then(
            cx,
            REST_PAT_IN_FULLY_BOUND_STRUCTS,
            pat.span,
            "unnecessary use of `..` pattern in struct binding. All fields were already bound",
            |diag| {
                diag.help("consider removing `..` from this binding");
            },
        );
    }
}
