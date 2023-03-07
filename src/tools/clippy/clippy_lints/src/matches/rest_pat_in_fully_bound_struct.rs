use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::{Pat, PatKind, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::REST_PAT_IN_FULLY_BOUND_STRUCTS;

pub(crate) fn check(cx: &LateContext<'_>, pat: &Pat<'_>) {
    if_chain! {
        if !pat.span.from_expansion();
        if let PatKind::Struct(QPath::Resolved(_, path), fields, true) = pat.kind;
        if let Some(def_id) = path.res.opt_def_id();
        let ty = cx.tcx.type_of(def_id).subst_identity();
        if let ty::Adt(def, _) = ty.kind();
        if def.is_struct() || def.is_union();
        if fields.len() == def.non_enum_variant().fields.len();
        if !def.non_enum_variant().is_field_list_non_exhaustive();

        then {
            span_lint_and_help(
                cx,
                REST_PAT_IN_FULLY_BOUND_STRUCTS,
                pat.span,
                "unnecessary use of `..` pattern in struct binding. All fields were already bound",
                None,
                "consider removing `..` from this binding",
            );
        }
    }
}
