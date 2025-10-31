use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::qpath_generic_tys;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::snippet;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::OPTION_OPTION;

pub(super) fn check(cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, qpath: &QPath<'_>, def_id: DefId) -> bool {
    if cx.tcx.is_diagnostic_item(sym::Option, def_id)
        && let Some(arg) = qpath_generic_tys(qpath).next()
        && arg.basic_res().opt_def_id() == Some(def_id)
    {
        span_lint_and_then(
            cx,
            OPTION_OPTION,
            hir_ty.span,
            // use just `T` here, as the inner type is not what's problematic
            "use of `Option<Option<T>>`",
            |diag| {
                // but use the specific type here, as:
                // - this is kind of a suggestion
                // - it's printed right after the linted type
                let inner_opt = snippet(cx, arg.span, "_");
                diag.help(format!(
                    "consider using `{inner_opt}`, or a custom enum if you need to distinguish all 3 cases"
                ));
            },
        );
        true
    } else {
        false
    }
}
