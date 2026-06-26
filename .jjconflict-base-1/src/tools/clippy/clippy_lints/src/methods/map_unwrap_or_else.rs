use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef as _;
use clippy_utils::source::snippet;
use clippy_utils::sym;
use clippy_utils::usage::mutated_variables;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::MAP_UNWRAP_OR;

/// lint use of `map().unwrap_or_else()` for `Option`s and `Result`s
///
/// Is part of the `map_unwrap_or` lint, split into separate files for readability.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    map_arg: &'tcx hir::Expr<'_>,
    unwrap_arg: &'tcx hir::Expr<'_>,
    msrv: Msrv,
) -> bool {
    let recv_ty = cx.typeck_results().expr_ty(recv).peel_refs();
    let recv_ty_kind = match recv_ty.opt_diag_name(cx) {
        Some(sym::Option) => sym::Option,
        Some(sym::Result) if msrv.meets(cx, msrvs::RESULT_MAP_OR_ELSE) => sym::Result,
        _ => return false,
    };

    // Don't make a suggestion that may fail to compile due to mutably borrowing
    // the same variable twice.
    let Some(map_mutated_vars) = mutated_variables(recv, cx) else {
        return false;
    };
    let Some(unwrap_mutated_vars) = mutated_variables(unwrap_arg, cx) else {
        return false;
    };
    if map_mutated_vars.intersection(&unwrap_mutated_vars).next().is_some() {
        return false;
    }

    // lint message
    let msg = if recv_ty_kind == sym::Option {
        "called `map(<f>).unwrap_or_else(<g>)` on an `Option` value"
    } else {
        "called `map(<f>).unwrap_or_else(<g>)` on a `Result` value"
    };
    // get snippets for args to map() and unwrap_or_else()
    let map_snippet = snippet(cx, map_arg.span, "..");
    let unwrap_snippet = snippet(cx, unwrap_arg.span, "..");
    // lint, with note if both map() and unwrap_or_else() have the same span
    if map_arg.span.eq_ctxt(unwrap_arg.span) {
        let var_snippet = snippet(cx, recv.span, "..");
        span_lint_and_sugg(
            cx,
            MAP_UNWRAP_OR,
            expr.span,
            msg,
            "try",
            format!("{var_snippet}.map_or_else({unwrap_snippet}, {map_snippet})"),
            Applicability::MachineApplicable,
        );
        return true;
    }

    false
}
