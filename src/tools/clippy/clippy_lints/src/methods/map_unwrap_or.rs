use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::usage::mutated_variables;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::MAP_UNWRAP_OR;

/// lint use of `map().unwrap_or_else()` for `Option`s and `Result`s
///
/// Returns true if the lint was emitted
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    map_arg: &'tcx hir::Expr<'_>,
    unwrap_arg: &'tcx hir::Expr<'_>,
    msrv: Msrv,
) -> bool {
    // lint if the caller of `map()` is an `Option` or a `Result`.
    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Option);
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result);

    if is_result && !msrv.meets(cx, msrvs::RESULT_MAP_OR_ELSE) {
        return false;
    }

    if is_option || is_result {
        // Don't make a suggestion that may fail to compile due to mutably borrowing
        // the same variable twice.
        let map_mutated_vars = mutated_variables(recv, cx);
        let unwrap_mutated_vars = mutated_variables(unwrap_arg, cx);
        if let (Some(map_mutated_vars), Some(unwrap_mutated_vars)) = (map_mutated_vars, unwrap_mutated_vars) {
            if map_mutated_vars.intersection(&unwrap_mutated_vars).next().is_some() {
                return false;
            }
        } else {
            return false;
        }

        // lint message
        let msg = if is_option {
            "called `map(<f>).unwrap_or_else(<g>)` on an `Option` value"
        } else {
            "called `map(<f>).unwrap_or_else(<g>)` on a `Result` value"
        };
        // get snippets for args to map() and unwrap_or_else()
        let map_snippet = snippet(cx, map_arg.span, "..");
        let unwrap_snippet = snippet(cx, unwrap_arg.span, "..");
        // lint, with note if neither arg is > 1 line and both map() and
        // unwrap_or_else() have the same span
        let multiline = map_snippet.lines().count() > 1 || unwrap_snippet.lines().count() > 1;
        let same_span = map_arg.span.eq_ctxt(unwrap_arg.span);
        if same_span && !multiline {
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
        } else if same_span && multiline {
            span_lint(cx, MAP_UNWRAP_OR, expr.span, msg);
            return true;
        }
    }

    false
}
