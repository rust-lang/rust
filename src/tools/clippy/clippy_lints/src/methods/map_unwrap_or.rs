use crate::utils::usage::mutated_variables;
use crate::utils::{is_type_diagnostic_item, meets_msrv, snippet, span_lint, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_semver::RustcVersion;
use rustc_span::symbol::sym;

use super::MAP_UNWRAP_OR;

const MAP_UNWRAP_OR_MSRV: RustcVersion = RustcVersion::new(1, 41, 0);

/// lint use of `map().unwrap_or_else()` for `Option`s and `Result`s
/// Return true if lint triggered
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    map_args: &'tcx [hir::Expr<'_>],
    unwrap_args: &'tcx [hir::Expr<'_>],
    msrv: Option<&RustcVersion>,
) -> bool {
    if !meets_msrv(msrv, &MAP_UNWRAP_OR_MSRV) {
        return false;
    }
    // lint if the caller of `map()` is an `Option`
    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_args[0]), sym::option_type);
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&map_args[0]), sym::result_type);

    if is_option || is_result {
        // Don't make a suggestion that may fail to compile due to mutably borrowing
        // the same variable twice.
        let map_mutated_vars = mutated_variables(&map_args[0], cx);
        let unwrap_mutated_vars = mutated_variables(&unwrap_args[1], cx);
        if let (Some(map_mutated_vars), Some(unwrap_mutated_vars)) = (map_mutated_vars, unwrap_mutated_vars) {
            if map_mutated_vars.intersection(&unwrap_mutated_vars).next().is_some() {
                return false;
            }
        } else {
            return false;
        }

        // lint message
        let msg = if is_option {
            "called `map(<f>).unwrap_or_else(<g>)` on an `Option` value. This can be done more directly by calling \
            `map_or_else(<g>, <f>)` instead"
        } else {
            "called `map(<f>).unwrap_or_else(<g>)` on a `Result` value. This can be done more directly by calling \
            `.map_or_else(<g>, <f>)` instead"
        };
        // get snippets for args to map() and unwrap_or_else()
        let map_snippet = snippet(cx, map_args[1].span, "..");
        let unwrap_snippet = snippet(cx, unwrap_args[1].span, "..");
        // lint, with note if neither arg is > 1 line and both map() and
        // unwrap_or_else() have the same span
        let multiline = map_snippet.lines().count() > 1 || unwrap_snippet.lines().count() > 1;
        let same_span = map_args[1].span.ctxt() == unwrap_args[1].span.ctxt();
        if same_span && !multiline {
            let var_snippet = snippet(cx, map_args[0].span, "..");
            span_lint_and_sugg(
                cx,
                MAP_UNWRAP_OR,
                expr.span,
                msg,
                "try this",
                format!("{}.map_or_else({}, {})", var_snippet, unwrap_snippet, map_snippet),
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
