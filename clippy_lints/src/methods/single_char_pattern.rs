use super::utils::get_hint_if_single_char_arg;
use clippy_utils::diagnostics::span_lint_and_sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::Symbol;

use super::SINGLE_CHAR_PATTERN;

const PATTERN_METHODS: [(&str, usize); 24] = [
    ("contains", 0),
    ("starts_with", 0),
    ("ends_with", 0),
    ("find", 0),
    ("rfind", 0),
    ("split", 0),
    ("split_inclusive", 0),
    ("rsplit", 0),
    ("split_terminator", 0),
    ("rsplit_terminator", 0),
    ("splitn", 1),
    ("rsplitn", 1),
    ("split_once", 0),
    ("rsplit_once", 0),
    ("matches", 0),
    ("rmatches", 0),
    ("match_indices", 0),
    ("rmatch_indices", 0),
    ("strip_prefix", 0),
    ("strip_suffix", 0),
    ("trim_start_matches", 0),
    ("trim_end_matches", 0),
    ("replace", 0),
    ("replacen", 0),
];

/// lint for length-1 `str`s for methods in `PATTERN_METHODS`
pub(super) fn check(
    cx: &LateContext<'_>,
    _expr: &hir::Expr<'_>,
    method_name: Symbol,
    receiver: &hir::Expr<'_>,
    args: &[hir::Expr<'_>],
) {
    for &(method, pos) in &PATTERN_METHODS {
        if_chain! {
            if let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty_adjusted(receiver).kind();
            if ty.is_str();
            if method_name.as_str() == method && args.len() > pos;
            let arg = &args[pos];
            let mut applicability = Applicability::MachineApplicable;
            if let Some(hint) = get_hint_if_single_char_arg(cx, arg, &mut applicability);
            then {
                span_lint_and_sugg(
                    cx,
                    SINGLE_CHAR_PATTERN,
                    arg.span,
                    "single-character string constant used as pattern",
                    "try using a `char` instead",
                    hint,
                    applicability,
                );
            }
        }
    }
}
