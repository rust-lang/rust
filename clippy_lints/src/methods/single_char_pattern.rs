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
    ("contains", 1),
    ("starts_with", 1),
    ("ends_with", 1),
    ("find", 1),
    ("rfind", 1),
    ("split", 1),
    ("split_inclusive", 1),
    ("rsplit", 1),
    ("split_terminator", 1),
    ("rsplit_terminator", 1),
    ("splitn", 2),
    ("rsplitn", 2),
    ("split_once", 1),
    ("rsplit_once", 1),
    ("matches", 1),
    ("rmatches", 1),
    ("match_indices", 1),
    ("rmatch_indices", 1),
    ("strip_prefix", 1),
    ("strip_suffix", 1),
    ("trim_start_matches", 1),
    ("trim_end_matches", 1),
    ("replace", 1),
    ("replacen", 1),
];

/// lint for length-1 `str`s for methods in `PATTERN_METHODS`
pub(super) fn check(cx: &LateContext<'_>, _expr: &hir::Expr<'_>, method_name: Symbol, args: &[hir::Expr<'_>]) {
    for &(method, pos) in &PATTERN_METHODS {
        if_chain! {
            if let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty_adjusted(&args[0]).kind();
            if *ty.kind() == ty::Str;
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
