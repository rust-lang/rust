use super::utils::get_hint_if_single_char_arg;
use clippy_utils::diagnostics::span_lint_and_sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::symbol::Symbol;

use super::SINGLE_CHAR_PATTERN;

/// lint for length-1 `str`s for methods in `PATTERN_METHODS`
pub(super) fn check(cx: &LateContext<'_>, _expr: &hir::Expr<'_>, method_name: Symbol, args: &[hir::Expr<'_>]) {
    for &(method, pos) in &crate::methods::PATTERN_METHODS {
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
