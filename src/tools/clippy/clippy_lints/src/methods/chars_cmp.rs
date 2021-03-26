use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{method_chain_args, single_segment_path};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_lint::Lint;
use rustc_middle::ty;
use rustc_span::sym;

/// Wrapper fn for `CHARS_NEXT_CMP` and `CHARS_LAST_CMP` lints.
pub(super) fn check(
    cx: &LateContext<'_>,
    info: &crate::methods::BinaryExprInfo<'_>,
    chain_methods: &[&str],
    lint: &'static Lint,
    suggest: &str,
) -> bool {
    if_chain! {
        if let Some(args) = method_chain_args(info.chain, chain_methods);
        if let hir::ExprKind::Call(ref fun, ref arg_char) = info.other.kind;
        if arg_char.len() == 1;
        if let hir::ExprKind::Path(ref qpath) = fun.kind;
        if let Some(segment) = single_segment_path(qpath);
        if segment.ident.name == sym::Some;
        then {
            let mut applicability = Applicability::MachineApplicable;
            let self_ty = cx.typeck_results().expr_ty_adjusted(&args[0][0]).peel_refs();

            if *self_ty.kind() != ty::Str {
                return false;
            }

            span_lint_and_sugg(
                cx,
                lint,
                info.expr.span,
                &format!("you should use the `{}` method", suggest),
                "like this",
                format!("{}{}.{}({})",
                        if info.eq { "" } else { "!" },
                        snippet_with_applicability(cx, args[0][0].span, "..", &mut applicability),
                        suggest,
                        snippet_with_applicability(cx, arg_char[0].span, "..", &mut applicability)),
                applicability,
            );

            return true;
        }
    }

    false
}
