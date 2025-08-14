use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{method_chain_args, path_def_id};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, Lint};
use rustc_middle::ty;
use rustc_span::Symbol;

/// Wrapper fn for `CHARS_NEXT_CMP` and `CHARS_LAST_CMP` lints.
pub(super) fn check(
    cx: &LateContext<'_>,
    info: &crate::methods::BinaryExprInfo<'_>,
    chain_methods: &[Symbol],
    lint: &'static Lint,
    suggest: &str,
) -> bool {
    if let Some(args) = method_chain_args(info.chain, chain_methods)
        && let hir::ExprKind::Call(fun, [arg_char]) = info.other.kind
        && let Some(id) = path_def_id(cx, fun).map(|ctor_id| cx.tcx.parent(ctor_id))
        && Some(id) == cx.tcx.lang_items().option_some_variant()
    {
        let mut applicability = Applicability::MachineApplicable;
        let self_ty = cx.typeck_results().expr_ty_adjusted(args[0].0).peel_refs();

        if *self_ty.kind() != ty::Str {
            return false;
        }

        span_lint_and_sugg(
            cx,
            lint,
            info.expr.span,
            format!("you should use the `{suggest}` method"),
            "like this",
            format!(
                "{}{}.{suggest}({})",
                if info.eq { "" } else { "!" },
                snippet_with_applicability(cx, args[0].0.span, "..", &mut applicability),
                snippet_with_applicability(cx, arg_char.span, "..", &mut applicability)
            ),
            applicability,
        );

        return true;
    }

    false
}
