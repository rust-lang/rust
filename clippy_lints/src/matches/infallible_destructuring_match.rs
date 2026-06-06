use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{peel_blocks, strip_pat_refs};
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, LetStmt, MatchSource, PatKind, QPath};
use rustc_lint::LateContext;

use super::INFALLIBLE_DESTRUCTURING_MATCH;

pub(crate) fn check(cx: &LateContext<'_>, local: &LetStmt<'_>) -> bool {
    // turn this:
    //
    // let local = match target {
    //     Variant(binding arg) => { arg }
    // };
    //
    // into this:
    //
    // let Variant(binding local) = target;
    if !local.span.from_expansion()
        && let Some(expr) = local.init
        && let ExprKind::Match(target, [arm], MatchSource::Normal) = expr.kind
        && arm.guard.is_none()
        && let PatKind::TupleStruct(QPath::Resolved(None, variant_name), [arg_lhs], _) = arm.pat.kind
        && let PatKind::Binding(binding, arg_lhs, ..) = strip_pat_refs(arg_lhs).kind
        && let arg_rhs = peel_blocks(arm.body)
        && Some(arg_lhs) == arg_rhs.res_local_id()
    {
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            INFALLIBLE_DESTRUCTURING_MATCH,
            local.span,
            "you seem to be trying to use `match` to destructure a single infallible pattern. \
            Consider using `let`",
            "try",
            format!(
                "let {}({}{}) = {};",
                snippet_with_applicability(cx, variant_name.span, "..", &mut applicability),
                binding.prefix_str(),
                snippet_with_applicability(cx, local.pat.span, "..", &mut applicability),
                snippet_with_applicability(cx, target.span, "..", &mut applicability),
            ),
            applicability,
        );
        return true;
    }
    false
}
