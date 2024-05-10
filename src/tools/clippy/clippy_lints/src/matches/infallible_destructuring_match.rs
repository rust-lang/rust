use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{path_to_local_id, peel_blocks, strip_pat_refs};
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, LetStmt, MatchSource, PatKind, QPath};
use rustc_lint::LateContext;

use super::INFALLIBLE_DESTRUCTURING_MATCH;

pub(crate) fn check(cx: &LateContext<'_>, local: &LetStmt<'_>) -> bool {
    if !local.span.from_expansion()
        && let Some(expr) = local.init
        && let ExprKind::Match(target, arms, MatchSource::Normal) = expr.kind
        && arms.len() == 1
        && arms[0].guard.is_none()
        && let PatKind::TupleStruct(QPath::Resolved(None, variant_name), args, _) = arms[0].pat.kind
        && args.len() == 1
        && let PatKind::Binding(binding, arg, ..) = strip_pat_refs(&args[0]).kind
        && let body = peel_blocks(arms[0].body)
        && path_to_local_id(body, arg)
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
