use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{path_to_local_id, peel_blocks, strip_pat_refs};
use rustc_errors::Applicability;
use rustc_hir::{ByRef, ExprKind, Local, MatchSource, PatKind, QPath};
use rustc_lint::LateContext;

use super::INFALLIBLE_DESTRUCTURING_MATCH;

pub(crate) fn check(cx: &LateContext<'_>, local: &Local<'_>) -> bool {
    if_chain! {
        if !local.span.from_expansion();
        if let Some(expr) = local.init;
        if let ExprKind::Match(target, arms, MatchSource::Normal) = expr.kind;
        if arms.len() == 1 && arms[0].guard.is_none();
        if let PatKind::TupleStruct(
            QPath::Resolved(None, variant_name), args, _) = arms[0].pat.kind;
        if args.len() == 1;
        if let PatKind::Binding(binding, arg, ..) = strip_pat_refs(&args[0]).kind;
        let body = peel_blocks(arms[0].body);
        if path_to_local_id(body, arg);

        then {
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
                    if binding.0 == ByRef::Yes { "ref " } else { "" },
                    snippet_with_applicability(cx, local.pat.span, "..", &mut applicability),
                    snippet_with_applicability(cx, target.span, "..", &mut applicability),
                ),
                applicability,
            );
            return true;
        }
    }
    false
}
