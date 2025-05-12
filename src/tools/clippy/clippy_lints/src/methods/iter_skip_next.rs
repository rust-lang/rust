use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::{is_trait_method, path_to_local};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{BindingMode, Node, PatKind};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::ITER_SKIP_NEXT;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, arg: &hir::Expr<'_>) {
    // lint if caller of skip is an Iterator
    if is_trait_method(cx, expr, sym::Iterator) {
        let mut application = Applicability::MachineApplicable;
        span_lint_and_then(
            cx,
            ITER_SKIP_NEXT,
            expr.span.trim_start(recv.span).unwrap(),
            "called `skip(..).next()` on an iterator",
            |diag| {
                if let Some(id) = path_to_local(recv)
                    && let Node::Pat(pat) = cx.tcx.hir_node(id)
                    && let PatKind::Binding(ann, _, _, _) = pat.kind
                    && ann != BindingMode::MUT
                {
                    application = Applicability::Unspecified;
                    diag.span_help(
                        pat.span,
                        format!("for this change `{}` has to be mutable", snippet(cx, pat.span, "..")),
                    );
                }

                diag.span_suggestion(
                    expr.span.trim_start(recv.span).unwrap(),
                    "use `nth` instead",
                    format!(".nth({})", snippet(cx, arg.span, "..")),
                    application,
                );
            },
        );
    }
}
