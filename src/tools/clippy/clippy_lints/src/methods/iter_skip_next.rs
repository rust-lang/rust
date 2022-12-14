use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_trait_method;
use clippy_utils::path_to_local;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{BindingAnnotation, Node, PatKind};
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
                if_chain! {
                    if let Some(id) = path_to_local(recv);
                    if let Node::Pat(pat) = cx.tcx.hir().get(id);
                    if let PatKind::Binding(ann, _, _, _)  = pat.kind;
                    if ann != BindingAnnotation::MUT;
                    then {
                        application = Applicability::Unspecified;
                        diag.span_help(
                            pat.span,
                            &format!("for this change `{}` has to be mutable", snippet(cx, pat.span, "..")),
                        );
                    }
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
