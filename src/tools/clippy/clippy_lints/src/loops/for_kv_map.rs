use super::FOR_KV_MAP;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::{snippet_with_applicability, walk_span_to_context};
use clippy_utils::{pat_is_wild, sugg};
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Span, sym};

/// Checks for the `FOR_KV_MAP` lint.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    span: Span,
) {
    let pat_span = pat.span;

    if let PatKind::Tuple(pat, _) = pat.kind
        && pat.len() == 2
    {
        let arg_span = arg.span;
        let (new_pat_span, kind, ty, mutbl) = match *cx.typeck_results().expr_ty(arg).kind() {
            ty::Ref(_, ty, mutbl) => match (&pat[0].kind, &pat[1].kind) {
                (key, _) if pat_is_wild(cx, key, body) => (pat[1].span, "value", ty, mutbl),
                (_, value) if pat_is_wild(cx, value, body) => (pat[0].span, "key", ty, Mutability::Not),
                _ => return,
            },
            _ => return,
        };
        let mutbl = match mutbl {
            Mutability::Not => "",
            Mutability::Mut => "_mut",
        };
        let arg = match arg.kind {
            ExprKind::AddrOf(BorrowKind::Ref, _, expr) => expr,
            _ => arg,
        };

        if matches!(ty.opt_diag_name(cx), Some(sym::HashMap | sym::BTreeMap))
            && let Some(arg_span) = walk_span_to_context(arg_span, span.ctxt())
        {
            span_lint_and_then(
                cx,
                FOR_KV_MAP,
                arg_span,
                format!("you seem to want to iterate on a map's {kind}s"),
                |diag| {
                    let mut applicability = Applicability::MachineApplicable;
                    let map = sugg::Sugg::hir_with_context(cx, arg, span.ctxt(), "map", &mut applicability);
                    let pat = snippet_with_applicability(cx, new_pat_span, kind, &mut applicability);
                    diag.multipart_suggestion(
                        "use the corresponding method",
                        vec![
                            (pat_span, pat.to_string()),
                            (arg_span, format!("{}.{kind}s{mutbl}()", map.maybe_paren())),
                        ],
                        applicability,
                    );
                },
            );
        }
    }
}
