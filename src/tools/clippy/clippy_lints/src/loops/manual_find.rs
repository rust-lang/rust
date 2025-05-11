use super::MANUAL_FIND;
use super::utils::make_iterator_snippet;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::implements_trait;
use clippy_utils::usage::contains_return_break_continue_macro;
use clippy_utils::{higher, is_res_lang_ctor, path_res, peel_blocks_with_stmt};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{BindingMode, Block, Expr, ExprKind, HirId, Node, Pat, PatKind, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_span::Span;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    span: Span,
    expr: &'tcx Expr<'_>,
) {
    let inner_expr = peel_blocks_with_stmt(body);
    // Check for the specific case that the result is returned and optimize suggestion for that (more
    // cases can be added later)
    if let Some(higher::If {
        cond,
        then,
        r#else: None,
    }) = higher::If::hir(inner_expr)
        && let Some(binding_id) = get_binding(pat)
        && let ExprKind::Block(block, _) = then.kind
        && let [stmt] = block.stmts
        && let StmtKind::Semi(semi) = stmt.kind
        && let ExprKind::Ret(Some(ret_value)) = semi.kind
        && let ExprKind::Call(ctor, [inner_ret]) = ret_value.kind
        && is_res_lang_ctor(cx, path_res(cx, ctor), LangItem::OptionSome)
        && path_res(cx, inner_ret) == Res::Local(binding_id)
        && !contains_return_break_continue_macro(cond)
        && let Some((last_stmt, last_ret)) = last_stmt_and_ret(cx, expr)
    {
        let mut applicability = Applicability::MachineApplicable;
        let mut snippet = make_iterator_snippet(cx, arg, &mut applicability);
        // Checks if `pat` is a single reference to a binding (`&x`)
        let is_ref_to_binding =
            matches!(pat.kind, PatKind::Ref(inner, _) if matches!(inner.kind, PatKind::Binding(..)));
        // If `pat` is not a binding or a reference to a binding (`x` or `&x`)
        // we need to map it to the binding returned by the function (i.e. `.map(|(x, _)| x)`)
        if !(matches!(pat.kind, PatKind::Binding(..)) || is_ref_to_binding) {
            snippet.push_str(
                &format!(
                    ".map(|{}| {})",
                    snippet_with_applicability(cx, pat.span, "..", &mut applicability),
                    snippet_with_applicability(cx, inner_ret.span, "..", &mut applicability),
                )[..],
            );
        }
        let ty = cx.typeck_results().expr_ty(inner_ret);
        if cx
            .tcx
            .lang_items()
            .copy_trait()
            .is_some_and(|id| implements_trait(cx, ty, id, &[]))
        {
            snippet.push_str(
                &format!(
                    ".find(|{}{}| {})",
                    "&".repeat(1 + usize::from(is_ref_to_binding)),
                    snippet_with_applicability(cx, inner_ret.span, "..", &mut applicability),
                    snippet_with_applicability(cx, cond.span, "..", &mut applicability),
                )[..],
            );
            if is_ref_to_binding {
                snippet.push_str(".copied()");
            }
        } else {
            applicability = Applicability::MaybeIncorrect;
            snippet.push_str(
                &format!(
                    ".find(|{}| {})",
                    snippet_with_applicability(cx, inner_ret.span, "..", &mut applicability),
                    snippet_with_applicability(cx, cond.span, "..", &mut applicability),
                )[..],
            );
        }
        // Extends to `last_stmt` to include semicolon in case of `return None;`
        let lint_span = span.to(last_stmt.span).to(last_ret.span);
        span_lint_and_then(
            cx,
            MANUAL_FIND,
            lint_span,
            "manual implementation of `Iterator::find`",
            |diag| {
                if applicability == Applicability::MaybeIncorrect {
                    diag.note("you may need to dereference some variables");
                }
                diag.span_suggestion(lint_span, "replace with an iterator", snippet, applicability);
            },
        );
    }
}

fn get_binding(pat: &Pat<'_>) -> Option<HirId> {
    let mut hir_id = None;
    let mut count = 0;
    pat.each_binding(|annotation, id, _, _| {
        count += 1;
        if count > 1 {
            hir_id = None;
            return;
        }
        if let BindingMode::NONE = annotation {
            hir_id = Some(id);
        }
    });
    hir_id
}

// Returns the last statement and last return if function fits format for lint
fn last_stmt_and_ret<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
) -> Option<(&'tcx Stmt<'tcx>, &'tcx Expr<'tcx>)> {
    // Returns last non-return statement and the last return
    fn extract<'tcx>(block: &Block<'tcx>) -> Option<(&'tcx Stmt<'tcx>, &'tcx Expr<'tcx>)> {
        if let [.., last_stmt] = block.stmts {
            if let Some(ret) = block.expr {
                return Some((last_stmt, ret));
            }
            if let [.., snd_last, _] = block.stmts
                && let StmtKind::Semi(last_expr) = last_stmt.kind
                && let ExprKind::Ret(Some(ret)) = last_expr.kind
            {
                return Some((snd_last, ret));
            }
        }
        None
    }
    let mut parent_iter = cx.tcx.hir_parent_iter(expr.hir_id);
    if let Some((node_hir, Node::Stmt(..))) = parent_iter.next()
        // This should be the loop
        // This should be the function body
        && let Some((_, Node::Block(block))) = parent_iter.next()
        && let Some((last_stmt, last_ret)) = extract(block)
        && last_stmt.hir_id == node_hir
        && is_res_lang_ctor(cx, path_res(cx, last_ret), LangItem::OptionNone)
        && let Some((_, Node::Expr(_block))) = parent_iter.next()
        // This includes the function header
        && let Some((_, func)) = parent_iter.next()
        && func.fn_kind().is_some()
    {
        Some((block.stmts.last().unwrap(), last_ret))
    } else {
        None
    }
}
