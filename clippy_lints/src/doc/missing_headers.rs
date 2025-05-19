use super::{DocHeaders, MISSING_ERRORS_DOC, MISSING_PANICS_DOC, MISSING_SAFETY_DOC, UNNECESSARY_SAFETY_DOC};
use clippy_utils::diagnostics::{span_lint, span_lint_and_note};
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::ty::{get_type_diagnostic_name, implements_trait_with_env, is_type_diagnostic_item};
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{fulfill_or_allowed, is_doc_hidden, method_chain_args, return_ty};
use rustc_hir::{BodyId, FnSig, OwnerId, Safety};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Span, sym};
use std::ops::ControlFlow;

pub fn check(
    cx: &LateContext<'_>,
    owner_id: OwnerId,
    sig: FnSig<'_>,
    headers: DocHeaders,
    body_id: Option<BodyId>,
    check_private_items: bool,
) {
    if !check_private_items && !cx.effective_visibilities.is_exported(owner_id.def_id) {
        return; // Private functions do not require doc comments
    }

    // do not lint if any parent has `#[doc(hidden)]` attribute (#7347)
    if !check_private_items
        && cx
            .tcx
            .hir_parent_iter(owner_id.into())
            .any(|(id, _node)| is_doc_hidden(cx.tcx.hir_attrs(id)))
    {
        return;
    }

    let span = cx.tcx.def_span(owner_id);
    match (headers.safety, sig.header.safety()) {
        (false, Safety::Unsafe) => span_lint(
            cx,
            MISSING_SAFETY_DOC,
            span,
            "unsafe function's docs are missing a `# Safety` section",
        ),
        (true, Safety::Safe) => span_lint(
            cx,
            UNNECESSARY_SAFETY_DOC,
            span,
            "safe function's docs have unnecessary `# Safety` section",
        ),
        _ => (),
    }
    if !headers.panics
        && let Some(body_id) = body_id
        && let Some(panic_span) = find_panic(cx, body_id)
    {
        span_lint_and_note(
            cx,
            MISSING_PANICS_DOC,
            span,
            "docs for function which may panic missing `# Panics` section",
            Some(panic_span),
            "first possible panic found here",
        );
    }
    if !headers.errors {
        if is_type_diagnostic_item(cx, return_ty(cx, owner_id), sym::Result) {
            span_lint(
                cx,
                MISSING_ERRORS_DOC,
                span,
                "docs for function returning `Result` missing `# Errors` section",
            );
        } else if let Some(body_id) = body_id
            && let Some(future) = cx.tcx.lang_items().future_trait()
            && let typeck = cx.tcx.typeck_body(body_id)
            && let body = cx.tcx.hir_body(body_id)
            && let ret_ty = typeck.expr_ty(body.value)
            && implements_trait_with_env(
                cx.tcx,
                ty::TypingEnv::non_body_analysis(cx.tcx, owner_id.def_id),
                ret_ty,
                future,
                Some(owner_id.def_id.to_def_id()),
                &[],
            )
            && let ty::Coroutine(_, subs) = ret_ty.kind()
            && is_type_diagnostic_item(cx, subs.as_coroutine().return_ty(), sym::Result)
        {
            span_lint(
                cx,
                MISSING_ERRORS_DOC,
                span,
                "docs for function returning `Result` missing `# Errors` section",
            );
        }
    }
}

fn find_panic(cx: &LateContext<'_>, body_id: BodyId) -> Option<Span> {
    let mut panic_span = None;
    let typeck = cx.tcx.typeck_body(body_id);
    for_each_expr(cx, cx.tcx.hir_body(body_id), |expr| {
        if let Some(macro_call) = root_macro_call_first_node(cx, expr)
            && (is_panic(cx, macro_call.def_id)
                || matches!(
                    cx.tcx.get_diagnostic_name(macro_call.def_id),
                    Some(sym::assert_macro | sym::assert_eq_macro | sym::assert_ne_macro)
                ))
            && !cx.tcx.hir_is_inside_const_context(expr.hir_id)
            && !fulfill_or_allowed(cx, MISSING_PANICS_DOC, [expr.hir_id])
            && panic_span.is_none()
        {
            panic_span = Some(macro_call.span);
        }

        // check for `unwrap` and `expect` for both `Option` and `Result`
        if let Some(arglists) =
            method_chain_args(expr, &[sym::unwrap]).or_else(|| method_chain_args(expr, &[sym::expect]))
            && let receiver_ty = typeck.expr_ty(arglists[0].0).peel_refs()
            && matches!(
                get_type_diagnostic_name(cx, receiver_ty),
                Some(sym::Option | sym::Result)
            )
            && !fulfill_or_allowed(cx, MISSING_PANICS_DOC, [expr.hir_id])
            && panic_span.is_none()
        {
            panic_span = Some(expr.span);
        }

        // Visit all nodes to fulfill any `#[expect]`s after the first linted panic
        ControlFlow::<!>::Continue(())
    });
    panic_span
}
