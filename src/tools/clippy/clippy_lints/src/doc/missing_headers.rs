use super::{DocHeaders, MISSING_ERRORS_DOC, MISSING_PANICS_DOC, MISSING_SAFETY_DOC, UNNECESSARY_SAFETY_DOC};
use clippy_utils::diagnostics::{span_lint, span_lint_and_note};
use clippy_utils::ty::{implements_trait_with_env, is_type_diagnostic_item};
use clippy_utils::{is_doc_hidden, return_ty};
use rustc_hir::{BodyId, FnSig, OwnerId, Safety};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Span, sym};

pub fn check(
    cx: &LateContext<'_>,
    owner_id: OwnerId,
    sig: FnSig<'_>,
    headers: DocHeaders,
    body_id: Option<BodyId>,
    panic_info: Option<(Span, bool)>,
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
    if !headers.panics && panic_info.is_some_and(|el| !el.1) {
        span_lint_and_note(
            cx,
            MISSING_PANICS_DOC,
            span,
            "docs for function which may panic missing `# Panics` section",
            panic_info.map(|el| el.0),
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
