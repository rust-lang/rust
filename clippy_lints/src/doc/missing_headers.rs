use super::{DocHeaders, MISSING_ERRORS_DOC, MISSING_PANICS_DOC, MISSING_SAFETY_DOC, UNNECESSARY_SAFETY_DOC};
use clippy_utils::diagnostics::{span_lint, span_lint_and_note};
use clippy_utils::macros::{is_panic, root_macro_call_first_node};
use clippy_utils::ty::{implements_trait_with_env, is_type_diagnostic_item};
use clippy_utils::visitors::Visitable;
use clippy_utils::{is_doc_hidden, method_chain_args, return_ty};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{AnonConst, BodyId, Expr, FnSig, OwnerId, Safety};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::ty;
use rustc_span::{Span, sym};

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
        && let Some((panic_span, false)) = FindPanicUnwrap::find_span(cx, body_id)
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

struct FindPanicUnwrap<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    is_const: bool,
    panic_span: Option<Span>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
}

impl<'a, 'tcx> FindPanicUnwrap<'a, 'tcx> {
    pub fn find_span(cx: &'a LateContext<'tcx>, body_id: BodyId) -> Option<(Span, bool)> {
        let mut vis = Self {
            cx,
            is_const: false,
            panic_span: None,
            typeck_results: cx.tcx.typeck_body(body_id),
        };
        cx.tcx.hir_body(body_id).visit(&mut vis);
        vis.panic_span.map(|el| (el, vis.is_const))
    }
}

impl<'tcx> Visitor<'tcx> for FindPanicUnwrap<'_, 'tcx> {
    type NestedFilter = OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.panic_span.is_some() {
            return;
        }

        if let Some(macro_call) = root_macro_call_first_node(self.cx, expr) {
            if is_panic(self.cx, macro_call.def_id)
                || matches!(
                    self.cx.tcx.item_name(macro_call.def_id).as_str(),
                    "assert" | "assert_eq" | "assert_ne"
                )
            {
                self.is_const = self.cx.tcx.hir_is_inside_const_context(expr.hir_id);
                self.panic_span = Some(macro_call.span);
            }
        }

        // check for `unwrap` and `expect` for both `Option` and `Result`
        if let Some(arglists) = method_chain_args(expr, &["unwrap"]).or(method_chain_args(expr, &["expect"])) {
            let receiver_ty = self.typeck_results.expr_ty(arglists[0].0).peel_refs();
            if is_type_diagnostic_item(self.cx, receiver_ty, sym::Option)
                || is_type_diagnostic_item(self.cx, receiver_ty, sym::Result)
            {
                self.panic_span = Some(expr.span);
            }
        }

        // and check sub-expressions
        intravisit::walk_expr(self, expr);
    }

    // Panics in const blocks will cause compilation to fail.
    fn visit_anon_const(&mut self, _: &'tcx AnonConst) {}

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}
