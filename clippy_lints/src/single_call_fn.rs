use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::{is_from_proc_macro, is_in_test_function};
use rustc_data_structures::fx::{FxIndexMap, IndexEntry};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Expr, ExprKind, HirId, Node};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that are only used once. Does not lint tests.
    ///
    /// ### Why is this bad?
    /// It's usually not, splitting a function into multiple parts often improves readability and in
    /// the case of generics, can prevent the compiler from duplicating the function dozens of
    /// time; instead, only duplicating a thunk. But this can prevent segmentation across a
    /// codebase, where many small functions are used only once.
    ///
    /// Note: If this lint is used, prepare to allow this a lot.
    ///
    /// ### Example
    /// ```no_run
    /// pub fn a<T>(t: &T)
    /// where
    ///     T: AsRef<str>,
    /// {
    ///     a_inner(t.as_ref())
    /// }
    ///
    /// fn a_inner(t: &str) {
    ///     /* snip */
    /// }
    ///
    /// ```
    /// Use instead:
    /// ```no_run
    /// pub fn a<T>(t: &T)
    /// where
    ///     T: AsRef<str>,
    /// {
    ///     let t = t.as_ref();
    ///     /* snip */
    /// }
    ///
    /// ```
    #[clippy::version = "1.72.0"]
    pub SINGLE_CALL_FN,
    restriction,
    "checks for functions that are only used once"
}
impl_lint_pass!(SingleCallFn => [SINGLE_CALL_FN]);

#[derive(Debug, Clone)]
pub enum CallState {
    Once { call_site: Span },
    Multiple,
}

#[derive(Clone)]
pub struct SingleCallFn {
    pub avoid_breaking_exported_api: bool,
    pub def_id_to_usage: FxIndexMap<LocalDefId, CallState>,
}

impl SingleCallFn {
    fn is_function_allowed(
        &self,
        cx: &LateContext<'_>,
        fn_def_id: LocalDefId,
        fn_hir_id: HirId,
        fn_span: Span,
    ) -> bool {
        (self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(fn_def_id))
            || in_external_macro(cx.sess(), fn_span)
            || cx
                .tcx
                .hir()
                .maybe_body_owned_by(fn_def_id)
                .map(|body| cx.tcx.hir().body(body))
                .map_or(true, |body| is_in_test_function(cx.tcx, body.value.hir_id))
            || match cx.tcx.hir_node(fn_hir_id) {
                Node::Item(item) => is_from_proc_macro(cx, item),
                Node::ImplItem(item) => is_from_proc_macro(cx, item),
                Node::TraitItem(item) => is_from_proc_macro(cx, item),
                _ => true,
            }
    }
}

/// Whether a called function is a kind of item that the lint cares about.
/// For example, calling an `extern "C" { fn fun(); }` only once is totally fine and does not
/// to be considered.
fn is_valid_item_kind(cx: &LateContext<'_>, def_id: LocalDefId) -> bool {
    matches!(
        cx.tcx.hir_node(cx.tcx.local_def_id_to_hir_id(def_id)),
        Node::Item(_) | Node::ImplItem(_) | Node::TraitItem(_)
    )
}

impl<'tcx> LateLintPass<'tcx> for SingleCallFn {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Path(qpath) = expr.kind
            && let res = cx.qpath_res(&qpath, expr.hir_id)
            && let Some(call_def_id) = res.opt_def_id()
            && let Some(def_id) = call_def_id.as_local()
            && let DefKind::Fn | DefKind::AssocFn = cx.tcx.def_kind(def_id)
            && is_valid_item_kind(cx, def_id)
        {
            match self.def_id_to_usage.entry(def_id) {
                IndexEntry::Occupied(mut entry) => {
                    if let CallState::Once { .. } = entry.get() {
                        entry.insert(CallState::Multiple);
                    }
                },
                IndexEntry::Vacant(entry) => {
                    entry.insert(CallState::Once { call_site: expr.span });
                },
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for (&def_id, usage) in &self.def_id_to_usage {
            if let CallState::Once { call_site } = *usage
                && let fn_hir_id = cx.tcx.local_def_id_to_hir_id(def_id)
                && let fn_span = cx.tcx.hir().span_with_body(fn_hir_id)
                && !self.is_function_allowed(cx, def_id, fn_hir_id, fn_span)
            {
                span_lint_hir_and_then(
                    cx,
                    SINGLE_CALL_FN,
                    fn_hir_id,
                    fn_span,
                    "this function is only used once",
                    |diag| {
                        diag.span_note(call_site, "used here");
                    },
                );
            }
        }
    }
}
