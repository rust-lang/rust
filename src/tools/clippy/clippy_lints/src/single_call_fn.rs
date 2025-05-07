use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::{is_from_proc_macro, is_in_test_function};
use rustc_data_structures::fx::{FxIndexMap, IndexEntry};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Expr, ExprKind, HirId, Node};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that are only used once. Does not lint tests.
    ///
    /// ### Why restrict this?
    /// If a function is only used once (perhaps because it used to be used more widely),
    /// then the code could be simplified by moving that function's code into its caller.
    ///
    /// However, there are reasons not to do this everywhere:
    ///
    /// * Splitting a large function into multiple parts often improves readability
    ///   by giving names to its parts.
    /// * A function’s signature might serve a necessary purpose, such as constraining
    ///   the type of a closure passed to it.
    /// * Generic functions might call non-generic functions to reduce duplication
    ///   in the produced machine code.
    ///
    /// If this lint is used, prepare to `#[allow]` it a lot.
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

pub struct SingleCallFn {
    avoid_breaking_exported_api: bool,
    def_id_to_usage: FxIndexMap<LocalDefId, CallState>,
}

impl SingleCallFn {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            def_id_to_usage: FxIndexMap::default(),
        }
    }

    fn is_function_allowed(
        &self,
        cx: &LateContext<'_>,
        fn_def_id: LocalDefId,
        fn_hir_id: HirId,
        fn_span: Span,
    ) -> bool {
        (self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(fn_def_id))
            || fn_span.in_external_macro(cx.sess().source_map())
            || cx
                .tcx
                .hir_maybe_body_owned_by(fn_def_id)
                .is_none_or(|body| is_in_test_function(cx.tcx, body.value.hir_id))
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
        cx.tcx.hir_node_by_def_id(def_id),
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
                && let fn_span = cx.tcx.hir_span_with_body(fn_hir_id)
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
