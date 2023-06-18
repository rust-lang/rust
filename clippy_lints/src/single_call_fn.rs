use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{is_from_proc_macro, is_in_test_function};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{intravisit::FnKind, Body, Expr, ExprKind, FnDecl};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
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
    /// ```rust
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
    /// ```rust
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

#[derive(Clone)]
pub struct SingleCallFn {
    pub avoid_breaking_exported_api: bool,
    pub def_id_to_usage: FxHashMap<LocalDefId, (Span, Vec<Span>)>,
}

impl<'tcx> LateLintPass<'tcx> for SingleCallFn {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(def_id)
            || in_external_macro(cx.sess(), span)
            || is_from_proc_macro(cx, &(&kind, body, cx.tcx.local_def_id_to_hir_id(def_id), span))
            || is_in_test_function(cx.tcx, body.value.hir_id)
        {
            return;
        }

        self.def_id_to_usage.insert(def_id, (span, vec![]));
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let mut v = FnUsageVisitor {
            cx,
            def_id_to_usage: &mut self.def_id_to_usage,
        };
        cx.tcx.hir().visit_all_item_likes_in_crate(&mut v);

        for usage in self.def_id_to_usage.values() {
            let single_call_fn_span = usage.0;
            if let [caller_span] = *usage.1 {
                span_lint_and_help(
                    cx,
                    SINGLE_CALL_FN,
                    single_call_fn_span,
                    "this function is only used once",
                    Some(caller_span),
                    "used here",
                );
            }
        }
    }
}

struct FnUsageVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    def_id_to_usage: &'a mut FxHashMap<LocalDefId, (Span, Vec<Span>)>,
}

impl<'a, 'tcx> Visitor<'tcx> for FnUsageVisitor<'a, 'tcx> {
    type NestedFilter = OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        let Self { cx, .. } = *self;

        if let ExprKind::Path(qpath) = expr.kind
            && let res = cx.qpath_res(&qpath, expr.hir_id)
            && let Some(call_def_id) = res.opt_def_id()
            && let Some(def_id) = call_def_id.as_local()
            && let Some(usage) = self.def_id_to_usage.get_mut(&def_id)
        {
            usage.1.push(expr.span);
        }

        walk_expr(self, expr);
    }
}
