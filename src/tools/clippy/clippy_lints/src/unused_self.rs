use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::usage::is_todo_unimplemented_stub;
use clippy_utils::visitors::is_local_used;
use rustc_hir::{Impl, ImplItem, ImplItemKind, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks methods that contain a `self` argument but don't use it
    ///
    /// ### Why is this bad?
    /// It may be clearer to define the method as an associated function instead
    /// of an instance method if it doesn't require `self`.
    ///
    /// ### Example
    /// ```rust,ignore
    /// struct A;
    /// impl A {
    ///     fn method(&self) {}
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust,ignore
    /// struct A;
    /// impl A {
    ///     fn method() {}
    /// }
    /// ```
    #[clippy::version = "1.40.0"]
    pub UNUSED_SELF,
    pedantic,
    "methods that contain a `self` argument but don't use it"
}

pub struct UnusedSelf {
    avoid_breaking_exported_api: bool,
}

impl_lint_pass!(UnusedSelf => [UNUSED_SELF]);

impl UnusedSelf {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for UnusedSelf {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &ImplItem<'_>) {
        if impl_item.span.from_expansion() {
            return;
        }
        let parent = cx.tcx.hir_get_parent_item(impl_item.hir_id()).def_id;
        let parent_item = cx.tcx.hir_expect_item(parent);
        let assoc_item = cx.tcx.associated_item(impl_item.owner_id);
        if let ItemKind::Impl(Impl { of_trait: None, .. }) = parent_item.kind
            && assoc_item.is_method()
            && let ImplItemKind::Fn(.., body_id) = &impl_item.kind
            && (!cx.effective_visibilities.is_exported(impl_item.owner_id.def_id) || !self.avoid_breaking_exported_api)
            && let body = cx.tcx.hir_body(*body_id)
            && let [self_param, ..] = body.params
            && !is_local_used(cx, body, self_param.pat.hir_id)
            && !is_todo_unimplemented_stub(cx, body.value)
        {
            span_lint_and_help(
                cx,
                UNUSED_SELF,
                self_param.span,
                "unused `self` argument",
                None,
                "consider refactoring to an associated function",
            );
        }
    }
}
