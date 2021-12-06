use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind, VisibilityKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for items declared `pub(crate)` that are not crate visible because they
    /// are inside a private module.
    ///
    /// ### Why is this bad?
    /// Writing `pub(crate)` is misleading when it's redundant due to the parent
    /// module's visibility.
    ///
    /// ### Example
    /// ```rust
    /// mod internal {
    ///     pub(crate) fn internal_fn() { }
    /// }
    /// ```
    /// This function is not visible outside the module and it can be declared with `pub` or
    /// private visibility
    /// ```rust
    /// mod internal {
    ///     pub fn internal_fn() { }
    /// }
    /// ```
    #[clippy::version = "1.44.0"]
    pub REDUNDANT_PUB_CRATE,
    nursery,
    "Using `pub(crate)` visibility on items that are not crate visible due to the visibility of the module that contains them."
}

#[derive(Default)]
pub struct RedundantPubCrate {
    is_exported: Vec<bool>,
}

impl_lint_pass!(RedundantPubCrate => [REDUNDANT_PUB_CRATE]);

impl<'tcx> LateLintPass<'tcx> for RedundantPubCrate {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let VisibilityKind::Crate { .. } = item.vis.node {
            if !cx.access_levels.is_exported(item.def_id) && self.is_exported.last() == Some(&false) {
                let span = item.span.with_hi(item.ident.span.hi());
                let descr = cx.tcx.def_kind(item.def_id).descr(item.def_id.to_def_id());
                span_lint_and_then(
                    cx,
                    REDUNDANT_PUB_CRATE,
                    span,
                    &format!("pub(crate) {} inside private module", descr),
                    |diag| {
                        diag.span_suggestion(
                            item.vis.span,
                            "consider using",
                            "pub".to_string(),
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }
        }

        if let ItemKind::Mod { .. } = item.kind {
            self.is_exported.push(cx.access_levels.is_exported(item.def_id));
        }
    }

    fn check_item_post(&mut self, _cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Mod { .. } = item.kind {
            self.is_exported.pop().expect("unbalanced check_item/check_item_post");
        }
    }
}
