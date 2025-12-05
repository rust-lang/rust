use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::HasSession;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Item, ItemKind, UseKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::def_id::CRATE_DEF_ID;

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
    /// ```no_run
    /// mod internal {
    ///     pub(crate) fn internal_fn() { }
    /// }
    /// ```
    /// This function is not visible outside the module and it can be declared with `pub` or
    /// private visibility
    /// ```no_run
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
        if cx.tcx.visibility(item.owner_id.def_id) == ty::Visibility::Restricted(CRATE_DEF_ID.to_def_id())
            && !cx.effective_visibilities.is_exported(item.owner_id.def_id)
            && self.is_exported.last() == Some(&false)
            && !is_ignorable_export(item)
            && !item.span.in_external_macro(cx.sess().source_map())
        {
            let span = item
                .kind
                .ident()
                .map_or(item.span, |ident| item.span.with_hi(ident.span.hi()));
            let descr = cx.tcx.def_kind(item.owner_id).descr(item.owner_id.to_def_id());
            span_lint_and_then(
                cx,
                REDUNDANT_PUB_CRATE,
                span,
                format!("pub(crate) {descr} inside private module"),
                |diag| {
                    diag.span_suggestion(
                        item.vis_span,
                        "consider using",
                        "pub".to_string(),
                        Applicability::MachineApplicable,
                    );
                },
            );
        }

        if let ItemKind::Mod { .. } = item.kind {
            self.is_exported
                .push(cx.effective_visibilities.is_exported(item.owner_id.def_id));
        }
    }

    fn check_item_post(&mut self, _cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Mod { .. } = item.kind {
            self.is_exported.pop().expect("unbalanced check_item/check_item_post");
        }
    }
}

// We ignore macro exports. And `ListStem` uses, which aren't interesting.
fn is_ignorable_export<'tcx>(item: &'tcx Item<'tcx>) -> bool {
    if let ItemKind::Use(path, kind) = item.kind {
        let ignore = matches!(path.res.macro_ns, Some(Res::Def(DefKind::Macro(_), _))) || kind == UseKind::ListStem;
        if ignore {
            return true;
        }
    } else if let ItemKind::Macro(..) = item.kind {
        return true;
    }

    false
}
