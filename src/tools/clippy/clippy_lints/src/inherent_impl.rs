//! lint on inherent implementations

use crate::utils::{in_macro, span_lint_and_then};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{def_id, Crate, Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for multiple inherent implementations of a struct
    ///
    /// **Why is this bad?** Splitting the implementation of a type makes the code harder to navigate.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// struct X;
    /// impl X {
    ///     fn one() {}
    /// }
    /// impl X {
    ///     fn other() {}
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// struct X;
    /// impl X {
    ///     fn one() {}
    ///     fn other() {}
    /// }
    /// ```
    pub MULTIPLE_INHERENT_IMPL,
    restriction,
    "Multiple inherent impl that could be grouped"
}

#[allow(clippy::module_name_repetitions)]
#[derive(Default)]
pub struct MultipleInherentImpl {
    impls: FxHashMap<def_id::DefId, Span>,
}

impl_lint_pass!(MultipleInherentImpl => [MULTIPLE_INHERENT_IMPL]);

impl<'tcx> LateLintPass<'tcx> for MultipleInherentImpl {
    fn check_item(&mut self, _: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Impl(Impl {
            ref generics,
            of_trait: None,
            ..
        }) = item.kind
        {
            // Remember for each inherent implementation encountered its span and generics
            // but filter out implementations that have generic params (type or lifetime)
            // or are derived from a macro
            if !in_macro(item.span) && generics.params.is_empty() {
                self.impls.insert(item.hir_id.owner.to_def_id(), item.span);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>, krate: &'tcx Crate<'_>) {
        if let Some(item) = krate.items.values().next() {
            // Retrieve all inherent implementations from the crate, grouped by type
            for impls in cx
                .tcx
                .crate_inherent_impls(item.hir_id.owner.to_def_id().krate)
                .inherent_impls
                .values()
            {
                // Filter out implementations that have generic params (type or lifetime)
                let mut impl_spans = impls.iter().filter_map(|impl_def| self.impls.get(impl_def));
                if let Some(initial_span) = impl_spans.next() {
                    impl_spans.for_each(|additional_span| {
                        span_lint_and_then(
                            cx,
                            MULTIPLE_INHERENT_IMPL,
                            *additional_span,
                            "multiple implementations of this structure",
                            |diag| {
                                diag.span_note(*initial_span, "first implementation here");
                            },
                        )
                    })
                }
            }
        }
    }
}
