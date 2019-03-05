//! lint on inherent implementations

use crate::utils::span_lint_and_then;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_data_structures::fx::FxHashMap;
use std::default::Default;
use syntax_pos::Span;

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

pub struct Pass {
    impls: FxHashMap<def_id::DefId, (Span, Generics)>,
}

impl Default for Pass {
    fn default() -> Self {
        Self {
            impls: FxHashMap::default(),
        }
    }
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MULTIPLE_INHERENT_IMPL)
    }

    fn name(&self) -> &'static str {
        "MultipleInherientImpl"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_item(&mut self, _: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let ItemKind::Impl(_, _, _, ref generics, None, _, _) = item.node {
            // Remember for each inherent implementation encoutered its span and generics
            self.impls
                .insert(item.hir_id.owner_def_id(), (item.span, generics.clone()));
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'a, 'tcx>, krate: &'tcx Crate) {
        if let Some(item) = krate.items.values().nth(0) {
            // Retrieve all inherent implementations from the crate, grouped by type
            for impls in cx
                .tcx
                .crate_inherent_impls(item.hir_id.owner_def_id().krate)
                .inherent_impls
                .values()
            {
                // Filter out implementations that have generic params (type or lifetime)
                let mut impl_spans = impls
                    .iter()
                    .filter_map(|impl_def| self.impls.get(impl_def))
                    .filter_map(|(span, generics)| if generics.params.len() == 0 { Some(span) } else { None });
                if let Some(initial_span) = impl_spans.nth(0) {
                    impl_spans.for_each(|additional_span| {
                        span_lint_and_then(
                            cx,
                            MULTIPLE_INHERENT_IMPL,
                            *additional_span,
                            "Multiple implementations of this structure",
                            |db| {
                                db.span_note(*initial_span, "First implementation here");
                            },
                        )
                    })
                }
            }
        }
    }
}
