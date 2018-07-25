//! lint on inherent implementations

use rustc::hir::*;
use rustc::lint::*;
use rustc::{declare_lint, lint_array};
use std::collections::HashMap;
use std::default::Default;
use syntax_pos::Span;

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
declare_clippy_lint! {
    pub MULTIPLE_INHERENT_IMPL,
    restriction,
    "Multiple inherent impl that could be grouped"
}

pub struct Pass {
    impls: HashMap<def_id::DefId, (Span, Generics)>,
}

impl Default for Pass {
    fn default() -> Self {
        Pass { impls: HashMap::new() }
    }
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(MULTIPLE_INHERENT_IMPL)
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
                    .filter(|(_, generics)| generics.params.len() == 0)
                    .map(|(span, _)| span);
                if let Some(initial_span) = impl_spans.nth(0) {
                    impl_spans.for_each(|additional_span| {
                        cx.span_lint_note(
                            MULTIPLE_INHERENT_IMPL,
                            *additional_span,
                            "Multiple implementations of this structure",
                            *initial_span,
                            "First implementation here",
                        )
                    })
                }
            }
        }
    }
}
