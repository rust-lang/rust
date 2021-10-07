//! lint on inherent implementations

use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::{in_macro, is_lint_allowed};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{def_id::LocalDefId, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;
use std::collections::hash_map::Entry;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for multiple inherent implementations of a struct
    ///
    /// ### Why is this bad?
    /// Splitting the implementation of a type makes the code harder to navigate.
    ///
    /// ### Example
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

declare_lint_pass!(MultipleInherentImpl => [MULTIPLE_INHERENT_IMPL]);

impl<'tcx> LateLintPass<'tcx> for MultipleInherentImpl {
    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        // Map from a type to it's first impl block. Needed to distinguish generic arguments.
        // e.g. `Foo<Bar>` and `Foo<Baz>`
        let mut type_map = FxHashMap::default();
        // List of spans to lint. (lint_span, first_span)
        let mut lint_spans = Vec::new();

        for (_, impl_ids) in cx
            .tcx
            .crate_inherent_impls(())
            .inherent_impls
            .iter()
            .filter(|(&id, impls)| {
                impls.len() > 1
                    // Check for `#[allow]` on the type definition
                    && !is_lint_allowed(
                        cx,
                        MULTIPLE_INHERENT_IMPL,
                        cx.tcx.hir().local_def_id_to_hir_id(id),
                    )
            })
        {
            for impl_id in impl_ids.iter().map(|id| id.expect_local()) {
                match type_map.entry(cx.tcx.type_of(impl_id)) {
                    Entry::Vacant(e) => {
                        // Store the id for the first impl block of this type. The span is retrieved lazily.
                        e.insert(IdOrSpan::Id(impl_id));
                    },
                    Entry::Occupied(mut e) => {
                        if let Some(span) = get_impl_span(cx, impl_id) {
                            let first_span = match *e.get() {
                                IdOrSpan::Span(s) => s,
                                IdOrSpan::Id(id) => {
                                    if let Some(s) = get_impl_span(cx, id) {
                                        // Remember the span of the first block.
                                        *e.get_mut() = IdOrSpan::Span(s);
                                        s
                                    } else {
                                        // The first impl block isn't considered by the lint. Replace it with the
                                        // current one.
                                        *e.get_mut() = IdOrSpan::Span(span);
                                        continue;
                                    }
                                },
                            };
                            lint_spans.push((span, first_span));
                        }
                    },
                }
            }

            // Switching to the next type definition, no need to keep the current entries around.
            type_map.clear();
        }

        // `TyCtxt::crate_inherent_impls` doesn't have a defined order. Sort the lint output first.
        lint_spans.sort_by_key(|x| x.0.lo());
        for (span, first_span) in lint_spans {
            span_lint_and_note(
                cx,
                MULTIPLE_INHERENT_IMPL,
                span,
                "multiple implementations of this structure",
                Some(first_span),
                "first implementation here",
            );
        }
    }
}

/// Gets the span for the given impl block unless it's not being considered by the lint.
fn get_impl_span(cx: &LateContext<'_>, id: LocalDefId) -> Option<Span> {
    let id = cx.tcx.hir().local_def_id_to_hir_id(id);
    if let Node::Item(&Item {
        kind: ItemKind::Impl(ref impl_item),
        span,
        ..
    }) = cx.tcx.hir().get(id)
    {
        (!in_macro(span) && impl_item.generics.params.is_empty() && !is_lint_allowed(cx, MULTIPLE_INHERENT_IMPL, id))
            .then(|| span)
    } else {
        None
    }
}

enum IdOrSpan {
    Id(LocalDefId),
    Span(Span),
}
