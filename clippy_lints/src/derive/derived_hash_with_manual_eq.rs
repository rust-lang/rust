use clippy_utils::diagnostics::span_lint_hir_and_then;
use rustc_hir::{HirId, TraitRef};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::{Span, sym};

use super::DERIVED_HASH_WITH_MANUAL_EQ;

/// Implementation of the `DERIVED_HASH_WITH_MANUAL_EQ` lint.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    trait_ref: &TraitRef<'_>,
    ty: Ty<'tcx>,
    adt_hir_id: HirId,
    hash_is_automatically_derived: bool,
) {
    if let Some(peq_trait_def_id) = cx.tcx.lang_items().eq_trait()
        && let Some(def_id) = trait_ref.trait_def_id()
        && cx.tcx.is_diagnostic_item(sym::Hash, def_id)
    {
        // Look for the PartialEq implementations for `ty`
        cx.tcx.for_each_relevant_impl(peq_trait_def_id, ty, |impl_id| {
            let peq_is_automatically_derived = cx.tcx.is_automatically_derived(impl_id);

            if !hash_is_automatically_derived || peq_is_automatically_derived {
                return;
            }

            let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

            // Only care about `impl PartialEq<Foo> for Foo`
            // For `impl PartialEq<B> for A, input_types is [A, B]
            if trait_ref.instantiate_identity().args.type_at(1) == ty {
                span_lint_hir_and_then(
                    cx,
                    DERIVED_HASH_WITH_MANUAL_EQ,
                    adt_hir_id,
                    span,
                    "you are deriving `Hash` but have implemented `PartialEq` explicitly",
                    |diag| {
                        if let Some(local_def_id) = impl_id.as_local() {
                            let hir_id = cx.tcx.local_def_id_to_hir_id(local_def_id);
                            diag.span_note(cx.tcx.hir_span(hir_id), "`PartialEq` implemented here");
                        }
                    },
                );
            }
        });
    }
}
