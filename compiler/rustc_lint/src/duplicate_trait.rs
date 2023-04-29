use rustc_data_structures::fx::FxHashSet;

use crate::hir;

use crate::{lints::DuplicateTraitDiag, LateContext, LateLintPass};

declare_lint! {
    /// The `lint_duplicate_trait` lints repetition of traits.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// fn foo(_: &(dyn MyTrait + Send + Send>) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Duplicate trait `Send` in trait object.
    pub DUPLICATE_TRAIT,
    Warn,
    "duplicate trait constraint in trait object"
}

declare_lint_pass!(DuplicateTrait => [DUPLICATE_TRAIT]);

impl<'tcx> LateLintPass<'tcx> for DuplicateTrait {
    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx>) {
        let hir::TyKind::Ref(
            ..,
            hir::MutTy {
                ty: hir::Ty {
                    kind: hir::TyKind::TraitObject(bounds, ..),
                    ..
                },
                ..
            }
        ) = ty.kind else { return; };

        if bounds.len() < 2 {
            return;
        }

        let mut seen_def_ids = FxHashSet::default();

        for bound in bounds.iter() {
            let Some(def_id) = bound.trait_ref.trait_def_id() else { continue; };

            let already_seen = !seen_def_ids.insert(def_id);

            if already_seen {
                cx.tcx.emit_spanned_lint(
                    DUPLICATE_TRAIT,
                    bound.trait_ref.hir_ref_id, // is this correct?
                    bound.span,
                    DuplicateTraitDiag {
                        trait_name: cx.tcx.item_name(def_id),
                        suggestion: bound.span
                    },
                )
            }
        }
    }
}
