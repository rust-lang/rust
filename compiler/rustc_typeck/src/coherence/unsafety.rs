//! Unsafety checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::Unsafety;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LocalDefId;

use crate::errors::{
    AttributeRequiresUnsafeKeyword, SafeTraitImplementedAsUnsafe,
    UnsafeTraitImplementedWithoutUnsafeKeyword,
};

pub(super) fn check_item(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    debug_assert!(matches!(tcx.def_kind(def_id), DefKind::Impl));
    let item = tcx.hir().expect_item(def_id);
    let hir::ItemKind::Impl(ref impl_) = item.kind else { bug!() };

    if let Some(trait_ref) = tcx.impl_trait_ref(item.def_id) {
        let trait_def = tcx.trait_def(trait_ref.def_id);
        let unsafe_attr =
            impl_.generics.params.iter().find(|p| p.pure_wrt_drop).map(|_| "may_dangle");
        match (trait_def.unsafety, unsafe_attr, impl_.unsafety, impl_.polarity) {
            (Unsafety::Normal, None, Unsafety::Unsafe, hir::ImplPolarity::Positive) => {
                tcx.sess.emit_err(SafeTraitImplementedAsUnsafe {
                    span: item.span,
                    trait_name: trait_ref.print_only_trait_path().to_string(),
                });
            }

            (Unsafety::Unsafe, _, Unsafety::Normal, hir::ImplPolarity::Positive) => {
                tcx.sess.emit_err(UnsafeTraitImplementedWithoutUnsafeKeyword {
                    span: item.span,
                    trait_name: trait_ref.print_only_trait_path().to_string(),
                });
            }

            (Unsafety::Normal, Some(attr_name), Unsafety::Normal, hir::ImplPolarity::Positive) => {
                tcx.sess.emit_err(AttributeRequiresUnsafeKeyword { span: item.span, attr_name });
            }

            (_, _, Unsafety::Unsafe, hir::ImplPolarity::Negative(_)) => {
                // Reported in AST validation
                tcx.sess.delay_span_bug(item.span, "unsafe negative impl");
            }
            (_, _, Unsafety::Normal, hir::ImplPolarity::Negative(_))
            | (Unsafety::Unsafe, _, Unsafety::Unsafe, hir::ImplPolarity::Positive)
            | (Unsafety::Normal, Some(_), Unsafety::Unsafe, hir::ImplPolarity::Positive)
            | (Unsafety::Normal, None, Unsafety::Normal, _) => {
                // OK
            }
        }
    }
}
