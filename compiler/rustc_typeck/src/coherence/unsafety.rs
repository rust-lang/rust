//! Unsafety checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::Unsafety;
use rustc_middle::ty::TyCtxt;

pub fn check(tcx: TyCtxt<'_>) {
    let mut unsafety = UnsafetyChecker { tcx };
    tcx.hir().visit_all_item_likes(&mut unsafety);
}

struct UnsafetyChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl UnsafetyChecker<'tcx> {
    fn check_unsafety_coherence(
        &mut self,
        item: &'v hir::Item<'v>,
        impl_generics: Option<&hir::Generics<'_>>,
        unsafety: hir::Unsafety,
        polarity: hir::ImplPolarity,
    ) {
        if let Some(trait_ref) = self.tcx.impl_trait_ref(item.def_id) {
            let trait_def = self.tcx.trait_def(trait_ref.def_id);
            let unsafe_attr = impl_generics.and_then(|generics| {
                generics.params.iter().find(|p| p.pure_wrt_drop).map(|_| "may_dangle")
            });
            match (trait_def.unsafety, unsafe_attr, unsafety, polarity) {
                (Unsafety::Normal, None, Unsafety::Unsafe, hir::ImplPolarity::Positive) => {
                    struct_span_err!(
                        self.tcx.sess,
                        item.span,
                        E0199,
                        "implementing the trait `{}` is not unsafe",
                        trait_ref.print_only_trait_path()
                    )
                    .emit();
                }

                (Unsafety::Unsafe, _, Unsafety::Normal, hir::ImplPolarity::Positive) => {
                    struct_span_err!(
                        self.tcx.sess,
                        item.span,
                        E0200,
                        "the trait `{}` requires an `unsafe impl` declaration",
                        trait_ref.print_only_trait_path()
                    )
                    .emit();
                }

                (
                    Unsafety::Normal,
                    Some(attr_name),
                    Unsafety::Normal,
                    hir::ImplPolarity::Positive,
                ) => {
                    struct_span_err!(
                        self.tcx.sess,
                        item.span,
                        E0569,
                        "requires an `unsafe impl` declaration due to `#[{}]` attribute",
                        attr_name
                    )
                    .emit();
                }

                (_, _, Unsafety::Unsafe, hir::ImplPolarity::Negative(_)) => {
                    // Reported in AST validation
                    self.tcx.sess.delay_span_bug(item.span, "unsafe negative impl");
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
}

impl ItemLikeVisitor<'v> for UnsafetyChecker<'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item<'v>) {
        if let hir::ItemKind::Impl(ref impl_) = item.kind {
            self.check_unsafety_coherence(
                item,
                Some(&impl_.generics),
                impl_.unsafety,
                impl_.polarity,
            );
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}
