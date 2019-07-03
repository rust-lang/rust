//! Unsafety checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use rustc::ty::TyCtxt;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::{self, Unsafety};

pub fn check(tcx: TyCtxt<'_>) {
    let mut unsafety = UnsafetyChecker { tcx };
    tcx.hir().krate().visit_all_item_likes(&mut unsafety);
}

struct UnsafetyChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl UnsafetyChecker<'tcx> {
    fn check_unsafety_coherence(&mut self,
                                item: &'v hir::Item,
                                impl_generics: Option<&hir::Generics>,
                                unsafety: hir::Unsafety,
                                polarity: hir::ImplPolarity)
    {
        let local_did = self.tcx.hir().local_def_id_from_hir_id(item.hir_id);
        if let Some(trait_ref) = self.tcx.impl_trait_ref(local_did) {
            let trait_def = self.tcx.trait_def(trait_ref.def_id);
            let unsafe_attr = impl_generics.and_then(|generics| {
                generics.params.iter().find(|p| p.pure_wrt_drop).map(|_| "may_dangle")
            });
            match (trait_def.unsafety, unsafe_attr, unsafety, polarity) {
                (Unsafety::Normal, None, Unsafety::Unsafe, hir::ImplPolarity::Positive) => {
                    span_err!(self.tcx.sess,
                              item.span,
                              E0199,
                              "implementing the trait `{}` is not unsafe",
                              trait_ref);
                }

                (Unsafety::Unsafe, _, Unsafety::Normal, hir::ImplPolarity::Positive) => {
                    span_err!(self.tcx.sess,
                              item.span,
                              E0200,
                              "the trait `{}` requires an `unsafe impl` declaration",
                              trait_ref);
                }

                (Unsafety::Normal, Some(attr_name), Unsafety::Normal,
                    hir::ImplPolarity::Positive) =>
                {
                    span_err!(self.tcx.sess,
                              item.span,
                              E0569,
                              "requires an `unsafe impl` declaration due to `#[{}]` attribute",
                              attr_name);
                }

                (_, _, Unsafety::Unsafe, hir::ImplPolarity::Negative) => {
                    // Reported in AST validation
                    self.tcx.sess.delay_span_bug(item.span, "unsafe negative impl");
                }
                (_, _, Unsafety::Normal, hir::ImplPolarity::Negative) |
                (Unsafety::Unsafe, _, Unsafety::Unsafe, hir::ImplPolarity::Positive) |
                (Unsafety::Normal, Some(_), Unsafety::Unsafe, hir::ImplPolarity::Positive) |
                (Unsafety::Normal, None, Unsafety::Normal, _) => {
                    // OK
                }
            }
        }
    }
}

impl ItemLikeVisitor<'v> for UnsafetyChecker<'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        if let hir::ItemKind::Impl(unsafety, polarity, _, ref generics, ..) = item.node {
            self.check_unsafety_coherence(item, Some(generics), unsafety, polarity);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
