//! Error Reporting for when the lifetime for a type doesn't match the `impl` selected for a predicate
//! to hold.

use crate::errors::{note_and_explain, IntroducesStaticBecauseUnmetLifetimeReq};
use crate::errors::{
    DoesNotOutliveStaticFromImpl, ImplicitStaticLifetimeSubdiag, MismatchedStaticLifetime,
};
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::ObligationCauseCode;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::intravisit::Visitor;
use rustc_middle::ty::TypeVisitor;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    pub(super) fn try_report_mismatched_static_lifetime(&self) -> Option<ErrorGuaranteed> {
        let error = self.error.as_ref()?;
        debug!("try_report_mismatched_static_lifetime {:?}", error);

        let RegionResolutionError::ConcreteFailure(origin, sub, sup) = error.clone() else {
            return None;
        };
        if !sub.is_static() {
            return None;
        }
        let SubregionOrigin::Subtype(box TypeTrace { ref cause, .. }) = origin else {
            return None;
        };
        // If we added a "points at argument expression" obligation, we remove it here, we care
        // about the original obligation only.
        let code = match cause.code() {
            ObligationCauseCode::FunctionArgumentObligation { parent_code, .. } => &*parent_code,
            code => code,
        };
        let ObligationCauseCode::MatchImpl(parent, impl_def_id) = code else {
            return None;
        };
        let (ObligationCauseCode::BindingObligation(_, binding_span) | ObligationCauseCode::ExprBindingObligation(_, binding_span, ..))
            = *parent.code() else {
            return None;
        };

        // FIXME: we should point at the lifetime
        let multi_span: MultiSpan = vec![binding_span].into();
        let multispan_subdiag = IntroducesStaticBecauseUnmetLifetimeReq {
            unmet_requirements: multi_span,
            binding_span,
        };

        let expl = note_and_explain::RegionExplanation::new(
            self.tcx(),
            sup,
            Some(binding_span),
            note_and_explain::PrefixKind::Empty,
            note_and_explain::SuffixKind::Continues,
        );
        let mut impl_span = None;
        let mut implicit_static_lifetimes = Vec::new();
        if let Some(impl_node) = self.tcx().hir().get_if_local(*impl_def_id) {
            // If an impl is local, then maybe this isn't what they want. Try to
            // be as helpful as possible with implicit lifetimes.

            // First, let's get the hir self type of the impl
            let hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Impl(hir::Impl { self_ty: impl_self_ty, .. }),
                ..
            }) = impl_node else {
                bug!("Node not an impl.");
            };

            // Next, let's figure out the set of trait objects with implicit static bounds
            let ty = self.tcx().type_of(*impl_def_id).subst_identity();
            let mut v = super::static_impl_trait::TraitObjectVisitor(FxIndexSet::default());
            v.visit_ty(ty);
            let mut traits = vec![];
            for matching_def_id in v.0 {
                let mut hir_v =
                    super::static_impl_trait::HirTraitObjectVisitor(&mut traits, matching_def_id);
                hir_v.visit_ty(&impl_self_ty);
            }

            if traits.is_empty() {
                // If there are no trait object traits to point at, either because
                // there aren't trait objects or because none are implicit, then just
                // write a single note on the impl itself.

                impl_span = Some(self.tcx().def_span(*impl_def_id));
            } else {
                // Otherwise, point at all implicit static lifetimes

                for span in &traits {
                    implicit_static_lifetimes
                        .push(ImplicitStaticLifetimeSubdiag::Note { span: *span });
                    // It would be nice to put this immediately under the above note, but they get
                    // pushed to the end.
                    implicit_static_lifetimes
                        .push(ImplicitStaticLifetimeSubdiag::Sugg { span: span.shrink_to_hi() });
                }
            }
        } else {
            // Otherwise just point out the impl.

            impl_span = Some(self.tcx().def_span(*impl_def_id));
        }
        let err = MismatchedStaticLifetime {
            cause_span: cause.span,
            unmet_lifetime_reqs: multispan_subdiag,
            expl,
            does_not_outlive_static_from_impl: impl_span
                .map(|span| DoesNotOutliveStaticFromImpl::Spanned { span })
                .unwrap_or(DoesNotOutliveStaticFromImpl::Unspanned),
            implicit_static_lifetimes,
        };
        let reported = self.tcx().sess.emit_err(err);
        Some(reported)
    }
}
