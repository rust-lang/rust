//! Error Reporting for when the lifetime for a type doesn't match the `impl` selected for a predicate
//! to hold.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::error_reporting::note_and_explain_region;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::ObligationCauseCode;
use rustc_data_structures::stable_set::FxHashSet;
use rustc_errors::{Applicability, ErrorReported};
use rustc_hir as hir;
use rustc_hir::intravisit::Visitor;
use rustc_middle::ty::{self, TypeVisitor};
use rustc_span::MultiSpan;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    pub(super) fn try_report_mismatched_static_lifetime(&self) -> Option<ErrorReported> {
        let error = self.error.as_ref()?;
        debug!("try_report_mismatched_static_lifetime {:?}", error);

        let (origin, sub, sup) = match error.clone() {
            RegionResolutionError::ConcreteFailure(origin, sub, sup) => (origin, sub, sup),
            _ => return None,
        };
        if *sub != ty::RegionKind::ReStatic {
            return None;
        }
        let cause = match origin {
            SubregionOrigin::Subtype(box TypeTrace { ref cause, .. }) => cause,
            _ => return None,
        };
        // If we added a "points at argument expression" obligation, we remove it here, we care
        // about the original obligation only.
        let code = match &cause.code {
            ObligationCauseCode::FunctionArgumentObligation { parent_code, .. } => &*parent_code,
            _ => &cause.code,
        };
        let (parent, impl_def_id) = match code {
            ObligationCauseCode::MatchImpl(parent, impl_def_id) => (parent, impl_def_id),
            _ => return None,
        };
        let binding_span = match parent.code {
            ObligationCauseCode::BindingObligation(_def_id, binding_span) => binding_span,
            _ => return None,
        };
        let mut err = self.tcx().sess.struct_span_err(cause.span, "incompatible lifetime on type");
        // FIXME: we should point at the lifetime
        let mut multi_span: MultiSpan = vec![binding_span].into();
        multi_span
            .push_span_label(binding_span, "introduces a `'static` lifetime requirement".into());
        err.span_note(multi_span, "because this has an unmet lifetime requirement");
        note_and_explain_region(self.tcx(), &mut err, "", sup, "...", Some(binding_span));
        if let Some(impl_node) = self.tcx().hir().get_if_local(*impl_def_id) {
            // If an impl is local, then maybe this isn't what they want. Try to
            // be as helpful as possible with implicit lifetimes.

            // First, let's get the hir self type of the impl
            let impl_self_ty = match impl_node {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(hir::Impl { self_ty, .. }),
                    ..
                }) => self_ty,
                _ => bug!("Node not an impl."),
            };

            // Next, let's figure out the set of trait objects with implict static bounds
            let ty = self.tcx().type_of(*impl_def_id);
            let mut v = super::static_impl_trait::TraitObjectVisitor(FxHashSet::default());
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

                let impl_span = self.tcx().def_span(*impl_def_id);
                err.span_note(impl_span, "...does not necessarily outlive the static lifetime introduced by the compatible `impl`");
            } else {
                // Otherwise, point at all implicit static lifetimes

                err.note("...does not necessarily outlive the static lifetime introduced by the compatible `impl`");
                for span in &traits {
                    err.span_note(*span, "this has an implicit `'static` lifetime requirement");
                    // It would be nice to put this immediately under the above note, but they get
                    // pushed to the end.
                    err.span_suggestion_verbose(
                        span.shrink_to_hi(),
                        "consider relaxing the implicit `'static` requirement",
                        " + '_".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        } else {
            // Otherwise just point out the impl.

            let impl_span = self.tcx().def_span(*impl_def_id);
            err.span_note(impl_span, "...does not necessarily outlive the static lifetime introduced by the compatible `impl`");
        }
        err.emit();
        Some(ErrorReported)
    }
}
