//! Error Reporting for when the lifetime for a type doesn't match the `impl` selected for a predicate
//! to hold.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::error_reporting::note_and_explain_region;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::ObligationCauseCode;
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
        let (parent, impl_def_id) = match &cause.code {
            ObligationCauseCode::MatchImpl(parent, impl_def_id) => (parent, impl_def_id),
            _ => return None,
        };
        let binding_span = match **parent {
            ObligationCauseCode::BindingObligation(_def_id, binding_span) => binding_span,
            _ => return None,
        };
        let mut err = self.tcx().sess.struct_span_err(cause.span, "incompatible lifetime on type");
        // FIXME: we should point at the lifetime
        let mut multi_span: MultiSpan = vec![binding_span].into();
        multi_span
            .push_span_label(binding_span, "introduces a `'static` lifetime requirement".into());
        err.span_note(multi_span, "because this has an unmet lifetime requirement");
        note_and_explain_region(self.tcx(), &mut err, "...", sup, "...");
        if let Some(impl_node) = self.tcx().hir().get_if_local(*impl_def_id) {
            let ty = self.tcx().type_of(*impl_def_id);
            let mut v = super::static_impl_trait::TraitObjectVisitor(vec![]);
            v.visit_ty(ty);
            let matching_def_ids = v.0;

            let impl_self_ty = match impl_node {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Impl(hir::Impl { self_ty, .. }),
                    ..
                }) => self_ty,
                _ => bug!("Node not an impl."),
            };

            for matching_def_id in matching_def_ids {
                let mut hir_v =
                    super::static_impl_trait::HirTraitObjectVisitor(vec![], matching_def_id);
                hir_v.visit_ty(&impl_self_ty);

                let mut multi_span: MultiSpan = hir_v.0.clone().into();
                for span in &hir_v.0 {
                    multi_span.push_span_label(
                        *span,
                        "this has an implicit `'static` lifetime requirement".to_string(),
                    );
                    err.span_suggestion_verbose(
                        span.shrink_to_hi(),
                        "consider relaxing the implicit `'static` requirement",
                        " + '_".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                }
                err.span_note(multi_span, "...does not necessarily outlive the static lifetime introduced by the compatible `impl`");
            }
        }
        err.emit();
        Some(ErrorReported)
    }
}
