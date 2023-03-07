use crate::{
    errors::PlaceholderRelationLfNotSatisfied,
    infer::{
        error_reporting::nice_region_error::NiceRegionError, RegionResolutionError, SubregionOrigin,
    },
};
use rustc_data_structures::intern::Interned;
use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed};
use rustc_middle::ty::{self, RePlaceholder, Region};

impl<'tcx> NiceRegionError<'_, 'tcx> {
    /// Emitted wwhen given a `ConcreteFailure` when relating two placeholders.
    pub(super) fn try_report_placeholder_relation(
        &self,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        match &self.error {
            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::RelateRegionParamBound(span),
                Region(Interned(RePlaceholder(ty::Placeholder { name: sub_name, .. }), _)),
                Region(Interned(RePlaceholder(ty::Placeholder { name: sup_name, .. }), _)),
            )) => {
                let span = *span;
                let (sub_span, sub_symbol) = match sub_name {
                    ty::BrNamed(def_id, symbol) => {
                        (Some(self.tcx().def_span(def_id)), Some(symbol))
                    }
                    ty::BrAnon(_, span) => (*span, None),
                    ty::BrEnv => (None, None),
                };
                let (sup_span, sup_symbol) = match sup_name {
                    ty::BrNamed(def_id, symbol) => {
                        (Some(self.tcx().def_span(def_id)), Some(symbol))
                    }
                    ty::BrAnon(_, span) => (*span, None),
                    ty::BrEnv => (None, None),
                };
                let diag = match (sub_span, sup_span, sub_symbol, sup_symbol) {
                    (Some(sub_span), Some(sup_span), Some(&sub_symbol), Some(&sup_symbol)) => {
                        PlaceholderRelationLfNotSatisfied::HasBoth {
                            span,
                            sub_span,
                            sup_span,
                            sub_symbol,
                            sup_symbol,
                            note: (),
                        }
                    }
                    (Some(sub_span), Some(sup_span), _, Some(&sup_symbol)) => {
                        PlaceholderRelationLfNotSatisfied::HasSup {
                            span,
                            sub_span,
                            sup_span,
                            sup_symbol,
                            note: (),
                        }
                    }
                    (Some(sub_span), Some(sup_span), Some(&sub_symbol), _) => {
                        PlaceholderRelationLfNotSatisfied::HasSub {
                            span,
                            sub_span,
                            sup_span,
                            sub_symbol,
                            note: (),
                        }
                    }
                    (Some(sub_span), Some(sup_span), _, _) => {
                        PlaceholderRelationLfNotSatisfied::HasNone {
                            span,
                            sub_span,
                            sup_span,
                            note: (),
                        }
                    }
                    _ => PlaceholderRelationLfNotSatisfied::OnlyPrimarySpan { span, note: () },
                };
                Some(self.tcx().sess.create_err(diag))
            }
            _ => None,
        }
    }
}
