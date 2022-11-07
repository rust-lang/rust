use crate::infer::{
    error_reporting::nice_region_error::NiceRegionError, RegionResolutionError, SubregionOrigin,
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
                let msg = "lifetime bound not satisfied";
                let mut err = self.tcx().sess.struct_span_err(*span, msg);
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
                match (sub_span, sup_span, sub_symbol, sup_symbol) {
                    (Some(sub_span), Some(sup_span), Some(sub_symbol), Some(sup_symbol)) => {
                        err.span_note(
                            sub_span,
                            format!("the lifetime `{sub_symbol}` defined here..."),
                        );
                        err.span_note(
                            sup_span,
                            format!("...must outlive the lifetime `{sup_symbol}` defined here"),
                        );
                    }
                    (Some(sub_span), Some(sup_span), _, Some(sup_symbol)) => {
                        err.span_note(sub_span, format!("the lifetime defined here..."));
                        err.span_note(
                            sup_span,
                            format!("...must outlive the lifetime `{sup_symbol}` defined here"),
                        );
                    }
                    (Some(sub_span), Some(sup_span), Some(sub_symbol), _) => {
                        err.span_note(
                            sub_span,
                            format!("the lifetime `{sub_symbol}` defined here..."),
                        );
                        err.span_note(
                            sup_span,
                            format!("...must outlive the lifetime defined here"),
                        );
                    }
                    (Some(sub_span), Some(sup_span), _, _) => {
                        err.span_note(sub_span, format!("the lifetime defined here..."));
                        err.span_note(
                            sup_span,
                            format!("...must outlive the lifetime defined here"),
                        );
                    }
                    _ => {}
                }
                err.note("this is a known limitation that will be removed in the future (see issue #100013 <https://github.com/rust-lang/rust/issues/100013> for more information)");
                Some(err)
            }

            _ => None,
        }
    }
}
