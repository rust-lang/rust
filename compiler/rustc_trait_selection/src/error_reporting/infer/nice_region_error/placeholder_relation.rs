use rustc_data_structures::intern::Interned;
use rustc_errors::Diag;
use rustc_middle::bug;
use rustc_middle::ty::{self, RePlaceholder, Region};

use crate::error_reporting::infer::nice_region_error::NiceRegionError;
use crate::errors::PlaceholderRelationLfNotSatisfied;
use crate::infer::{RegionResolutionError, SubregionOrigin};

impl<'tcx> NiceRegionError<'_, 'tcx> {
    /// Emitted wwhen given a `ConcreteFailure` when relating two placeholders.
    pub(super) fn try_report_placeholder_relation(&self) -> Option<Diag<'tcx>> {
        match &self.error {
            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::RelateRegionParamBound(span, _),
                Region(Interned(
                    RePlaceholder(ty::Placeholder {
                        bound: ty::BoundRegion { kind: sub_name, .. },
                        ..
                    }),
                    _,
                )),
                Region(Interned(
                    RePlaceholder(ty::Placeholder {
                        bound: ty::BoundRegion { kind: sup_name, .. },
                        ..
                    }),
                    _,
                )),
            )) => {
                let span = *span;
                let (sub_span, sub_symbol) = match *sub_name {
                    ty::BoundRegionKind::Named(def_id) => {
                        (Some(self.tcx().def_span(def_id)), Some(self.tcx().item_name(def_id)))
                    }
                    ty::BoundRegionKind::Anon | ty::BoundRegionKind::ClosureEnv => (None, None),
                    ty::BoundRegionKind::NamedAnon(_) => bug!("only used for pretty printing"),
                };
                let (sup_span, sup_symbol) = match *sup_name {
                    ty::BoundRegionKind::Named(def_id) => {
                        (Some(self.tcx().def_span(def_id)), Some(self.tcx().item_name(def_id)))
                    }
                    ty::BoundRegionKind::Anon | ty::BoundRegionKind::ClosureEnv => (None, None),
                    ty::BoundRegionKind::NamedAnon(_) => bug!("only used for pretty printing"),
                };
                let diag = match (sub_span, sup_span, sub_symbol, sup_symbol) {
                    (Some(sub_span), Some(sup_span), Some(sub_symbol), Some(sup_symbol)) => {
                        PlaceholderRelationLfNotSatisfied::HasBoth {
                            span,
                            sub_span,
                            sup_span,
                            sub_symbol,
                            sup_symbol,
                            note: (),
                        }
                    }
                    (Some(sub_span), Some(sup_span), _, Some(sup_symbol)) => {
                        PlaceholderRelationLfNotSatisfied::HasSup {
                            span,
                            sub_span,
                            sup_span,
                            sup_symbol,
                            note: (),
                        }
                    }
                    (Some(sub_span), Some(sup_span), Some(sub_symbol), _) => {
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
                Some(self.tcx().dcx().create_err(diag))
            }
            _ => None,
        }
    }
}
