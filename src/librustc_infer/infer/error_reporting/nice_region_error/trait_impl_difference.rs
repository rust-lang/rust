//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use crate::hir::def_id::DefId;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{Subtype, ValuePairs};
use crate::traits::ObligationCauseCode::CompareImplMethodObligation;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::ErrorReported;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the `impl` doesn't conform to the `trait`.
    pub(super) fn try_report_impl_not_conforming_to_trait(&self) -> Option<ErrorReported> {
        if let Some(ref error) = self.error {
            debug!("try_report_impl_not_conforming_to_trait {:?}", error);
            if let RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                _sub,
                sup_origin,
                _sup,
            ) = error.clone()
            {
                if let (&Subtype(ref sup_trace), &Subtype(ref sub_trace)) =
                    (&sup_origin, &sub_origin)
                {
                    if let (
                        ValuePairs::Types(sub_expected_found),
                        ValuePairs::Types(sup_expected_found),
                        CompareImplMethodObligation { trait_item_def_id, .. },
                    ) = (&sub_trace.values, &sup_trace.values, &sub_trace.cause.code)
                    {
                        if sup_expected_found == sub_expected_found {
                            self.emit_err(
                                var_origin.span(),
                                sub_expected_found.expected,
                                sub_expected_found.found,
                                self.tcx().def_span(*trait_item_def_id),
                            );
                            return Some(ErrorReported);
                        }
                    }
                }
            }
        }
        None
    }

    fn emit_err(&self, sp: Span, expected: Ty<'tcx>, found: Ty<'tcx>, impl_sp: Span) {
        let mut err = self
            .tcx()
            .sess
            .struct_span_err(sp, "`impl` item signature doesn't match `trait` item signature");
        err.span_label(sp, &format!("found {:?}", found));
        err.span_label(impl_sp, &format!("expected {:?}", expected));

        struct EarlyBoundRegionHighlighter(FxHashSet<DefId>);
        impl<'tcx> ty::fold::TypeVisitor<'tcx> for EarlyBoundRegionHighlighter {
            fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
                debug!("LateBoundRegionNameCollector visit_region {:?}", r);
                match *r {
                    ty::ReFree(free) => {
                        self.0.insert(free.scope);
                    }

                    ty::ReEarlyBound(bound) => {
                        self.0.insert(bound.def_id);
                    }
                    _ => {}
                }
                r.super_visit_with(self)
            }
        }

        let mut visitor = EarlyBoundRegionHighlighter(FxHashSet::default());
        expected.visit_with(&mut visitor);

        let note = !visitor.0.is_empty();

        if let Some((expected, found)) = self
            .tcx()
            .infer_ctxt()
            .enter(|infcx| infcx.expected_found_str_ty(&ExpectedFound { expected, found }))
        {
            err.note_expected_found(&"", expected, &"", found);
        } else {
            // This fallback shouldn't be necessary, but let's keep it in just in case.
            err.note(&format!("expected `{:?}`\n   found `{:?}`", expected, found));
        }
        if note {
            err.note(
                "the lifetime requirements from the `trait` could not be fulfilled by the \
                      `impl`",
            );
            err.help(
                "consider adding a named lifetime to the `trait` that constrains the item's \
                      `self` argument, its inputs and its output with it",
            );
        }
        err.emit();
    }
}
