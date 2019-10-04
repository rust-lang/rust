//! Error Reporting for `impl` items that do not match the obligations from their `trait`.

use syntax_pos::Span;
use crate::ty::Ty;
use crate::infer::{ValuePairs, Subtype};
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::util::common::ErrorReported;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the `impl` doesn't conform to the `trait`.
    pub(super) fn try_report_impl_not_conforming_to_trait(&self) -> Option<ErrorReported> {
        if let Some(ref error) = self.error {
            if let RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                _sub,
                sup_origin,
                _sup,
            ) = error.clone() {
                match (&sup_origin, &sub_origin) {
                    (&Subtype(ref sup_trace), &Subtype(ref sub_trace)) => {
                        if let (
                            ValuePairs::Types(sub_expected_found),
                            ValuePairs::Types(sup_expected_found),
                        ) = (&sub_trace.values, &sup_trace.values) {
                            if sup_expected_found == sub_expected_found {
                                let sp = var_origin.span();
                                self.emit_err(
                                    sp,
                                    sub_expected_found.expected,
                                    sub_expected_found.found,
                                );
                                return Some(ErrorReported);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    fn emit_err(&self, sp: Span, expected: Ty<'tcx>, found: Ty<'tcx>) {
        let mut err = self.tcx().sess.struct_span_err(
            sp,
            "`impl` item signature doesn't match `trait` item signature",
        );
        err.note(&format!("expected: {:?}\n   found: {:?}", expected, found));
        err.span_label(sp, &format!("found {:?}", found));
        // FIXME: recover the `FnPtr`'s `HirId`/`Node` to point to it.
        err.emit();
    }
}
