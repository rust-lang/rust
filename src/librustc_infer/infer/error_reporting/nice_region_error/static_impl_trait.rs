//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::msg_span_from_free_region;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use rustc_errors::Applicability;
use rustc_middle::ty::{BoundRegion, FreeRegion, RegionKind};
use rustc_middle::util::common::ErrorReported;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the return type is a static impl Trait.
    pub(super) fn try_report_static_impl_trait(&self) -> Option<ErrorReported> {
        if let Some(ref error) = self.error {
            if let RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                sub_r,
                sup_origin,
                sup_r,
            ) = error.clone()
            {
                let anon_reg_sup = self.tcx().is_suitable_region(sup_r)?;
                let return_ty = self.tcx().return_type_impl_trait(anon_reg_sup.def_id);
                if sub_r == &RegionKind::ReStatic && return_ty.is_some() {
                    let sp = var_origin.span();
                    let return_sp = sub_origin.span();
                    let mut err =
                        self.tcx().sess.struct_span_err(sp, "cannot infer an appropriate lifetime");
                    err.span_label(
                        return_sp,
                        "this return type evaluates to the `'static` lifetime...",
                    );
                    err.span_label(sup_origin.span(), "...but this borrow...");

                    let (lifetime, lt_sp_opt) = msg_span_from_free_region(self.tcx(), sup_r);
                    if let Some(lifetime_sp) = lt_sp_opt {
                        err.span_note(lifetime_sp, &format!("...can't outlive {}", lifetime));
                    }

                    let lifetime_name = match sup_r {
                        RegionKind::ReFree(FreeRegion {
                            bound_region: BoundRegion::BrNamed(_, ref name),
                            ..
                        }) => name.to_string(),
                        _ => "'_".to_owned(),
                    };
                    let fn_return_span = return_ty.unwrap().1;
                    if let Ok(snippet) =
                        self.tcx().sess.source_map().span_to_snippet(fn_return_span)
                    {
                        // only apply this suggestion onto functions with
                        // explicit non-desugar'able return.
                        if fn_return_span.desugaring_kind().is_none() {
                            err.span_suggestion(
                                fn_return_span,
                                &format!(
                                    "you can add a bound to the return type to make it last \
                                 less than `'static` and match {}",
                                    lifetime,
                                ),
                                format!("{} + {}", snippet, lifetime_name),
                                Applicability::Unspecified,
                            );
                        }
                    }
                    err.emit();
                    return Some(ErrorReported);
                }
            }
        }
        None
    }
}
