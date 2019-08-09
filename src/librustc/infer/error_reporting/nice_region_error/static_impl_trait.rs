//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::ty::{BoundRegion, FreeRegion, RegionKind};
use crate::util::common::ErrorReported;
use errors::Applicability;

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
                if sub_r == &RegionKind::ReStatic &&
                    self.tcx().return_type_impl_trait(anon_reg_sup.def_id).is_some()
                {
                    let sp = var_origin.span();
                    let return_sp = sub_origin.span();
                    let mut err = self.tcx().sess.struct_span_err(
                        sp,
                        "cannot infer an appropriate lifetime",
                    );
                    err.span_label(
                        return_sp,
                        "this return type evaluates to the `'static` lifetime...",
                    );
                    err.span_label(
                        sup_origin.span(),
                        "...but this borrow...",
                    );

                    let (lifetime, lt_sp_opt) = self.tcx().msg_span_from_free_region(sup_r);
                    if let Some(lifetime_sp) = lt_sp_opt {
                        err.span_note(
                            lifetime_sp,
                            &format!("...can't outlive {}", lifetime),
                        );
                    }

                    let lifetime_name = match sup_r {
                        RegionKind::ReFree(FreeRegion {
                            bound_region: BoundRegion::BrNamed(_, ref name), ..
                        }) => name.to_string(),
                        _ => "'_".to_owned(),
                    };
                    if let Ok(snippet) = self.tcx().sess.source_map().span_to_snippet(return_sp) {
                        err.span_suggestion(
                            return_sp,
                            &format!(
                                "you can add a constraint to the return type to make it last \
                                 less than `'static` and match {}",
                                lifetime,
                            ),
                            format!("{} + {}", snippet, lifetime_name),
                            Applicability::Unspecified,
                        );
                    }
                    err.emit();
                    return Some(ErrorReported);
                }
            }
        }
        None
    }
}
