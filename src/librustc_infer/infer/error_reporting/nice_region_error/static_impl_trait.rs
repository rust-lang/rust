//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::msg_span_from_free_region;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use rustc_errors::{Applicability, ErrorReported};
use rustc_middle::ty::RegionKind;

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
                let (fn_return_span, is_dyn) =
                    self.tcx().return_type_impl_or_dyn_trait(anon_reg_sup.def_id)?;
                if sub_r == &RegionKind::ReStatic {
                    let sp = var_origin.span();
                    let return_sp = sub_origin.span();
                    let mut err =
                        self.tcx().sess.struct_span_err(sp, "cannot infer an appropriate lifetime");
                    let param_info = self.find_param_with_region(sup_r, sub_r)?;
                    err.span_label(param_info.param_ty_span, "data with this lifetime...");

                    // We try to make the output have fewer overlapping spans if possible.
                    if (sp == sup_origin.span() || !return_sp.overlaps(sup_origin.span()))
                        && sup_origin.span() != return_sp
                    {
                        // FIXME: account for `async fn` like in `async-await/issues/issue-62097.rs`

                        // Customize the spans and labels depending on their relative order so
                        // that split sentences flow correctly.
                        if sup_origin.span().shrink_to_hi() <= return_sp.shrink_to_lo() {
                            err.span_label(sup_origin.span(), "...is captured here...");
                            err.span_label(return_sp, "...and required to be `'static` by this");
                        } else {
                            err.span_label(return_sp, "...is required to be `'static` by this...");
                            err.span_label(sup_origin.span(), "...and is captured here");
                        }
                    } else {
                        err.span_label(
                            return_sp,
                            "...is captured and required to be `'static` here",
                        );
                    }

                    let (lifetime, _) = msg_span_from_free_region(self.tcx(), sup_r);

                    let lifetime_name =
                        if sup_r.has_name() { sup_r.to_string() } else { "'_".to_owned() };
                    // only apply this suggestion onto functions with
                    // explicit non-desugar'able return.
                    if fn_return_span.desugaring_kind().is_none() {
                        let msg = format!(
                            "to permit non-static references in {} `{} Trait` value, you can add \
                             an explicit bound for {}",
                            if is_dyn { "a" } else { "an" },
                            if is_dyn { "dyn" } else { "impl" },
                            lifetime,
                        );
                        // FIXME: account for the need of parens in `&(dyn Trait + '_)`
                        err.span_suggestion_verbose(
                            fn_return_span.shrink_to_hi(),
                            &msg,
                            format!(" + {}", lifetime_name),
                            Applicability::MaybeIncorrect,
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
