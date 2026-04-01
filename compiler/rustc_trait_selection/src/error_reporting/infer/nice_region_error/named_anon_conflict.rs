//! Error Reporting for Anonymous Region Lifetime Errors
//! where one region is named and the other is anonymous.

use rustc_errors::Diag;
use tracing::debug;

use crate::error_reporting::infer::nice_region_error::NiceRegionError;
use crate::error_reporting::infer::nice_region_error::find_anon_type::find_anon_type;
use crate::errors::ExplicitLifetimeRequired;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// When given a `ConcreteFailure` for a function with parameters containing a named region and
    /// an anonymous region, emit an descriptive diagnostic error.
    pub(super) fn try_report_named_anon_conflict(&self) -> Option<Diag<'tcx>> {
        let (span, sub, sup) = self.regions()?;

        debug!(
            "try_report_named_anon_conflict(sub={:?}, sup={:?}, error={:?})",
            sub, sup, self.error,
        );

        // Determine whether the sub and sup consist of one named region ('a)
        // and one anonymous (elided) region. If so, find the parameter arg
        // where the anonymous region appears (there must always be one; we
        // only introduced anonymous regions in parameters) as well as a
        // version new_ty of its type where the anonymous region is replaced
        // with the named one.
        let (named, anon, anon_param_info, region_info) = if sub.is_named(self.tcx())
            && let Some(region_info) = self.tcx().is_suitable_region(self.generic_param_scope, sup)
            && let Some(anon_param_info) = self.find_param_with_region(sup, sub)
        {
            (sub, sup, anon_param_info, region_info)
        } else if sup.is_named(self.tcx())
            && let Some(region_info) = self.tcx().is_suitable_region(self.generic_param_scope, sub)
            && let Some(anon_param_info) = self.find_param_with_region(sub, sup)
        {
            (sup, sub, anon_param_info, region_info)
        } else {
            return None; // inapplicable
        };

        // Suggesting to add a `'static` lifetime to a parameter is nearly always incorrect,
        // and can steer users down the wrong path.
        if named.is_static() {
            return None;
        }

        debug!("try_report_named_anon_conflict: named = {:?}", named);
        debug!("try_report_named_anon_conflict: anon_param_info = {:?}", anon_param_info);
        debug!("try_report_named_anon_conflict: region_info = {:?}", region_info);

        let param = anon_param_info.param;
        let new_ty = anon_param_info.param_ty;
        let new_ty_span = anon_param_info.param_ty_span;
        let is_first = anon_param_info.is_first;
        let scope_def_id = region_info.scope;
        let is_impl_item = region_info.is_impl_item;

        if anon_param_info.kind.is_named(self.tcx()) {
            /* not an anonymous region */
            debug!("try_report_named_anon_conflict: not an anonymous region");
            return None;
        }

        if is_impl_item {
            debug!("try_report_named_anon_conflict: impl item, bail out");
            return None;
        }

        if find_anon_type(self.tcx(), self.generic_param_scope, anon).is_some()
            && self.is_self_anon(is_first, scope_def_id)
        {
            return None;
        }
        let named = named.to_string();
        let err = match param.pat.simple_ident() {
            Some(simple_ident) => ExplicitLifetimeRequired::WithIdent {
                span,
                simple_ident,
                named,
                new_ty_span,
                new_ty,
            },
            None => ExplicitLifetimeRequired::WithParamType { span, named, new_ty_span, new_ty },
        };
        Some(self.tcx().sess.dcx().create_err(err))
    }
}
