//! Error Reporting for Anonymous Region Lifetime Errors
//! where one region is named and the other is anonymous.
use crate::infer::error_reporting::nice_region_error::find_anon_type::find_anon_type;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir::intravisit::Visitor;
use rustc_hir::FnRetTy;
use rustc_middle::ty;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// When given a `ConcreteFailure` for a function with parameters containing a named region and
    /// an anonymous region, emit an descriptive diagnostic error.
    pub(super) fn try_report_named_anon_conflict(&self) -> Option<DiagnosticBuilder<'tcx>> {
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
        let (named, anon, anon_param_info, region_info) = if sub.has_name()
            && self.tcx().is_suitable_region(sup).is_some()
            && self.find_param_with_region(sup, sub).is_some()
        {
            (
                sub,
                sup,
                self.find_param_with_region(sup, sub).unwrap(),
                self.tcx().is_suitable_region(sup).unwrap(),
            )
        } else if sup.has_name()
            && self.tcx().is_suitable_region(sub).is_some()
            && self.find_param_with_region(sub, sup).is_some()
        {
            (
                sup,
                sub,
                self.find_param_with_region(sub, sup).unwrap(),
                self.tcx().is_suitable_region(sub).unwrap(),
            )
        } else {
            return None; // inapplicable
        };

        debug!("try_report_named_anon_conflict: named = {:?}", named);
        debug!("try_report_named_anon_conflict: anon_param_info = {:?}", anon_param_info);
        debug!("try_report_named_anon_conflict: region_info = {:?}", region_info);

        let (param, new_ty, new_ty_span, br, is_first, scope_def_id, is_impl_item) = (
            anon_param_info.param,
            anon_param_info.param_ty,
            anon_param_info.param_ty_span,
            anon_param_info.bound_region,
            anon_param_info.is_first,
            region_info.def_id,
            region_info.is_impl_item,
        );
        match br {
            ty::BrAnon(_) => {}
            _ => {
                /* not an anonymous region */
                debug!("try_report_named_anon_conflict: not an anonymous region");
                return None;
            }
        }

        if is_impl_item {
            debug!("try_report_named_anon_conflict: impl item, bail out");
            return None;
        }

        if let Some((_, fndecl)) = find_anon_type(self.tcx(), anon, &br) {
            if self.is_self_anon(is_first, scope_def_id) {
                return None;
            }

            if let FnRetTy::Return(ty) = &fndecl.output {
                let mut v = ty::TraitObjectVisitor(vec![], self.tcx().hir());
                v.visit_ty(ty);

                debug!("try_report_named_anon_conflict: ret ty {:?}", ty);
                if sub == &ty::ReStatic
                    && v.0.into_iter().any(|t| t.span.desugaring_kind().is_none())
                {
                    // If the failure is due to a `'static` requirement coming from a `dyn` or
                    // `impl` Trait that *isn't* caused by `async fn` desugaring, handle this case
                    // better in `static_impl_trait`.
                    debug!("try_report_named_anon_conflict: impl Trait + 'static");
                    return None;
                }
            }
        }

        let (error_var, span_label_var) = match param.pat.simple_ident() {
            Some(simple_ident) => (
                format!("the type of `{}`", simple_ident),
                format!("the type of `{}`", simple_ident),
            ),
            None => ("parameter type".to_owned(), "type".to_owned()),
        };

        let mut diag = struct_span_err!(
            self.tcx().sess,
            span,
            E0621,
            "explicit lifetime required in {}",
            error_var
        );

        diag.span_label(span, format!("lifetime `{}` required", named));
        diag.span_suggestion(
            new_ty_span,
            &format!("add explicit lifetime `{}` to {}", named, span_label_var),
            new_ty.to_string(),
            Applicability::Unspecified,
        );

        Some(diag)
    }
}
