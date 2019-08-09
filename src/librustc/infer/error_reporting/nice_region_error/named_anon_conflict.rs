//! Error Reporting for Anonymous Region Lifetime Errors
//! where one region is named and the other is anonymous.
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::hir::{FunctionRetTy, TyKind};
use crate::ty;
use errors::{Applicability, DiagnosticBuilder};

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// When given a `ConcreteFailure` for a function with arguments containing a named region and
    /// an anonymous region, emit an descriptive diagnostic error.
    pub(super) fn try_report_named_anon_conflict(&self) -> Option<DiagnosticBuilder<'a>> {
        let (span, sub, sup) = self.get_regions();

        debug!(
            "try_report_named_anon_conflict(sub={:?}, sup={:?}, error={:?})",
            sub,
            sup,
            self.error,
        );

        // Determine whether the sub and sup consist of one named region ('a)
        // and one anonymous (elided) region. If so, find the parameter arg
        // where the anonymous region appears (there must always be one; we
        // only introduced anonymous regions in parameters) as well as a
        // version new_ty of its type where the anonymous region is replaced
        // with the named one.//scope_def_id
        let (named, anon, anon_arg_info, region_info) = if self.is_named_region(sub)
            && self.tcx().is_suitable_region(sup).is_some()
            && self.find_arg_with_region(sup, sub).is_some()
        {
            (
                sub,
                sup,
                self.find_arg_with_region(sup, sub).unwrap(),
                self.tcx().is_suitable_region(sup).unwrap(),
            )
        } else if self.is_named_region(sup) && self.tcx().is_suitable_region(sub).is_some()
            && self.find_arg_with_region(sub, sup).is_some()
        {
            (
                sup,
                sub,
                self.find_arg_with_region(sub, sup).unwrap(),
                self.tcx().is_suitable_region(sub).unwrap(),
            )
        } else {
            return None; // inapplicable
        };

        debug!("try_report_named_anon_conflict: named = {:?}", named);
        debug!(
            "try_report_named_anon_conflict: anon_arg_info = {:?}",
            anon_arg_info
        );
        debug!(
            "try_report_named_anon_conflict: region_info = {:?}",
            region_info
        );

        let (arg, new_ty, new_ty_span, br, is_first, scope_def_id, is_impl_item) = (
            anon_arg_info.arg,
            anon_arg_info.arg_ty,
            anon_arg_info.arg_ty_span,
            anon_arg_info.bound_region,
            anon_arg_info.is_first,
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

        if let Some((_, fndecl)) = self.find_anon_type(anon, &br) {
            if self.is_return_type_anon(scope_def_id, br, fndecl).is_some()
                || self.is_self_anon(is_first, scope_def_id)
            {
                return None;
            }
            if let FunctionRetTy::Return(ty) = &fndecl.output {
                if let (TyKind::Def(_, _), ty::ReStatic) = (&ty.node, sub) {
                    // This is an impl Trait return that evaluates de need of 'static.
                    // We handle this case better in `static_impl_trait`.
                    return None;
                }
            }
        }

        let (error_var, span_label_var) = match arg.pat.simple_ident() {
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

    // This method returns whether the given Region is Named
    pub(super) fn is_named_region(&self, region: ty::Region<'tcx>) -> bool {
        match *region {
            ty::ReStatic => true,
            ty::ReFree(ref free_region) => match free_region.bound_region {
                ty::BrNamed(..) => true,
                _ => false,
            },
            ty::ReEarlyBound(ebr) => ebr.has_name(),
            _ => false,
        }
    }
}
