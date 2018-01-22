// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting for Anonymous Region Lifetime Errors
//! where one region is named and the other is anonymous.
use infer::error_reporting::nice_region_error::NiceRegionError;
use ty;
use util::common::ErrorReported;

impl<'a, 'gcx, 'tcx> NiceRegionError<'a, 'gcx, 'tcx> {
    /// When given a `ConcreteFailure` for a function with arguments containing a named region and
    /// an anonymous region, emit an descriptive diagnostic error.
    pub(super) fn try_report_named_anon_conflict(&self) -> Option<ErrorReported> {
        let (span, sub, sup) = self.get_regions();

        debug!(
            "try_report_named_anon_conflict(sub={:?}, sup={:?})",
            sub,
            sup
        );

        // Determine whether the sub and sup consist of one named region ('a)
        // and one anonymous (elided) region. If so, find the parameter arg
        // where the anonymous region appears (there must always be one; we
        // only introduced anonymous regions in parameters) as well as a
        // version new_ty of its type where the anonymous region is replaced
        // with the named one.//scope_def_id
        let (named, anon, anon_arg_info, region_info) = if self.is_named_region(sub)
            && self.is_suitable_region(sup).is_some()
            && self.find_arg_with_region(sup, sub).is_some()
        {
            (
                sub,
                sup,
                self.find_arg_with_region(sup, sub).unwrap(),
                self.is_suitable_region(sup).unwrap(),
            )
        } else if self.is_named_region(sup) && self.is_suitable_region(sub).is_some()
            && self.find_arg_with_region(sub, sup).is_some()
        {
            (
                sup,
                sub,
                self.find_arg_with_region(sub, sup).unwrap(),
                self.is_suitable_region(sub).unwrap(),
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

        let (arg, new_ty, br, is_first, scope_def_id, is_impl_item) = (
            anon_arg_info.arg,
            anon_arg_info.arg_ty,
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
        }

        let (error_var, span_label_var) = if let Some(simple_name) = arg.pat.simple_name() {
            (
                format!("the type of `{}`", simple_name),
                format!("the type of `{}`", simple_name),
            )
        } else {
            ("parameter type".to_owned(), "type".to_owned())
        };

        struct_span_err!(
            self.tcx.sess,
            span,
            E0621,
            "explicit lifetime required in {}",
            error_var
        ).span_label(
            arg.pat.span,
            format!("consider changing {} to `{}`", span_label_var, new_ty),
        )
            .span_label(span, format!("lifetime `{}` required", named))
            .emit();
        return Some(ErrorReported);
    }

    // This method returns whether the given Region is Named
    pub(super) fn is_named_region(&self, region: ty::Region<'tcx>) -> bool {
        match *region {
            ty::ReStatic => true,
            ty::ReFree(ref free_region) => match free_region.bound_region {
                ty::BrNamed(..) => true,
                _ => false,
            },
            ty::ReEarlyBound(_) => true,
            _ => false,
        }
    }
}
