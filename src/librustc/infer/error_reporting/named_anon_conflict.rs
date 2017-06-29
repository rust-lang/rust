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
use infer::InferCtxt;
use ty;
use infer::region_inference::RegionResolutionError::*;
use infer::region_inference::RegionResolutionError;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    // This method generates the error message for the case when
    // the function arguments consist of a named region and an anonymous
    // region and corresponds to `ConcreteFailure(..)`
    pub fn try_report_named_anon_conflict(&self, error: &RegionResolutionError<'tcx>) -> bool {
        let (span, sub, sup) = match *error {
            ConcreteFailure(ref origin, sub, sup) => (origin.span(), sub, sup),
            _ => return false, // inapplicable
        };

        // Determine whether the sub and sup consist of one named region ('a)
        // and one anonymous (elided) region. If so, find the parameter arg
        // where the anonymous region appears (there must always be one; we
        // only introduced anonymous regions in parameters) as well as a
        // version new_ty of its type where the anonymous region is replaced
        // with the named one.
        let (named, (arg, new_ty, br, is_first), (scope_def_id, _)) =
            if sub.is_named_region() && self.is_suitable_anonymous_region(sup).is_some() {
                (sub,
                 self.find_arg_with_anonymous_region(sup, sub).unwrap(),
                 self.is_suitable_anonymous_region(sup).unwrap())
            } else if sup.is_named_region() && self.is_suitable_anonymous_region(sub).is_some() {
                (sup,
                 self.find_arg_with_anonymous_region(sub, sup).unwrap(),
                 self.is_suitable_anonymous_region(sub).unwrap())
            } else {
                return false; // inapplicable
            };

        // Here, we check for the case where the anonymous region
        // is in the return type.
        // FIXME(#42703) - Need to handle certain cases here.
        let ret_ty = self.tcx.type_of(scope_def_id);
        match ret_ty.sty {
            ty::TyFnDef(_, _) => {
                let sig = ret_ty.fn_sig(self.tcx);
                let late_bound_regions = self.tcx
                    .collect_referenced_late_bound_regions(&sig.output());
                if late_bound_regions.iter().any(|r| *r == br) {
                    return false;
                }
            }
            _ => {}
        }

        // Here we check for the case where anonymous region
        // corresponds to self and if yes, we display E0312.
        // FIXME(#42700) - Need to format self properly to
        // enable E0621 for it.
        if is_first &&
           self.tcx
               .opt_associated_item(scope_def_id)
               .map(|i| i.method_has_self_argument)
               .unwrap_or(false) {
            return false;
        }

        let (error_var, span_label_var) = if let Some(simple_name) = arg.pat.simple_name() {
            (format!("the type of `{}`", simple_name), format!("the type of `{}`", simple_name))
        } else {
            ("parameter type".to_owned(), "type".to_owned())
        };

        struct_span_err!(self.tcx.sess,
                         span,
                         E0621,
                         "explicit lifetime required in {}",
                         error_var)
                .span_label(arg.pat.span,
                            format!("consider changing {} to `{}`", span_label_var, new_ty))
                .span_label(span, format!("lifetime `{}` required", named))
                .emit();

        return true;
    }
}
