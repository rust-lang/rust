// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting for Anonymous Region Lifetime Errors.
use hir;
use infer::InferCtxt;
use ty::{self, Region};
use infer::region_inference::RegionResolutionError::*;
use infer::region_inference::RegionResolutionError;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    // This method walks the Type of the function body arguments using
    // `fold_regions()` function and returns the
    // &hir::Arg of the function argument corresponding to the anonymous
    // region and the Ty corresponding to the named region.
    // Currently only the case where the function declaration consists of
    // one named region and one anonymous region is handled.
    // Consider the example `fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32`
    // Here, the `y` and the `Ty` of `y` is returned after being substituted
    // by that of the named region.
    pub fn find_arg_with_anonymous_region(&self,
                                          anon_region: Region<'tcx>,
                                          named_region: Region<'tcx>)
                                          -> Option<(&hir::Arg, ty::Ty<'tcx>)> {

        match *anon_region {
            ty::ReFree(ref free_region) => {

                let id = free_region.scope;
                let node_id = self.tcx.hir.as_local_node_id(id).unwrap();
                let body_id = self.tcx.hir.maybe_body_owned_by(node_id).unwrap();

                let body = self.tcx.hir.body(body_id);
                body.arguments
                    .iter()
                    .filter_map(|arg| if let Some(tables) = self.in_progress_tables {
                                    let ty = tables.borrow().node_id_to_type(arg.id);
                                    let mut found_anon_region = false;
                                    let new_arg_ty = self.tcx
                                        .fold_regions(&ty,
                                                      &mut false,
                                                      |r, _| if *r == *anon_region {
                                                          found_anon_region = true;
                                                          named_region
                                                      } else {
                                                          r
                                                      });
                                    if found_anon_region {
                                        return Some((arg, new_arg_ty));
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                })
                    .next()
            }
            _ => None,
        }

    }

    // This method generates the error message for the case when
    // the function arguments consist of a named region and an anonymous
    // region and corresponds to `ConcreteFailure(..)`
    pub fn report_named_anon_conflict(&self, error: &RegionResolutionError<'tcx>) -> bool {

        let (span, sub, sup) = match *error {
            ConcreteFailure(ref origin, sub, sup) => (origin.span(), sub, sup),
            _ => return false, // inapplicable
        };

        let (named, (var, new_ty)) =
            if self.is_named_region(sub) && self.is_anonymous_region(sup) {
                (sub, self.find_arg_with_anonymous_region(sup, sub).unwrap())
            } else if self.is_named_region(sup) && self.is_anonymous_region(sub) {
                (sup, self.find_arg_with_anonymous_region(sub, sup).unwrap())
            } else {
                return false; // inapplicable
            };

        if let Some(simple_name) = var.pat.simple_name() {
            struct_span_err!(self.tcx.sess,
                             span,
                             E0611,
                             "explicit lifetime required in the type of `{}`",
                             simple_name)
                    .span_label(var.pat.span,
                                format!("consider changing the type of `{}` to `{}`",
                                        simple_name,
                                        new_ty))
                    .span_label(span, format!("lifetime `{}` required", named))
                    .emit();

        } else {
            struct_span_err!(self.tcx.sess,
                             span,
                             E0611,
                             "explicit lifetime required in parameter type")
                    .span_label(var.pat.span,
                                format!("consider changing type to `{}`", new_ty))
                    .span_label(span, format!("lifetime `{}` required", named))
                    .emit();
        }
        return true;

    }
}
