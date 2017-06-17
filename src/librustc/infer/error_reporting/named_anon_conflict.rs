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
use hir::map as hir_map;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    // This method walks the Type of the function body arguments using
    // `fold_regions()` function and returns the
    // &hir::Arg of the function argument corresponding to the anonymous
    // region and the Ty corresponding to the named region.
    // Currently only the case where the function declaration consists of
    // one named region and one anonymous region is handled.
    // Consider the example `fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32`
    // Here, we would return the hir::Arg for y, and we return the type &'a
    // i32, which is the type of y but with the anonymous region replaced
    // with 'a.
    fn find_arg_with_anonymous_region(&self,
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
        let (named, (arg, new_ty)) =
            if self.is_named_region(sub) && self.is_suitable_anonymous_region(sup) {
                (sub, self.find_arg_with_anonymous_region(sup, sub).unwrap())
            } else if self.is_named_region(sup) && self.is_suitable_anonymous_region(sub) {
                (sup, self.find_arg_with_anonymous_region(sub, sup).unwrap())
            } else {
                return false; // inapplicable
            };

        if let Some(simple_name) = arg.pat.simple_name() {
            struct_span_err!(self.tcx.sess,
                             span,
                             E0611,
                             "explicit lifetime required in the type of `{}`",
                             simple_name)
                    .span_label(arg.pat.span,
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
                    .span_label(arg.pat.span,
                                format!("consider changing type to `{}`", new_ty))
                    .span_label(span, format!("lifetime `{}` required", named))
                    .emit();
        }
        return true;

    }

    // This method returns whether the given Region is Anonymous
    pub fn is_suitable_anonymous_region(&self, region: Region<'tcx>) -> bool {

        match *region {
            ty::ReFree(ref free_region) => {
                match free_region.bound_region {
                    ty::BrAnon(..) => {
                        let anonymous_region_binding_scope = free_region.scope;
                        let node_id = self.tcx
                            .hir
                            .as_local_node_id(anonymous_region_binding_scope)
                            .unwrap();
                        match self.tcx.hir.find(node_id) {
                            Some(hir_map::NodeItem(..)) |
                            Some(hir_map::NodeTraitItem(..)) => {
                                // proceed ahead //
                            }
                            Some(hir_map::NodeImplItem(..)) => {
                                if self.tcx.impl_trait_ref(self.tcx.
associated_item(anonymous_region_binding_scope).container.id()).is_some() {
                                    // For now, we do not try to target impls of traits. This is
                                    // because this message is going to suggest that the user
                                    // change the fn signature, but they may not be free to do so,
                                    // since the signature must match the trait.
                                    //
                                    // FIXME(#42706) -- in some cases, we could do better here.
                                    return false;//None;
                                }
                              else{  }

                            }
                            _ => return false, // inapplicable
                            // we target only top-level functions
                        }
                        return true;
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}
