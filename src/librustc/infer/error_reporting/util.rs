// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper functions corresponding to lifetime errors due to
//! anonymous regions.
use hir;
use infer::InferCtxt;
use ty::{self, Region};
use hir::def_id::DefId;
use hir::map as hir_map;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    // This method walks the Type of the function body arguments using
    // `fold_regions()` function and returns the
    // &hir::Arg of the function argument corresponding to the anonymous
    // region and the Ty corresponding to the named region.
    // Currently only the case where the function declaration consists of
    // one named region and one anonymous region is handled.
    // Consider the example `fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32`
    // Here, we would return the hir::Arg for y, we return the type &'a
    // i32, which is the type of y but with the anonymous region replaced
    // with 'a, the corresponding bound region and is_first which is true if
    // the hir::Arg is the first argument in the function declaration.
    pub fn find_arg_with_anonymous_region
        (&self,
         anon_region: Region<'tcx>,
         replace_region: Region<'tcx>)
         -> Option<(&hir::Arg, ty::Ty<'tcx>, ty::BoundRegion, bool)> {

        if let ty::ReFree(ref free_region) = *anon_region {

            let id = free_region.scope;
            let hir = &self.tcx.hir;
            if let Some(node_id) = hir.as_local_node_id(id) {
                if let Some(body_id) = hir.maybe_body_owned_by(node_id) {
                    let body = hir.body(body_id);
                    if let Some(tables) = self.in_progress_tables {
                        body.arguments
                            .iter()
                            .enumerate()
                            .filter_map(|(index, arg)| {
                                let ty = tables.borrow().node_id_to_type(arg.hir_id);
                                let mut found_anon_region = false;
                                let new_arg_ty = self.tcx
                                    .fold_regions(&ty, &mut false, |r, _| if *r == *anon_region {
                                        found_anon_region = true;
                                        replace_region
                                    } else {
                                        r
                                    });
                                if found_anon_region {
                                    let is_first = index == 0;
                                    Some((arg, new_arg_ty, free_region.bound_region, is_first))
                                } else {
                                    None
                                }
                            })
                            .next()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    // This method returns whether the given Region is Anonymous
    // and returns the DefId and the BoundRegion corresponding to the given region.
    pub fn is_suitable_anonymous_region(&self,
                                        region: Region<'tcx>)
                                        -> Option<(DefId, ty::BoundRegion)> {
        if let ty::ReFree(ref free_region) = *region {
            if let ty::BrAnon(..) = free_region.bound_region{
                    let anonymous_region_binding_scope = free_region.scope;
                    let node_id = self.tcx
                        .hir
                        .as_local_node_id(anonymous_region_binding_scope)
                        .unwrap();
                    match self.tcx.hir.find(node_id) {
                        Some(hir_map::NodeItem(..)) |
                        Some(hir_map::NodeTraitItem(..)) => {
                            // Success -- proceed to return Some below
                        }
                        Some(hir_map::NodeImplItem(..)) => {
                            let container_id = self.tcx
                                .associated_item(anonymous_region_binding_scope)
                                .container
                                .id();
                            if self.tcx.impl_trait_ref(container_id).is_some() {
                                // For now, we do not try to target impls of traits. This is
                                // because this message is going to suggest that the user
                                // change the fn signature, but they may not be free to do so,
                                // since the signature must match the trait.
                                //
                                // FIXME(#42706) -- in some cases, we could do better here.
                                return None;
                            }
                        }
                        _ => return None, // inapplicable
                        // we target only top-level functions
                    }
                    return Some((anonymous_region_binding_scope, free_region.bound_region));
                }
            }
            None
        }
}
