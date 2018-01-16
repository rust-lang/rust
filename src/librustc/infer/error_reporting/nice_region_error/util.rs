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
use infer::error_reporting::nice_region_error::NiceRegionError;
use ty::{self, Region, Ty};
use hir::def_id::DefId;
use hir::map as hir_map;
use syntax_pos::Span;

// The struct contains the information about the anonymous region
// we are searching for.
#[derive(Debug)]
pub(super) struct AnonymousArgInfo<'tcx> {
    // the argument corresponding to the anonymous region
    pub arg: &'tcx hir::Arg,
    // the type corresponding to the anonymopus region argument
    pub arg_ty: Ty<'tcx>,
    // the ty::BoundRegion corresponding to the anonymous region
    pub bound_region: ty::BoundRegion,
    // corresponds to id the argument is the first parameter
    // in the declaration
    pub is_first: bool,
}

// This struct contains information regarding the
// Refree((FreeRegion) corresponding to lifetime conflict
#[derive(Debug)]
pub(super) struct FreeRegionInfo {
    // def id corresponding to FreeRegion
    pub def_id: DefId,
    // the bound region corresponding to FreeRegion
    pub boundregion: ty::BoundRegion,
    // checks if bound region is in Impl Item
    pub is_impl_item: bool,
}

impl<'a, 'gcx, 'tcx> NiceRegionError<'a, 'gcx, 'tcx> {
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
    pub(super) fn find_arg_with_region(
        &self,
        anon_region: Region<'tcx>,
        replace_region: Region<'tcx>,
    ) -> Option<AnonymousArgInfo> {
        let (id, bound_region) = match *anon_region {
            ty::ReFree(ref free_region) => (free_region.scope, free_region.bound_region),
            ty::ReEarlyBound(ref ebr) => (
                self.tcx.parent_def_id(ebr.def_id).unwrap(),
                ty::BoundRegion::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let hir = &self.tcx.hir;
        if let Some(node_id) = hir.as_local_node_id(id) {
            if let Some(body_id) = hir.maybe_body_owned_by(node_id) {
                let body = hir.body(body_id);
                if let Some(tables) = self.tables {
                    body.arguments
                        .iter()
                        .enumerate()
                        .filter_map(|(index, arg)| {
                            // May return None; sometimes the tables are not yet populated.
                            let ty = tables.node_id_to_type_opt(arg.hir_id)?;
                            let mut found_anon_region = false;
                            let new_arg_ty = self.tcx.fold_regions(&ty, &mut false, |r, _| {
                                if *r == *anon_region {
                                    found_anon_region = true;
                                    replace_region
                                } else {
                                    r
                                }
                            });
                            if found_anon_region {
                                let is_first = index == 0;
                                Some(AnonymousArgInfo {
                                    arg: arg,
                                    arg_ty: new_arg_ty,
                                    bound_region: bound_region,
                                    is_first: is_first,
                                })
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
    }

    // This method returns the DefId and the BoundRegion corresponding to the given region.
    pub(super) fn is_suitable_region(&self, region: Region<'tcx>) -> Option<FreeRegionInfo> {
        let (suitable_region_binding_scope, bound_region) = match *region {
            ty::ReFree(ref free_region) => (free_region.scope, free_region.bound_region),
            ty::ReEarlyBound(ref ebr) => (
                self.tcx.parent_def_id(ebr.def_id).unwrap(),
                ty::BoundRegion::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let node_id = self.tcx
            .hir
            .as_local_node_id(suitable_region_binding_scope)
            .unwrap();
        let is_impl_item = match self.tcx.hir.find(node_id) {
            Some(hir_map::NodeItem(..)) | Some(hir_map::NodeTraitItem(..)) => false,
            Some(hir_map::NodeImplItem(..)) => {
                self.is_bound_region_in_impl_item(suitable_region_binding_scope)
            }
            _ => return None,
        };

        return Some(FreeRegionInfo {
            def_id: suitable_region_binding_scope,
            boundregion: bound_region,
            is_impl_item: is_impl_item,
        });
    }

    // Here, we check for the case where the anonymous region
    // is in the return type.
    // FIXME(#42703) - Need to handle certain cases here.
    pub(super) fn is_return_type_anon(
        &self,
        scope_def_id: DefId,
        br: ty::BoundRegion,
        decl: &hir::FnDecl,
    ) -> Option<Span> {
        let ret_ty = self.tcx.type_of(scope_def_id);
        match ret_ty.sty {
            ty::TyFnDef(_, _) => {
                let sig = ret_ty.fn_sig(self.tcx);
                let late_bound_regions = self.tcx
                    .collect_referenced_late_bound_regions(&sig.output());
                if late_bound_regions.iter().any(|r| *r == br) {
                    return Some(decl.output.span());
                }
            }
            _ => {}
        }
        None
    }
    // Here we check for the case where anonymous region
    // corresponds to self and if yes, we display E0312.
    // FIXME(#42700) - Need to format self properly to
    // enable E0621 for it.
    pub(super) fn is_self_anon(&self, is_first: bool, scope_def_id: DefId) -> bool {
        is_first
            && self.tcx
                .opt_associated_item(scope_def_id)
                .map(|i| i.method_has_self_argument) == Some(true)
    }

    // Here we check if the bound region is in Impl Item.
    pub(super) fn is_bound_region_in_impl_item(
        &self,
        suitable_region_binding_scope: DefId,
    ) -> bool {
        let container_id = self.tcx
            .associated_item(suitable_region_binding_scope)
            .container
            .id();
        if self.tcx.impl_trait_ref(container_id).is_some() {
            // For now, we do not try to target impls of traits. This is
            // because this message is going to suggest that the user
            // change the fn signature, but they may not be free to do so,
            // since the signature must match the trait.
            //
            // FIXME(#42706) -- in some cases, we could do better here.
            return true;
        }
        false
    }
}
