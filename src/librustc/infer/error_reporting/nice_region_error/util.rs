//! Helper functions corresponding to lifetime errors due to
//! anonymous regions.

use crate::hir;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::ty::{self, DefIdTree, Region, Ty};
use crate::hir::def_id::DefId;
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
    // arg_ty_span contains span of argument type
    pub arg_ty_span : Span,
    // corresponds to id the argument is the first parameter
    // in the declaration
    pub is_first: bool,
}

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
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
    ) -> Option<AnonymousArgInfo<'_>> {
        let (id, bound_region) = match *anon_region {
            ty::ReFree(ref free_region) => (free_region.scope, free_region.bound_region),
            ty::ReEarlyBound(ebr) => (
                self.tcx().parent(ebr.def_id).unwrap(),
                ty::BoundRegion::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let hir = &self.tcx().hir();
        if let Some(hir_id) = hir.as_local_hir_id(id) {
            if let Some(body_id) = hir.maybe_body_owned_by(hir_id) {
                let body = hir.body(body_id);
                let owner_id = hir.body_owner(body_id);
                let fn_decl = hir.fn_decl_by_hir_id(owner_id).unwrap();
                if let Some(tables) = self.tables {
                    body.arguments
                        .iter()
                        .enumerate()
                        .filter_map(|(index, arg)| {
                            // May return None; sometimes the tables are not yet populated.
                            let ty_hir_id = fn_decl.inputs[index].hir_id;
                            let arg_ty_span = hir.span(ty_hir_id);
                            let ty = tables.node_type_opt(arg.hir_id)?;
                            let mut found_anon_region = false;
                            let new_arg_ty = self.tcx().fold_regions(&ty, &mut false, |r, _| {
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
                                    arg_ty_span : arg_ty_span,
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

    // Here, we check for the case where the anonymous region
    // is in the return type.
    // FIXME(#42703) - Need to handle certain cases here.
    pub(super) fn is_return_type_anon(
        &self,
        scope_def_id: DefId,
        br: ty::BoundRegion,
        decl: &hir::FnDecl,
    ) -> Option<Span> {
        let ret_ty = self.tcx().type_of(scope_def_id);
        if let ty::FnDef(_, _) = ret_ty.sty {
            let sig = ret_ty.fn_sig(self.tcx());
            let late_bound_regions = self.tcx()
                .collect_referenced_late_bound_regions(&sig.output());
            if late_bound_regions.iter().any(|r| *r == br) {
                return Some(decl.output.span());
            }
        }
        None
    }

    // Here we check for the case where anonymous region
    // corresponds to self and if yes, we display E0312.
    // FIXME(#42700) - Need to format self properly to
    // enable E0621 for it.
    pub(super) fn is_self_anon(&self, is_first: bool, scope_def_id: DefId) -> bool {
        is_first
            && self.tcx()
                   .opt_associated_item(scope_def_id)
                   .map(|i| i.method_has_self_argument) == Some(true)
    }

}
