//! Helper functions corresponding to lifetime errors due to
//! anonymous regions.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, DefIdTree, Region, Ty};
use rustc_span::Span;

/// Information about the anonymous region we are searching for.
#[derive(Debug)]
pub(super) struct AnonymousParamInfo<'tcx> {
    /// The parameter corresponding to the anonymous region.
    pub param: &'tcx hir::Param<'tcx>,
    /// The type corresponding to the anonymous region parameter.
    pub param_ty: Ty<'tcx>,
    /// The ty::BoundRegion corresponding to the anonymous region.
    pub bound_region: ty::BoundRegion,
    /// The `Span` of the parameter type.
    pub param_ty_span: Span,
    /// Signals that the argument is the first parameter in the declaration.
    pub is_first: bool,
}

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    // This method walks the Type of the function body parameters using
    // `fold_regions()` function and returns the
    // &hir::Param of the function parameter corresponding to the anonymous
    // region and the Ty corresponding to the named region.
    // Currently only the case where the function declaration consists of
    // one named region and one anonymous region is handled.
    // Consider the example `fn foo<'a>(x: &'a i32, y: &i32) -> &'a i32`
    // Here, we would return the hir::Param for y, we return the type &'a
    // i32, which is the type of y but with the anonymous region replaced
    // with 'a, the corresponding bound region and is_first which is true if
    // the hir::Param is the first parameter in the function declaration.
    pub(super) fn find_param_with_region(
        &self,
        anon_region: Region<'tcx>,
        replace_region: Region<'tcx>,
    ) -> Option<AnonymousParamInfo<'_>> {
        let (id, bound_region) = match *anon_region {
            ty::ReFree(ref free_region) => (free_region.scope, free_region.bound_region),
            ty::ReEarlyBound(ebr) => (
                self.tcx().parent(ebr.def_id).unwrap(),
                ty::BoundRegion::BrNamed(ebr.def_id, ebr.name),
            ),
            _ => return None, // not a free region
        };

        let hir = &self.tcx().hir();
        let hir_id = hir.local_def_id_to_hir_id(id.as_local()?);
        let body_id = hir.maybe_body_owned_by(hir_id)?;
        let body = hir.body(body_id);
        let owner_id = hir.body_owner(body_id);
        let fn_decl = hir.fn_decl_by_hir_id(owner_id).unwrap();
        let poly_fn_sig = self.tcx().fn_sig(id);
        let fn_sig = self.tcx().liberate_late_bound_regions(id, poly_fn_sig);
        body.params.iter().enumerate().find_map(|(index, param)| {
            // May return None; sometimes the tables are not yet populated.
            let ty = fn_sig.inputs()[index];
            let mut found_anon_region = false;
            let new_param_ty = self.tcx().fold_regions(ty, &mut false, |r, _| {
                if *r == *anon_region {
                    found_anon_region = true;
                    replace_region
                } else {
                    r
                }
            });
            if found_anon_region {
                let ty_hir_id = fn_decl.inputs[index].hir_id;
                let param_ty_span = hir.span(ty_hir_id);
                let is_first = index == 0;
                Some(AnonymousParamInfo {
                    param,
                    param_ty: new_param_ty,
                    param_ty_span,
                    bound_region,
                    is_first,
                })
            } else {
                None
            }
        })
    }

    pub(super) fn future_return_type(
        &self,
        local_def_id: LocalDefId,
    ) -> Option<&rustc_hir::Ty<'_>> {
        if let Some(hir::IsAsync::Async) = self.asyncness(local_def_id) {
            if let rustc_middle::ty::Opaque(def_id, _) =
                self.tcx().type_of(local_def_id).fn_sig(self.tcx()).output().skip_binder().kind()
            {
                match self.tcx().hir().get_if_local(*def_id) {
                    Some(hir::Node::Item(hir::Item {
                        kind:
                            hir::ItemKind::OpaqueTy(hir::OpaqueTy {
                                bounds,
                                origin: hir::OpaqueTyOrigin::AsyncFn,
                                ..
                            }),
                        ..
                    })) => {
                        for b in bounds.iter() {
                            if let hir::GenericBound::LangItemTrait(
                                hir::LangItem::Future,
                                _span,
                                _hir_id,
                                generic_args,
                            ) = b
                            {
                                for type_binding in generic_args.bindings.iter() {
                                    if type_binding.ident.name == rustc_span::sym::Output {
                                        if let hir::TypeBindingKind::Equality { ty } =
                                            type_binding.kind
                                        {
                                            return Some(ty);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    pub(super) fn asyncness(&self, local_def_id: LocalDefId) -> Option<hir::IsAsync> {
        // similar to the asyncness fn in rustc_ty::ty
        let hir_id = self.tcx().hir().local_def_id_to_hir_id(local_def_id);
        let node = self.tcx().hir().get(hir_id);
        let fn_like = rustc_middle::hir::map::blocks::FnLikeNode::from_node(node)?;

        Some(fn_like.asyncness())
    }

    // Here, we check for the case where the anonymous region
    // is in the return type.
    // FIXME(#42703) - Need to handle certain cases here.
    pub(super) fn is_return_type_anon(
        &self,
        scope_def_id: LocalDefId,
        br: ty::BoundRegion,
        decl: &hir::FnDecl<'_>,
    ) -> Option<Span> {
        let ret_ty = self.tcx().type_of(scope_def_id);
        if let ty::FnDef(_, _) = ret_ty.kind() {
            let sig = ret_ty.fn_sig(self.tcx());
            let late_bound_regions =
                self.tcx().collect_referenced_late_bound_regions(&sig.output());
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
    pub(super) fn is_self_anon(&self, is_first: bool, scope_def_id: LocalDefId) -> bool {
        is_first
            && self
                .tcx()
                .opt_associated_item(scope_def_id.to_def_id())
                .map(|i| i.fn_has_self_parameter)
                == Some(true)
    }
}
