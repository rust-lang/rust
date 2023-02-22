//! Helper functions corresponding to lifetime errors due to
//! anonymous regions.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::TyCtxt;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::{self, Binder, DefIdTree, Region, Ty, TypeVisitable};
use rustc_span::Span;

/// Information about the anonymous region we are searching for.
#[derive(Debug)]
pub struct AnonymousParamInfo<'tcx> {
    /// The parameter corresponding to the anonymous region.
    pub param: &'tcx hir::Param<'tcx>,
    /// The type corresponding to the anonymous region parameter.
    pub param_ty: Ty<'tcx>,
    /// The ty::BoundRegionKind corresponding to the anonymous region.
    pub bound_region: ty::BoundRegionKind,
    /// The `Span` of the parameter type.
    pub param_ty_span: Span,
    /// Signals that the argument is the first parameter in the declaration.
    pub is_first: bool,
}

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
#[instrument(skip(tcx), level = "debug")]
pub fn find_param_with_region<'tcx>(
    tcx: TyCtxt<'tcx>,
    anon_region: Region<'tcx>,
    replace_region: Region<'tcx>,
) -> Option<AnonymousParamInfo<'tcx>> {
    let (id, bound_region) = match *anon_region {
        ty::ReFree(ref free_region) => (free_region.scope, free_region.bound_region),
        ty::ReEarlyBound(ebr) => {
            (tcx.parent(ebr.def_id), ty::BoundRegionKind::BrNamed(ebr.def_id, ebr.name))
        }
        _ => return None, // not a free region
    };

    let hir = &tcx.hir();
    let def_id = id.as_local()?;
    let hir_id = hir.local_def_id_to_hir_id(def_id);

    // FIXME: use def_kind
    // Don't perform this on closures
    match hir.get(hir_id) {
        hir::Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => {
            return None;
        }
        _ => {}
    }

    let body_id = hir.maybe_body_owned_by(def_id)?;

    let owner_id = hir.body_owner(body_id);
    let fn_decl = hir.fn_decl_by_hir_id(owner_id).unwrap();
    let poly_fn_sig = tcx.fn_sig(id).subst_identity();

    let fn_sig = tcx.liberate_late_bound_regions(id, poly_fn_sig);
    let body = hir.body(body_id);
    body.params
        .iter()
        .take(if fn_sig.c_variadic {
            fn_sig.inputs().len()
        } else {
            assert_eq!(fn_sig.inputs().len(), body.params.len());
            body.params.len()
        })
        .enumerate()
        .find_map(|(index, param)| {
            // May return None; sometimes the tables are not yet populated.
            let ty = fn_sig.inputs()[index];
            let mut found_anon_region = false;
            let new_param_ty = tcx.fold_regions(ty, |r, _| {
                if r == anon_region {
                    found_anon_region = true;
                    replace_region
                } else {
                    r
                }
            });
            found_anon_region.then(|| {
                let ty_hir_id = fn_decl.inputs[index].hir_id;
                let param_ty_span = hir.span(ty_hir_id);
                let is_first = index == 0;
                AnonymousParamInfo {
                    param,
                    param_ty: new_param_ty,
                    param_ty_span,
                    bound_region,
                    is_first,
                }
            })
        })
}

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    pub(super) fn find_param_with_region(
        &self,
        anon_region: Region<'tcx>,
        replace_region: Region<'tcx>,
    ) -> Option<AnonymousParamInfo<'tcx>> {
        find_param_with_region(self.tcx(), anon_region, replace_region)
    }

    // Here, we check for the case where the anonymous region
    // is in the return type as written by the user.
    // FIXME(#42703) - Need to handle certain cases here.
    pub(super) fn is_return_type_anon(
        &self,
        scope_def_id: LocalDefId,
        br: ty::BoundRegionKind,
        hir_sig: &hir::FnSig<'_>,
    ) -> Option<Span> {
        let fn_ty = self.tcx().type_of(scope_def_id).subst_identity();
        if let ty::FnDef(_, _) = fn_ty.kind() {
            let ret_ty = fn_ty.fn_sig(self.tcx()).output();
            let span = hir_sig.decl.output.span();
            let future_output = if hir_sig.header.is_async() {
                ret_ty.map_bound(|ty| self.cx.get_impl_future_output_ty(ty)).transpose()
            } else {
                None
            };
            return match future_output {
                Some(output) if self.includes_region(output, br) => Some(span),
                None if self.includes_region(ret_ty, br) => Some(span),
                _ => None,
            };
        }
        None
    }

    fn includes_region(
        &self,
        ty: Binder<'tcx, impl TypeVisitable<TyCtxt<'tcx>>>,
        region: ty::BoundRegionKind,
    ) -> bool {
        let late_bound_regions = self.tcx().collect_referenced_late_bound_regions(&ty);
        // We are only checking is any region meets the condition so order doesn't matter
        #[allow(rustc::potential_query_instability)]
        late_bound_regions.iter().any(|r| *r == region)
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
