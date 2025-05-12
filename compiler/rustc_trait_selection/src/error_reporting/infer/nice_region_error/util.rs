//! Helper functions corresponding to lifetime errors due to
//! anonymous regions.

use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{self, Binder, Region, Ty, TyCtxt, TypeFoldable, fold_regions};
use rustc_span::Span;
use tracing::instrument;

use crate::error_reporting::infer::nice_region_error::NiceRegionError;

/// Information about the anonymous region we are searching for.
#[derive(Debug)]
pub struct AnonymousParamInfo<'tcx> {
    /// The parameter corresponding to the anonymous region.
    pub param: &'tcx hir::Param<'tcx>,
    /// The type corresponding to the anonymous region parameter.
    pub param_ty: Ty<'tcx>,
    /// The `ty::LateParamRegionKind` corresponding to the anonymous region.
    pub kind: ty::LateParamRegionKind,
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
    generic_param_scope: LocalDefId,
    anon_region: Region<'tcx>,
    replace_region: Region<'tcx>,
) -> Option<AnonymousParamInfo<'tcx>> {
    let (id, kind) = match anon_region.kind() {
        ty::ReLateParam(late_param) => (late_param.scope, late_param.kind),
        ty::ReEarlyParam(ebr) => {
            let region_def = tcx.generics_of(generic_param_scope).region_param(ebr, tcx).def_id;
            (tcx.parent(region_def), ty::LateParamRegionKind::Named(region_def, ebr.name))
        }
        _ => return None, // not a free region
    };

    let def_id = id.as_local()?;

    // FIXME: use def_kind
    // Don't perform this on closures
    match tcx.hir_node_by_def_id(generic_param_scope) {
        hir::Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => {
            return None;
        }
        _ => {}
    }

    let body = tcx.hir_maybe_body_owned_by(def_id)?;

    let owner_id = tcx.hir_body_owner(body.id());
    let fn_decl = tcx.hir_fn_decl_by_hir_id(owner_id)?;
    let poly_fn_sig = tcx.fn_sig(id).instantiate_identity();

    let fn_sig = tcx.liberate_late_bound_regions(id, poly_fn_sig);
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
            let new_param_ty = fold_regions(tcx, ty, |r, _| {
                if r == anon_region {
                    found_anon_region = true;
                    replace_region
                } else {
                    r
                }
            });
            found_anon_region.then(|| {
                let ty_hir_id = fn_decl.inputs[index].hir_id;
                let param_ty_span = tcx.hir_span(ty_hir_id);
                let is_first = index == 0;
                AnonymousParamInfo { param, param_ty: new_param_ty, param_ty_span, kind, is_first }
            })
        })
}

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    pub(super) fn find_param_with_region(
        &self,
        anon_region: Region<'tcx>,
        replace_region: Region<'tcx>,
    ) -> Option<AnonymousParamInfo<'tcx>> {
        find_param_with_region(self.tcx(), self.generic_param_scope, anon_region, replace_region)
    }

    // Here, we check for the case where the anonymous region
    // is in the return type as written by the user.
    // FIXME(#42703) - Need to handle certain cases here.
    pub(super) fn is_return_type_anon(
        &self,
        scope_def_id: LocalDefId,
        region_def_id: DefId,
        hir_sig: &hir::FnSig<'_>,
    ) -> Option<Span> {
        let fn_ty = self.tcx().type_of(scope_def_id).instantiate_identity();
        if let ty::FnDef(_, _) = fn_ty.kind() {
            let ret_ty = fn_ty.fn_sig(self.tcx()).output();
            let span = hir_sig.decl.output.span();
            let future_output = if hir_sig.header.is_async() {
                ret_ty.map_bound(|ty| self.cx.get_impl_future_output_ty(ty)).transpose()
            } else {
                None
            };
            return match future_output {
                Some(output) if self.includes_region(output, region_def_id) => Some(span),
                None if self.includes_region(ret_ty, region_def_id) => Some(span),
                _ => None,
            };
        }
        None
    }

    fn includes_region(
        &self,
        ty: Binder<'tcx, impl TypeFoldable<TyCtxt<'tcx>>>,
        region_def_id: DefId,
    ) -> bool {
        let late_bound_regions = self.tcx().collect_referenced_late_bound_regions(ty);
        // We are only checking is any region meets the condition so order doesn't matter
        #[allow(rustc::potential_query_instability)]
        late_bound_regions.iter().any(|r| match *r {
            ty::BoundRegionKind::Named(def_id, _) => def_id == region_def_id,
            _ => false,
        })
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
                .is_some_and(|i| i.is_method())
    }
}
