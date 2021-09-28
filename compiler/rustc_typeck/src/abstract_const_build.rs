use rustc_middle::thir::abstract_const::Node as ACNode;
use rustc_middle::ty::{self, DefIdTree, TyCtxt, TypeFoldable};
use rustc_span::def_id::LocalDefId;

/// Builds an abstract const, do not use this directly, but use `AbstractConst::new` instead.
pub(super) fn abstract_const_from_fully_qualif_assoc<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> Option<Option<&'tcx [ACNode<'tcx>]>> {
    let anon_ct_hir_id = tcx.hir().local_def_id_to_hir_id(def.did);
    tcx.hir()
        .get(anon_ct_hir_id)
        .is_anon_const()
        .and_then(|ct| tcx.hir().is_fully_qualif_assoc_const_proj(ct.body))
        .map(|(this, path)| {
            let trait_did = tcx.parent(path.res.def_id()).unwrap();
            debug!("trait_did: {:?}", trait_did);
            let item_ctxt: &dyn crate::astconv::AstConv<'_> =
                &crate::collect::ItemCtxt::new(tcx, trait_did);
            let self_ty = item_ctxt.ast_ty_to_ty(this);
            let trait_ref_substs = <dyn crate::astconv::AstConv<'_>>::ast_path_to_mono_trait_ref(
                item_ctxt,
                path.span,
                trait_did,
                self_ty,
                &path.segments[0],
            )
            .substs;
            debug!("trait_ref_substs: {:?}", trait_ref_substs);

            // there is no such thing as `feature(generic_associated_consts)` yet so we dont need
            // to handle substs for the const on the trait i.e. `N` in `<T as Trait<U>>::ASSOC::<N>`
            assert!(path.segments[1].args.is_none());

            trait_ref_substs.definitely_has_param_types_or_consts(tcx).then(|| {
                let ct = tcx.mk_const(ty::Const {
                    val: ty::ConstKind::Unevaluated(ty::Unevaluated::new(
                        ty::WithOptConstParam {
                            did: path.res.def_id(),
                            const_param_did: def.const_param_did,
                        },
                        trait_ref_substs,
                    )),
                    ty: tcx.type_of(path.res.def_id()),
                });
                &*tcx.arena.alloc_from_iter([ACNode::Leaf(ct)])
            })
        })
}
