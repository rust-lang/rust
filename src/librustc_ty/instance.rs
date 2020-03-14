use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, Instance, TyCtxt, TypeFoldable};
use rustc_hir::def_id::DefId;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;

use log::debug;

pub fn resolve_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
) -> Option<Instance<'tcx>> {
    debug!("resolve(def_id={:?}, substs={:?})", def_id, substs);
    let result = if let Some(trait_def_id) = tcx.trait_of_item(def_id) {
        debug!(" => associated item, attempting to find impl in param_env {:#?}", param_env);
        let item = tcx.associated_item(def_id);
        resolve_associated_item(tcx, &item, param_env, trait_def_id, substs)
    } else {
        let ty = tcx.type_of(def_id);
        let item_type = tcx.subst_and_normalize_erasing_regions(substs, param_env, &ty);

        let def = match item_type.kind {
            ty::FnDef(..)
                if {
                    let f = item_type.fn_sig(tcx);
                    f.abi() == Abi::RustIntrinsic || f.abi() == Abi::PlatformIntrinsic
                } =>
            {
                debug!(" => intrinsic");
                ty::InstanceDef::Intrinsic(def_id)
            }
            _ => {
                if Some(def_id) == tcx.lang_items().drop_in_place_fn() {
                    let ty = substs.type_at(0);
                    if ty.needs_drop(tcx, param_env.with_reveal_all()) {
                        debug!(" => nontrivial drop glue");
                        ty::InstanceDef::DropGlue(def_id, Some(ty))
                    } else {
                        debug!(" => trivial drop glue");
                        ty::InstanceDef::DropGlue(def_id, None)
                    }
                } else {
                    debug!(" => free item");
                    ty::InstanceDef::Item(def_id)
                }
            }
        };
        Some(Instance { def, substs })
    };
    debug!("resolve(def_id={:?}, substs={:?}) = {:?}", def_id, substs, result);
    result
}

fn resolve_associated_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item: &ty::AssocItem,
    param_env: ty::ParamEnv<'tcx>,
    trait_id: DefId,
    rcvr_substs: SubstsRef<'tcx>,
) -> Option<Instance<'tcx>> {
    let def_id = trait_item.def_id;
    debug!(
        "resolve_associated_item(trait_item={:?}, \
            param_env={:?}, \
            trait_id={:?}, \
            rcvr_substs={:?})",
        def_id, param_env, trait_id, rcvr_substs
    );

    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);
    let vtbl = tcx.codegen_fulfill_obligation((param_env, ty::Binder::bind(trait_ref)))?;

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        traits::VtableImpl(impl_data) => {
            let (def_id, substs) =
                traits::find_associated_item(tcx, param_env, trait_item, rcvr_substs, &impl_data);

            let resolved_item = tcx.associated_item(def_id);

            // Since this is a trait item, we need to see if the item is either a trait default item
            // or a specialization because we can't resolve those unless we can `Reveal::All`.
            // NOTE: This should be kept in sync with the similar code in
            // `rustc::traits::project::assemble_candidates_from_impls()`.
            let eligible = if !resolved_item.defaultness.is_default() {
                true
            } else if param_env.reveal == traits::Reveal::All {
                !trait_ref.needs_subst()
            } else {
                false
            };

            if !eligible {
                return None;
            }

            let substs = tcx.erase_regions(&substs);
            Some(ty::Instance::new(def_id, substs))
        }
        traits::VtableGenerator(generator_data) => Some(Instance {
            def: ty::InstanceDef::Item(generator_data.generator_def_id),
            substs: generator_data.substs,
        }),
        traits::VtableClosure(closure_data) => {
            let trait_closure_kind = tcx.fn_trait_kind_from_lang_item(trait_id).unwrap();
            Some(Instance::resolve_closure(
                tcx,
                closure_data.closure_def_id,
                closure_data.substs,
                trait_closure_kind,
            ))
        }
        traits::VtableFnPointer(ref data) => Some(Instance {
            def: ty::InstanceDef::FnPtrShim(trait_item.def_id, data.fn_ty),
            substs: rcvr_substs,
        }),
        traits::VtableObject(ref data) => {
            let index = traits::get_vtable_index_of_object_method(tcx, data, def_id);
            Some(Instance { def: ty::InstanceDef::Virtual(def_id, index), substs: rcvr_substs })
        }
        traits::VtableBuiltin(..) => {
            if tcx.lang_items().clone_trait().is_some() {
                Some(Instance {
                    def: ty::InstanceDef::CloneShim(def_id, trait_ref.self_ty()),
                    substs: rcvr_substs,
                })
            } else {
                None
            }
        }
        traits::VtableAutoImpl(..) | traits::VtableParam(..) | traits::VtableTraitAlias(..) => None,
    }
}
