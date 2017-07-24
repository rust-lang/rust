use rustc::hir::def_id::DefId;
use rustc::ty::{Region, Ty, TyCtxt};
use rustc::ty::fold::{BottomUpFolder, TypeFoldable};

use semcheck::mapping::IdMapping;

use std::collections::HashMap;

/// Translate all old `DefId`s in the object to their new counterparts, if possible.
pub fn translate<'a, 'tcx, T>(id_mapping: &IdMapping,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              index_map: &HashMap<u32, DefId>,
                              old: &T) -> T
    where T: TypeFoldable<'tcx>
{
    use rustc::ty::{AdtDef, Binder, ExistentialProjection, ExistentialTraitRef};
    use rustc::ty::ExistentialPredicate::*;
    use rustc::ty::TypeVariants::*;

    old.fold_with(&mut BottomUpFolder { tcx: tcx, fldop: |ty| {
        match ty.sty {
            TyAdt(&AdtDef { ref did, .. }, substs) if id_mapping.in_old_crate(*did) => {
                let new_def_id = id_mapping.get_new_id(*did);
                let new_adt = tcx.adt_def(new_def_id);
                tcx.mk_adt(new_adt, substs)
            },
            TyRef(region, type_and_mut) => {
                tcx.mk_ref(translate_region(tcx, id_mapping, region), type_and_mut)
            },
            TyFnDef(did, substs) => {
                tcx.mk_fn_def(id_mapping.get_new_id(did), substs)
            },
            TyDynamic(preds, region) => {
                let new_preds = tcx.mk_existential_predicates(preds.iter().map(|p| {
                    match *p.skip_binder() {
                        Trait(ExistentialTraitRef { def_id: did, substs }) => {
                            let new_def_id = id_mapping.get_new_id(did);

                            Trait(ExistentialTraitRef {
                                def_id: new_def_id,
                                substs: substs
                            })
                        },
                        Projection(ExistentialProjection { item_def_id, substs, ty }) => {
                            let new_def_id = id_mapping.get_new_id(item_def_id);

                            Projection(ExistentialProjection {
                                item_def_id: new_def_id,
                                substs: substs,
                                ty: ty,
                            })
                        },
                        AutoTrait(did) => {
                            AutoTrait(id_mapping.get_new_id(did))
                        },
                    }
                }));

                tcx.mk_dynamic(Binder(new_preds), region)
            },
            TyProjection(proj) => {
                let trait_def_id = tcx.associated_item(proj.item_def_id).container.id();
                let new_def_id =
                    id_mapping.get_new_trait_item_id(proj.item_def_id, trait_def_id);

                tcx.mk_projection(new_def_id, proj.substs)
            },
            TyAnon(did, substs) => {
                tcx.mk_anon(id_mapping.get_new_id(did), substs)
            },
            TyParam(param) => {
                if param.idx != 0 { // `Self` is special
                    let old_def_id = index_map[&param.idx];
                    if id_mapping.in_old_crate(old_def_id) {
                        let new_def_id = id_mapping.get_new_id(old_def_id);
                        tcx.mk_param_from_def(&id_mapping.get_type_param(&new_def_id))
                    } else {
                        tcx.mk_ty(TyParam(param))
                    }
                } else {
                    tcx.mk_ty(TyParam(param))
                }
            },
            _ => ty,
        }
    }})
}

/// Translate all old `DefId`s in the region to their new counterparts, if possible.
pub fn translate_region<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  id_mapping: &IdMapping,
                                  region: Region<'tcx>) -> Region<'tcx> {
    use rustc::ty::{EarlyBoundRegion, FreeRegion};
    use rustc::ty::BoundRegion::BrNamed;
    use rustc::ty::RegionKind::*;

    tcx.mk_region(match *region {
        ReEarlyBound(early) => {
            let new_early = EarlyBoundRegion {
                def_id: id_mapping.get_new_id(early.def_id),
                index: early.index,
                name: early.name,
            };

            ReEarlyBound(new_early)
        },
        ReFree(FreeRegion { scope, bound_region }) => {
            ReFree(FreeRegion {
                scope: id_mapping.get_new_id(scope),
                bound_region: match bound_region {
                    BrNamed(def_id, name) => BrNamed(id_mapping.get_new_id(def_id), name),
                    reg => reg,
                },
            })
        },
        reg => reg,
    })
}

/// Translate all old `DefId`s in the type to their new counterparts, if possible.
///
/// This computes the mapping of type parameters needed as well.
pub fn translate_item_type<'a, 'tcx>(id_mapping: &IdMapping,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     old_def_id: DefId,
                                     old: Ty<'tcx>) -> Ty<'tcx> {
    let mut index_map = HashMap::new();
    let old_generics = tcx.generics_of(old_def_id);

    for type_ in &old_generics.types {
        index_map.insert(type_.index, type_.def_id);
    }

    if let Some(did) = old_generics.parent {
        let parent_generics = tcx.generics_of(did);

        for type_ in &parent_generics.types {
            index_map.insert(type_.index, type_.def_id);
        }
    }

    translate(id_mapping, tcx, &index_map, &old)
}
