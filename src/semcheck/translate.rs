use rustc::hir::def_id::DefId;
use rustc::ty::{ParamEnv, Predicate, Region, Ty, TyCtxt};
use rustc::ty::fold::{BottomUpFolder, TypeFoldable, TypeFolder};

use rustc_data_structures::accumulate_vec::AccumulateVec;

use semcheck::mapping::IdMapping;

use std::collections::HashMap;

/// Construct an parameter index map for an item.
fn construct_index_map<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, old_def_id: DefId)
    -> HashMap<u32, DefId>
{
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

    index_map
}

/// Translate all old `DefId`s in the object to their new counterparts, if possible.
fn translate<'a, 'tcx, T>(id_mapping: &IdMapping,
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
                tcx.mk_ref(translate_region(id_mapping, tcx, region), type_and_mut)
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
fn translate_region<'a, 'tcx>(id_mapping: &IdMapping,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
    translate(id_mapping, tcx, &construct_index_map(tcx, old_def_id), &old)
}

/// Translate all old `DefId`s in the predicate to their new counterparts, if possible.
fn translate_predicate<'a, 'tcx>(id_mapping: &IdMapping,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     index_map: &HashMap<u32, DefId>,
                                     predicate: Predicate<'tcx>) -> Predicate<'tcx> {
    use rustc::ty::{EquatePredicate, OutlivesPredicate, ProjectionPredicate, ProjectionTy,
                    SubtypePredicate, TraitPredicate, TraitRef};

    match predicate {
        Predicate::Trait(trait_predicate) => {
            Predicate::Trait(trait_predicate.map_bound(|t_pred| {
                TraitPredicate {
                    trait_ref: TraitRef {
                        def_id: id_mapping.get_new_id(t_pred.trait_ref.def_id),
                        substs: t_pred.trait_ref.substs,
                    }
                }
            }))
        },
        Predicate::Equate(equate_predicate) => {
            Predicate::Equate(equate_predicate.map_bound(|e_pred| {
                let l = translate(id_mapping, tcx, index_map, &e_pred.0);
                let r = translate(id_mapping, tcx, index_map, &e_pred.1);
                EquatePredicate(l, r)
            }))
        },
        Predicate::RegionOutlives(region_outlives_predicate) => {
            Predicate::RegionOutlives(region_outlives_predicate.map_bound(|r_pred| {
                let l = translate_region(id_mapping, tcx, r_pred.0);
                let r = translate_region(id_mapping, tcx, r_pred.1);
                OutlivesPredicate(l, r)
            }))
        },
        Predicate::TypeOutlives(type_outlives_predicate) => {
            Predicate::TypeOutlives(type_outlives_predicate.map_bound(|r_pred| {
                let l = translate(id_mapping, tcx, index_map, &r_pred.0);
                let r = translate_region(id_mapping, tcx, r_pred.1);
                OutlivesPredicate(l, r)
            }))
        },
        Predicate::Projection(projection_predicate) => {
            Predicate::Projection(projection_predicate.map_bound(|p_pred| {
                ProjectionPredicate {
                    projection_ty: ProjectionTy {
                        substs: p_pred.projection_ty.substs, // TODO: maybe this needs handling
                        item_def_id: id_mapping.get_new_id(p_pred.projection_ty.item_def_id),
                    },
                    ty: translate(id_mapping, tcx, index_map, &p_pred.ty),
                }
            }))
        },
        Predicate::WellFormed(ty) =>
            Predicate::WellFormed(translate(id_mapping, tcx, index_map, &ty)),
        Predicate::ObjectSafe(did) => Predicate::ObjectSafe(id_mapping.get_new_id(did)),
        Predicate::ClosureKind(did, kind) =>
            Predicate::ClosureKind(id_mapping.get_new_id(did), kind),
        Predicate::Subtype(subtype_predicate) => {
            Predicate::Subtype(subtype_predicate.map_bound(|s_pred| {
                let l = translate(id_mapping, tcx, index_map, &s_pred.a);
                let r = translate(id_mapping, tcx, index_map, &s_pred.b);
                SubtypePredicate {
                    a_is_expected: s_pred.a_is_expected,
                    a: l,
                    b: r,
                }
            }))
        },
    }
}

/// Translate all old `DefId`s in the `ParamEnv` to their new counterparts, if possible.
///
/// This computes the mapping of type parameters needed as well.
pub fn translate_param_env<'a, 'tcx>(id_mapping: &IdMapping,
                                     tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     old_def_id: DefId,
                                     param_env: ParamEnv<'tcx>) -> ParamEnv<'tcx> {
    let index_map = construct_index_map(tcx, old_def_id);
    let res = param_env
        .caller_bounds
        .iter()
        .map(|p| translate_predicate(id_mapping, tcx, &index_map, *p))
        .collect::<AccumulateVec<[_; 8]>>();

    ParamEnv {
        caller_bounds: tcx.intern_predicates(&res),
        reveal: param_env.reveal,
    }
}

/// A simple closure folder for regions and types.
pub struct BottomUpRegionFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a, F, G>
    where F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
          G: FnMut(Region<'tcx>) -> Region<'tcx>,
{
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
    pub fldop_t: F,
    pub fldop_r: G,
}

impl<'a, 'gcx, 'tcx, F, G> TypeFolder<'gcx, 'tcx> for BottomUpRegionFolder<'a, 'gcx, 'tcx, F, G>
    where F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
          G: FnMut(Region<'tcx>) -> Region<'tcx>,
{
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.tcx }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t1 = ty.super_fold_with(self);
        (self.fldop_t)(t1)
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        let r1 = r.super_fold_with(self);
        (self.fldop_r)(r1)
    }
}
