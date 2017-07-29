use rustc::hir::def_id::DefId;
use rustc::ty::{ParamEnv, Predicate, Region, Ty, TyCtxt};
use rustc::ty::fold::{BottomUpFolder, TypeFoldable, TypeFolder};

use rustc_data_structures::accumulate_vec::AccumulateVec;

use semcheck::mapping::IdMapping;

use std::collections::HashMap;

/// The context in which `DefId` translation happens.
pub struct TranslationContext<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> {
    /// The type context to use.
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    /// The id mapping to use.
    id_mapping: &'a IdMapping,
    /// Elementary operation to decide whether to translate a `DefId`.
    needs_translation: fn(&IdMapping, DefId) -> bool,
    /// Elementary operation to translate a `DefId`.
    translate_orig: fn(&IdMapping, DefId) -> DefId,
    /// Elementary operation to translate a `DefId` of a trait item.
    translate_orig_trait: fn(&IdMapping, DefId, DefId) -> DefId,
}

impl<'a, 'gcx, 'tcx> TranslationContext<'a, 'gcx, 'tcx> {
    /// Construct a translation context translating to the new crate's `DefId`s.
    // TODO: check whether the function pointers force us to use dynamic dispatch
    pub fn to_new(tcx: TyCtxt<'a, 'gcx, 'tcx>, id_mapping: &'a IdMapping)
        -> TranslationContext<'a, 'gcx, 'tcx/*, F1, F2, F3*/>
    {
        TranslationContext {
            tcx: tcx,
            id_mapping: id_mapping,
            needs_translation: |id_mapping, orig_def_id| {
                id_mapping.in_old_crate(orig_def_id)
            },
            translate_orig: |id_mapping, orig_def_id| {
                id_mapping.get_new_id(orig_def_id)
            },
            translate_orig_trait: |id_mapping, orig_def_id, trait_def_id| {
                id_mapping.get_new_trait_item_id(orig_def_id, trait_def_id)
            },
        }
    }

    /// Construct a translation context translating to the old crate's `DefId`s.
    // TODO: check whether the function pointers force us to use dynamic dispatch
    pub fn to_old(tcx: TyCtxt<'a, 'gcx, 'tcx>, id_mapping: &'a IdMapping)
        -> TranslationContext<'a, 'gcx, 'tcx/*, F1, F2, F3*/>
    {
        TranslationContext {
            tcx: tcx,
            id_mapping: id_mapping,
            needs_translation: |id_mapping, orig_def_id| {
                id_mapping.in_new_crate(orig_def_id)
            },
            translate_orig: |id_mapping, orig_def_id| {
                id_mapping.get_old_id(orig_def_id)
            },
            translate_orig_trait: |id_mapping, orig_def_id, trait_def_id| {
                id_mapping.get_old_trait_item_id(orig_def_id, trait_def_id)
            },
        }
    }

    /// Construct a type parameter index map for translation.
    fn construct_index_map(&self, orig_def_id: DefId) -> HashMap<u32, DefId> {
        let mut index_map = HashMap::new();
        let orig_generics = self.tcx.generics_of(orig_def_id);

        for type_ in &orig_generics.types {
            index_map.insert(type_.index, type_.def_id);
        }

        if let Some(did) = orig_generics.parent {
            let parent_generics = self.tcx.generics_of(did);

            for type_ in &parent_generics.types {
                index_map.insert(type_.index, type_.def_id);
            }
        }

        index_map
    }

    /// Check whether a `DefId` needs translation.
    fn needs_translation(&self, def_id: DefId) -> bool {
        (self.needs_translation)(self.id_mapping, def_id)
    }

    /// Translate a `DefId`.
    fn translate_orig(&self, def_id: DefId) -> DefId {
        (self.translate_orig)(self.id_mapping, def_id)
    }

    /// Translate a `DefId` of a trait item.
    fn translate_orig_trait(&self, item_def_id: DefId, trait_def_id: DefId) -> DefId {
        (self.translate_orig_trait)(self.id_mapping, item_def_id, trait_def_id)
    }

    /// Fold a structure, translating all `DefId`s reachable by the folder.
    fn translate<T: TypeFoldable<'tcx>>(&self, index_map: &HashMap<u32, DefId>, orig: &T) -> T {
        use rustc::ty::{AdtDef, Binder, ExistentialProjection, ExistentialTraitRef};
        use rustc::ty::ExistentialPredicate::*;
        use rustc::ty::TypeVariants::*;

        orig.fold_with(&mut BottomUpFolder { tcx: self.tcx, fldop: |ty| {
            match ty.sty {
                TyAdt(&AdtDef { ref did, .. }, substs) if self.needs_translation(*did) => {
                    let target_def_id = self.id_mapping.get_new_id(*did);
                    let target_adt = self.tcx.adt_def(target_def_id);
                    self.tcx.mk_adt(target_adt, substs)
                },
                TyRef(region, type_and_mut) => {
                    self.tcx.mk_ref(self.translate_region(region), type_and_mut)
                },
                TyFnDef(did, substs) => {
                    self.tcx.mk_fn_def(self.translate_orig(did), substs)
                },
                TyDynamic(preds, region) => {
                    let target_preds = self.tcx.mk_existential_predicates(preds.iter().map(|p| {
                        match *p.skip_binder() {
                            Trait(ExistentialTraitRef { def_id: did, substs }) => {
                                let target_def_id = self.translate_orig(did);

                                Trait(ExistentialTraitRef {
                                    def_id: target_def_id,
                                    substs: substs
                                })
                            },
                            Projection(ExistentialProjection { item_def_id, substs, ty }) => {
                                let target_def_id = self.translate_orig(item_def_id);

                                Projection(ExistentialProjection {
                                    item_def_id: target_def_id,
                                    substs: substs,
                                    ty: ty,
                                })
                            },
                            AutoTrait(did) => {
                                AutoTrait(self.translate_orig(did))
                            },
                        }
                    }));

                    self.tcx.mk_dynamic(Binder(target_preds), region)
                },
                TyProjection(proj) => {
                    let trait_def_id = self.tcx.associated_item(proj.item_def_id).container.id();
                    let target_def_id =
                        self.translate_orig_trait(proj.item_def_id, trait_def_id);

                    self.tcx.mk_projection(target_def_id, proj.substs)
                },
                TyAnon(did, substs) => {
                    self.tcx.mk_anon(self.translate_orig(did), substs)
                },
                TyParam(param) => {
                    if param.idx != 0 { // `Self` is special
                        let orig_def_id = index_map[&param.idx];
                        if self.needs_translation(orig_def_id) {
                            let target_def_id = self.translate_orig(orig_def_id);
                            let type_param = self.id_mapping.get_type_param(&target_def_id);
                            self.tcx.mk_param_from_def(&type_param)
                        } else {
                            self.tcx.mk_ty(TyParam(param))
                        }
                    } else {
                        self.tcx.mk_ty(TyParam(param))
                    }
                },
                _ => ty,
            }
        }})
    }

    /// Translate a region.
    fn translate_region(&self, region: Region<'tcx>) -> Region<'tcx> {
        use rustc::ty::{EarlyBoundRegion, FreeRegion};
        use rustc::ty::BoundRegion::BrNamed;
        use rustc::ty::RegionKind::*;

        self.tcx.mk_region(match *region {
            ReEarlyBound(early) => {
                let target_early = EarlyBoundRegion {
                    def_id: self.translate_orig(early.def_id),
                    index: early.index,
                    name: early.name,
                };

                ReEarlyBound(target_early)
            },
            ReFree(FreeRegion { scope, bound_region }) => {
                ReFree(FreeRegion {
                    scope: self.translate_orig(scope),
                    bound_region: match bound_region {
                        BrNamed(def_id, name) => BrNamed(self.translate_orig(def_id), name),
                        reg => reg,
                    },
                })
            },
            reg => reg,
        })
    }

    /// Translate an item's type.
    pub fn translate_item_type(&self, orig_def_id: DefId, orig: Ty<'tcx>) -> Ty<'tcx> {
        self.translate(&self.construct_index_map(orig_def_id), &orig)
    }

    /// Translate a predicate.
    fn translate_predicate(&self, index_map: &HashMap<u32, DefId>, predicate: Predicate<'tcx>)
        -> Predicate<'tcx>
    {
        use rustc::ty::{EquatePredicate, OutlivesPredicate, ProjectionPredicate, ProjectionTy,
                        SubtypePredicate, TraitPredicate, TraitRef};

        match predicate {
            Predicate::Trait(trait_predicate) => {
                Predicate::Trait(trait_predicate.map_bound(|t_pred| {
                    TraitPredicate {
                        trait_ref: TraitRef {
                            def_id: self.translate_orig(t_pred.trait_ref.def_id),
                            substs: self.translate(index_map, &t_pred.trait_ref.substs),
                        }
                    }
                }))
            },
            Predicate::Equate(equate_predicate) => {
                Predicate::Equate(equate_predicate.map_bound(|e_pred| {
                    let l = self.translate(index_map, &e_pred.0);
                    let r = self.translate(index_map, &e_pred.1);
                    EquatePredicate(l, r)
                }))
            },
            Predicate::RegionOutlives(region_outlives_predicate) => {
                Predicate::RegionOutlives(region_outlives_predicate.map_bound(|r_pred| {
                    let l = self.translate_region(r_pred.0);
                    let r = self.translate_region(r_pred.1);
                    OutlivesPredicate(l, r)
                }))
            },
            Predicate::TypeOutlives(type_outlives_predicate) => {
                Predicate::TypeOutlives(type_outlives_predicate.map_bound(|r_pred| {
                    let l = self.translate(index_map, &r_pred.0);
                    let r = self.translate_region(r_pred.1);
                    OutlivesPredicate(l, r)
                }))
            },
            Predicate::Projection(projection_predicate) => {
                Predicate::Projection(projection_predicate.map_bound(|p_pred| {
                    ProjectionPredicate {
                        projection_ty: ProjectionTy {
                            substs: self.translate(index_map, &p_pred.projection_ty.substs),
                            item_def_id: self.translate_orig(p_pred.projection_ty.item_def_id),
                        },
                        ty: self.translate(index_map, &p_pred.ty),
                    }
                }))
            },
            Predicate::WellFormed(ty) =>
                Predicate::WellFormed(self.translate(index_map, &ty)),
            Predicate::ObjectSafe(did) => Predicate::ObjectSafe(self.translate_orig(did)),
            Predicate::ClosureKind(did, kind) =>
                Predicate::ClosureKind(self.translate_orig(did), kind),
            Predicate::Subtype(subtype_predicate) => {
                Predicate::Subtype(subtype_predicate.map_bound(|s_pred| {
                    let l = self.translate(index_map, &s_pred.a);
                    let r = self.translate(index_map, &s_pred.b);
                    SubtypePredicate {
                        a_is_expected: s_pred.a_is_expected,
                        a: l,
                        b: r,
                    }
                }))
            },
        }
    }

    /// Translate a vector of predicates.
    pub fn translate_predicates(&self, orig_def_id: DefId, orig_preds: Vec<Predicate<'tcx>>)
        -> Vec<Predicate<'tcx>>
    {
        let index_map = self.construct_index_map(orig_def_id);
        orig_preds.iter().map(|p| self.translate_predicate(&index_map, *p)).collect()
    }

    /// Translate a `ParamEnv`.
    pub fn translate_param_env(&self, orig_def_id: DefId, param_env: ParamEnv<'tcx>)
        -> ParamEnv<'tcx>
    {
        let index_map = self.construct_index_map(orig_def_id);
        let res = param_env
            .caller_bounds
            .iter()
            .map(|p| self.translate_predicate(&index_map, *p))
            .collect::<AccumulateVec<[_; 8]>>();

        ParamEnv {
            caller_bounds: self.tcx.intern_predicates(&res),
            reveal: param_env.reveal,
        }
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
