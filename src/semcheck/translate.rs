//! The translation machinery used to lift items into the context of the other crate for
//! comparison and inference.

use rustc::hir::def_id::DefId;
use rustc::ty::{GenericParamDefKind, ParamEnv, Predicate, Region, TraitRef, Ty, TyCtxt};
use rustc::ty::fold::{BottomUpFolder, TypeFoldable, TypeFolder};
use rustc::ty::subst::Kind;
use rustc::infer::InferCtxt;
use rustc::ty::subst::Substs;

use semcheck::mapping::{IdMapping, InherentEntry};

use std::collections::HashMap;

/// The context in which `DefId` translation happens.
pub struct TranslationContext<'a, 'gcx: 'tcx + 'a, 'tcx: 'a> {
    /// The type context to use.
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    /// The id mapping to use.
    id_mapping: &'a IdMapping,
    /// Whether to translate type and region parameters.
    translate_params: bool,
    /// Elementary operation to decide whether to translate a `DefId`.
    needs_translation: fn(&IdMapping, DefId) -> bool,
    /// Elementary operation to translate a `DefId`.
    translate_orig: fn(&IdMapping, DefId) -> Option<DefId>,
}

impl<'a, 'gcx, 'tcx> TranslationContext<'a, 'gcx, 'tcx> {
    /// Construct a translation context translating to the new crate's `DefId`s.
    pub fn target_new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      id_mapping: &'a IdMapping,
                      translate_params: bool) -> TranslationContext<'a, 'gcx, 'tcx>
    {
        TranslationContext {
            tcx: tcx,
            id_mapping: id_mapping,
            translate_params: translate_params,
            needs_translation: IdMapping::in_old_crate,
            translate_orig: IdMapping::get_new_id,
        }
    }

    /// Construct a translation context translating to the old crate's `DefId`s.
    pub fn target_old(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                      id_mapping: &'a IdMapping,
                      translate_params: bool) -> TranslationContext<'a, 'gcx, 'tcx> {
        TranslationContext {
            tcx: tcx,
            id_mapping: id_mapping,
            translate_params: translate_params,
            needs_translation: IdMapping::in_new_crate,
            translate_orig: IdMapping::get_old_id,
        }
    }

    /// Construct a type parameter index map for translation.
    fn construct_index_map(&self, orig_def_id: DefId) -> HashMap<u32, DefId> {
        let mut index_map = HashMap::new();
        let orig_generics = self.tcx.generics_of(orig_def_id);

        for param in &orig_generics.params {
            match param.kind {
                GenericParamDefKind::Type { .. } => {
                    index_map.insert(param.index, param.def_id);
                },
                _ => (),
            };
        }

        if let Some(did) = orig_generics.parent {
            let parent_generics = self.tcx.generics_of(did);

            for param in &parent_generics.params {
                match param.kind {
                    GenericParamDefKind::Type { .. } => {
                        index_map.insert(param.index, param.def_id);
                    },
                    _ => (),
                };
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
        (self.translate_orig)(self.id_mapping, def_id).unwrap_or_else(|| {
            info!("not mapped: {:?}", def_id);
            def_id
        })
    }

    /// Translate the `DefId` and substs of an item.
    fn translate_orig_substs(&self,
                             index_map: &HashMap<u32, DefId>,
                             orig_def_id: DefId,
                             orig_substs: &Substs<'tcx>) -> Option<(DefId, &'tcx Substs<'tcx>)> {
        debug!("translating w/ substs: did: {:?}, substs: {:?}",
               orig_def_id, orig_substs);

        use rustc::ty::ReEarlyBound;
        use rustc::ty::subst::UnpackedKind;
        use std::cell::Cell;

        let target_def_id = (self.translate_orig)(self.id_mapping, orig_def_id);

        if let Some(target_def_id) = target_def_id {
            let success = Cell::new(true);

            let target_substs = Substs::for_item(self.tcx, target_def_id, |def, _| {
                match def.kind {
                    GenericParamDefKind::Lifetime => {
                        Kind::from(if !success.get() {
                            self.tcx.mk_region(ReEarlyBound(def.to_early_bound_region_data()))
                        } else if let Some(UnpackedKind::Lifetime(region)) = orig_substs
                            .get(def.index as usize)
                            .map(|k| k.unpack())
                        {
                            self.translate_region(region)
                        } else {
                            success.set(false);
                            self.tcx.mk_region(ReEarlyBound(def.to_early_bound_region_data()))
                        })
                    },
                    GenericParamDefKind::Type { .. } => {
                        if !success.get() {
                            self.tcx.mk_param_from_def(def)
                        } else if let Some(UnpackedKind::Type(type_)) = orig_substs
                            .get(def.index as usize)
                            .map(|k| k.unpack())
                        {
                            self.translate(index_map, &Kind::from(type_))
                        } else if self.id_mapping
                                      .is_non_mapped_defaulted_type_param(&def.def_id) {
                            Kind::from(self.tcx.type_of(def.def_id))
                        } else if self.tcx
                                      .generics_of(target_def_id).has_self && def.index == 0 {
                            self.tcx.mk_param_from_def(def)
                        } else {
                            success.set(false);
                            self.tcx.mk_param_from_def(def)
                        }
                    },
                }
            });

            if success.get() {
                return Some((target_def_id, target_substs));
            }
        }

        None
    }

    /// Fold a structure, translating all `DefId`s reachable by the folder.
    fn translate<T: TypeFoldable<'tcx>>(&self, index_map: &HashMap<u32, DefId>, orig: &T) -> T {
        use rustc::ty::{AdtDef, Binder, ExistentialProjection, ExistentialTraitRef};
        use rustc::ty::ExistentialPredicate::*;
        use rustc::ty::TypeAndMut;
        use rustc::ty::TypeVariants::*;

        orig.fold_with(&mut BottomUpFolder { tcx: self.tcx, fldop: |ty| {
            match ty.sty {
                TyAdt(&AdtDef { ref did, .. }, substs) if self.needs_translation(*did) => {
                    // we fold bottom-up, so the code above is invalid, as it assumes the
                    // substs (that have been folded already) are yet untranslated
                    if let Some(target_def_id) = (self.translate_orig)(self.id_mapping, *did) {
                        let target_adt = self.tcx.adt_def(target_def_id);
                        self.tcx.mk_adt(target_adt, substs)
                    } else {
                        ty
                    }
                },
                TyRef(region, ty, mutbl) => {
                    let ty_and_mut = TypeAndMut { ty, mutbl };
                    self.tcx.mk_ref(self.translate_region(region), ty_and_mut)
                },
                TyFnDef(did, substs) => {
                    // TODO: this might be buggy as *technically* the substs are
                    // already translated (see TyAdt for a possible fix)
                    if let Some((target_def_id, target_substs)) =
                        self.translate_orig_substs(index_map, did, substs)
                    {
                        self.tcx.mk_fn_def(target_def_id, target_substs)
                    } else {
                        ty
                    }
                },
                TyDynamic(preds, region) => {
                    // hacky error catching mechanism
                    use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
                    use std::cell::Cell;

                    let success = Cell::new(true);
                    let err_pred = AutoTrait(DefId::local(CRATE_DEF_INDEX));

                    let res: Vec<_> = preds.iter().map(|p| {
                        match *p.skip_binder() {
                            Trait(existential_trait_ref) => {
                                let trait_ref = Binder::bind(existential_trait_ref)
                                    .with_self_ty(self.tcx, self.tcx.types.err);
                                let did = trait_ref.skip_binder().def_id;
                                let substs = trait_ref.skip_binder().substs;

                                // TODO: here, the substs could also be already translated
                                if let Some((target_def_id, target_substs)) =
                                    self.translate_orig_substs(index_map, did, substs)
                                {
                                    let target_trait_ref = TraitRef {
                                        def_id: target_def_id,
                                        substs: target_substs,
                                    };
                                    Trait(ExistentialTraitRef::erase_self_ty(self.tcx,
                                                                             target_trait_ref))
                                } else  {
                                    success.set(false);
                                    err_pred
                                }
                            },
                            Projection(existential_projection) => {
                                let projection_pred = Binder::bind(existential_projection)
                                    .with_self_ty(self.tcx, self.tcx.types.err);
                                let item_def_id =
                                    projection_pred.skip_binder().projection_ty.item_def_id;
                                let substs = projection_pred.skip_binder().projection_ty.substs;

                                // TODO: here, the substs could also be already translated
                                if let Some((target_def_id, target_substs)) =
                                    self.translate_orig_substs(index_map, item_def_id, substs)
                                {
                                    Projection(ExistentialProjection {
                                        item_def_id: target_def_id,
                                        // TODO: should be it's own method in rustc
                                        substs: self.tcx.intern_substs(&target_substs[1..]),
                                        ty: ty,
                                    })
                                } else {
                                    success.set(false);
                                    err_pred
                                }
                            },
                            AutoTrait(did) => {
                                AutoTrait(self.translate_orig(did))
                            },
                        }
                    }).collect();

                    if success.get() {
                        let target_preds = self.tcx.mk_existential_predicates(res.iter());
                        self.tcx.mk_dynamic(Binder::bind(target_preds), region)
                    } else {
                        ty
                    }
                },
                TyProjection(proj) => {
                    if let Some((target_def_id, target_substs)) =
                        self.translate_orig_substs(index_map,
                                                   proj.item_def_id,
                                                   proj.substs) {
                        self.tcx.mk_projection(target_def_id, target_substs)
                    } else {
                        ty
                    }
                },
                TyAnon(did, substs) => {
                    if let Some((target_def_id, target_substs)) =
                        self.translate_orig_substs(index_map, did, substs)
                    {
                        self.tcx.mk_anon(target_def_id, target_substs)
                    } else {
                        ty
                    }
                },
                TyParam(param) => {
                    // FIXME: we should check `has_self` if this gets used again!
                    if param.idx != 0 && self.translate_params { // `Self` is special
                        let orig_def_id = index_map[&param.idx];
                        if self.needs_translation(orig_def_id) {
                            use rustc::ty::subst::UnpackedKind;

                            let target_def_id = self.translate_orig(orig_def_id);
                            debug!("translating type param: {:?}", param);
                            let type_param = self.id_mapping.get_type_param(&target_def_id);
                            debug!("translated type param: {:?}", type_param);
                            match self.tcx.mk_param_from_def(&type_param).unpack() {
                                UnpackedKind::Type(param_t) => param_t,
                                _ => unreachable!(),
                            }
                        } else {
                            ty
                        }
                    } else {
                        ty
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

        if !self.translate_params {
            return region;
        }

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

    /// Translate a predicate using a type parameter index map.
    fn translate_predicate(&self, index_map: &HashMap<u32, DefId>, predicate: Predicate<'tcx>)
        -> Option<Predicate<'tcx>>
    {
        use rustc::ty::{Binder, /*EquatePredicate,*/ OutlivesPredicate, ProjectionPredicate,
                        ProjectionTy, SubtypePredicate, TraitPredicate, TraitRef};

        Some(match predicate {
            Predicate::Trait(trait_predicate) => {
                Predicate::Trait(Binder::bind(if let Some((target_def_id, target_substs)) =
                    self.translate_orig_substs(index_map,
                                               trait_predicate.skip_binder().trait_ref.def_id,
                                               trait_predicate.skip_binder().trait_ref.substs) {
                    TraitPredicate {
                        trait_ref: TraitRef {
                            def_id: target_def_id,
                            substs: target_substs,
                        }
                    }
                } else {
                    return None;
                }))
            },
            /*Predicate::Equate(equate_predicate) => {
                Predicate::Equate(equate_predicate.map_bound(|e_pred| {
                    let l = self.translate(index_map, &e_pred.0);
                    let r = self.translate(index_map, &e_pred.1);
                    EquatePredicate(l, r)
                }))
            },*/
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
                Predicate::Projection(Binder::bind(if let Some((target_def_id, target_substs)) =
                    self.translate_orig_substs(index_map,
                                               projection_predicate
                                                   .skip_binder()
                                                   .projection_ty
                                                   .item_def_id,
                                               projection_predicate
                                                   .skip_binder()
                                                   .projection_ty
                                                   .substs) {
                    ProjectionPredicate {
                        projection_ty: ProjectionTy {
                            substs: target_substs,
                            item_def_id: target_def_id,
                        },
                        ty: self.translate(index_map, &projection_predicate.skip_binder().ty),
                    }
                } else {
                    return None;
                }))
            },
            Predicate::WellFormed(ty) =>
                Predicate::WellFormed(self.translate(index_map, &ty)),
            Predicate::ObjectSafe(did) => Predicate::ObjectSafe(self.translate_orig(did)),
            Predicate::ClosureKind(did, substs, kind) =>
                Predicate::ClosureKind(
                    self.translate_orig(did),
                    self.translate(index_map, &substs),
                    kind),
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
            Predicate::ConstEvaluatable(orig_did, orig_substs) => {
                if let Some((target_def_id, target_substs)) =
                        self.translate_orig_substs(index_map, orig_did, orig_substs) {
                    Predicate::ConstEvaluatable(target_def_id, target_substs)
                } else {
                    return None;
                }
            },
        })
    }

    /// Translate a slice of predicates in the context of an item.
    fn translate_predicates(&self, orig_def_id: DefId, orig_preds: &[Predicate<'tcx>])
        -> Option<Vec<Predicate<'tcx>>>
    {
        let index_map = self.construct_index_map(orig_def_id);
        let mut target_preds = Vec::with_capacity(orig_preds.len());

        for orig_pred in orig_preds {
            if let Some(target_pred) = self.translate_predicate(&index_map, *orig_pred) {
                target_preds.push(target_pred);
            } else {
                return None;
            }
        }

        Some(target_preds)
    }

    /// Translate a `ParamEnv` in the context of an item.
    pub fn translate_param_env(&self, orig_def_id: DefId, param_env: ParamEnv<'tcx>)
        -> Option<ParamEnv<'tcx>>
    {
        self.translate_predicates(orig_def_id, param_env.caller_bounds)
            .map(|target_preds| ParamEnv {
                caller_bounds: self.tcx.intern_predicates(&target_preds),
                reveal: param_env.reveal,
            })
    }

    /// Translate a `TraitRef` in the context of an item.
    pub fn translate_trait_ref(&self, orig_def_id: DefId, orig_trait_ref: &TraitRef<'tcx>)
        -> TraitRef<'tcx>
    {
        let index_map = self.construct_index_map(orig_def_id);
        TraitRef {
            def_id: self.translate_orig(orig_trait_ref.def_id),
            substs: self.translate(&index_map, &orig_trait_ref.substs),
        }
    }

    /// Translate an `InherentEntry`.
    pub fn translate_inherent_entry(&self, orig_entry: &InherentEntry) -> Option<InherentEntry> {
        (self.translate_orig)(self.id_mapping, orig_entry.parent_def_id)
            .map(|parent_def_id| InherentEntry {
                parent_def_id: parent_def_id,
                kind: orig_entry.kind,
                name: orig_entry.name,
            })
    }

    /// Check whether a given `DefId` can be translated.
    pub fn can_translate(&self, def_id: DefId) -> bool {
        (self.translate_orig)(self.id_mapping, def_id).is_some()
    }
}

/// A type folder that removes inference artifacts.
///
/// Used to lift type errors and predicates to wrap them in an error type.
#[derive(Clone)]
pub struct InferenceCleanupFolder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    /// The inference context used.
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> InferenceCleanupFolder<'a, 'gcx, 'tcx> {
    /// Construct a new folder.
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        InferenceCleanupFolder {
            infcx: infcx,
        }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for InferenceCleanupFolder<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> { self.infcx.tcx }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        use rustc::ty::TypeAndMut;
        use rustc::ty::TypeVariants::{TyError, TyInfer, TyRef};

        let t1 = ty.super_fold_with(self);
        match t1.sty {
            TyRef(region, ty, mutbl) if region.needs_infer() => {
                let ty_and_mut = TypeAndMut { ty, mutbl };
                self.infcx.tcx.mk_ref(self.infcx.tcx.types.re_erased, ty_and_mut)
            },
            TyInfer(_) => self.infcx.tcx.mk_ty(TyError),
            _ => t1,
        }
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        let r1 = r.super_fold_with(self);
        if r1.needs_infer() {
            self.infcx.tcx.types.re_erased
        } else {
            r1
        }
    }
}
