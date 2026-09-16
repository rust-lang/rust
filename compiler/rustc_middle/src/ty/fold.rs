use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;
use rustc_type_ir::PredicateProxy;
use rustc_type_ir::data_structures::DelayedMap;

use crate::traits::solve::TraitEvidence;
use crate::ty::{
    self, Binder, BoundTy, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
};

///////////////////////////////////////////////////////////////////////////
// Some sample folders

pub struct BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(ty::Const<'tcx>) -> ty::Const<'tcx>,
{
    pub tcx: TyCtxt<'tcx>,
    pub ty_op: F,
    pub lt_op: G,
    pub ct_op: H,
}

impl<'tcx, F, G, H> TypeFolder<TyCtxt<'tcx>> for BottomUpFolder<'tcx, F, G, H>
where
    F: FnMut(Ty<'tcx>) -> Ty<'tcx>,
    G: FnMut(ty::Region<'tcx>) -> ty::Region<'tcx>,
    H: FnMut(ty::Const<'tcx>) -> ty::Const<'tcx>,
{
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let t = ty.super_fold_with(self);
        (self.ty_op)(t)
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        // This one is a little different, because `super_fold_with` is not
        // implemented on non-recursive `Region`.
        (self.lt_op)(r)
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let ct = ct.super_fold_with(self);
        (self.ct_op)(ct)
    }
}

///////////////////////////////////////////////////////////////////////////
// Bound vars replacer

/// A delegate used when instantiating bound vars.
///
/// Any implementation must make sure that each bound variable always
/// gets mapped to the same result. `BoundVarReplacer` caches by using
/// a `DelayedMap` which does not cache the first few types it encounters.
pub trait BoundVarReplacerDelegate<'tcx> {
    fn replace_region(&mut self, br: ty::BoundRegion<'tcx>) -> ty::Region<'tcx>;
    fn replace_ty(&mut self, bt: ty::BoundTy<'tcx>) -> Ty<'tcx>;
    fn replace_const(&mut self, bc: ty::BoundConst<'tcx>) -> ty::Const<'tcx>;

    /// Replaces evidence bound by the binder currently being instantiated.
    ///
    /// `trait_ref` has already had its ordinary bound variables replaced and
    /// is expressed at the replacer's current binder depth. The returned
    /// evidence must be expressed at that same depth and prove exactly this
    /// trait ref. Returning `Bound(INNERMOST, ..)` is also accepted: the
    /// replacer adjusts that bound-evidence index to its current depth.
    fn replace_evidence(
        &mut self,
        trait_ref: ty::TraitRef<'tcx>,
        bound: ty::BoundEvidence<'tcx>,
    ) -> TraitEvidence<'tcx> {
        panic!("bound evidence requires an explicit replacement: {bound:?} proving {trait_ref:?}")
    }
}

/// A simple delegate taking 3 mutable functions. The used functions must
/// always return the same result for each bound variable, no matter how
/// frequently they are called.
pub struct FnMutDelegate<'a, 'tcx> {
    pub regions: &'a mut (dyn FnMut(ty::BoundRegion<'tcx>) -> ty::Region<'tcx> + 'a),
    pub types: &'a mut (dyn FnMut(ty::BoundTy<'tcx>) -> Ty<'tcx> + 'a),
    pub consts: &'a mut (dyn FnMut(ty::BoundConst<'tcx>) -> ty::Const<'tcx> + 'a),
}

impl<'a, 'tcx> BoundVarReplacerDelegate<'tcx> for FnMutDelegate<'a, 'tcx> {
    fn replace_region(&mut self, br: ty::BoundRegion<'tcx>) -> ty::Region<'tcx> {
        (self.regions)(br)
    }
    fn replace_ty(&mut self, bt: ty::BoundTy<'tcx>) -> Ty<'tcx> {
        (self.types)(bt)
    }
    fn replace_const(&mut self, bc: ty::BoundConst<'tcx>) -> ty::Const<'tcx> {
        (self.consts)(bc)
    }
}

/// Replaces the escaping bound vars (late bound regions or bound types) in a type.
struct BoundVarReplacer<'tcx, D> {
    tcx: TyCtxt<'tcx>,

    /// As with `RegionFolder`, represents the index of a binder *just outside*
    /// the ones we have visited.
    current_index: ty::DebruijnIndex,

    delegate: D,

    /// This cache only tracks the `DebruijnIndex` and assumes that it does not matter
    /// for the delegate how often its methods get used.
    cache: DelayedMap<(ty::DebruijnIndex, Ty<'tcx>), Ty<'tcx>>,
}

impl<'tcx, D: BoundVarReplacerDelegate<'tcx>> BoundVarReplacer<'tcx, D> {
    fn new(tcx: TyCtxt<'tcx>, delegate: D) -> Self {
        BoundVarReplacer { tcx, current_index: ty::INNERMOST, delegate, cache: Default::default() }
    }
}

impl<'tcx, D> TypeFolder<TyCtxt<'tcx>> for BoundVarReplacer<'tcx, D>
where
    D: BoundVarReplacerDelegate<'tcx>,
{
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn == self.current_index =>
            {
                let ty = self.delegate.replace_ty(bound_ty);
                debug_assert!(!ty.has_vars_bound_above(ty::INNERMOST));
                ty::shift_vars(self.tcx, ty, self.current_index.as_u32())
            }
            _ => {
                if !t.has_vars_bound_at_or_above(self.current_index) {
                    t
                } else if let Some(&t) = self.cache.get(&(self.current_index, t)) {
                    t
                } else {
                    let res = t.super_fold_with(self);
                    assert!(self.cache.insert((self.current_index, t), res));
                    res
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            ty::ReBound(ty::BoundVarIndexKind::Bound(debruijn), br)
                if debruijn == self.current_index =>
            {
                let region = self.delegate.replace_region(br);
                if let ty::ReBound(ty::BoundVarIndexKind::Bound(debruijn1), br) = region.kind() {
                    // If the callback returns a bound region,
                    // that region should always use the INNERMOST
                    // debruijn index. Then we adjust it to the
                    // correct depth.
                    assert_eq!(debruijn1, ty::INNERMOST);
                    ty::Region::new_bound(self.tcx, debruijn, br)
                } else {
                    region
                }
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_const)
                if debruijn == self.current_index =>
            {
                let ct = self.delegate.replace_const(bound_const);
                debug_assert!(!ct.has_vars_bound_above(ty::INNERMOST));
                ty::shift_vars(self.tcx, ct, self.current_index.as_u32())
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_trait_evidence(&mut self, evidence: TraitEvidence<'tcx>) -> TraitEvidence<'tcx> {
        match &evidence.kind {
            ty::solve::TraitEvidenceKind::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound)
                if *debruijn == self.current_index =>
            {
                let trait_ref = evidence.trait_ref.fold_with(self);
                let replacement = self.delegate.replace_evidence(trait_ref, *bound);
                let replacement = match &replacement.kind {
                    ty::solve::TraitEvidenceKind::Bound(
                        ty::BoundVarIndexKind::Bound(debruijn),
                        replacement_bound,
                    ) if *debruijn == ty::INNERMOST && self.current_index != ty::INNERMOST => {
                        self.tcx.mk_trait_evidence_kind(
                            replacement.trait_ref,
                            ty::solve::TraitEvidenceKind::Bound(
                                ty::BoundVarIndexKind::Bound(self.current_index),
                                *replacement_bound,
                            ),
                        )
                    }
                    _ => replacement,
                };
                replacement.assert_well_formed();
                assert_eq!(
                    replacement.trait_ref, trait_ref,
                    "bound evidence replacement proves a different predicate"
                );
                replacement
            }
            _ if evidence.has_vars_bound_at_or_above(self.current_index) => {
                self.tcx.mk_trait_evidence_data((*evidence).clone().fold_with(self))
            }
            _ => evidence,
        }
    }

    fn fold_predicate<P: PredicateProxy<TyCtxt<'tcx>>>(&mut self, p: P) -> P {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: ty::Clauses<'tcx>) -> ty::Clauses<'tcx> {
        if c.has_vars_bound_at_or_above(self.current_index) { c.super_fold_with(self) } else { c }
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// Replaces all regions bound by the given `Binder` with the
    /// results returned by the closure; the closure is expected to
    /// return a free region (relative to this binder), and hence the
    /// binder is removed in the return type. The closure is invoked
    /// once for each unique `BoundRegionKind`; multiple references to the
    /// same `BoundRegionKind` will reuse the previous result. A map is
    /// returned at the end with each bound region and the free region
    /// that replaced it.
    ///
    /// # Panics
    ///
    /// This method only replaces late bound regions. Any types or
    /// constants bound by `value` will cause an ICE.
    pub fn instantiate_bound_regions<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut fld_r: F,
    ) -> (T, FxIndexMap<ty::BoundRegion<'tcx>, ty::Region<'tcx>>)
    where
        F: FnMut(ty::BoundRegion<'tcx>) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let mut region_map = FxIndexMap::default();
        let real_fld_r =
            |br: ty::BoundRegion<'tcx>| *region_map.entry(br).or_insert_with(|| fld_r(br));
        let value = self.instantiate_bound_regions_uncached(value, real_fld_r);
        (value, region_map)
    }

    /// Replaces every ordinary region entry in `value` while retaining the
    /// clauses owned by its dependent telescope.
    ///
    /// Unlike [`TyCtxt::instantiate_bound_regions`], this operation retains
    /// evidence clauses in a dependent telescope. It only substitutes the
    /// ordinary region prefix and returns the instantiated clauses so the caller
    /// can add them as assumptions or obligations.
    ///
    /// With telescope clauses, every ordinary declaration must be a region.
    /// Type and const declarations are rejected, including unused declarations.
    /// Without telescope clauses, this delegates to `instantiate_bound_regions`,
    /// which only visits variables used in the value.
    pub fn instantiate_bound_regions_with_telescope_clauses<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut fld_r: F,
    ) -> (
        T,
        FxIndexMap<ty::BoundRegion<'tcx>, ty::Region<'tcx>>,
        Vec<ty::InstantiatedTelescopeClause<TyCtxt<'tcx>>>,
    )
    where
        F: FnMut(ty::BoundRegion<'tcx>) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        if !value.has_telescope_clauses() {
            let (value, region_map) = self.instantiate_bound_regions(value, fld_r);
            return (value, region_map, Vec::new());
        }

        let mut region_map = FxIndexMap::default();
        let mut args = Vec::with_capacity(value.ordinary_bound_var_count());

        for (index, bound_var) in value.ordinary_bound_vars().enumerate() {
            let var = ty::BoundVar::from_usize(index);
            let arg = match bound_var {
                ty::BoundVariableKind::Region(kind) => {
                    let bound_region = ty::BoundRegion { var, kind };
                    let region =
                        *region_map.entry(bound_region).or_insert_with(|| fld_r(bound_region));
                    region.into()
                }
                ty::BoundVariableKind::Ty(kind) => {
                    bug!("unexpected bound ty in region-only binder: {kind:?}")
                }
                ty::BoundVariableKind::Const(kind) => {
                    bug!("unexpected bound const in region-only binder: {kind:?}")
                }
                ty::BoundVariableKind::Evidence(_) => {
                    unreachable!("evidence entries are not ordinary binder variables")
                }
            };
            args.push(arg);
        }

        let args = self.mk_args(&args);
        let (value, clauses) =
            value.instantiate_with_args_and_binder_assumption_evidence(self, args);
        (value, region_map, clauses)
    }

    pub fn instantiate_bound_regions_uncached<T, F>(
        self,
        value: Binder<'tcx, T>,
        mut replace_regions: F,
    ) -> T
    where
        F: FnMut(ty::BoundRegion<'tcx>) -> ty::Region<'tcx>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        assert!(
            !value.has_telescope_clauses(),
            "region-only binder instantiation cannot discharge telescope clauses: {value:?}"
        );
        let value = value.skip_binder();
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let delegate = FnMutDelegate {
                regions: &mut replace_regions,
                types: &mut |b| bug!("unexpected bound ty in binder: {b:?}"),
                consts: &mut |b| bug!("unexpected bound ct in binder: {b:?}"),
            };
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    /// Replaces all escaping bound vars. The `fld_r` closure replaces escaping
    /// bound regions; the `fld_t` closure replaces escaping bound types and the `fld_c`
    /// closure replaces escaping bound consts.
    pub fn replace_escaping_bound_vars_uncached<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        value: T,
        delegate: impl BoundVarReplacerDelegate<'tcx>,
    ) -> T {
        if !value.has_escaping_bound_vars() {
            value
        } else {
            let mut replacer = BoundVarReplacer::new(self, delegate);
            value.fold_with(&mut replacer)
        }
    }

    /// Replaces all types or regions bound by the given `Binder`. The `fld_r`
    /// closure replaces bound regions, the `fld_t` closure replaces bound
    /// types, and `fld_c` replaces bound constants.
    pub fn replace_bound_vars_uncached<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        value: Binder<'tcx, T>,
        delegate: impl BoundVarReplacerDelegate<'tcx>,
    ) -> T {
        assert!(
            !value.has_telescope_clauses(),
            "replacing bound variables without their telescope clauses: {value:?}"
        );
        self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate)
    }

    /// Replaces any late-bound regions bound in `value` with
    /// free variants attached to `all_outlive_scope`.
    pub fn liberate_late_bound_regions<T>(
        self,
        all_outlive_scope: DefId,
        value: ty::Binder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.instantiate_bound_regions_uncached(value, |br| {
            let kind = ty::LateParamRegionKind::from_bound(br.var, br.kind);
            ty::Region::new_late_param(self, all_outlive_scope, kind)
        })
    }

    /// Liberates the ordinary late-bound regions while retaining every
    /// clause in the binder's dependent telescope.
    ///
    /// Consumers which can encounter evidence entries must use this instead
    /// of `liberate_late_bound_regions`, then explicitly decide whether the
    /// returned clauses are assumptions or obligations at that boundary.
    pub fn liberate_late_bound_regions_with_telescope_clauses<T>(
        self,
        all_outlive_scope: DefId,
        value: ty::Binder<'tcx, T>,
    ) -> (T, Vec<ty::InstantiatedTelescopeClause<TyCtxt<'tcx>>>)
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let (value, _, clauses) =
            self.instantiate_bound_regions_with_telescope_clauses(value, |br| {
                let kind = ty::LateParamRegionKind::from_bound(br.var, br.kind);
                ty::Region::new_late_param(self, all_outlive_scope, kind)
            });
        (value, clauses)
    }

    pub fn shift_bound_var_indices<T>(self, bound_vars: usize, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        struct ShiftBoundVarIndices<'tcx> {
            tcx: TyCtxt<'tcx>,
            amount: usize,
        }

        impl<'tcx> BoundVarReplacerDelegate<'tcx> for ShiftBoundVarIndices<'tcx> {
            fn replace_region(&mut self, r: ty::BoundRegion<'tcx>) -> ty::Region<'tcx> {
                ty::Region::new_bound(
                    self.tcx,
                    ty::INNERMOST,
                    ty::BoundRegion { var: r.var + self.amount, kind: r.kind },
                )
            }

            fn replace_ty(&mut self, t: ty::BoundTy<'tcx>) -> Ty<'tcx> {
                Ty::new_bound(
                    self.tcx,
                    ty::INNERMOST,
                    ty::BoundTy { var: t.var + self.amount, kind: t.kind },
                )
            }

            fn replace_const(&mut self, c: ty::BoundConst<'tcx>) -> ty::Const<'tcx> {
                ty::Const::new_bound(
                    self.tcx,
                    ty::INNERMOST,
                    ty::BoundConst::new(c.var + self.amount),
                )
            }

            fn replace_evidence(
                &mut self,
                trait_ref: ty::TraitRef<'tcx>,
                evidence: ty::BoundEvidence<'tcx>,
            ) -> TraitEvidence<'tcx> {
                self.tcx.mk_trait_evidence_kind(
                    trait_ref,
                    ty::solve::TraitEvidenceKind::Bound(
                        ty::BoundVarIndexKind::Bound(ty::INNERMOST),
                        ty::BoundEvidence::new(evidence.var + self.amount),
                    ),
                )
            }
        }

        self.replace_escaping_bound_vars_uncached(
            value,
            ShiftBoundVarIndices { tcx: self, amount: bound_vars },
        )
    }

    /// Replaces any late-bound regions bound in `value` with `'erased`. Useful in codegen but also
    /// method lookup and a few other places where precise region relationships are not required.
    pub fn instantiate_bound_regions_with_erased<T>(self, value: Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.instantiate_bound_regions(value, |_| self.lifetimes.re_erased).0
    }

    /// Anonymize all bound variables in `value`, this is mostly used to improve caching.
    pub fn anonymize_bound_vars<T>(self, value: Binder<'tcx, T>) -> Binder<'tcx, T>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        struct Anonymize<'a, 'tcx> {
            tcx: TyCtxt<'tcx>,
            map: &'a mut FxIndexMap<ty::BoundVar, ty::BoundVariableKind<'tcx>>,
        }
        impl<'tcx> BoundVarReplacerDelegate<'tcx> for Anonymize<'_, 'tcx> {
            fn replace_region(&mut self, br: ty::BoundRegion<'tcx>) -> ty::Region<'tcx> {
                let entry = self.map.entry(br.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon))
                    .expect_region();
                let br = ty::BoundRegion { var, kind };
                ty::Region::new_bound(self.tcx, ty::INNERMOST, br)
            }
            fn replace_ty(&mut self, bt: ty::BoundTy<'tcx>) -> Ty<'tcx> {
                let entry = self.map.entry(bt.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let kind = entry
                    .or_insert_with(|| ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon))
                    .expect_ty();
                Ty::new_bound(self.tcx, ty::INNERMOST, BoundTy { var, kind })
            }
            fn replace_const(&mut self, bc: ty::BoundConst<'tcx>) -> ty::Const<'tcx> {
                let entry = self.map.entry(bc.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let () = entry.or_insert_with(|| ty::BoundVariableKind::Const(None)).expect_const();
                ty::Const::new_bound(self.tcx, ty::INNERMOST, ty::BoundConst::new(var))
            }

            fn replace_evidence(
                &mut self,
                trait_ref: ty::TraitRef<'tcx>,
                bound: ty::BoundEvidence<'tcx>,
            ) -> TraitEvidence<'tcx> {
                let entry = self.map.entry(bound.var);
                let index = entry.index();
                let var = ty::BoundVar::from_usize(index);
                let clause = ty::ClauseKind::Trait(ty::TraitClause {
                    trait_ref,
                    polarity: ty::ClausePolarity::Positive,
                });
                let _ = entry
                    .or_insert_with(|| {
                        ty::BoundVariableKind::Evidence(ty::EvidenceVariable::principal(clause))
                    })
                    .expect_evidence();
                self.tcx.mk_trait_evidence_kind(
                    trait_ref,
                    ty::solve::TraitEvidenceKind::Bound(
                        ty::BoundVarIndexKind::Bound(ty::INNERMOST),
                        ty::BoundEvidence::new(var),
                    ),
                )
            }
        }

        if value.has_telescope_clauses() {
            // A dependent telescope's metadata is semantic: typed const
            // declarations and evidence assumptions must survive
            // anonymization. Keep its ordinary prefix in stable source order,
            // pre-seed the replacement map, and fold the value and every entry
            // with one substitution. This also preserves the invariant that an
            // evidence suffix can only refer to preceding ordinary entries.
            let original_bound_vars = value.bound_vars();
            let mut map: FxIndexMap<_, _> = value
                .bound_vars()
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    let anonymized = match entry {
                        ty::BoundVariableKind::Ty(_) => {
                            ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)
                        }
                        ty::BoundVariableKind::Region(_) => {
                            ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon)
                        }
                        ty::BoundVariableKind::Const(_) => ty::BoundVariableKind::Const(None),
                        // Pre-seeding the complete suffix preserves its stable
                        // telescope indices even when an evidence variable is
                        // unused by the bound value.
                        ty::BoundVariableKind::Evidence(evidence) => {
                            ty::BoundVariableKind::Evidence(evidence)
                        }
                    };
                    (ty::BoundVar::from_usize(index), anonymized)
                })
                .collect();
            let delegate = Anonymize { tcx: self, map: &mut map };
            let (inner, bound_vars) = self.replace_escaping_bound_vars_uncached(
                (value.skip_binder(), original_bound_vars.to_vec()),
                delegate,
            );
            let bound_vars =
                self.mk_bound_variable_kinds_from_iter(bound_vars.into_iter().map(|entry| {
                    match entry {
                        ty::BoundVariableKind::Ty(_) => {
                            ty::BoundVariableKind::Ty(ty::BoundTyKind::Anon)
                        }
                        ty::BoundVariableKind::Region(_) => {
                            ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon)
                        }
                        entry @ (ty::BoundVariableKind::Const(_)
                        | ty::BoundVariableKind::Evidence(_)) => entry,
                    }
                }));
            Binder::bind_with_vars(inner, bound_vars)
        } else {
            let mut map = Default::default();
            let delegate = Anonymize { tcx: self, map: &mut map };
            let inner = self.replace_escaping_bound_vars_uncached(value.skip_binder(), delegate);
            let bound_vars = self.mk_bound_variable_kinds_from_iter(map.into_values());
            Binder::bind_with_vars(inner, bound_vars)
        }
    }
}
