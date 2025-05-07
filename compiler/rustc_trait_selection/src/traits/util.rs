use std::collections::{BTreeMap, VecDeque};

use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::InferCtxt;
pub use rustc_infer::traits::util::*;
use rustc_middle::bug;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};
use rustc_span::Span;
use smallvec::{SmallVec, smallvec};
use tracing::debug;

/// Return the trait and projection predicates that come from eagerly expanding the
/// trait aliases in the list of clauses. For each trait predicate, record a stack
/// of spans that trace from the user-written trait alias bound. For projection predicates,
/// just record the span of the projection itself.
///
/// For trait aliases, we don't deduplicte the predicates, since we currently do not
/// consider duplicated traits as a single trait for the purposes of our "one trait principal"
/// restriction; however, for projections we do deduplicate them.
///
/// ```rust,ignore (fails)
/// trait Bar {}
/// trait Foo = Bar + Bar;
///
/// let dyn_incompatible: dyn Foo; // bad, two `Bar` principals.
/// ```
pub fn expand_trait_aliases<'tcx>(
    tcx: TyCtxt<'tcx>,
    clauses: impl IntoIterator<Item = (ty::Clause<'tcx>, Span)>,
) -> (
    Vec<(ty::PolyTraitPredicate<'tcx>, SmallVec<[Span; 1]>)>,
    Vec<(ty::PolyProjectionPredicate<'tcx>, Span)>,
) {
    let mut trait_preds = vec![];
    let mut projection_preds = vec![];
    let mut seen_projection_preds = FxHashSet::default();

    let mut queue: VecDeque<_> = clauses.into_iter().map(|(p, s)| (p, smallvec![s])).collect();

    while let Some((clause, spans)) = queue.pop_front() {
        match clause.kind().skip_binder() {
            ty::ClauseKind::Trait(trait_pred) => {
                if tcx.is_trait_alias(trait_pred.def_id()) {
                    queue.extend(
                        tcx.explicit_super_predicates_of(trait_pred.def_id())
                            .iter_identity_copied()
                            .map(|(super_clause, span)| {
                                let mut spans = spans.clone();
                                spans.push(span);
                                (
                                    super_clause.instantiate_supertrait(
                                        tcx,
                                        clause.kind().rebind(trait_pred.trait_ref),
                                    ),
                                    spans,
                                )
                            }),
                    );
                } else {
                    trait_preds.push((clause.kind().rebind(trait_pred), spans));
                }
            }
            ty::ClauseKind::Projection(projection_pred) => {
                let projection_pred = clause.kind().rebind(projection_pred);
                if !seen_projection_preds.insert(tcx.anonymize_bound_vars(projection_pred)) {
                    continue;
                }
                projection_preds.push((projection_pred, *spans.last().unwrap()));
            }
            ty::ClauseKind::RegionOutlives(..)
            | ty::ClauseKind::TypeOutlives(..)
            | ty::ClauseKind::ConstArgHasType(_, _)
            | ty::ClauseKind::WellFormed(_)
            | ty::ClauseKind::ConstEvaluatable(_)
            | ty::ClauseKind::HostEffect(..) => {}
        }
    }

    (trait_preds, projection_preds)
}

///////////////////////////////////////////////////////////////////////////
// Other
///////////////////////////////////////////////////////////////////////////

/// Casts a trait reference into a reference to one of its super
/// traits; returns `None` if `target_trait_def_id` is not a
/// supertrait.
pub fn upcast_choices<'tcx>(
    tcx: TyCtxt<'tcx>,
    source_trait_ref: ty::PolyTraitRef<'tcx>,
    target_trait_def_id: DefId,
) -> Vec<ty::PolyTraitRef<'tcx>> {
    if source_trait_ref.def_id() == target_trait_def_id {
        return vec![source_trait_ref]; // Shortcut the most common case.
    }

    supertraits(tcx, source_trait_ref).filter(|r| r.def_id() == target_trait_def_id).collect()
}

pub(crate) fn closure_trait_ref_and_return_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::PolyFnSig<'tcx>,
    tuple_arguments: TupleArgumentsFlag,
) -> ty::Binder<'tcx, (ty::TraitRef<'tcx>, Ty<'tcx>)> {
    assert!(!self_ty.has_escaping_bound_vars());
    let arguments_tuple = match tuple_arguments {
        TupleArgumentsFlag::No => sig.skip_binder().inputs()[0],
        TupleArgumentsFlag::Yes => Ty::new_tup(tcx, sig.skip_binder().inputs()),
    };
    let trait_ref = ty::TraitRef::new(tcx, fn_trait_def_id, [self_ty, arguments_tuple]);
    sig.map_bound(|sig| (trait_ref, sig.output()))
}

pub(crate) fn coroutine_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::GenSig<TyCtxt<'tcx>>,
) -> (ty::TraitRef<'tcx>, Ty<'tcx>, Ty<'tcx>) {
    assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef::new(tcx, fn_trait_def_id, [self_ty, sig.resume_ty]);
    (trait_ref, sig.yield_ty, sig.return_ty)
}

pub(crate) fn future_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_trait_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::GenSig<TyCtxt<'tcx>>,
) -> (ty::TraitRef<'tcx>, Ty<'tcx>) {
    assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef::new(tcx, fn_trait_def_id, [self_ty]);
    (trait_ref, sig.return_ty)
}

pub(crate) fn iterator_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    iterator_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::GenSig<TyCtxt<'tcx>>,
) -> (ty::TraitRef<'tcx>, Ty<'tcx>) {
    assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef::new(tcx, iterator_def_id, [self_ty]);
    (trait_ref, sig.yield_ty)
}

pub(crate) fn async_iterator_trait_ref_and_outputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    async_iterator_def_id: DefId,
    self_ty: Ty<'tcx>,
    sig: ty::GenSig<TyCtxt<'tcx>>,
) -> (ty::TraitRef<'tcx>, Ty<'tcx>) {
    assert!(!self_ty.has_escaping_bound_vars());
    let trait_ref = ty::TraitRef::new(tcx, async_iterator_def_id, [self_ty]);
    (trait_ref, sig.yield_ty)
}

pub fn impl_item_is_final(tcx: TyCtxt<'_>, assoc_item: &ty::AssocItem) -> bool {
    assoc_item.defaultness(tcx).is_final()
        && tcx.defaultness(assoc_item.container_id(tcx)).is_final()
}

pub(crate) enum TupleArgumentsFlag {
    Yes,
    No,
}

/// Executes `f` on `value` after replacing all escaping bound variables with placeholders
/// and then replaces these placeholders with the original bound variables in the result.
///
/// In most places, bound variables should be replaced right when entering a binder, making
/// this function unnecessary. However, normalization currently does not do that, so we have
/// to do this lazily.
///
/// You should not add any additional uses of this function, at least not without first
/// discussing it with t-types.
///
/// FIXME(@lcnr): We may even consider experimenting with eagerly replacing bound vars during
/// normalization as well, at which point this function will be unnecessary and can be removed.
pub fn with_replaced_escaping_bound_vars<
    'a,
    'tcx,
    T: TypeFoldable<TyCtxt<'tcx>>,
    R: TypeFoldable<TyCtxt<'tcx>>,
>(
    infcx: &'a InferCtxt<'tcx>,
    universe_indices: &'a mut Vec<Option<ty::UniverseIndex>>,
    value: T,
    f: impl FnOnce(T) -> R,
) -> R {
    if value.has_escaping_bound_vars() {
        let (value, mapped_regions, mapped_types, mapped_consts) =
            BoundVarReplacer::replace_bound_vars(infcx, universe_indices, value);
        let result = f(value);
        PlaceholderReplacer::replace_placeholders(
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            universe_indices,
            result,
        )
    } else {
        f(value)
    }
}

pub struct BoundVarReplacer<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    // These three maps track the bound variable that were replaced by placeholders. It might be
    // nice to remove these since we already have the `kind` in the placeholder; we really just need
    // the `var` (but we *could* bring that into scope if we were to track them as we pass them).
    mapped_regions: FxIndexMap<ty::PlaceholderRegion, ty::BoundRegion>,
    mapped_types: FxIndexMap<ty::PlaceholderType, ty::BoundTy>,
    mapped_consts: BTreeMap<ty::PlaceholderConst, ty::BoundVar>,
    // The current depth relative to *this* folding, *not* the entire normalization. In other words,
    // the depth of binders we've passed here.
    current_index: ty::DebruijnIndex,
    // The `UniverseIndex` of the binding levels above us. These are optional, since we are lazy:
    // we don't actually create a universe until we see a bound var we have to replace.
    universe_indices: &'a mut Vec<Option<ty::UniverseIndex>>,
}

impl<'a, 'tcx> BoundVarReplacer<'a, 'tcx> {
    /// Returns `Some` if we *were* able to replace bound vars. If there are any bound vars that
    /// use a binding level above `universe_indices.len()`, we fail.
    pub fn replace_bound_vars<T: TypeFoldable<TyCtxt<'tcx>>>(
        infcx: &'a InferCtxt<'tcx>,
        universe_indices: &'a mut Vec<Option<ty::UniverseIndex>>,
        value: T,
    ) -> (
        T,
        FxIndexMap<ty::PlaceholderRegion, ty::BoundRegion>,
        FxIndexMap<ty::PlaceholderType, ty::BoundTy>,
        BTreeMap<ty::PlaceholderConst, ty::BoundVar>,
    ) {
        let mapped_regions: FxIndexMap<ty::PlaceholderRegion, ty::BoundRegion> =
            FxIndexMap::default();
        let mapped_types: FxIndexMap<ty::PlaceholderType, ty::BoundTy> = FxIndexMap::default();
        let mapped_consts: BTreeMap<ty::PlaceholderConst, ty::BoundVar> = BTreeMap::new();

        let mut replacer = BoundVarReplacer {
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            current_index: ty::INNERMOST,
            universe_indices,
        };

        let value = value.fold_with(&mut replacer);

        (value, replacer.mapped_regions, replacer.mapped_types, replacer.mapped_consts)
    }

    fn universe_for(&mut self, debruijn: ty::DebruijnIndex) -> ty::UniverseIndex {
        let infcx = self.infcx;
        let index =
            self.universe_indices.len() + self.current_index.as_usize() - debruijn.as_usize() - 1;
        let universe = self.universe_indices[index].unwrap_or_else(|| {
            for i in self.universe_indices.iter_mut().take(index + 1) {
                *i = i.or_else(|| Some(infcx.create_next_universe()))
            }
            self.universe_indices[index].unwrap()
        });
        universe
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for BoundVarReplacer<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
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

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            ty::ReBound(debruijn, _)
                if debruijn.as_usize()
                    >= self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!(
                    "Bound vars {r:#?} outside of `self.universe_indices`: {:#?}",
                    self.universe_indices
                );
            }
            ty::ReBound(debruijn, br) if debruijn >= self.current_index => {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderRegion { universe, bound: br };
                self.mapped_regions.insert(p, br);
                ty::Region::new_placeholder(self.infcx.tcx, p)
            }
            _ => r,
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Bound(debruijn, _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!(
                    "Bound vars {t:#?} outside of `self.universe_indices`: {:#?}",
                    self.universe_indices
                );
            }
            ty::Bound(debruijn, bound_ty) if debruijn >= self.current_index => {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderType { universe, bound: bound_ty };
                self.mapped_types.insert(p, bound_ty);
                Ty::new_placeholder(self.infcx.tcx, p)
            }
            _ if t.has_vars_bound_at_or_above(self.current_index) => t.super_fold_with(self),
            _ => t,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Bound(debruijn, _)
                if debruijn.as_usize() + 1
                    > self.current_index.as_usize() + self.universe_indices.len() =>
            {
                bug!(
                    "Bound vars {ct:#?} outside of `self.universe_indices`: {:#?}",
                    self.universe_indices
                );
            }
            ty::ConstKind::Bound(debruijn, bound_const) if debruijn >= self.current_index => {
                let universe = self.universe_for(debruijn);
                let p = ty::PlaceholderConst { universe, bound: bound_const };
                self.mapped_consts.insert(p, bound_const);
                ty::Const::new_placeholder(self.infcx.tcx, p)
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.has_vars_bound_at_or_above(self.current_index) { p.super_fold_with(self) } else { p }
    }
}

/// The inverse of [`BoundVarReplacer`]: replaces placeholders with the bound vars from which they came.
pub struct PlaceholderReplacer<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    mapped_regions: FxIndexMap<ty::PlaceholderRegion, ty::BoundRegion>,
    mapped_types: FxIndexMap<ty::PlaceholderType, ty::BoundTy>,
    mapped_consts: BTreeMap<ty::PlaceholderConst, ty::BoundVar>,
    universe_indices: &'a [Option<ty::UniverseIndex>],
    current_index: ty::DebruijnIndex,
}

impl<'a, 'tcx> PlaceholderReplacer<'a, 'tcx> {
    pub fn replace_placeholders<T: TypeFoldable<TyCtxt<'tcx>>>(
        infcx: &'a InferCtxt<'tcx>,
        mapped_regions: FxIndexMap<ty::PlaceholderRegion, ty::BoundRegion>,
        mapped_types: FxIndexMap<ty::PlaceholderType, ty::BoundTy>,
        mapped_consts: BTreeMap<ty::PlaceholderConst, ty::BoundVar>,
        universe_indices: &'a [Option<ty::UniverseIndex>],
        value: T,
    ) -> T {
        let mut replacer = PlaceholderReplacer {
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            universe_indices,
            current_index: ty::INNERMOST,
        };
        value.fold_with(&mut replacer)
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for PlaceholderReplacer<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        if !t.has_placeholders() && !t.has_infer() {
            return t;
        }
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r0: ty::Region<'tcx>) -> ty::Region<'tcx> {
        let r1 = match r0.kind() {
            ty::ReVar(vid) => self
                .infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(self.infcx.tcx, vid),
            _ => r0,
        };

        let r2 = match r1.kind() {
            ty::RePlaceholder(p) => {
                let replace_var = self.mapped_regions.get(&p);
                match replace_var {
                    Some(replace_var) => {
                        let index = self
                            .universe_indices
                            .iter()
                            .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                            .unwrap_or_else(|| bug!("Unexpected placeholder universe."));
                        let db = ty::DebruijnIndex::from_usize(
                            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                        );
                        ty::Region::new_bound(self.cx(), db, *replace_var)
                    }
                    None => r1,
                }
            }
            _ => r1,
        };

        debug!(?r0, ?r1, ?r2, "fold_region");

        r2
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.infcx.shallow_resolve(ty);
        match *ty.kind() {
            ty::Placeholder(p) => {
                let replace_var = self.mapped_types.get(&p);
                match replace_var {
                    Some(replace_var) => {
                        let index = self
                            .universe_indices
                            .iter()
                            .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                            .unwrap_or_else(|| bug!("Unexpected placeholder universe."));
                        let db = ty::DebruijnIndex::from_usize(
                            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                        );
                        Ty::new_bound(self.infcx.tcx, db, *replace_var)
                    }
                    None => {
                        if ty.has_infer() {
                            ty.super_fold_with(self)
                        } else {
                            ty
                        }
                    }
                }
            }

            _ if ty.has_placeholders() || ty.has_infer() => ty.super_fold_with(self),
            _ => ty,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let ct = self.infcx.shallow_resolve_const(ct);
        if let ty::ConstKind::Placeholder(p) = ct.kind() {
            let replace_var = self.mapped_consts.get(&p);
            match replace_var {
                Some(replace_var) => {
                    let index = self
                        .universe_indices
                        .iter()
                        .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                        .unwrap_or_else(|| bug!("Unexpected placeholder universe."));
                    let db = ty::DebruijnIndex::from_usize(
                        self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                    );
                    ty::Const::new_bound(self.infcx.tcx, db, *replace_var)
                }
                None => {
                    if ct.has_infer() {
                        ct.super_fold_with(self)
                    } else {
                        ct
                    }
                }
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}

pub fn sizedness_fast_path<'tcx>(tcx: TyCtxt<'tcx>, predicate: ty::Predicate<'tcx>) -> bool {
    // Proving `Sized` very often on "obviously sized" types like `&T`, accounts for about 60%
    // percentage of the predicates we have to prove. No need to canonicalize and all that for
    // such cases.
    if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_ref)) =
        predicate.kind().skip_binder()
    {
        if tcx.is_lang_item(trait_ref.def_id(), LangItem::Sized)
            && trait_ref.self_ty().is_trivially_sized(tcx)
        {
            debug!("fast path -- trivial sizedness");
            return true;
        }
    }

    false
}
