//! Things related to predicates.

use std::cmp::Ordering;

use macros::{TypeFoldable, TypeVisitable};
use rustc_type_ir::{
    self as ty, CollectAndApply, DebruijnIndex, EarlyBinder, FlagComputation, Flags,
    PredicatePolarity, TypeFlags, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable,
    TypeVisitable, Upcast, UpcastFrom, WithCachedTypeInfo,
    elaborate::Elaboratable,
    error::{ExpectedFound, TypeError},
    inherent::{IntoKind, SliceLike},
};
use smallvec::SmallVec;

use crate::next_solver::{GenericArg, InternedWrapperNoDebug, TraitIdWrapper};

use super::{Binder, BoundVarKinds, DbInterner, Region, Ty, interned_vec_db};

pub type BoundExistentialPredicate<'db> = Binder<'db, ExistentialPredicate<'db>>;

pub type TraitRef<'db> = ty::TraitRef<DbInterner<'db>>;
pub type AliasTerm<'db> = ty::AliasTerm<DbInterner<'db>>;
pub type ProjectionPredicate<'db> = ty::ProjectionPredicate<DbInterner<'db>>;
pub type ExistentialPredicate<'db> = ty::ExistentialPredicate<DbInterner<'db>>;
pub type ExistentialTraitRef<'db> = ty::ExistentialTraitRef<DbInterner<'db>>;
pub type ExistentialProjection<'db> = ty::ExistentialProjection<DbInterner<'db>>;
pub type TraitPredicate<'db> = ty::TraitPredicate<DbInterner<'db>>;
pub type ClauseKind<'db> = ty::ClauseKind<DbInterner<'db>>;
pub type PredicateKind<'db> = ty::PredicateKind<DbInterner<'db>>;
pub type NormalizesTo<'db> = ty::NormalizesTo<DbInterner<'db>>;
pub type CoercePredicate<'db> = ty::CoercePredicate<DbInterner<'db>>;
pub type SubtypePredicate<'db> = ty::SubtypePredicate<DbInterner<'db>>;
pub type OutlivesPredicate<'db, T> = ty::OutlivesPredicate<DbInterner<'db>, T>;
pub type RegionOutlivesPredicate<'db> = OutlivesPredicate<'db, Region<'db>>;
pub type TypeOutlivesPredicate<'db> = OutlivesPredicate<'db, Ty<'db>>;
pub type PolyTraitPredicate<'db> = Binder<'db, TraitPredicate<'db>>;
pub type PolyRegionOutlivesPredicate<'db> = Binder<'db, RegionOutlivesPredicate<'db>>;
pub type PolyTypeOutlivesPredicate<'db> = Binder<'db, TypeOutlivesPredicate<'db>>;
pub type PolySubtypePredicate<'db> = Binder<'db, SubtypePredicate<'db>>;
pub type PolyCoercePredicate<'db> = Binder<'db, CoercePredicate<'db>>;
pub type PolyProjectionPredicate<'db> = Binder<'db, ProjectionPredicate<'db>>;
pub type PolyTraitRef<'db> = Binder<'db, TraitRef<'db>>;
pub type PolyExistentialTraitRef<'db> = Binder<'db, ExistentialTraitRef<'db>>;
pub type PolyExistentialProjection<'db> = Binder<'db, ExistentialProjection<'db>>;
pub type ArgOutlivesPredicate<'db> = OutlivesPredicate<'db, GenericArg<'db>>;

/// Compares via an ordering that will not change if modules are reordered or other changes are
/// made to the tree. In particular, this ordering is preserved across incremental compilations.
fn stable_cmp_existential_predicate<'db>(
    a: &ExistentialPredicate<'db>,
    b: &ExistentialPredicate<'db>,
) -> Ordering {
    // FIXME: this is actual unstable - see impl in predicate.rs in `rustc_middle`
    match (a, b) {
        (ExistentialPredicate::Trait(_), ExistentialPredicate::Trait(_)) => Ordering::Equal,
        (ExistentialPredicate::Projection(_a), ExistentialPredicate::Projection(_b)) => {
            // Should sort by def path hash
            Ordering::Equal
        }
        (ExistentialPredicate::AutoTrait(_a), ExistentialPredicate::AutoTrait(_b)) => {
            // Should sort by def path hash
            Ordering::Equal
        }
        (ExistentialPredicate::Trait(_), _) => Ordering::Less,
        (ExistentialPredicate::Projection(_), ExistentialPredicate::Trait(_)) => Ordering::Greater,
        (ExistentialPredicate::Projection(_), _) => Ordering::Less,
        (ExistentialPredicate::AutoTrait(_), _) => Ordering::Greater,
    }
}
interned_vec_db!(BoundExistentialPredicates, BoundExistentialPredicate);

impl<'db> rustc_type_ir::inherent::BoundExistentialPredicates<DbInterner<'db>>
    for BoundExistentialPredicates<'db>
{
    fn principal_def_id(self) -> Option<TraitIdWrapper> {
        self.principal().map(|trait_ref| trait_ref.skip_binder().def_id)
    }

    fn principal(
        self,
    ) -> Option<
        rustc_type_ir::Binder<DbInterner<'db>, rustc_type_ir::ExistentialTraitRef<DbInterner<'db>>>,
    > {
        self.inner()[0]
            .map_bound(|this| match this {
                ExistentialPredicate::Trait(tr) => Some(tr),
                _ => None,
            })
            .transpose()
    }

    fn auto_traits(self) -> impl IntoIterator<Item = TraitIdWrapper> {
        self.iter().filter_map(|predicate| match predicate.skip_binder() {
            ExistentialPredicate::AutoTrait(did) => Some(did),
            _ => None,
        })
    }

    fn projection_bounds(
        self,
    ) -> impl IntoIterator<
        Item = rustc_type_ir::Binder<
            DbInterner<'db>,
            rustc_type_ir::ExistentialProjection<DbInterner<'db>>,
        >,
    > {
        self.iter().filter_map(|predicate| {
            predicate
                .map_bound(|pred| match pred {
                    ExistentialPredicate::Projection(projection) => Some(projection),
                    _ => None,
                })
                .transpose()
        })
    }
}

impl<'db> rustc_type_ir::relate::Relate<DbInterner<'db>> for BoundExistentialPredicates<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        let interner = relation.cx();

        // We need to perform this deduplication as we sometimes generate duplicate projections in `a`.
        let mut a_v: Vec<_> = a.into_iter().collect();
        let mut b_v: Vec<_> = b.into_iter().collect();
        // `skip_binder` here is okay because `stable_cmp` doesn't look at binders
        a_v.sort_by(|a, b| {
            stable_cmp_existential_predicate(a.as_ref().skip_binder(), b.as_ref().skip_binder())
        });
        a_v.dedup();
        b_v.sort_by(|a, b| {
            stable_cmp_existential_predicate(a.as_ref().skip_binder(), b.as_ref().skip_binder())
        });
        b_v.dedup();
        if a_v.len() != b_v.len() {
            return Err(TypeError::ExistentialMismatch(ExpectedFound::new(a, b)));
        }

        let v = std::iter::zip(a_v, b_v).map(
            |(ep_a, ep_b): (
                Binder<'_, ty::ExistentialPredicate<_>>,
                Binder<'_, ty::ExistentialPredicate<_>>,
            )| {
                match (ep_a.skip_binder(), ep_b.skip_binder()) {
                    (ty::ExistentialPredicate::Trait(a), ty::ExistentialPredicate::Trait(b)) => {
                        Ok(ep_a.rebind(ty::ExistentialPredicate::Trait(
                            relation.relate(ep_a.rebind(a), ep_b.rebind(b))?.skip_binder(),
                        )))
                    }
                    (
                        ty::ExistentialPredicate::Projection(a),
                        ty::ExistentialPredicate::Projection(b),
                    ) => Ok(ep_a.rebind(ty::ExistentialPredicate::Projection(
                        relation.relate(ep_a.rebind(a), ep_b.rebind(b))?.skip_binder(),
                    ))),
                    (
                        ty::ExistentialPredicate::AutoTrait(a),
                        ty::ExistentialPredicate::AutoTrait(b),
                    ) if a == b => Ok(ep_a.rebind(ty::ExistentialPredicate::AutoTrait(a))),
                    _ => Err(TypeError::ExistentialMismatch(ExpectedFound::new(a, b))),
                }
            },
        );

        CollectAndApply::collect_and_apply(v, |g| {
            BoundExistentialPredicates::new_from_iter(interner, g.iter().cloned())
        })
    }
}

#[salsa::interned(constructor = new_)]
pub struct Predicate<'db> {
    #[returns(ref)]
    kind_: InternedWrapperNoDebug<WithCachedTypeInfo<Binder<'db, PredicateKind<'db>>>>,
}

impl<'db> std::fmt::Debug for Predicate<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().internee.fmt(f)
    }
}

impl<'db> std::fmt::Debug
    for InternedWrapperNoDebug<WithCachedTypeInfo<Binder<'db, PredicateKind<'db>>>>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Binder<")?;
        match self.0.internee.skip_binder() {
            rustc_type_ir::PredicateKind::Clause(clause_kind) => {
                write!(f, "{clause_kind:?}")
            }
            rustc_type_ir::PredicateKind::DynCompatible(trait_def_id) => {
                write!(f, "the trait `{trait_def_id:?}` is dyn-compatible")
            }
            rustc_type_ir::PredicateKind::Subtype(subtype_predicate) => {
                write!(f, "{subtype_predicate:?}")
            }
            rustc_type_ir::PredicateKind::Coerce(coerce_predicate) => {
                write!(f, "{coerce_predicate:?}")
            }
            rustc_type_ir::PredicateKind::ConstEquate(c1, c2) => {
                write!(f, "the constant `{c1:?}` equals `{c2:?}`")
            }
            rustc_type_ir::PredicateKind::Ambiguous => write!(f, "ambiguous"),
            rustc_type_ir::PredicateKind::NormalizesTo(data) => write!(f, "{data:?}"),
            rustc_type_ir::PredicateKind::AliasRelate(t1, t2, dir) => {
                write!(f, "{t1:?} {dir:?} {t2:?}")
            }
        }?;
        write!(f, ", [{:?}]>", self.0.internee.bound_vars())?;
        Ok(())
    }
}

impl<'db> Predicate<'db> {
    pub fn new(interner: DbInterner<'db>, kind: Binder<'db, PredicateKind<'db>>) -> Self {
        let flags = FlagComputation::for_predicate(kind);
        let cached = WithCachedTypeInfo {
            internee: kind,
            flags: flags.flags,
            outer_exclusive_binder: flags.outer_exclusive_binder,
        };
        Predicate::new_(interner.db(), InternedWrapperNoDebug(cached))
    }

    pub fn inner(&self) -> &WithCachedTypeInfo<Binder<'db, PredicateKind<'db>>> {
        crate::with_attached_db(|db| {
            let inner = &self.kind_(db).0;
            // SAFETY: The caller already has access to a `Predicate<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
    }

    /// Flips the polarity of a Predicate.
    ///
    /// Given `T: Trait` predicate it returns `T: !Trait` and given `T: !Trait` returns `T: Trait`.
    pub fn flip_polarity(self) -> Option<Predicate<'db>> {
        let kind = self
            .kind()
            .map_bound(|kind| match kind {
                PredicateKind::Clause(ClauseKind::Trait(TraitPredicate {
                    trait_ref,
                    polarity,
                })) => Some(PredicateKind::Clause(ClauseKind::Trait(TraitPredicate {
                    trait_ref,
                    polarity: polarity.flip(),
                }))),

                _ => None,
            })
            .transpose()?;

        Some(Predicate::new(DbInterner::conjure(), kind))
    }
}

// FIXME: should make a "header" in interned_vec

#[derive(Debug, Clone)]
pub struct InternedClausesWrapper<'db>(SmallVec<[Clause<'db>; 2]>, TypeFlags, DebruijnIndex);

impl<'db> PartialEq for InternedClausesWrapper<'db> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<'db> Eq for InternedClausesWrapper<'db> {}

impl<'db> std::hash::Hash for InternedClausesWrapper<'db> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[salsa::interned(constructor = new_)]
pub struct Clauses<'db> {
    #[returns(ref)]
    inner_: InternedClausesWrapper<'db>,
}

impl<'db> Clauses<'db> {
    pub fn new_from_iter(
        interner: DbInterner<'db>,
        data: impl IntoIterator<Item = Clause<'db>>,
    ) -> Self {
        let clauses: SmallVec<_> = data.into_iter().collect();
        let flags = FlagComputation::<DbInterner<'db>>::for_clauses(&clauses);
        let wrapper = InternedClausesWrapper(clauses, flags.flags, flags.outer_exclusive_binder);
        Clauses::new_(interner.db(), wrapper)
    }

    pub fn inner(&self) -> &InternedClausesWrapper<'db> {
        crate::with_attached_db(|db| {
            let inner = self.inner_(db);
            // SAFETY: The caller already has access to a `Clauses<'db>`, so borrowchecking will
            // make sure that our returned value is valid for the lifetime `'db`.
            unsafe { std::mem::transmute(inner) }
        })
    }
}

impl<'db> std::fmt::Debug for Clauses<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().0.fmt(f)
    }
}

impl<'db> rustc_type_ir::inherent::Clauses<DbInterner<'db>> for Clauses<'db> {}

impl<'db> rustc_type_ir::inherent::SliceLike for Clauses<'db> {
    type Item = Clause<'db>;

    type IntoIter = <smallvec::SmallVec<[Clause<'db>; 2]> as IntoIterator>::IntoIter;

    fn iter(self) -> Self::IntoIter {
        self.inner().0.clone().into_iter()
    }

    fn as_slice(&self) -> &[Self::Item] {
        self.inner().0.as_slice()
    }
}

impl<'db> IntoIterator for Clauses<'db> {
    type Item = Clause<'db>;
    type IntoIter = <Self as rustc_type_ir::inherent::SliceLike>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        rustc_type_ir::inherent::SliceLike::iter(self)
    }
}

impl<'db> Default for Clauses<'db> {
    fn default() -> Self {
        Clauses::new_from_iter(DbInterner::conjure(), [])
    }
}

impl<'db> rustc_type_ir::TypeSuperFoldable<DbInterner<'db>> for Clauses<'db> {
    fn try_super_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let mut clauses: SmallVec<[_; 2]> = SmallVec::with_capacity(self.inner().0.len());
        for c in self {
            clauses.push(c.try_fold_with(folder)?);
        }
        Ok(Clauses::new_from_iter(folder.cx(), clauses))
    }

    fn super_fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Self {
        let mut clauses: SmallVec<[_; 2]> = SmallVec::with_capacity(self.inner().0.len());
        for c in self {
            clauses.push(c.fold_with(folder));
        }
        Clauses::new_from_iter(folder.cx(), clauses)
    }
}

impl<'db> rustc_type_ir::TypeFoldable<DbInterner<'db>> for Clauses<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        use rustc_type_ir::inherent::SliceLike as _;
        let inner: smallvec::SmallVec<[_; 2]> =
            self.iter().map(|v| v.try_fold_with(folder)).collect::<Result<_, _>>()?;
        Ok(Clauses::new_from_iter(folder.cx(), inner))
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        use rustc_type_ir::inherent::SliceLike as _;
        let inner: smallvec::SmallVec<[_; 2]> = self.iter().map(|v| v.fold_with(folder)).collect();
        Clauses::new_from_iter(folder.cx(), inner)
    }
}

impl<'db> rustc_type_ir::TypeVisitable<DbInterner<'db>> for Clauses<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        use rustc_ast_ir::visit::VisitorResult;
        use rustc_type_ir::inherent::SliceLike as _;
        rustc_ast_ir::walk_visitable_list!(visitor, self.as_slice().iter());
        V::Result::output()
    }
}

impl<'db> rustc_type_ir::Flags for Clauses<'db> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        self.inner().1
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        self.inner().2
    }
}

impl<'db> rustc_type_ir::TypeSuperVisitable<DbInterner<'db>> for Clauses<'db> {
    fn super_visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        self.as_slice().visit_with(visitor)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)] // TODO implement Debug by hand
pub struct Clause<'db>(pub(crate) Predicate<'db>);

// We could cram the reveal into the clauses like rustc does, probably
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, TypeVisitable, TypeFoldable)]
pub struct ParamEnv<'db> {
    pub(crate) clauses: Clauses<'db>,
}

impl<'db> ParamEnv<'db> {
    pub fn empty() -> Self {
        ParamEnv { clauses: Clauses::new_from_iter(DbInterner::conjure(), []) }
    }

    pub fn clauses(self) -> Clauses<'db> {
        self.clauses
    }
}

impl<'db> rustc_type_ir::inherent::ParamEnv<DbInterner<'db>> for ParamEnv<'db> {
    fn caller_bounds(self) -> impl rustc_type_ir::inherent::SliceLike<Item = Clause<'db>> {
        self.clauses
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParamEnvAnd<'db, T> {
    pub param_env: ParamEnv<'db>,
    pub value: T,
}

impl<'db, T> ParamEnvAnd<'db, T> {
    pub fn into_parts(self) -> (ParamEnv<'db>, T) {
        (self.param_env, self.value)
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for Predicate<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        visitor.visit_predicate(*self)
    }
}

impl<'db> TypeSuperVisitable<DbInterner<'db>> for Predicate<'db> {
    fn super_visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        (*self).kind().visit_with(visitor)
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for Predicate<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_predicate(self)
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self)
    }
}

impl<'db> TypeSuperFoldable<DbInterner<'db>> for Predicate<'db> {
    fn try_super_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let new = self.kind().try_fold_with(folder)?;
        Ok(Predicate::new(folder.cx(), new))
    }
    fn super_fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Self {
        let new = self.kind().fold_with(folder);
        Predicate::new(folder.cx(), new)
    }
}

impl<'db> Elaboratable<DbInterner<'db>> for Predicate<'db> {
    fn predicate(&self) -> <DbInterner<'db> as rustc_type_ir::Interner>::Predicate {
        *self
    }

    fn child(&self, clause: <DbInterner<'db> as rustc_type_ir::Interner>::Clause) -> Self {
        clause.as_predicate()
    }

    fn child_with_derived_cause(
        &self,
        clause: <DbInterner<'db> as rustc_type_ir::Interner>::Clause,
        _span: <DbInterner<'db> as rustc_type_ir::Interner>::Span,
        _parent_trait_pred: rustc_type_ir::Binder<
            DbInterner<'db>,
            rustc_type_ir::TraitPredicate<DbInterner<'db>>,
        >,
        _index: usize,
    ) -> Self {
        clause.as_predicate()
    }
}

impl<'db> Flags for Predicate<'db> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        self.inner().flags
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        self.inner().outer_exclusive_binder
    }
}

impl<'db> IntoKind for Predicate<'db> {
    type Kind = Binder<'db, PredicateKind<'db>>;

    fn kind(self) -> Self::Kind {
        self.inner().internee
    }
}

impl<'db> UpcastFrom<DbInterner<'db>, ty::PredicateKind<DbInterner<'db>>> for Predicate<'db> {
    fn upcast_from(from: ty::PredicateKind<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        Binder::dummy(from).upcast(interner)
    }
}
impl<'db>
    UpcastFrom<DbInterner<'db>, ty::Binder<DbInterner<'db>, ty::PredicateKind<DbInterner<'db>>>>
    for Predicate<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::PredicateKind<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Predicate::new(interner, from)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::ClauseKind<DbInterner<'db>>> for Predicate<'db> {
    fn upcast_from(from: ty::ClauseKind<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        Binder::dummy(PredicateKind::Clause(from)).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::Binder<DbInterner<'db>, ty::ClauseKind<DbInterner<'db>>>>
    for Predicate<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::ClauseKind<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        from.map_bound(PredicateKind::Clause).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, Clause<'db>> for Predicate<'db> {
    fn upcast_from(from: Clause<'db>, _interner: DbInterner<'db>) -> Self {
        from.0
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::NormalizesTo<DbInterner<'db>>> for Predicate<'db> {
    fn upcast_from(from: ty::NormalizesTo<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        PredicateKind::NormalizesTo(from).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::TraitRef<DbInterner<'db>>> for Predicate<'db> {
    fn upcast_from(from: ty::TraitRef<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        Binder::dummy(from).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::Binder<DbInterner<'db>, ty::TraitRef<DbInterner<'db>>>>
    for Predicate<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::TraitRef<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        from.map_bound(|trait_ref| TraitPredicate {
            trait_ref,
            polarity: PredicatePolarity::Positive,
        })
        .upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, Binder<'db, ty::TraitPredicate<DbInterner<'db>>>>
    for Predicate<'db>
{
    fn upcast_from(
        from: Binder<'db, ty::TraitPredicate<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        from.map_bound(|it| PredicateKind::Clause(ClauseKind::Trait(it))).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, Binder<'db, ProjectionPredicate<'db>>> for Predicate<'db> {
    fn upcast_from(from: Binder<'db, ProjectionPredicate<'db>>, interner: DbInterner<'db>) -> Self {
        from.map_bound(|it| PredicateKind::Clause(ClauseKind::Projection(it))).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ProjectionPredicate<'db>> for Predicate<'db> {
    fn upcast_from(from: ProjectionPredicate<'db>, interner: DbInterner<'db>) -> Self {
        PredicateKind::Clause(ClauseKind::Projection(from)).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::TraitPredicate<DbInterner<'db>>> for Predicate<'db> {
    fn upcast_from(from: ty::TraitPredicate<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        PredicateKind::Clause(ClauseKind::Trait(from)).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::OutlivesPredicate<DbInterner<'db>, Ty<'db>>>
    for Predicate<'db>
{
    fn upcast_from(
        from: ty::OutlivesPredicate<DbInterner<'db>, Ty<'db>>,
        interner: DbInterner<'db>,
    ) -> Self {
        PredicateKind::Clause(ClauseKind::TypeOutlives(from)).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::OutlivesPredicate<DbInterner<'db>, Region<'db>>>
    for Predicate<'db>
{
    fn upcast_from(
        from: ty::OutlivesPredicate<DbInterner<'db>, Region<'db>>,
        interner: DbInterner<'db>,
    ) -> Self {
        PredicateKind::Clause(ClauseKind::RegionOutlives(from)).upcast(interner)
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::OutlivesPredicate<DbInterner<'db>, Ty<'db>>>
    for Clause<'db>
{
    fn upcast_from(
        from: ty::OutlivesPredicate<DbInterner<'db>, Ty<'db>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.upcast(interner))
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::OutlivesPredicate<DbInterner<'db>, Region<'db>>>
    for Clause<'db>
{
    fn upcast_from(
        from: ty::OutlivesPredicate<DbInterner<'db>, Region<'db>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.upcast(interner))
    }
}

impl<'db> UpcastFrom<DbInterner<'db>, PolyRegionOutlivesPredicate<'db>> for Predicate<'db> {
    fn upcast_from(from: PolyRegionOutlivesPredicate<'db>, tcx: DbInterner<'db>) -> Self {
        from.map_bound(|p| PredicateKind::Clause(ClauseKind::RegionOutlives(p))).upcast(tcx)
    }
}

impl<'db> rustc_type_ir::inherent::Predicate<DbInterner<'db>> for Predicate<'db> {
    fn as_clause(self) -> Option<<DbInterner<'db> as rustc_type_ir::Interner>::Clause> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Some(self.expect_clause()),
            _ => None,
        }
    }

    /// Whether this projection can be soundly normalized.
    ///
    /// Wf predicates must not be normalized, as normalization
    /// can remove required bounds which would cause us to
    /// unsoundly accept some programs. See #91068.
    fn allow_normalization(self) -> bool {
        // TODO: this should probably live in rustc_type_ir
        match self.inner().as_ref().skip_binder() {
            PredicateKind::Clause(ClauseKind::WellFormed(_))
            | PredicateKind::AliasRelate(..)
            | PredicateKind::NormalizesTo(..) => false,
            PredicateKind::Clause(ClauseKind::Trait(_))
            | PredicateKind::Clause(ClauseKind::RegionOutlives(_))
            | PredicateKind::Clause(ClauseKind::TypeOutlives(_))
            | PredicateKind::Clause(ClauseKind::Projection(_))
            | PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::Clause(ClauseKind::HostEffect(..))
            | PredicateKind::Clause(ClauseKind::UnstableFeature(_))
            | PredicateKind::DynCompatible(_)
            | PredicateKind::Subtype(_)
            | PredicateKind::Coerce(_)
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(_))
            | PredicateKind::ConstEquate(_, _)
            | PredicateKind::Ambiguous => true,
        }
    }
}

impl<'db> Predicate<'db> {
    pub fn as_trait_clause(self) -> Option<PolyTraitPredicate<'db>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Trait(t)) => Some(predicate.rebind(t)),
            _ => None,
        }
    }

    pub fn as_projection_clause(self) -> Option<PolyProjectionPredicate<'db>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Projection(t)) => Some(predicate.rebind(t)),
            _ => None,
        }
    }

    /// Matches a `PredicateKind::Clause` and turns it into a `Clause`, otherwise returns `None`.
    pub fn as_clause(self) -> Option<Clause<'db>> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Some(self.expect_clause()),
            _ => None,
        }
    }

    /// Assert that the predicate is a clause.
    pub fn expect_clause(self) -> Clause<'db> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Clause(self),
            _ => panic!("{self:?} is not a clause"),
        }
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for Clause<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        visitor.visit_predicate((*self).as_predicate())
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for Clause<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(folder.try_fold_predicate(self.as_predicate())?.expect_clause())
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        folder.fold_predicate(self.as_predicate()).expect_clause()
    }
}

impl<'db> IntoKind for Clause<'db> {
    type Kind = Binder<'db, ClauseKind<'db>>;

    fn kind(self) -> Self::Kind {
        self.0.kind().map_bound(|pk| match pk {
            PredicateKind::Clause(kind) => kind,
            _ => unreachable!(),
        })
    }
}

impl<'db> Clause<'db> {
    pub fn as_predicate(self) -> Predicate<'db> {
        self.0
    }
}

impl<'db> Elaboratable<DbInterner<'db>> for Clause<'db> {
    fn predicate(&self) -> <DbInterner<'db> as rustc_type_ir::Interner>::Predicate {
        self.0
    }

    fn child(&self, clause: <DbInterner<'db> as rustc_type_ir::Interner>::Clause) -> Self {
        clause
    }

    fn child_with_derived_cause(
        &self,
        clause: <DbInterner<'db> as rustc_type_ir::Interner>::Clause,
        _span: <DbInterner<'db> as rustc_type_ir::Interner>::Span,
        _parent_trait_pred: rustc_type_ir::Binder<
            DbInterner<'db>,
            rustc_type_ir::TraitPredicate<DbInterner<'db>>,
        >,
        _index: usize,
    ) -> Self {
        clause
    }
}

impl<'db> UpcastFrom<DbInterner<'db>, ty::Binder<DbInterner<'db>, ty::ClauseKind<DbInterner<'db>>>>
    for Clause<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::ClauseKind<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.map_bound(PredicateKind::Clause).upcast(interner))
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::TraitRef<DbInterner<'db>>> for Clause<'db> {
    fn upcast_from(from: ty::TraitRef<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        Clause(from.upcast(interner))
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::Binder<DbInterner<'db>, ty::TraitRef<DbInterner<'db>>>>
    for Clause<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::TraitRef<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.upcast(interner))
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::TraitPredicate<DbInterner<'db>>> for Clause<'db> {
    fn upcast_from(from: ty::TraitPredicate<DbInterner<'db>>, interner: DbInterner<'db>) -> Self {
        Clause(from.upcast(interner))
    }
}
impl<'db>
    UpcastFrom<DbInterner<'db>, ty::Binder<DbInterner<'db>, ty::TraitPredicate<DbInterner<'db>>>>
    for Clause<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::TraitPredicate<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.upcast(interner))
    }
}
impl<'db> UpcastFrom<DbInterner<'db>, ty::ProjectionPredicate<DbInterner<'db>>> for Clause<'db> {
    fn upcast_from(
        from: ty::ProjectionPredicate<DbInterner<'db>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.upcast(interner))
    }
}
impl<'db>
    UpcastFrom<
        DbInterner<'db>,
        ty::Binder<DbInterner<'db>, ty::ProjectionPredicate<DbInterner<'db>>>,
    > for Clause<'db>
{
    fn upcast_from(
        from: ty::Binder<DbInterner<'db>, ty::ProjectionPredicate<DbInterner<'db>>>,
        interner: DbInterner<'db>,
    ) -> Self {
        Clause(from.upcast(interner))
    }
}

impl<'db> rustc_type_ir::inherent::Clause<DbInterner<'db>> for Clause<'db> {
    fn as_predicate(self) -> <DbInterner<'db> as rustc_type_ir::Interner>::Predicate {
        self.0
    }

    fn instantiate_supertrait(
        self,
        cx: DbInterner<'db>,
        trait_ref: rustc_type_ir::Binder<DbInterner<'db>, rustc_type_ir::TraitRef<DbInterner<'db>>>,
    ) -> Self {
        tracing::debug!(?self, ?trait_ref);
        // See the rustc impl for a long comment
        let bound_pred = self.kind();
        let pred_bound_vars = bound_pred.bound_vars();
        let trait_bound_vars = trait_ref.bound_vars();
        // 1) Self: Bar1<'a, '^0.0> -> Self: Bar1<'a, '^0.1>
        let shifted_pred =
            cx.shift_bound_var_indices(trait_bound_vars.len(), bound_pred.skip_binder());
        // 2) Self: Bar1<'a, '^0.1> -> T: Bar1<'^0.0, '^0.1>
        let new = EarlyBinder::bind(shifted_pred).instantiate(cx, trait_ref.skip_binder().args);
        // 3) ['x] + ['b] -> ['x, 'b]
        let bound_vars =
            BoundVarKinds::new_from_iter(cx, trait_bound_vars.iter().chain(pred_bound_vars.iter()));

        let predicate: Predicate<'db> =
            ty::Binder::bind_with_vars(PredicateKind::Clause(new), bound_vars).upcast(cx);
        predicate.expect_clause()
    }
}
