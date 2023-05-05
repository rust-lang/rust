use std::fmt;

use rustc_data_structures::intern::Interned;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_type_ir::WithCachedTypeInfo;

use crate::ty::{
    self, AliasRelationDirection, AliasTy, Binder, ClosureKind, Const, DebruijnIndex, EarlyBinder,
    GenericArg, ImplPolarity, ParamEnv, PolyTraitRef, SubstsRef, Term, TraitRef, Ty, TyCtxt,
    TypeFlags,
};

mod crate_predicates_map;
mod instantiated_predicates;
mod to_predicate;

pub use crate_predicates_map::CratePredicatesMap;
pub use instantiated_predicates::InstantiatedPredicates;
pub use to_predicate::ToPredicate;

/// Use this rather than `PredicateKind`, whenever possible.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Predicate<'tcx>(
    pub(super) Interned<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
);

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum PredicateKind<'tcx> {
    /// Prove a clause
    Clause(Clause<'tcx>),

    /// No syntax: `T` well-formed.
    WellFormed(GenericArg<'tcx>),

    /// Trait must be object-safe.
    ObjectSafe(DefId),

    /// No direct syntax. May be thought of as `where T: FnFoo<...>`
    /// for some substitutions `...` and `T` being a closure type.
    /// Satisfied (or refuted) once we know the closure's kind.
    ClosureKind(DefId, SubstsRef<'tcx>, ClosureKind),

    /// `T1 <: T2`
    ///
    /// This obligation is created most often when we have two
    /// unresolved type variables and hence don't have enough
    /// information to process the subtyping obligation yet.
    Subtype(SubtypePredicate<'tcx>),

    /// `T1` coerced to `T2`
    ///
    /// Like a subtyping obligation, this is created most often
    /// when we have two unresolved type variables and hence
    /// don't have enough information to process the coercion
    /// obligation yet. At the moment, we actually process coercions
    /// very much like subtyping and don't handle the full coercion
    /// logic.
    Coerce(CoercePredicate<'tcx>),

    /// Constant initializer must evaluate successfully.
    ConstEvaluatable(ty::Const<'tcx>),

    /// Constants must be equal. The first component is the const that is expected.
    ConstEquate(Const<'tcx>, Const<'tcx>),

    /// Represents a type found in the environment that we can use for implied bounds.
    ///
    /// Only used for Chalk.
    TypeWellFormedFromEnv(Ty<'tcx>),

    /// A marker predicate that is always ambiguous.
    /// Used for coherence to mark opaque types as possibly equal to each other but ambiguous.
    Ambiguous,

    /// Separate from `Clause::Projection` which is used for normalization in new solver.
    /// This predicate requires two terms to be equal to eachother.
    ///
    /// Only used for new solver
    AliasRelate(Term<'tcx>, Term<'tcx>, AliasRelationDirection),
}

/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum Clause<'tcx> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(TraitPredicate<'tcx>),

    /// `where 'a: 'b`
    RegionOutlives(RegionOutlivesPredicate<'tcx>),

    /// `where T: 'a`
    TypeOutlives(TypeOutlivesPredicate<'tcx>),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(ProjectionPredicate<'tcx>),

    /// Ensures that a const generic argument to a parameter `const N: u8`
    /// is of type `u8`.
    ConstArgHasType(Const<'tcx>, Ty<'tcx>),
}

pub type PolySubtypePredicate<'tcx> = ty::Binder<'tcx, SubtypePredicate<'tcx>>;

/// Encodes that `a` must be a subtype of `b`. The `a_is_expected` flag indicates
/// whether the `a` type is the type that we should label as "expected" when
/// presenting user diagnostics.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct SubtypePredicate<'tcx> {
    pub a_is_expected: bool,
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}

pub type PolyCoercePredicate<'tcx> = ty::Binder<'tcx, CoercePredicate<'tcx>>;

/// Encodes that we have to coerce *from* the `a` type to the `b` type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct CoercePredicate<'tcx> {
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}

pub type PolyTraitPredicate<'tcx> = ty::Binder<'tcx, TraitPredicate<'tcx>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: TraitRef<'tcx>,

    pub constness: BoundConstness,

    /// If polarity is Positive: we are proving that the trait is implemented.
    ///
    /// If polarity is Negative: we are proving that a negative impl of this trait
    /// exists. (Note that coherence also checks whether negative impls of supertraits
    /// exist via a series of predicates.)
    ///
    /// If polarity is Reserved: that's a bug.
    pub polarity: ImplPolarity,
}

pub type RegionOutlivesPredicate<'tcx> = OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>;
pub type TypeOutlivesPredicate<'tcx> = OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>;
pub type PolyRegionOutlivesPredicate<'tcx> = ty::Binder<'tcx, RegionOutlivesPredicate<'tcx>>;
pub type PolyTypeOutlivesPredicate<'tcx> = ty::Binder<'tcx, TypeOutlivesPredicate<'tcx>>;

/// `A: B`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct OutlivesPredicate<A, B>(pub A, pub B);

pub type PolyProjectionPredicate<'tcx> = ty::Binder<'tcx, ProjectionPredicate<'tcx>>;

/// This kind of predicate has no *direct* correspondent in the
/// syntax, but it roughly corresponds to the syntactic forms:
///
/// 1. `T: TraitRef<..., Item = Type>`
/// 2. `<T as TraitRef<...>>::Item == Type` (NYI)
///
/// In particular, form #1 is "desugared" to the combination of a
/// normal trait predicate (`T: TraitRef<...>`) and one of these
/// predicates. Form #2 is a broader form in that it also permits
/// equality between arbitrary types. Processing an instance of
/// Form #2 eventually yields one of these `ProjectionPredicate`
/// instances to normalize the LHS.
#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: AliasTy<'tcx>,
    pub term: Term<'tcx>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable)]
pub enum BoundConstness {
    /// `T: Trait`
    NotConst,
    /// `T: ~const Trait`
    ///
    /// Requires resolving to const only when we are in a const context.
    ConstIfConst,
}

impl<'tcx> Predicate<'tcx> {
    /// Gets the inner `Binder<'tcx, PredicateKind<'tcx>>`.
    #[inline]
    pub fn kind(self) -> Binder<'tcx, PredicateKind<'tcx>> {
        self.0.internee
    }

    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    #[inline(always)]
    pub fn outer_exclusive_binder(self) -> DebruijnIndex {
        self.0.outer_exclusive_binder
    }

    /// Flips the polarity of a Predicate.
    ///
    /// Given `T: Trait` predicate it returns `T: !Trait` and given `T: !Trait` returns `T: Trait`.
    pub fn flip_polarity(self, tcx: TyCtxt<'tcx>) -> Option<Predicate<'tcx>> {
        let kind = self
            .kind()
            .map_bound(|kind| match kind {
                PredicateKind::Clause(Clause::Trait(TraitPredicate {
                    trait_ref,
                    constness,
                    polarity,
                })) => Some(PredicateKind::Clause(Clause::Trait(TraitPredicate {
                    trait_ref,
                    constness,
                    polarity: polarity.flip()?,
                }))),

                _ => None,
            })
            .transpose()?;

        Some(tcx.mk_predicate(kind))
    }

    pub fn without_const(mut self, tcx: TyCtxt<'tcx>) -> Self {
        if let PredicateKind::Clause(Clause::Trait(TraitPredicate { trait_ref, constness, polarity })) = self.kind().skip_binder()
            && constness != BoundConstness::NotConst
        {
            self = tcx.mk_predicate(self.kind().rebind(PredicateKind::Clause(Clause::Trait(TraitPredicate {
                trait_ref,
                constness: BoundConstness::NotConst,
                polarity,
            }))));
        }
        self
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn is_coinductive(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(data)) => {
                tcx.trait_is_coinductive(data.def_id())
            }
            ty::PredicateKind::WellFormed(_) => true,
            _ => false,
        }
    }

    /// Whether this projection can be soundly normalized.
    ///
    /// Wf predicates must not be normalized, as normalization
    /// can remove required bounds which would cause us to
    /// unsoundly accept some programs. See #91068.
    #[inline]
    pub fn allow_normalization(self) -> bool {
        match self.kind().skip_binder() {
            PredicateKind::WellFormed(_) => false,
            PredicateKind::Clause(Clause::Trait(_))
            | PredicateKind::Clause(Clause::RegionOutlives(_))
            | PredicateKind::Clause(Clause::TypeOutlives(_))
            | PredicateKind::Clause(Clause::Projection(_))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::AliasRelate(..)
            | PredicateKind::ObjectSafe(_)
            | PredicateKind::ClosureKind(_, _, _)
            | PredicateKind::Subtype(_)
            | PredicateKind::Coerce(_)
            | PredicateKind::ConstEvaluatable(_)
            | PredicateKind::ConstEquate(_, _)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(_) => true,
        }
    }

    /// Performs a substitution suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// substitution in terms of what happens with bound regions. See
    /// lengthy comment below for details.
    pub fn subst_supertrait(
        self,
        tcx: TyCtxt<'tcx>,
        trait_ref: &PolyTraitRef<'tcx>,
    ) -> Predicate<'tcx> {
        // The interaction between HRTB and supertraits is not entirely
        // obvious. Let me walk you (and myself) through an example.
        //
        // Let's start with an easy case. Consider two traits:
        //
        //     trait Foo<'a>: Bar<'a,'a> { }
        //     trait Bar<'b,'c> { }
        //
        // Now, if we have a trait reference `for<'x> T: Foo<'x>`, then
        // we can deduce that `for<'x> T: Bar<'x,'x>`. Basically, if we
        // knew that `Foo<'x>` (for any 'x) then we also know that
        // `Bar<'x,'x>` (for any 'x). This more-or-less falls out from
        // normal substitution.
        //
        // In terms of why this is sound, the idea is that whenever there
        // is an impl of `T:Foo<'a>`, it must show that `T:Bar<'a,'a>`
        // holds. So if there is an impl of `T:Foo<'a>` that applies to
        // all `'a`, then we must know that `T:Bar<'a,'a>` holds for all
        // `'a`.
        //
        // Another example to be careful of is this:
        //
        //     trait Foo1<'a>: for<'b> Bar1<'a,'b> { }
        //     trait Bar1<'b,'c> { }
        //
        // Here, if we have `for<'x> T: Foo1<'x>`, then what do we know?
        // The answer is that we know `for<'x,'b> T: Bar1<'x,'b>`. The
        // reason is similar to the previous example: any impl of
        // `T:Foo1<'x>` must show that `for<'b> T: Bar1<'x, 'b>`. So
        // basically we would want to collapse the bound lifetimes from
        // the input (`trait_ref`) and the supertraits.
        //
        // To achieve this in practice is fairly straightforward. Let's
        // consider the more complicated scenario:
        //
        // - We start out with `for<'x> T: Foo1<'x>`. In this case, `'x`
        //   has a De Bruijn index of 1. We want to produce `for<'x,'b> T: Bar1<'x,'b>`,
        //   where both `'x` and `'b` would have a DB index of 1.
        //   The substitution from the input trait-ref is therefore going to be
        //   `'a => 'x` (where `'x` has a DB index of 1).
        // - The supertrait-ref is `for<'b> Bar1<'a,'b>`, where `'a` is an
        //   early-bound parameter and `'b` is a late-bound parameter with a
        //   DB index of 1.
        // - If we replace `'a` with `'x` from the input, it too will have
        //   a DB index of 1, and thus we'll have `for<'x,'b> Bar1<'x,'b>`
        //   just as we wanted.
        //
        // There is only one catch. If we just apply the substitution `'a
        // => 'x` to `for<'b> Bar1<'a,'b>`, the substitution code will
        // adjust the DB index because we substituting into a binder (it
        // tries to be so smart...) resulting in `for<'x> for<'b>
        // Bar1<'x,'b>` (we have no syntax for this, so use your
        // imagination). Basically the 'x will have DB index of 2 and 'b
        // will have DB index of 1. Not quite what we want. So we apply
        // the substitution to the *contents* of the trait reference,
        // rather than the trait reference itself (put another way, the
        // substitution code expects equal binding levels in the values
        // from the substitution and the value being substituted into, and
        // this trick achieves that).

        // Working through the second example:
        // trait_ref: for<'x> T: Foo1<'^0.0>; substs: [T, '^0.0]
        // predicate: for<'b> Self: Bar1<'a, '^0.0>; substs: [Self, 'a, '^0.0]
        // We want to end up with:
        //     for<'x, 'b> T: Bar1<'^0.0, '^0.1>
        // To do this:
        // 1) We must shift all bound vars in predicate by the length
        //    of trait ref's bound vars. So, we would end up with predicate like
        //    Self: Bar1<'a, '^0.1>
        // 2) We can then apply the trait substs to this, ending up with
        //    T: Bar1<'^0.0, '^0.1>
        // 3) Finally, to create the final bound vars, we concatenate the bound
        //    vars of the trait ref with those of the predicate:
        //    ['x, 'b]
        let bound_pred = self.kind();
        let pred_bound_vars = bound_pred.bound_vars();
        let trait_bound_vars = trait_ref.bound_vars();
        // 1) Self: Bar1<'a, '^0.0> -> Self: Bar1<'a, '^0.1>
        let shifted_pred =
            tcx.shift_bound_var_indices(trait_bound_vars.len(), bound_pred.skip_binder());
        // 2) Self: Bar1<'a, '^0.1> -> T: Bar1<'^0.0, '^0.1>
        let new = EarlyBinder(shifted_pred).subst(tcx, trait_ref.skip_binder().substs);
        // 3) ['x] + ['b] -> ['x, 'b]
        let bound_vars =
            tcx.mk_bound_variable_kinds_from_iter(trait_bound_vars.iter().chain(pred_bound_vars));
        tcx.reuse_or_mk_predicate(self, ty::Binder::bind_with_vars(new, bound_vars))
    }

    pub fn to_opt_poly_trait_pred(self) -> Option<PolyTraitPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(Clause::Trait(t)) => Some(predicate.rebind(t)),
            PredicateKind::Clause(Clause::Projection(..))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::AliasRelate(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(Clause::RegionOutlives(..))
            | PredicateKind::WellFormed(..)
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::ClosureKind(..)
            | PredicateKind::Clause(Clause::TypeOutlives(..))
            | PredicateKind::ConstEvaluatable(..)
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(..) => None,
        }
    }

    pub fn to_opt_poly_projection_pred(self) -> Option<PolyProjectionPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(Clause::Projection(t)) => Some(predicate.rebind(t)),
            PredicateKind::Clause(Clause::Trait(..))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::AliasRelate(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(Clause::RegionOutlives(..))
            | PredicateKind::WellFormed(..)
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::ClosureKind(..)
            | PredicateKind::Clause(Clause::TypeOutlives(..))
            | PredicateKind::ConstEvaluatable(..)
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(..) => None,
        }
    }

    pub fn to_opt_type_outlives(self) -> Option<PolyTypeOutlivesPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(Clause::TypeOutlives(data)) => Some(predicate.rebind(data)),
            PredicateKind::Clause(Clause::Trait(..))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::Clause(Clause::Projection(..))
            | PredicateKind::AliasRelate(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(Clause::RegionOutlives(..))
            | PredicateKind::WellFormed(..)
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::ClosureKind(..)
            | PredicateKind::ConstEvaluatable(..)
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(..) => None,
        }
    }
}

impl rustc_errors::IntoDiagnosticArg for Predicate<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        rustc_errors::DiagnosticArgValue::Str(std::borrow::Cow::Owned(self.to_string()))
    }
}

impl<'tcx> TraitPredicate<'tcx> {
    pub fn remap_constness(&mut self, param_env: &mut ParamEnv<'tcx>) {
        *param_env = param_env.with_constness(self.constness.and(param_env.constness()))
    }

    /// Remap the constness of this predicate before emitting it for diagnostics.
    pub fn remap_constness_diag(&mut self, param_env: ParamEnv<'tcx>) {
        // this is different to `remap_constness` that callees want to print this predicate
        // in case of selection errors. `T: ~const Drop` bounds cannot end up here when the
        // param_env is not const because it is always satisfied in non-const contexts.
        if let hir::Constness::NotConst = param_env.constness() {
            self.constness = ty::BoundConstness::NotConst;
        }
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        Self { trait_ref: self.trait_ref.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn def_id(self) -> DefId {
        self.trait_ref.def_id
    }

    pub fn self_ty(self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
    }

    #[inline]
    pub fn is_const_if_const(self) -> bool {
        self.constness == BoundConstness::ConstIfConst
    }

    pub fn is_constness_satisfied_by(self, constness: hir::Constness) -> bool {
        match (self.constness, constness) {
            (BoundConstness::NotConst, _)
            | (BoundConstness::ConstIfConst, hir::Constness::Const) => true,
            (BoundConstness::ConstIfConst, hir::Constness::NotConst) => false,
        }
    }

    pub fn without_const(mut self) -> Self {
        self.constness = BoundConstness::NotConst;
        self
    }
}

impl<'tcx> PolyTraitPredicate<'tcx> {
    pub fn def_id(self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().def_id()
    }

    pub fn self_ty(self) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.map_bound(|trait_ref| trait_ref.self_ty())
    }

    /// Remap the constness of this predicate before emitting it for diagnostics.
    pub fn remap_constness_diag(&mut self, param_env: ParamEnv<'tcx>) {
        *self = self.map_bound(|mut p| {
            p.remap_constness_diag(param_env);
            p
        });
    }

    #[inline]
    pub fn is_const_if_const(self) -> bool {
        self.skip_binder().is_const_if_const()
    }

    #[inline]
    pub fn polarity(self) -> ImplPolarity {
        self.skip_binder().polarity
    }
}

impl<'tcx> ProjectionPredicate<'tcx> {
    pub fn self_ty(self) -> Ty<'tcx> {
        self.projection_ty.self_ty()
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ProjectionPredicate<'tcx> {
        Self { projection_ty: self.projection_ty.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId {
        self.projection_ty.trait_def_id(tcx)
    }

    pub fn def_id(self) -> DefId {
        self.projection_ty.def_id
    }
}

impl<'tcx> PolyProjectionPredicate<'tcx> {
    /// Returns the `DefId` of the trait of the associated item being projected.
    #[inline]
    pub fn trait_def_id(&self, tcx: TyCtxt<'tcx>) -> DefId {
        self.skip_binder().projection_ty.trait_def_id(tcx)
    }

    /// Get the [PolyTraitRef] required for this projection to be well formed.
    /// Note that for generic associated types the predicates of the associated
    /// type also need to be checked.
    #[inline]
    pub fn required_poly_trait_ref(&self, tcx: TyCtxt<'tcx>) -> PolyTraitRef<'tcx> {
        // Note: unlike with `TraitRef::to_poly_trait_ref()`,
        // `self.0.trait_ref` is permitted to have escaping regions.
        // This is because here `self` has a `Binder` and so does our
        // return value, so we are preserving the number of binding
        // levels.
        self.map_bound(|predicate| predicate.projection_ty.trait_ref(tcx))
    }

    pub fn term(&self) -> ty::Binder<'tcx, Term<'tcx>> {
        self.map_bound(|predicate| predicate.term)
    }

    /// The `DefId` of the `TraitItem` for the associated type.
    ///
    /// Note that this is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `tcx.associated_item(projection_def_id()).container`.
    pub fn projection_def_id(&self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().projection_ty.def_id
    }
}

impl BoundConstness {
    /// Reduce `self` and `constness` to two possible combined states instead of four.
    pub fn and(&mut self, constness: hir::Constness) -> hir::Constness {
        match (constness, self) {
            (hir::Constness::Const, BoundConstness::ConstIfConst) => hir::Constness::Const,
            (_, this) => {
                *this = BoundConstness::NotConst;
                hir::Constness::NotConst
            }
        }
    }
}

impl fmt::Display for BoundConstness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConst => f.write_str("normal"),
            Self::ConstIfConst => f.write_str("`~const`"),
        }
    }
}
