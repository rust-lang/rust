pub use crate::ty::context::{
    tls, CtxtInterners, DeducedParamAttrs, FreeRegionInfo, GlobalCtxt, Lift, TyCtxt, TyCtxtFeed,
};
use rustc_data_structures::intern::Interned;
use rustc_hir::def_id::DefId;
use rustc_type_ir::WithCachedTypeInfo;

use crate::ty::{
    self, AliasRelationDirection, Binder, BoundConstness, Clause, ClosureKind, CoercePredicate,
    Const, DebruijnIndex, EarlyBinder, GenericArg, PolyProjectionPredicate,
    PolyTypeOutlivesPredicate, SubstsRef, SubtypePredicate, Term, Ty, TypeFlags,
};

mod instantiated_predicates;
mod trait_predicate;

pub use instantiated_predicates::InstantiatedPredicates;
pub use trait_predicate::{PolyTraitPredicate, TraitPredicate};

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
        trait_ref: &ty::PolyTraitRef<'tcx>,
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
