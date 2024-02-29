use rustc_data_structures::captures::Captures;
use rustc_data_structures::intern::Interned;
use rustc_errors::{DiagArgValue, IntoDiagnosticArg};
use rustc_hir::def_id::DefId;
use rustc_hir::LangItem;
use rustc_span::Span;
use rustc_type_ir::ClauseKind as IrClauseKind;
use rustc_type_ir::PredicateKind as IrPredicateKind;
use std::cmp::Ordering;

use crate::ty::visit::TypeVisitableExt;
use crate::ty::{
    self, AliasTy, Binder, DebruijnIndex, DebugWithInfcx, EarlyBinder, GenericArg, GenericArgs,
    GenericArgsRef, ImplPolarity, Term, Ty, TyCtxt, TypeFlags, WithCachedTypeInfo,
};

pub type ClauseKind<'tcx> = IrClauseKind<TyCtxt<'tcx>>;
pub type PredicateKind<'tcx> = IrPredicateKind<TyCtxt<'tcx>>;

/// A statement that can be proven by a trait solver. This includes things that may
/// show up in where clauses, such as trait predicates and projection predicates,
/// and also things that are emitted as part of type checking such as `ObjectSafe`
/// predicate which is emitted when a type is coerced to a trait object.
///
/// Use this rather than `PredicateKind`, whenever possible.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Predicate<'tcx>(
    pub(super) Interned<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
);

impl<'tcx> rustc_type_ir::visit::Flags for Predicate<'tcx> {
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    fn outer_exclusive_binder(&self) -> ty::DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl<'tcx> Predicate<'tcx> {
    /// Gets the inner `ty::Binder<'tcx, PredicateKind<'tcx>>`.
    #[inline]
    pub fn kind(self) -> ty::Binder<'tcx, PredicateKind<'tcx>> {
        self.0.internee
    }

    // FIXME(compiler-errors): Think about removing this.
    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    // FIXME(compiler-errors): Think about removing this.
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
                PredicateKind::Clause(ClauseKind::Trait(TraitPredicate {
                    trait_ref,
                    polarity,
                })) => Some(PredicateKind::Clause(ClauseKind::Trait(TraitPredicate {
                    trait_ref,
                    polarity: polarity.flip()?,
                }))),

                _ => None,
            })
            .transpose()?;

        Some(tcx.mk_predicate(kind))
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn is_coinductive(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(data)) => {
                tcx.trait_is_coinductive(data.def_id())
            }
            ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(_)) => true,
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
            PredicateKind::Clause(ClauseKind::WellFormed(_)) => false,
            // `NormalizesTo` is only used in the new solver, so this shouldn't
            // matter. Normalizing `term` would be 'wrong' however, as it changes whether
            // `normalizes-to(<T as Trait>::Assoc, <T as Trait>::Assoc)` holds.
            PredicateKind::NormalizesTo(..) => false,
            PredicateKind::Clause(ClauseKind::Trait(_))
            | PredicateKind::Clause(ClauseKind::RegionOutlives(_))
            | PredicateKind::Clause(ClauseKind::TypeOutlives(_))
            | PredicateKind::Clause(ClauseKind::Projection(_))
            | PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::AliasRelate(..)
            | PredicateKind::ObjectSafe(_)
            | PredicateKind::Subtype(_)
            | PredicateKind::Coerce(_)
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(_))
            | PredicateKind::ConstEquate(_, _)
            | PredicateKind::Ambiguous => true,
        }
    }
}

impl rustc_errors::IntoDiagnosticArg for Predicate<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagArgValue {
        rustc_errors::DiagArgValue::Str(std::borrow::Cow::Owned(self.to_string()))
    }
}

impl rustc_errors::IntoDiagnosticArg for Clause<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagArgValue {
        rustc_errors::DiagArgValue::Str(std::borrow::Cow::Owned(self.to_string()))
    }
}

/// A subset of predicates which can be assumed by the trait solver. They show up in
/// an item's where clauses, hence the name `Clause`, and may either be user-written
/// (such as traits) or may be inserted during lowering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Clause<'tcx>(
    pub(super) Interned<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
);

impl<'tcx> Clause<'tcx> {
    pub fn as_predicate(self) -> Predicate<'tcx> {
        Predicate(self.0)
    }

    pub fn kind(self) -> ty::Binder<'tcx, ClauseKind<'tcx>> {
        self.0.internee.map_bound(|kind| match kind {
            PredicateKind::Clause(clause) => clause,
            _ => unreachable!(),
        })
    }

    pub fn as_trait_clause(self) -> Option<ty::Binder<'tcx, TraitPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::Trait(trait_clause) = clause.skip_binder() {
            Some(clause.rebind(trait_clause))
        } else {
            None
        }
    }

    pub fn as_projection_clause(self) -> Option<ty::Binder<'tcx, ProjectionPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::Projection(projection_clause) = clause.skip_binder() {
            Some(clause.rebind(projection_clause))
        } else {
            None
        }
    }

    pub fn as_type_outlives_clause(self) -> Option<ty::Binder<'tcx, TypeOutlivesPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::TypeOutlives(o) = clause.skip_binder() {
            Some(clause.rebind(o))
        } else {
            None
        }
    }

    pub fn as_region_outlives_clause(
        self,
    ) -> Option<ty::Binder<'tcx, RegionOutlivesPredicate<'tcx>>> {
        let clause = self.kind();
        if let ty::ClauseKind::RegionOutlives(o) = clause.skip_binder() {
            Some(clause.rebind(o))
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum ExistentialPredicate<'tcx> {
    /// E.g., `Iterator`.
    Trait(ExistentialTraitRef<'tcx>),
    /// E.g., `Iterator::Item = T`.
    Projection(ExistentialProjection<'tcx>),
    /// E.g., `Send`.
    AutoTrait(DefId),
}

impl<'tcx> DebugWithInfcx<TyCtxt<'tcx>> for ExistentialPredicate<'tcx> {
    fn fmt<Infcx: rustc_type_ir::InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: rustc_type_ir::WithInfcx<'_, Infcx, &Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        std::fmt::Debug::fmt(&this.data, f)
    }
}

impl<'tcx> ExistentialPredicate<'tcx> {
    /// Compares via an ordering that will not change if modules are reordered or other changes are
    /// made to the tree. In particular, this ordering is preserved across incremental compilations.
    pub fn stable_cmp(&self, tcx: TyCtxt<'tcx>, other: &Self) -> Ordering {
        use self::ExistentialPredicate::*;
        match (*self, *other) {
            (Trait(_), Trait(_)) => Ordering::Equal,
            (Projection(ref a), Projection(ref b)) => {
                tcx.def_path_hash(a.def_id).cmp(&tcx.def_path_hash(b.def_id))
            }
            (AutoTrait(ref a), AutoTrait(ref b)) => {
                tcx.def_path_hash(*a).cmp(&tcx.def_path_hash(*b))
            }
            (Trait(_), _) => Ordering::Less,
            (Projection(_), Trait(_)) => Ordering::Greater,
            (Projection(_), _) => Ordering::Less,
            (AutoTrait(_), _) => Ordering::Greater,
        }
    }
}

pub type PolyExistentialPredicate<'tcx> = ty::Binder<'tcx, ExistentialPredicate<'tcx>>;

impl<'tcx> PolyExistentialPredicate<'tcx> {
    /// Given an existential predicate like `?Self: PartialEq<u32>` (e.g., derived from `dyn PartialEq<u32>`),
    /// and a concrete type `self_ty`, returns a full predicate where the existentially quantified variable `?Self`
    /// has been replaced with `self_ty` (e.g., `self_ty: PartialEq<u32>`, in our example).
    pub fn with_self_ty(&self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ty::Clause<'tcx> {
        match self.skip_binder() {
            ExistentialPredicate::Trait(tr) => {
                self.rebind(tr).with_self_ty(tcx, self_ty).to_predicate(tcx)
            }
            ExistentialPredicate::Projection(p) => {
                self.rebind(p.with_self_ty(tcx, self_ty)).to_predicate(tcx)
            }
            ExistentialPredicate::AutoTrait(did) => {
                let generics = tcx.generics_of(did);
                let trait_ref = if generics.params.len() == 1 {
                    ty::TraitRef::new(tcx, did, [self_ty])
                } else {
                    // If this is an ill-formed auto trait, then synthesize
                    // new error args for the missing generics.
                    let err_args = ty::GenericArgs::extend_with_error(tcx, did, &[self_ty.into()]);
                    ty::TraitRef::new(tcx, did, err_args)
                };
                self.rebind(trait_ref).to_predicate(tcx)
            }
        }
    }
}

impl<'tcx> ty::List<ty::PolyExistentialPredicate<'tcx>> {
    /// Returns the "principal `DefId`" of this set of existential predicates.
    ///
    /// A Rust trait object type consists (in addition to a lifetime bound)
    /// of a set of trait bounds, which are separated into any number
    /// of auto-trait bounds, and at most one non-auto-trait bound. The
    /// non-auto-trait bound is called the "principal" of the trait
    /// object.
    ///
    /// Only the principal can have methods or type parameters (because
    /// auto traits can have neither of them). This is important, because
    /// it means the auto traits can be treated as an unordered set (methods
    /// would force an order for the vtable, while relating traits with
    /// type parameters without knowing the order to relate them in is
    /// a rather non-trivial task).
    ///
    /// For example, in the trait object `dyn std::fmt::Debug + Sync`, the
    /// principal bound is `Some(std::fmt::Debug)`, while the auto-trait bounds
    /// are the set `{Sync}`.
    ///
    /// It is also possible to have a "trivial" trait object that
    /// consists only of auto traits, with no principal - for example,
    /// `dyn Send + Sync`. In that case, the set of auto-trait bounds
    /// is `{Send, Sync}`, while there is no principal. These trait objects
    /// have a "trivial" vtable consisting of just the size, alignment,
    /// and destructor.
    pub fn principal(&self) -> Option<ty::Binder<'tcx, ExistentialTraitRef<'tcx>>> {
        self[0]
            .map_bound(|this| match this {
                ExistentialPredicate::Trait(tr) => Some(tr),
                _ => None,
            })
            .transpose()
    }

    pub fn principal_def_id(&self) -> Option<DefId> {
        self.principal().map(|trait_ref| trait_ref.skip_binder().def_id)
    }

    #[inline]
    pub fn projection_bounds<'a>(
        &'a self,
    ) -> impl Iterator<Item = ty::Binder<'tcx, ExistentialProjection<'tcx>>> + 'a {
        self.iter().filter_map(|predicate| {
            predicate
                .map_bound(|pred| match pred {
                    ExistentialPredicate::Projection(projection) => Some(projection),
                    _ => None,
                })
                .transpose()
        })
    }

    #[inline]
    pub fn auto_traits<'a>(&'a self) -> impl Iterator<Item = DefId> + Captures<'tcx> + 'a {
        self.iter().filter_map(|predicate| match predicate.skip_binder() {
            ExistentialPredicate::AutoTrait(did) => Some(did),
            _ => None,
        })
    }
}

/// A complete reference to a trait. These take numerous guises in syntax,
/// but perhaps the most recognizable form is in a where-clause:
/// ```ignore (illustrative)
/// T: Foo<U>
/// ```
/// This would be represented by a trait-reference where the `DefId` is the
/// `DefId` for the trait `Foo` and the args define `T` as parameter 0,
/// and `U` as parameter 1.
///
/// Trait references also appear in object types like `Foo<U>`, but in
/// that case the `Self` parameter is absent from the generic parameters.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitRef<'tcx> {
    pub def_id: DefId,
    pub args: GenericArgsRef<'tcx>,
    /// This field exists to prevent the creation of `TraitRef` without
    /// calling [`TraitRef::new`].
    pub(super) _use_trait_ref_new_instead: (),
}

impl<'tcx> TraitRef<'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        args: impl IntoIterator<Item: Into<GenericArg<'tcx>>>,
    ) -> Self {
        let args = tcx.check_and_mk_args(trait_def_id, args);
        Self { def_id: trait_def_id, args, _use_trait_ref_new_instead: () }
    }

    pub fn from_lang_item(
        tcx: TyCtxt<'tcx>,
        trait_lang_item: LangItem,
        span: Span,
        args: impl IntoIterator<Item: Into<ty::GenericArg<'tcx>>>,
    ) -> Self {
        let trait_def_id = tcx.require_lang_item(trait_lang_item, Some(span));
        Self::new(tcx, trait_def_id, args)
    }

    pub fn from_method(
        tcx: TyCtxt<'tcx>,
        trait_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> ty::TraitRef<'tcx> {
        let defs = tcx.generics_of(trait_id);
        ty::TraitRef::new(tcx, trait_id, tcx.mk_args(&args[..defs.params.len()]))
    }

    /// Returns a `TraitRef` of the form `P0: Foo<P1..Pn>` where `Pi`
    /// are the parameters defined on trait.
    pub fn identity(tcx: TyCtxt<'tcx>, def_id: DefId) -> TraitRef<'tcx> {
        ty::TraitRef::new(tcx, def_id, GenericArgs::identity_for_item(tcx, def_id))
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        ty::TraitRef::new(
            tcx,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.iter().skip(1)),
        )
    }

    #[inline]
    pub fn self_ty(&self) -> Ty<'tcx> {
        self.args.type_at(0)
    }
}

pub type PolyTraitRef<'tcx> = ty::Binder<'tcx, TraitRef<'tcx>>;

impl<'tcx> PolyTraitRef<'tcx> {
    pub fn self_ty(&self) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.map_bound_ref(|tr| tr.self_ty())
    }

    pub fn def_id(&self) -> DefId {
        self.skip_binder().def_id
    }
}

impl<'tcx> IntoDiagnosticArg for TraitRef<'tcx> {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        self.to_string().into_diagnostic_arg()
    }
}

/// An existential reference to a trait, where `Self` is erased.
/// For example, the trait object `Trait<'a, 'b, X, Y>` is:
/// ```ignore (illustrative)
/// exists T. T: Trait<'a, 'b, X, Y>
/// ```
/// The generic parameters don't include the erased `Self`, only trait
/// type and lifetime parameters (`[X, Y]` and `['a, 'b]` above).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct ExistentialTraitRef<'tcx> {
    pub def_id: DefId,
    pub args: GenericArgsRef<'tcx>,
}

impl<'tcx> ExistentialTraitRef<'tcx> {
    pub fn erase_self_ty(
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
    ) -> ty::ExistentialTraitRef<'tcx> {
        // Assert there is a Self.
        trait_ref.args.type_at(0);

        ty::ExistentialTraitRef {
            def_id: trait_ref.def_id,
            args: tcx.mk_args(&trait_ref.args[1..]),
        }
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ty::TraitRef<'tcx> {
        // otherwise the escaping vars would be captured by the binder
        // debug_assert!(!self_ty.has_escaping_bound_vars());

        ty::TraitRef::new(tcx, self.def_id, [self_ty.into()].into_iter().chain(self.args.iter()))
    }
}

impl<'tcx> IntoDiagnosticArg for ExistentialTraitRef<'tcx> {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        self.to_string().into_diagnostic_arg()
    }
}

pub type PolyExistentialTraitRef<'tcx> = ty::Binder<'tcx, ExistentialTraitRef<'tcx>>;

impl<'tcx> PolyExistentialTraitRef<'tcx> {
    pub fn def_id(&self) -> DefId {
        self.skip_binder().def_id
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ty::PolyTraitRef<'tcx> {
        self.map_bound(|trait_ref| trait_ref.with_self_ty(tcx, self_ty))
    }
}

/// A `ProjectionPredicate` for an `ExistentialTraitRef`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct ExistentialProjection<'tcx> {
    pub def_id: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub term: Term<'tcx>,
}

pub type PolyExistentialProjection<'tcx> = ty::Binder<'tcx, ExistentialProjection<'tcx>>;

impl<'tcx> ExistentialProjection<'tcx> {
    /// Extracts the underlying existential trait reference from this projection.
    /// For example, if this is a projection of `exists T. <T as Iterator>::Item == X`,
    /// then this function would return an `exists T. T: Iterator` existential trait
    /// reference.
    pub fn trait_ref(&self, tcx: TyCtxt<'tcx>) -> ty::ExistentialTraitRef<'tcx> {
        let def_id = tcx.parent(self.def_id);
        let args_count = tcx.generics_of(def_id).count() - 1;
        let args = tcx.mk_args(&self.args[..args_count]);
        ty::ExistentialTraitRef { def_id, args }
    }

    pub fn with_self_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        self_ty: Ty<'tcx>,
    ) -> ty::ProjectionPredicate<'tcx> {
        // otherwise the escaping regions would be captured by the binders
        debug_assert!(!self_ty.has_escaping_bound_vars());

        ty::ProjectionPredicate {
            projection_ty: AliasTy::new(
                tcx,
                self.def_id,
                [self_ty.into()].into_iter().chain(self.args),
            ),
            term: self.term,
        }
    }

    pub fn erase_self_ty(
        tcx: TyCtxt<'tcx>,
        projection_predicate: ty::ProjectionPredicate<'tcx>,
    ) -> Self {
        // Assert there is a Self.
        projection_predicate.projection_ty.args.type_at(0);

        Self {
            def_id: projection_predicate.projection_ty.def_id,
            args: tcx.mk_args(&projection_predicate.projection_ty.args[1..]),
            term: projection_predicate.term,
        }
    }
}

impl<'tcx> PolyExistentialProjection<'tcx> {
    pub fn with_self_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        self_ty: Ty<'tcx>,
    ) -> ty::PolyProjectionPredicate<'tcx> {
        self.map_bound(|p| p.with_self_ty(tcx, self_ty))
    }

    pub fn item_def_id(&self) -> DefId {
        self.skip_binder().def_id
    }
}

impl<'tcx> Clause<'tcx> {
    /// Performs a instantiation suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// instantiation in terms of what happens with bound regions. See
    /// lengthy comment below for details.
    pub fn instantiate_supertrait(
        self,
        tcx: TyCtxt<'tcx>,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    ) -> Clause<'tcx> {
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
        // normal instantiation.
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
        //   The instantiation from the input trait-ref is therefore going to be
        //   `'a => 'x` (where `'x` has a DB index of 1).
        // - The supertrait-ref is `for<'b> Bar1<'a,'b>`, where `'a` is an
        //   early-bound parameter and `'b` is a late-bound parameter with a
        //   DB index of 1.
        // - If we replace `'a` with `'x` from the input, it too will have
        //   a DB index of 1, and thus we'll have `for<'x,'b> Bar1<'x,'b>`
        //   just as we wanted.
        //
        // There is only one catch. If we just apply the instantiation `'a
        // => 'x` to `for<'b> Bar1<'a,'b>`, the instantiation code will
        // adjust the DB index because we instantiating into a binder (it
        // tries to be so smart...) resulting in `for<'x> for<'b>
        // Bar1<'x,'b>` (we have no syntax for this, so use your
        // imagination). Basically the 'x will have DB index of 2 and 'b
        // will have DB index of 1. Not quite what we want. So we apply
        // the instantiation to the *contents* of the trait reference,
        // rather than the trait reference itself (put another way, the
        // instantiation code expects equal binding levels in the values
        // from the instantiation and the value being instantiated into, and
        // this trick achieves that).

        // Working through the second example:
        // trait_ref: for<'x> T: Foo1<'^0.0>; args: [T, '^0.0]
        // predicate: for<'b> Self: Bar1<'a, '^0.0>; args: [Self, 'a, '^0.0]
        // We want to end up with:
        //     for<'x, 'b> T: Bar1<'^0.0, '^0.1>
        // To do this:
        // 1) We must shift all bound vars in predicate by the length
        //    of trait ref's bound vars. So, we would end up with predicate like
        //    Self: Bar1<'a, '^0.1>
        // 2) We can then apply the trait args to this, ending up with
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
        let new = EarlyBinder::bind(shifted_pred).instantiate(tcx, trait_ref.skip_binder().args);
        // 3) ['x] + ['b] -> ['x, 'b]
        let bound_vars =
            tcx.mk_bound_variable_kinds_from_iter(trait_bound_vars.iter().chain(pred_bound_vars));

        // FIXME: Is it really perf sensitive to use reuse_or_mk_predicate here?
        tcx.reuse_or_mk_predicate(
            self.as_predicate(),
            ty::Binder::bind_with_vars(PredicateKind::Clause(new), bound_vars),
        )
        .expect_clause()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: TraitRef<'tcx>,

    /// If polarity is Positive: we are proving that the trait is implemented.
    ///
    /// If polarity is Negative: we are proving that a negative impl of this trait
    /// exists. (Note that coherence also checks whether negative impls of supertraits
    /// exist via a series of predicates.)
    ///
    /// If polarity is Reserved: that's a bug.
    pub polarity: ImplPolarity,
}

pub type PolyTraitPredicate<'tcx> = ty::Binder<'tcx, TraitPredicate<'tcx>>;

impl<'tcx> TraitPredicate<'tcx> {
    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        Self { trait_ref: self.trait_ref.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn def_id(self) -> DefId {
        self.trait_ref.def_id
    }

    pub fn self_ty(self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
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

    #[inline]
    pub fn polarity(self) -> ImplPolarity {
        self.skip_binder().polarity
    }
}

/// `A: B`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct OutlivesPredicate<A, B>(pub A, pub B);
pub type RegionOutlivesPredicate<'tcx> = OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>;
pub type TypeOutlivesPredicate<'tcx> = OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>;
pub type PolyRegionOutlivesPredicate<'tcx> = ty::Binder<'tcx, RegionOutlivesPredicate<'tcx>>;
pub type PolyTypeOutlivesPredicate<'tcx> = ty::Binder<'tcx, TypeOutlivesPredicate<'tcx>>;

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
pub type PolySubtypePredicate<'tcx> = ty::Binder<'tcx, SubtypePredicate<'tcx>>;

/// Encodes that we have to coerce *from* the `a` type to the `b` type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct CoercePredicate<'tcx> {
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}
pub type PolyCoercePredicate<'tcx> = ty::Binder<'tcx, CoercePredicate<'tcx>>;

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

pub type PolyProjectionPredicate<'tcx> = Binder<'tcx, ProjectionPredicate<'tcx>>;

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

    pub fn term(&self) -> Binder<'tcx, Term<'tcx>> {
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

/// Used by the new solver. Unlike a `ProjectionPredicate` this can only be
/// proven by actually normalizing `alias`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct NormalizesTo<'tcx> {
    pub alias: AliasTy<'tcx>,
    pub term: Term<'tcx>,
}

impl<'tcx> NormalizesTo<'tcx> {
    pub fn self_ty(self) -> Ty<'tcx> {
        self.alias.self_ty()
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> NormalizesTo<'tcx> {
        Self { alias: self.alias.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId {
        self.alias.trait_def_id(tcx)
    }

    pub fn def_id(self) -> DefId {
        self.alias.def_id
    }
}

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref)
    }
}

pub trait ToPredicate<'tcx, P = Predicate<'tcx>> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> P;
}

impl<'tcx, T> ToPredicate<'tcx, T> for T {
    fn to_predicate(self, _tcx: TyCtxt<'tcx>) -> T {
        self
    }
}

impl<'tcx> ToPredicate<'tcx> for PredicateKind<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        ty::Binder::dummy(self).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for Binder<'tcx, PredicateKind<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        tcx.mk_predicate(self)
    }
}

impl<'tcx> ToPredicate<'tcx> for ClauseKind<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        tcx.mk_predicate(ty::Binder::dummy(ty::PredicateKind::Clause(self)))
    }
}

impl<'tcx> ToPredicate<'tcx> for Binder<'tcx, ClauseKind<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        tcx.mk_predicate(self.map_bound(ty::PredicateKind::Clause))
    }
}

impl<'tcx> ToPredicate<'tcx> for Clause<'tcx> {
    #[inline(always)]
    fn to_predicate(self, _tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.as_predicate()
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for ClauseKind<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        tcx.mk_predicate(Binder::dummy(ty::PredicateKind::Clause(self))).expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for Binder<'tcx, ClauseKind<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        tcx.mk_predicate(self.map_bound(|clause| ty::PredicateKind::Clause(clause))).expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx> for TraitRef<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        ty::Binder::dummy(self).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, TraitPredicate<'tcx>> for TraitRef<'tcx> {
    #[inline(always)]
    fn to_predicate(self, _tcx: TyCtxt<'tcx>) -> TraitPredicate<'tcx> {
        TraitPredicate { trait_ref: self, polarity: ImplPolarity::Positive }
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for TraitRef<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        let p: Predicate<'tcx> = self.to_predicate(tcx);
        p.expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx> for Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        let pred: PolyTraitPredicate<'tcx> = self.to_predicate(tcx);
        pred.to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        let pred: PolyTraitPredicate<'tcx> = self.to_predicate(tcx);
        pred.to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, PolyTraitPredicate<'tcx>> for Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, _: TyCtxt<'tcx>) -> PolyTraitPredicate<'tcx> {
        self.map_bound(|trait_ref| TraitPredicate {
            trait_ref,
            polarity: ty::ImplPolarity::Positive,
        })
    }
}

impl<'tcx> ToPredicate<'tcx> for TraitPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        PredicateKind::Clause(ClauseKind::Trait(self)).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(ClauseKind::Trait(p))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for TraitPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        let p: Predicate<'tcx> = self.to_predicate(tcx);
        p.expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for PolyTraitPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        let p: Predicate<'tcx> = self.to_predicate(tcx);
        p.expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyRegionOutlivesPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(ClauseKind::RegionOutlives(p))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::TypeOutlives(self))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for ProjectionPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        ty::Binder::dummy(PredicateKind::Clause(ClauseKind::Projection(self))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(ClauseKind::Projection(p))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for ProjectionPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        let p: Predicate<'tcx> = self.to_predicate(tcx);
        p.expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx, Clause<'tcx>> for PolyProjectionPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Clause<'tcx> {
        let p: Predicate<'tcx> = self.to_predicate(tcx);
        p.expect_clause()
    }
}

impl<'tcx> ToPredicate<'tcx> for NormalizesTo<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        PredicateKind::NormalizesTo(self).to_predicate(tcx)
    }
}

impl<'tcx> Predicate<'tcx> {
    pub fn to_opt_poly_trait_pred(self) -> Option<PolyTraitPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Trait(t)) => Some(predicate.rebind(t)),
            PredicateKind::Clause(ClauseKind::Projection(..))
            | PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::NormalizesTo(..)
            | PredicateKind::AliasRelate(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(ClauseKind::RegionOutlives(..))
            | PredicateKind::Clause(ClauseKind::WellFormed(..))
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::Clause(ClauseKind::TypeOutlives(..))
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(..))
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous => None,
        }
    }

    pub fn to_opt_poly_projection_pred(self) -> Option<PolyProjectionPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(ClauseKind::Projection(t)) => Some(predicate.rebind(t)),
            PredicateKind::Clause(ClauseKind::Trait(..))
            | PredicateKind::Clause(ClauseKind::ConstArgHasType(..))
            | PredicateKind::NormalizesTo(..)
            | PredicateKind::AliasRelate(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(ClauseKind::RegionOutlives(..))
            | PredicateKind::Clause(ClauseKind::WellFormed(..))
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::Clause(ClauseKind::TypeOutlives(..))
            | PredicateKind::Clause(ClauseKind::ConstEvaluatable(..))
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous => None,
        }
    }

    /// Matches a `PredicateKind::Clause` and turns it into a `Clause`, otherwise returns `None`.
    pub fn as_clause(self) -> Option<Clause<'tcx>> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Some(self.expect_clause()),
            _ => None,
        }
    }

    /// Assert that the predicate is a clause.
    pub fn expect_clause(self) -> Clause<'tcx> {
        match self.kind().skip_binder() {
            PredicateKind::Clause(..) => Clause(self.0),
            _ => bug!("{self} is not a clause"),
        }
    }
}
