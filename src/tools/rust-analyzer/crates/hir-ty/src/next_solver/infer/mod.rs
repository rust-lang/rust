//! Infer context the next-trait-solver.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

pub use BoundRegionConversionTime::*;
use ena::unify as ut;
use hir_def::GenericParamId;
use opaque_types::{OpaqueHiddenType, OpaqueTypeStorage};
use region_constraints::{RegionConstraintCollector, RegionConstraintStorage};
use rustc_next_trait_solver::solve::SolverDelegateEvalExt;
use rustc_pattern_analysis::Captures;
use rustc_type_ir::{
    ClosureKind, ConstVid, FloatVarValue, FloatVid, GenericArgKind, InferConst, InferTy,
    IntVarValue, IntVid, OutlivesPredicate, RegionVid, TermKind, TyVid, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt, UniverseIndex,
    error::{ExpectedFound, TypeError},
    inherent::{
        Const as _, GenericArg as _, GenericArgs as _, IntoKind, SliceLike, Term as _, Ty as _,
    },
};
use snapshot::undo_log::InferCtxtUndoLogs;
use tracing::{debug, instrument};
use traits::{ObligationCause, PredicateObligations};
use type_variable::TypeVariableOrigin;
use unify_key::{ConstVariableOrigin, ConstVariableValue, ConstVidKey};

use crate::next_solver::{
    ArgOutlivesPredicate, BoundConst, BoundRegion, BoundTy, BoundVarKind, Goal, Predicate,
    SolverContext,
    fold::BoundVarReplacerDelegate,
    infer::{at::ToTrace, select::EvaluationResult, traits::PredicateObligation},
    obligation_ctxt::ObligationCtxt,
};

use super::{
    AliasTerm, Binder, CanonicalQueryInput, CanonicalVarValues, Const, ConstKind, DbInterner,
    ErrorGuaranteed, GenericArg, GenericArgs, OpaqueTypeKey, ParamEnv, PolyCoercePredicate,
    PolyExistentialProjection, PolyExistentialTraitRef, PolyFnSig, PolyRegionOutlivesPredicate,
    PolySubtypePredicate, Region, SolverDefId, SubtypePredicate, Term, TraitRef, Ty, TyKind,
    TypingMode,
};

pub mod at;
pub mod canonical;
mod context;
pub mod opaque_types;
mod outlives;
pub mod region_constraints;
pub mod relate;
pub mod resolve;
pub(crate) mod select;
pub(crate) mod snapshot;
pub(crate) mod traits;
mod type_variable;
mod unify_key;

/// `InferOk<'db, ()>` is used a lot. It may seem like a useless wrapper
/// around `PredicateObligations`, but it has one important property:
/// because `InferOk` is marked with `#[must_use]`, if you have a method
/// `InferCtxt::f` that returns `InferResult<()>` and you call it with
/// `infcx.f()?;` you'll get a warning about the obligations being discarded
/// without use, which is probably unintentional and has been a source of bugs
/// in the past.
#[must_use]
#[derive(Debug)]
pub struct InferOk<'db, T> {
    pub value: T,
    pub obligations: PredicateObligations<'db>,
}
pub type InferResult<'db, T> = Result<InferOk<'db, T>, TypeError<DbInterner<'db>>>;

pub(crate) type UnificationTable<'a, 'db, T> = ut::UnificationTable<
    ut::InPlace<T, &'a mut ut::UnificationStorage<T>, &'a mut InferCtxtUndoLogs<'db>>,
>;

fn iter_idx_range<T: From<u32> + Into<u32>>(range: Range<T>) -> impl Iterator<Item = T> {
    (range.start.into()..range.end.into()).map(Into::into)
}

/// This type contains all the things within `InferCtxt` that sit within a
/// `RefCell` and are involved with taking/rolling back snapshots. Snapshot
/// operations are hot enough that we want only one call to `borrow_mut` per
/// call to `start_snapshot` and `rollback_to`.
#[derive(Clone)]
pub struct InferCtxtInner<'db> {
    pub(crate) undo_log: InferCtxtUndoLogs<'db>,

    /// We instantiate `UnificationTable` with `bounds<Ty>` because the types
    /// that might instantiate a general type variable have an order,
    /// represented by its upper and lower bounds.
    pub(crate) type_variable_storage: type_variable::TypeVariableStorage<'db>,

    /// Map from const parameter variable to the kind of const it represents.
    pub(crate) const_unification_storage: ut::UnificationTableStorage<ConstVidKey<'db>>,

    /// Map from integral variable to the kind of integer it represents.
    pub(crate) int_unification_storage: ut::UnificationTableStorage<IntVid>,

    /// Map from floating variable to the kind of float it represents.
    pub(crate) float_unification_storage: ut::UnificationTableStorage<FloatVid>,

    /// Tracks the set of region variables and the constraints between them.
    ///
    /// This is initially `Some(_)` but when
    /// `resolve_regions_and_report_errors` is invoked, this gets set to `None`
    /// -- further attempts to perform unification, etc., may fail if new
    /// region constraints would've been added.
    pub(crate) region_constraint_storage: Option<RegionConstraintStorage<'db>>,

    /// A set of constraints that regionck must validate.
    ///
    /// Each constraint has the form `T:'a`, meaning "some type `T` must
    /// outlive the lifetime 'a". These constraints derive from
    /// instantiated type parameters. So if you had a struct defined
    /// like the following:
    /// ```ignore (illustrative)
    /// struct Foo<T: 'static> { ... }
    /// ```
    /// In some expression `let x = Foo { ... }`, it will
    /// instantiate the type parameter `T` with a fresh type `$0`. At
    /// the same time, it will record a region obligation of
    /// `$0: 'static`. This will get checked later by regionck. (We
    /// can't generally check these things right away because we have
    /// to wait until types are resolved.)
    ///
    /// These are stored in a map keyed to the id of the innermost
    /// enclosing fn body / static initializer expression. This is
    /// because the location where the obligation was incurred can be
    /// relevant with respect to which sublifetime assumptions are in
    /// place. The reason that we store under the fn-id, and not
    /// something more fine-grained, is so that it is easier for
    /// regionck to be sure that it has found *all* the region
    /// obligations (otherwise, it's easy to fail to walk to a
    /// particular node-id).
    ///
    /// Before running `resolve_regions_and_report_errors`, the creator
    /// of the inference context is expected to invoke
    /// [`InferCtxt::process_registered_region_obligations`]
    /// for each body-id in this map, which will process the
    /// obligations within. This is expected to be done 'late enough'
    /// that all type inference variables have been bound and so forth.
    pub(crate) region_obligations: Vec<TypeOutlivesConstraint<'db>>,

    /// The outlives bounds that we assume must hold about placeholders that
    /// come from instantiating the binder of coroutine-witnesses. These bounds
    /// are deduced from the well-formedness of the witness's types, and are
    /// necessary because of the way we anonymize the regions in a coroutine,
    /// which may cause types to no longer be considered well-formed.
    region_assumptions: Vec<ArgOutlivesPredicate<'db>>,

    /// Caches for opaque type inference.
    pub(crate) opaque_type_storage: OpaqueTypeStorage<'db>,
}

impl<'db> InferCtxtInner<'db> {
    fn new() -> InferCtxtInner<'db> {
        InferCtxtInner {
            undo_log: InferCtxtUndoLogs::default(),

            type_variable_storage: Default::default(),
            const_unification_storage: Default::default(),
            int_unification_storage: Default::default(),
            float_unification_storage: Default::default(),
            region_constraint_storage: Some(Default::default()),
            region_obligations: vec![],
            region_assumptions: Default::default(),
            opaque_type_storage: Default::default(),
        }
    }

    #[inline]
    pub fn region_obligations(&self) -> &[TypeOutlivesConstraint<'db>] {
        &self.region_obligations
    }

    #[inline]
    fn try_type_variables_probe_ref(
        &self,
        vid: TyVid,
    ) -> Option<&type_variable::TypeVariableValue<'db>> {
        // Uses a read-only view of the unification table, this way we don't
        // need an undo log.
        self.type_variable_storage.eq_relations_ref().try_probe_value(vid)
    }

    #[inline]
    fn type_variables(&mut self) -> type_variable::TypeVariableTable<'_, 'db> {
        self.type_variable_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    pub(crate) fn opaque_types(&mut self) -> opaque_types::OpaqueTypeTable<'_, 'db> {
        self.opaque_type_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    pub(crate) fn int_unification_table(&mut self) -> UnificationTable<'_, 'db, IntVid> {
        tracing::debug!(?self.int_unification_storage);
        self.int_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    pub(crate) fn float_unification_table(&mut self) -> UnificationTable<'_, 'db, FloatVid> {
        self.float_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    fn const_unification_table(&mut self) -> UnificationTable<'_, 'db, ConstVidKey<'db>> {
        self.const_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    pub fn unwrap_region_constraints(&mut self) -> RegionConstraintCollector<'db, '_> {
        self.region_constraint_storage
            .as_mut()
            .expect("region constraints already solved")
            .with_log(&mut self.undo_log)
    }
}

#[derive(Clone)]
pub struct InferCtxt<'db> {
    pub interner: DbInterner<'db>,

    /// The mode of this inference context, see the struct documentation
    /// for more details.
    typing_mode: TypingMode<'db>,

    pub inner: RefCell<InferCtxtInner<'db>>,

    /// When an error occurs, we want to avoid reporting "derived"
    /// errors that are due to this original failure. We have this
    /// flag that one can set whenever one creates a type-error that
    /// is due to an error in a prior pass.
    ///
    /// Don't read this flag directly, call `is_tainted_by_errors()`
    /// and `set_tainted_by_errors()`.
    tainted_by_errors: Cell<Option<ErrorGuaranteed>>,

    /// What is the innermost universe we have created? Starts out as
    /// `UniverseIndex::root()` but grows from there as we enter
    /// universal quantifiers.
    ///
    /// N.B., at present, we exclude the universal quantifiers on the
    /// item we are type-checking, and just consider those names as
    /// part of the root universe. So this would only get incremented
    /// when we enter into a higher-ranked (`for<..>`) type or trait
    /// bound.
    universe: Cell<UniverseIndex>,
}

/// See the `error_reporting` module for more details.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValuePairs<'db> {
    Regions(ExpectedFound<Region<'db>>),
    Terms(ExpectedFound<Term<'db>>),
    Aliases(ExpectedFound<AliasTerm<'db>>),
    TraitRefs(ExpectedFound<TraitRef<'db>>),
    PolySigs(ExpectedFound<PolyFnSig<'db>>),
    ExistentialTraitRef(ExpectedFound<PolyExistentialTraitRef<'db>>),
    ExistentialProjection(ExpectedFound<PolyExistentialProjection<'db>>),
}

impl<'db> ValuePairs<'db> {
    pub fn ty(&self) -> Option<(Ty<'db>, Ty<'db>)> {
        if let ValuePairs::Terms(ExpectedFound { expected, found }) = self
            && let Some(expected) = expected.as_type()
            && let Some(found) = found.as_type()
        {
            return Some((expected, found));
        }
        None
    }
}

/// The trace designates the path through inference that we took to
/// encounter an error or subtyping constraint.
///
/// See the `error_reporting` module for more details.
#[derive(Clone, Debug)]
pub struct TypeTrace<'db> {
    pub cause: ObligationCause,
    pub values: ValuePairs<'db>,
}

/// Times when we replace bound regions with existentials:
#[derive(Clone, Copy, Debug)]
pub enum BoundRegionConversionTime {
    /// when a fn is called
    FnCall,

    /// when two higher-ranked types are compared
    HigherRankedType,

    /// when projecting an associated type
    AssocTypeProjection(SolverDefId),
}

#[derive(Copy, Clone, Debug)]
pub struct FixupError {
    unresolved: TyOrConstInferVar,
}

impl fmt::Display for FixupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TyOrConstInferVar::*;

        match self.unresolved {
            TyInt(_) => write!(
                f,
                "cannot determine the type of this integer; \
                 add a suffix to specify the type explicitly"
            ),
            TyFloat(_) => write!(
                f,
                "cannot determine the type of this number; \
                 add a suffix to specify the type explicitly"
            ),
            Ty(_) => write!(f, "unconstrained type"),
            Const(_) => write!(f, "unconstrained const value"),
        }
    }
}

/// See the `region_obligations` field for more information.
#[derive(Clone, Debug)]
pub struct TypeOutlivesConstraint<'db> {
    pub sub_region: Region<'db>,
    pub sup_type: Ty<'db>,
}

/// Used to configure inference contexts before their creation.
pub struct InferCtxtBuilder<'db> {
    interner: DbInterner<'db>,
}

pub trait DbInternerInferExt<'db> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'db>;
}

impl<'db> DbInternerInferExt<'db> for DbInterner<'db> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'db> {
        InferCtxtBuilder { interner: self }
    }
}

impl<'db> InferCtxtBuilder<'db> {
    /// Given a canonical value `C` as a starting point, create an
    /// inference context that contains each of the bound values
    /// within instantiated as a fresh variable. The `f` closure is
    /// invoked with the new infcx, along with the instantiated value
    /// `V` and a instantiation `S`. This instantiation `S` maps from
    /// the bound values in `C` to their instantiated values in `V`
    /// (in other words, `S(C) = V`).
    pub fn build_with_canonical<T>(
        mut self,
        input: &CanonicalQueryInput<'db, T>,
    ) -> (InferCtxt<'db>, T, CanonicalVarValues<'db>)
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        let infcx = self.build(input.typing_mode);
        let (value, args) = infcx.instantiate_canonical(&input.canonical);
        (infcx, value, args)
    }

    pub fn build(&mut self, typing_mode: TypingMode<'db>) -> InferCtxt<'db> {
        let InferCtxtBuilder { interner } = *self;
        InferCtxt {
            interner,
            typing_mode,
            inner: RefCell::new(InferCtxtInner::new()),
            tainted_by_errors: Cell::new(None),
            universe: Cell::new(UniverseIndex::ROOT),
        }
    }
}

impl<'db> InferOk<'db, ()> {
    pub fn into_obligations(self) -> PredicateObligations<'db> {
        self.obligations
    }
}

impl<'db> InferCtxt<'db> {
    #[inline(always)]
    pub fn typing_mode(&self) -> TypingMode<'db> {
        self.typing_mode
    }

    #[inline(always)]
    pub fn typing_mode_unchecked(&self) -> TypingMode<'db> {
        self.typing_mode
    }

    /// Evaluates whether the predicate can be satisfied (by any means)
    /// in the given `ParamEnv`.
    pub fn predicate_may_hold(&self, obligation: &PredicateObligation<'db>) -> bool {
        self.evaluate_obligation(obligation).may_apply()
    }

    /// See the comment on `GeneralAutoderef::overloaded_deref_ty`
    /// for more details.
    pub fn predicate_may_hold_opaque_types_jank(
        &self,
        obligation: &PredicateObligation<'db>,
    ) -> bool {
        <&SolverContext<'db>>::from(self).root_goal_may_hold_opaque_types_jank(Goal::new(
            self.interner,
            obligation.param_env,
            obligation.predicate,
        ))
    }

    pub(crate) fn insert_type_vars<T>(&self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        struct Folder<'a, 'db> {
            infcx: &'a InferCtxt<'db>,
        }
        impl<'db> TypeFolder<DbInterner<'db>> for Folder<'_, 'db> {
            fn cx(&self) -> DbInterner<'db> {
                self.infcx.interner
            }

            fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
                if !ty.references_error() {
                    return ty;
                }

                if ty.is_ty_error() { self.infcx.next_ty_var() } else { ty.super_fold_with(self) }
            }

            fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
                if !ct.references_error() {
                    return ct;
                }

                if ct.is_ct_error() {
                    self.infcx.next_const_var()
                } else {
                    ct.super_fold_with(self)
                }
            }

            fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
                if r.is_error() { self.infcx.next_region_var() } else { r }
            }
        }

        ty.fold_with(&mut Folder { infcx: self })
    }

    /// Evaluates whether the predicate can be satisfied in the given
    /// `ParamEnv`, and returns `false` if not certain. However, this is
    /// not entirely accurate if inference variables are involved.
    ///
    /// This version may conservatively fail when outlives obligations
    /// are required. Therefore, this version should only be used for
    /// optimizations or diagnostics and be treated as if it can always
    /// return `false`.
    ///
    /// # Example
    ///
    /// ```
    /// # #![allow(dead_code)]
    /// trait Trait {}
    ///
    /// fn check<T: Trait>() {}
    ///
    /// fn foo<T: 'static>()
    /// where
    ///     &'static T: Trait,
    /// {
    ///     // Evaluating `&'?0 T: Trait` adds a `'?0: 'static` outlives obligation,
    ///     // which means that `predicate_must_hold_considering_regions` will return
    ///     // `false`.
    ///     check::<&'_ T>();
    /// }
    /// ```
    #[expect(dead_code, reason = "this is used in rustc")]
    fn predicate_must_hold_considering_regions(
        &self,
        obligation: &PredicateObligation<'db>,
    ) -> bool {
        self.evaluate_obligation(obligation).must_apply_considering_regions()
    }

    /// Evaluates whether the predicate can be satisfied in the given
    /// `ParamEnv`, and returns `false` if not certain. However, this is
    /// not entirely accurate if inference variables are involved.
    ///
    /// This version ignores all outlives constraints.
    #[expect(dead_code, reason = "this is used in rustc")]
    fn predicate_must_hold_modulo_regions(&self, obligation: &PredicateObligation<'db>) -> bool {
        self.evaluate_obligation(obligation).must_apply_modulo_regions()
    }

    /// Evaluate a given predicate, capturing overflow and propagating it back.
    fn evaluate_obligation(&self, obligation: &PredicateObligation<'db>) -> EvaluationResult {
        self.probe(|snapshot| {
            let mut ocx = ObligationCtxt::new(self);
            ocx.register_obligation(obligation.clone());
            let mut result = EvaluationResult::EvaluatedToOk;
            for error in ocx.evaluate_obligations_error_on_ambiguity() {
                if error.is_true_error() {
                    return EvaluationResult::EvaluatedToErr;
                } else {
                    result = result.max(EvaluationResult::EvaluatedToAmbig);
                }
            }
            if self.opaque_types_added_in_snapshot(snapshot) {
                result = result.max(EvaluationResult::EvaluatedToOkModuloOpaqueTypes);
            } else if self.region_constraints_added_in_snapshot(snapshot) {
                result = result.max(EvaluationResult::EvaluatedToOkModuloRegions);
            }
            result
        })
    }

    pub fn can_eq<T: ToTrace<'db>>(&self, param_env: ParamEnv<'db>, a: T, b: T) -> bool {
        self.probe(|_| {
            let mut ocx = ObligationCtxt::new(self);
            let Ok(()) = ocx.eq(&ObligationCause::dummy(), param_env, a, b) else {
                return false;
            };
            ocx.try_evaluate_obligations().is_empty()
        })
    }

    /// See the comment on `GeneralAutoderef::overloaded_deref_ty`
    /// for more details.
    pub fn goal_may_hold_opaque_types_jank(&self, goal: Goal<'db, Predicate<'db>>) -> bool {
        <&SolverContext<'db>>::from(self).root_goal_may_hold_opaque_types_jank(goal)
    }

    pub fn type_is_copy_modulo_regions(&self, param_env: ParamEnv<'db>, ty: Ty<'db>) -> bool {
        let ty = self.resolve_vars_if_possible(ty);

        let Some(copy_def_id) = self.interner.lang_items().Copy else {
            return false;
        };

        // This can get called from typeck (by euv), and `moves_by_default`
        // rightly refuses to work with inference variables, but
        // moves_by_default has a cache, which we want to use in other
        // cases.
        traits::type_known_to_meet_bound_modulo_regions(self, param_env, ty, copy_def_id)
    }

    pub fn unresolved_variables(&self) -> Vec<Ty<'db>> {
        let mut inner = self.inner.borrow_mut();
        let mut vars: Vec<Ty<'db>> = inner
            .type_variables()
            .unresolved_variables()
            .into_iter()
            .map(|t| Ty::new_var(self.interner, t))
            .collect();
        vars.extend(
            (0..inner.int_unification_table().len())
                .map(IntVid::from_usize)
                .filter(|&vid| inner.int_unification_table().probe_value(vid).is_unknown())
                .map(|v| Ty::new_int_var(self.interner, v)),
        );
        vars.extend(
            (0..inner.float_unification_table().len())
                .map(FloatVid::from_usize)
                .filter(|&vid| inner.float_unification_table().probe_value(vid).is_unknown())
                .map(|v| Ty::new_float_var(self.interner, v)),
        );
        vars
    }

    #[instrument(skip(self), level = "debug")]
    pub fn sub_regions(&self, a: Region<'db>, b: Region<'db>) {
        self.inner.borrow_mut().unwrap_region_constraints().make_subregion(a, b);
    }

    /// Processes a `Coerce` predicate from the fulfillment context.
    /// This is NOT the preferred way to handle coercion, which is to
    /// invoke `FnCtxt::coerce` or a similar method (see `coercion.rs`).
    ///
    /// This method here is actually a fallback that winds up being
    /// invoked when `FnCtxt::coerce` encounters unresolved type variables
    /// and records a coercion predicate. Presently, this method is equivalent
    /// to `subtype_predicate` -- that is, "coercing" `a` to `b` winds up
    /// actually requiring `a <: b`. This is of course a valid coercion,
    /// but it's not as flexible as `FnCtxt::coerce` would be.
    ///
    /// (We may refactor this in the future, but there are a number of
    /// practical obstacles. Among other things, `FnCtxt::coerce` presently
    /// records adjustments that are required on the HIR in order to perform
    /// the coercion, and we don't currently have a way to manage that.)
    pub fn coerce_predicate(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        predicate: PolyCoercePredicate<'db>,
    ) -> Result<InferResult<'db, ()>, (TyVid, TyVid)> {
        let subtype_predicate = predicate.map_bound(|p| SubtypePredicate {
            a_is_expected: false, // when coercing from `a` to `b`, `b` is expected
            a: p.a,
            b: p.b,
        });
        self.subtype_predicate(cause, param_env, subtype_predicate)
    }

    pub fn subtype_predicate(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        predicate: PolySubtypePredicate<'db>,
    ) -> Result<InferResult<'db, ()>, (TyVid, TyVid)> {
        // Check for two unresolved inference variables, in which case we can
        // make no progress. This is partly a micro-optimization, but it's
        // also an opportunity to "sub-unify" the variables. This isn't
        // *necessary* to prevent cycles, because they would eventually be sub-unified
        // anyhow during generalization, but it helps with diagnostics (we can detect
        // earlier that they are sub-unified).
        //
        // Note that we can just skip the binders here because
        // type variables can't (at present, at
        // least) capture any of the things bound by this binder.
        //
        // Note that this sub here is not just for diagnostics - it has semantic
        // effects as well.
        let r_a = self.shallow_resolve(predicate.skip_binder().a);
        let r_b = self.shallow_resolve(predicate.skip_binder().b);
        match (r_a.kind(), r_b.kind()) {
            (TyKind::Infer(InferTy::TyVar(a_vid)), TyKind::Infer(InferTy::TyVar(b_vid))) => {
                return Err((a_vid, b_vid));
            }
            _ => {}
        }

        self.enter_forall(predicate, |SubtypePredicate { a_is_expected, a, b }| {
            if a_is_expected {
                Ok(self.at(cause, param_env).sub(a, b))
            } else {
                Ok(self.at(cause, param_env).sup(b, a))
            }
        })
    }

    pub fn region_outlives_predicate(
        &self,
        _cause: &traits::ObligationCause,
        predicate: PolyRegionOutlivesPredicate<'db>,
    ) {
        self.enter_forall(predicate, |OutlivesPredicate(r_a, r_b)| {
            self.sub_regions(r_b, r_a); // `b : a` ==> `a <= b`
        })
    }

    /// Number of type variables created so far.
    pub fn num_ty_vars(&self) -> usize {
        self.inner.borrow_mut().type_variables().num_vars()
    }

    pub fn next_var_for_param(&self, id: GenericParamId) -> GenericArg<'db> {
        match id {
            GenericParamId::TypeParamId(_) => self.next_ty_var().into(),
            GenericParamId::ConstParamId(_) => self.next_const_var().into(),
            GenericParamId::LifetimeParamId(_) => self.next_region_var().into(),
        }
    }

    pub fn next_ty_var(&self) -> Ty<'db> {
        self.next_ty_var_with_origin(TypeVariableOrigin { param_def_id: None })
    }

    pub fn next_ty_vid(&self) -> TyVid {
        self.inner
            .borrow_mut()
            .type_variables()
            .new_var(self.universe(), TypeVariableOrigin { param_def_id: None })
    }

    pub fn next_ty_var_with_origin(&self, origin: TypeVariableOrigin) -> Ty<'db> {
        let vid = self.inner.borrow_mut().type_variables().new_var(self.universe(), origin);
        Ty::new_var(self.interner, vid)
    }

    pub fn next_ty_var_id_in_universe(&self, universe: UniverseIndex) -> TyVid {
        let origin = TypeVariableOrigin { param_def_id: None };
        self.inner.borrow_mut().type_variables().new_var(universe, origin)
    }

    pub fn next_ty_var_in_universe(&self, universe: UniverseIndex) -> Ty<'db> {
        let vid = self.next_ty_var_id_in_universe(universe);
        Ty::new_var(self.interner, vid)
    }

    pub fn next_const_var(&self) -> Const<'db> {
        self.next_const_var_with_origin(ConstVariableOrigin {})
    }

    pub fn next_const_vid(&self) -> ConstVid {
        self.inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown {
                origin: ConstVariableOrigin {},
                universe: self.universe(),
            })
            .vid
    }

    pub fn next_const_var_with_origin(&self, origin: ConstVariableOrigin) -> Const<'db> {
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown { origin, universe: self.universe() })
            .vid;
        Const::new_var(self.interner, vid)
    }

    pub fn next_const_var_in_universe(&self, universe: UniverseIndex) -> Const<'db> {
        let origin = ConstVariableOrigin {};
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown { origin, universe })
            .vid;
        Const::new_var(self.interner, vid)
    }

    pub fn next_int_var(&self) -> Ty<'db> {
        let next_int_var_id =
            self.inner.borrow_mut().int_unification_table().new_key(IntVarValue::Unknown);
        Ty::new_int_var(self.interner, next_int_var_id)
    }

    pub fn next_int_vid(&self) -> IntVid {
        self.inner.borrow_mut().int_unification_table().new_key(IntVarValue::Unknown)
    }

    pub fn next_float_var(&self) -> Ty<'db> {
        Ty::new_float_var(self.interner, self.next_float_vid())
    }

    pub fn next_float_vid(&self) -> FloatVid {
        self.inner.borrow_mut().float_unification_table().new_key(FloatVarValue::Unknown)
    }

    /// Creates a fresh region variable with the next available index.
    /// The variable will be created in the maximum universe created
    /// thus far, allowing it to name any region created thus far.
    pub fn next_region_var(&self) -> Region<'db> {
        self.next_region_var_in_universe(self.universe())
    }

    pub fn next_region_vid(&self) -> RegionVid {
        self.inner.borrow_mut().unwrap_region_constraints().new_region_var(self.universe())
    }

    /// Creates a fresh region variable with the next available index
    /// in the given universe; typically, you can use
    /// `next_region_var` and just use the maximal universe.
    pub fn next_region_var_in_universe(&self, universe: UniverseIndex) -> Region<'db> {
        let region_var =
            self.inner.borrow_mut().unwrap_region_constraints().new_region_var(universe);
        Region::new_var(self.interner, region_var)
    }

    pub fn next_term_var_of_kind(&self, term: Term<'db>) -> Term<'db> {
        match term.kind() {
            TermKind::Ty(_) => self.next_ty_var().into(),
            TermKind::Const(_) => self.next_const_var().into(),
        }
    }

    /// Return the universe that the region `r` was created in. For
    /// most regions (e.g., `'static`, named regions from the user,
    /// etc) this is the root universe U0. For inference variables or
    /// placeholders, however, it will return the universe which they
    /// are associated.
    pub fn universe_of_region(&self, r: Region<'db>) -> UniverseIndex {
        self.inner.borrow_mut().unwrap_region_constraints().universe(r)
    }

    /// Number of region variables created so far.
    pub fn num_region_vars(&self) -> usize {
        self.inner.borrow_mut().unwrap_region_constraints().num_region_vars()
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    #[instrument(skip(self), level = "debug")]
    pub fn next_nll_region_var(&self) -> Region<'db> {
        self.next_region_var()
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    #[instrument(skip(self), level = "debug")]
    pub fn next_nll_region_var_in_universe(&self, universe: UniverseIndex) -> Region<'db> {
        self.next_region_var_in_universe(universe)
    }

    fn var_for_def(&self, id: GenericParamId) -> GenericArg<'db> {
        match id {
            GenericParamId::LifetimeParamId(_) => {
                // Create a region inference variable for the given
                // region parameter definition.
                self.next_region_var().into()
            }
            GenericParamId::TypeParamId(_) => {
                // Create a type inference variable for the given
                // type parameter definition. The generic parameters are
                // for actual parameters that may be referred to by
                // the default of this type parameter, if it exists.
                // e.g., `struct Foo<A, B, C = (A, B)>(...);` when
                // used in a path such as `Foo::<T, U>::new()` will
                // use an inference variable for `C` with `[T, U]`
                // as the generic parameters for the default, `(T, U)`.
                let ty_var_id = self
                    .inner
                    .borrow_mut()
                    .type_variables()
                    .new_var(self.universe(), TypeVariableOrigin { param_def_id: None });

                Ty::new_var(self.interner, ty_var_id).into()
            }
            GenericParamId::ConstParamId(_) => {
                let origin = ConstVariableOrigin {};
                let const_var_id = self
                    .inner
                    .borrow_mut()
                    .const_unification_table()
                    .new_key(ConstVariableValue::Unknown { origin, universe: self.universe() })
                    .vid;
                Const::new_var(self.interner, const_var_id).into()
            }
        }
    }

    /// Given a set of generics defined on a type or impl, returns the generic parameters mapping
    /// each type/region parameter to a fresh inference variable.
    pub fn fresh_args_for_item(&self, def_id: SolverDefId) -> GenericArgs<'db> {
        GenericArgs::for_item(self.interner, def_id, |_index, kind, _| self.var_for_def(kind))
    }

    /// Like `fresh_args_for_item()`, but first uses the args from `first`.
    pub fn fill_rest_fresh_args(
        &self,
        def_id: SolverDefId,
        first: impl IntoIterator<Item = GenericArg<'db>>,
    ) -> GenericArgs<'db> {
        GenericArgs::fill_rest(self.interner, def_id, first, |_index, kind, _| {
            self.var_for_def(kind)
        })
    }

    /// Returns `true` if errors have been reported since this infcx was
    /// created. This is sometimes used as a heuristic to skip
    /// reporting errors that often occur as a result of earlier
    /// errors, but where it's hard to be 100% sure (e.g., unresolved
    /// inference variables, regionck errors).
    #[must_use = "this method does not have any side effects"]
    pub fn tainted_by_errors(&self) -> Option<ErrorGuaranteed> {
        self.tainted_by_errors.get()
    }

    /// Set the "tainted by errors" flag to true. We call this when we
    /// observe an error from a prior pass.
    pub fn set_tainted_by_errors(&self, e: ErrorGuaranteed) {
        debug!("set_tainted_by_errors(ErrorGuaranteed)");
        self.tainted_by_errors.set(Some(e));
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn take_opaque_types(&self) -> Vec<(OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        self.inner.borrow_mut().opaque_type_storage.take_opaque_types().collect()
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn clone_opaque_types(&self) -> Vec<(OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        self.inner.borrow_mut().opaque_type_storage.iter_opaque_types().collect()
    }

    pub fn has_opaques_with_sub_unified_hidden_type(&self, ty_vid: TyVid) -> bool {
        let ty_sub_vid = self.sub_unification_table_root_var(ty_vid);
        let inner = &mut *self.inner.borrow_mut();
        let mut type_variables = inner.type_variable_storage.with_log(&mut inner.undo_log);
        inner.opaque_type_storage.iter_opaque_types().any(|(_, hidden_ty)| {
            if let TyKind::Infer(InferTy::TyVar(hidden_vid)) = hidden_ty.ty.kind() {
                let opaque_sub_vid = type_variables.sub_unification_table_root_var(hidden_vid);
                if opaque_sub_vid == ty_sub_vid {
                    return true;
                }
            }

            false
        })
    }

    #[inline(always)]
    pub fn can_define_opaque_ty(&self, id: impl Into<SolverDefId>) -> bool {
        match self.typing_mode_unchecked() {
            TypingMode::Analysis { defining_opaque_types_and_generators } => {
                defining_opaque_types_and_generators.contains(&id.into())
            }
            TypingMode::Coherence | TypingMode::PostAnalysis => false,
            TypingMode::Borrowck { defining_opaque_types: _ } => unimplemented!(),
            TypingMode::PostBorrowckAnalysis { defined_opaque_types: _ } => unimplemented!(),
        }
    }

    /// If `TyVar(vid)` resolves to a type, return that type. Else, return the
    /// universe index of `TyVar(vid)`.
    pub fn probe_ty_var(&self, vid: TyVid) -> Result<Ty<'db>, UniverseIndex> {
        use self::type_variable::TypeVariableValue;

        match self.inner.borrow_mut().type_variables().probe(vid) {
            TypeVariableValue::Known { value } => Ok(value),
            TypeVariableValue::Unknown { universe } => Err(universe),
        }
    }

    pub fn shallow_resolve(&self, ty: Ty<'db>) -> Ty<'db> {
        if let TyKind::Infer(v) = ty.kind() {
            match v {
                InferTy::TyVar(v) => {
                    // Not entirely obvious: if `typ` is a type variable,
                    // it can be resolved to an int/float variable, which
                    // can then be recursively resolved, hence the
                    // recursion. Note though that we prevent type
                    // variables from unifying to other type variables
                    // directly (though they may be embedded
                    // structurally), and we prevent cycles in any case,
                    // so this recursion should always be of very limited
                    // depth.
                    //
                    // Note: if these two lines are combined into one we get
                    // dynamic borrow errors on `self.inner`.
                    let known = self.inner.borrow_mut().type_variables().probe(v).known();
                    known.map_or(ty, |t| self.shallow_resolve(t))
                }

                InferTy::IntVar(v) => {
                    match self.inner.borrow_mut().int_unification_table().probe_value(v) {
                        IntVarValue::IntType(ty) => Ty::new_int(self.interner, ty),
                        IntVarValue::UintType(ty) => Ty::new_uint(self.interner, ty),
                        IntVarValue::Unknown => ty,
                    }
                }

                InferTy::FloatVar(v) => {
                    match self.inner.borrow_mut().float_unification_table().probe_value(v) {
                        FloatVarValue::Known(ty) => Ty::new_float(self.interner, ty),
                        FloatVarValue::Unknown => ty,
                    }
                }

                InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_) => ty,
            }
        } else {
            ty
        }
    }

    pub fn shallow_resolve_const(&self, ct: Const<'db>) -> Const<'db> {
        match ct.kind() {
            ConstKind::Infer(infer_ct) => match infer_ct {
                InferConst::Var(vid) => self
                    .inner
                    .borrow_mut()
                    .const_unification_table()
                    .probe_value(vid)
                    .known()
                    .unwrap_or(ct),
                InferConst::Fresh(_) => ct,
            },
            ConstKind::Param(_)
            | ConstKind::Bound(_, _)
            | ConstKind::Placeholder(_)
            | ConstKind::Unevaluated(_)
            | ConstKind::Value(_)
            | ConstKind::Error(_)
            | ConstKind::Expr(_) => ct,
        }
    }

    pub fn root_var(&self, var: TyVid) -> TyVid {
        self.inner.borrow_mut().type_variables().root_var(var)
    }

    pub fn root_const_var(&self, var: ConstVid) -> ConstVid {
        self.inner.borrow_mut().const_unification_table().find(var).vid
    }

    /// Resolves an int var to a rigid int type, if it was constrained to one,
    /// or else the root int var in the unification table.
    pub fn opportunistic_resolve_int_var(&self, vid: IntVid) -> Ty<'db> {
        let mut inner = self.inner.borrow_mut();
        let value = inner.int_unification_table().probe_value(vid);
        match value {
            IntVarValue::IntType(ty) => Ty::new_int(self.interner, ty),
            IntVarValue::UintType(ty) => Ty::new_uint(self.interner, ty),
            IntVarValue::Unknown => {
                Ty::new_int_var(self.interner, inner.int_unification_table().find(vid))
            }
        }
    }

    pub fn resolve_int_var(&self, vid: IntVid) -> Option<Ty<'db>> {
        let mut inner = self.inner.borrow_mut();
        let value = inner.int_unification_table().probe_value(vid);
        match value {
            IntVarValue::IntType(ty) => Some(Ty::new_int(self.interner, ty)),
            IntVarValue::UintType(ty) => Some(Ty::new_uint(self.interner, ty)),
            IntVarValue::Unknown => None,
        }
    }

    /// Resolves a float var to a rigid int type, if it was constrained to one,
    /// or else the root float var in the unification table.
    pub fn opportunistic_resolve_float_var(&self, vid: FloatVid) -> Ty<'db> {
        let mut inner = self.inner.borrow_mut();
        let value = inner.float_unification_table().probe_value(vid);
        match value {
            FloatVarValue::Known(ty) => Ty::new_float(self.interner, ty),
            FloatVarValue::Unknown => {
                Ty::new_float_var(self.interner, inner.float_unification_table().find(vid))
            }
        }
    }

    pub fn resolve_float_var(&self, vid: FloatVid) -> Option<Ty<'db>> {
        let mut inner = self.inner.borrow_mut();
        let value = inner.float_unification_table().probe_value(vid);
        match value {
            FloatVarValue::Known(ty) => Some(Ty::new_float(self.interner, ty)),
            FloatVarValue::Unknown => None,
        }
    }

    /// Where possible, replaces type/const variables in
    /// `value` with their final value. Note that region variables
    /// are unaffected. If a type/const variable has not been unified, it
    /// is left as is. This is an idempotent operation that does
    /// not affect inference state in any way and so you can do it
    /// at will.
    pub fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        if let Err(guar) = value.error_reported() {
            self.set_tainted_by_errors(guar);
        }
        if !value.has_non_region_infer() {
            return value;
        }
        let mut r = resolve::OpportunisticVarResolver::new(self);
        value.fold_with(&mut r)
    }

    pub fn probe_const_var(&self, vid: ConstVid) -> Result<Const<'db>, UniverseIndex> {
        match self.inner.borrow_mut().const_unification_table().probe_value(vid) {
            ConstVariableValue::Known { value } => Ok(value),
            ConstVariableValue::Unknown { origin: _, universe } => Err(universe),
        }
    }

    // Instantiates the bound variables in a given binder with fresh inference
    // variables in the current universe.
    //
    // Use this method if you'd like to find some generic parameters of the binder's
    // variables (e.g. during a method call). If there isn't a [`BoundRegionConversionTime`]
    // that corresponds to your use case, consider whether or not you should
    // use [`InferCtxt::enter_forall`] instead.
    pub fn instantiate_binder_with_fresh_vars<T>(
        &self,
        _lbrct: BoundRegionConversionTime,
        value: Binder<'db, T>,
    ) -> T
    where
        T: TypeFoldable<DbInterner<'db>> + Clone,
    {
        if let Some(inner) = value.clone().no_bound_vars() {
            return inner;
        }

        let bound_vars = value.clone().bound_vars();
        let mut args = Vec::with_capacity(bound_vars.len());

        for bound_var_kind in bound_vars {
            let arg: GenericArg<'db> = match bound_var_kind {
                BoundVarKind::Ty(_) => self.next_ty_var().into(),
                BoundVarKind::Region(_) => self.next_region_var().into(),
                BoundVarKind::Const => self.next_const_var().into(),
            };
            args.push(arg);
        }

        struct ToFreshVars<'db> {
            args: Vec<GenericArg<'db>>,
        }

        impl<'db> BoundVarReplacerDelegate<'db> for ToFreshVars<'db> {
            fn replace_region(&mut self, br: BoundRegion) -> Region<'db> {
                self.args[br.var.index()].expect_region()
            }
            fn replace_ty(&mut self, bt: BoundTy) -> Ty<'db> {
                self.args[bt.var.index()].expect_ty()
            }
            fn replace_const(&mut self, bv: BoundConst) -> Const<'db> {
                self.args[bv.var.index()].expect_const()
            }
        }
        let delegate = ToFreshVars { args };
        self.interner.replace_bound_vars_uncached(value, delegate)
    }

    /// Obtains the latest type of the given closure; this may be a
    /// closure in the current function, in which case its
    /// `ClosureKind` may not yet be known.
    pub fn closure_kind(&self, closure_ty: Ty<'db>) -> Option<ClosureKind> {
        let unresolved_kind_ty = match closure_ty.kind() {
            TyKind::Closure(_, args) => args.as_closure().kind_ty(),
            TyKind::CoroutineClosure(_, args) => args.as_coroutine_closure().kind_ty(),
            _ => panic!("unexpected type {closure_ty:?}"),
        };
        let closure_kind_ty = self.shallow_resolve(unresolved_kind_ty);
        closure_kind_ty.to_opt_closure_kind()
    }

    pub fn universe(&self) -> UniverseIndex {
        self.universe.get()
    }

    /// Creates and return a fresh universe that extends all previous
    /// universes. Updates `self.universe` to that new universe.
    pub fn create_next_universe(&self) -> UniverseIndex {
        let u = self.universe.get().next_universe();
        debug!("create_next_universe {u:?}");
        self.universe.set(u);
        u
    }

    /// The returned function is used in a fast path. If it returns `true` the variable is
    /// unchanged, `false` indicates that the status is unknown.
    #[inline]
    pub fn is_ty_infer_var_definitely_unchanged<'a>(
        &'a self,
    ) -> impl Fn(TyOrConstInferVar) -> bool + Captures<'db> + 'a {
        // This hoists the borrow/release out of the loop body.
        let inner = self.inner.try_borrow();

        move |infer_var: TyOrConstInferVar| match (infer_var, &inner) {
            (TyOrConstInferVar::Ty(ty_var), Ok(inner)) => {
                use self::type_variable::TypeVariableValue;

                matches!(
                    inner.try_type_variables_probe_ref(ty_var),
                    Some(TypeVariableValue::Unknown { .. })
                )
            }
            _ => false,
        }
    }

    /// `ty_or_const_infer_var_changed` is equivalent to one of these two:
    ///   * `shallow_resolve(ty) != ty` (where `ty.kind = Infer(_)`)
    ///   * `shallow_resolve(ct) != ct` (where `ct.kind = ConstKind::Infer(_)`)
    ///
    /// However, `ty_or_const_infer_var_changed` is more efficient. It's always
    /// inlined, despite being large, because it has only two call sites that
    /// are extremely hot (both in `traits::fulfill`'s checking of `stalled_on`
    /// inference variables), and it handles both `Ty` and `Const` without
    /// having to resort to storing full `GenericArg`s in `stalled_on`.
    #[inline(always)]
    pub fn ty_or_const_infer_var_changed(&self, infer_var: TyOrConstInferVar) -> bool {
        match infer_var {
            TyOrConstInferVar::Ty(v) => {
                use self::type_variable::TypeVariableValue;

                // If `inlined_probe` returns a `Known` value, it never equals
                // `Infer(TyVar(v))`.
                match self.inner.borrow_mut().type_variables().inlined_probe(v) {
                    TypeVariableValue::Unknown { .. } => false,
                    TypeVariableValue::Known { .. } => true,
                }
            }

            TyOrConstInferVar::TyInt(v) => {
                // If `inlined_probe_value` returns a value it's always a
                // `Int(_)` or `UInt(_)`, which never matches a
                // `Infer(_)`.
                self.inner.borrow_mut().int_unification_table().inlined_probe_value(v).is_known()
            }

            TyOrConstInferVar::TyFloat(v) => {
                // If `probe_value` returns a value it's always a
                // `Float(_)`, which never matches a `Infer(_)`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                self.inner.borrow_mut().float_unification_table().probe_value(v).is_known()
            }

            TyOrConstInferVar::Const(v) => {
                // If `probe_value` returns a `Known` value, it never equals
                // `ConstKind::Infer(InferConst::Var(v))`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                match self.inner.borrow_mut().const_unification_table().probe_value(v) {
                    ConstVariableValue::Unknown { .. } => false,
                    ConstVariableValue::Known { .. } => true,
                }
            }
        }
    }

    fn sub_unification_table_root_var(&self, var: rustc_type_ir::TyVid) -> rustc_type_ir::TyVid {
        self.inner.borrow_mut().type_variables().sub_unification_table_root_var(var)
    }

    fn sub_unify_ty_vids_raw(&self, a: rustc_type_ir::TyVid, b: rustc_type_ir::TyVid) {
        self.inner.borrow_mut().type_variables().sub_unify(a, b);
    }
}

/// Helper for [InferCtxt::ty_or_const_infer_var_changed] (see comment on that), currently
/// used only for `traits::fulfill`'s list of `stalled_on` inference variables.
#[derive(Copy, Clone, Debug)]
pub enum TyOrConstInferVar {
    /// Equivalent to `Infer(TyVar(_))`.
    Ty(TyVid),
    /// Equivalent to `Infer(IntVar(_))`.
    TyInt(IntVid),
    /// Equivalent to `Infer(FloatVar(_))`.
    TyFloat(FloatVid),

    /// Equivalent to `ConstKind::Infer(InferConst::Var(_))`.
    Const(ConstVid),
}

impl TyOrConstInferVar {
    /// Tries to extract an inference variable from a type or a constant, returns `None`
    /// for types other than `Infer(_)` (or `InferTy::Fresh*`) and
    /// for constants other than `ConstKind::Infer(_)` (or `InferConst::Fresh`).
    pub fn maybe_from_generic_arg<'db>(arg: GenericArg<'db>) -> Option<Self> {
        match arg.kind() {
            GenericArgKind::Type(ty) => Self::maybe_from_ty(ty),
            GenericArgKind::Const(ct) => Self::maybe_from_const(ct),
            GenericArgKind::Lifetime(_) => None,
        }
    }

    /// Tries to extract an inference variable from a type, returns `None`
    /// for types other than `Infer(_)` (or `InferTy::Fresh*`).
    fn maybe_from_ty<'db>(ty: Ty<'db>) -> Option<Self> {
        match ty.kind() {
            TyKind::Infer(InferTy::TyVar(v)) => Some(TyOrConstInferVar::Ty(v)),
            TyKind::Infer(InferTy::IntVar(v)) => Some(TyOrConstInferVar::TyInt(v)),
            TyKind::Infer(InferTy::FloatVar(v)) => Some(TyOrConstInferVar::TyFloat(v)),
            _ => None,
        }
    }

    /// Tries to extract an inference variable from a constant, returns `None`
    /// for constants other than `ConstKind::Infer(_)` (or `InferConst::Fresh`).
    fn maybe_from_const<'db>(ct: Const<'db>) -> Option<Self> {
        match ct.kind() {
            ConstKind::Infer(InferConst::Var(v)) => Some(TyOrConstInferVar::Const(v)),
            _ => None,
        }
    }
}

impl<'db> TypeTrace<'db> {
    pub fn types(cause: &ObligationCause, a: Ty<'db>, b: Ty<'db>) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(a.into(), b.into())),
        }
    }

    pub fn trait_refs(
        cause: &ObligationCause,
        a: TraitRef<'db>,
        b: TraitRef<'db>,
    ) -> TypeTrace<'db> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::TraitRefs(ExpectedFound::new(a, b)) }
    }

    pub fn consts(cause: &ObligationCause, a: Const<'db>, b: Const<'db>) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(a.into(), b.into())),
        }
    }
}

/// Requires that `region` must be equal to one of the regions in `choice_regions`.
/// We often denote this using the syntax:
///
/// ```text
/// R0 member of [O1..On]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemberConstraint<'db> {
    /// The `DefId` and args of the opaque type causing this constraint.
    /// Used for error reporting.
    pub key: OpaqueTypeKey<'db>,

    /// The hidden type in which `member_region` appears: used for error reporting.
    pub hidden_ty: Ty<'db>,

    /// The region `R0`.
    pub member_region: Region<'db>,

    /// The options `O1..On`.
    pub choice_regions: Arc<Vec<Region<'db>>>,
}
