use std::cell::{Cell, RefCell};
use std::fmt;

pub use BoundRegionConversionTime::*;
pub use RegionVariableOrigin::*;
pub use SubregionOrigin::*;
pub use at::DefineOpaqueTypes;
use free_regions::RegionRelations;
pub use freshen::TypeFreshener;
use lexical_region_resolve::LexicalRegionResolutions;
pub use lexical_region_resolve::RegionResolutionError;
pub use opaque_types::{OpaqueTypeStorage, OpaqueTypeStorageEntries, OpaqueTypeTable};
use region_constraints::{
    GenericKind, RegionConstraintCollector, RegionConstraintStorage, VarInfos, VerifyBound,
};
pub use relate::StructurallyRelateAliases;
pub use relate::combine::PredicateEmittingRelation;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::undo_log::{Rollback, UndoLogs};
use rustc_data_structures::unify as ut;
use rustc_errors::{DiagCtxtHandle, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_macros::extension;
pub use rustc_macros::{TypeFoldable, TypeVisitable};
use rustc_middle::bug;
use rustc_middle::infer::canonical::{CanonicalQueryInput, CanonicalVarValues};
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::traits::select;
use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{
    self, BoundVarReplacerDelegate, ConstVid, FloatVid, GenericArg, GenericArgKind, GenericArgs,
    GenericArgsRef, GenericParamDefKind, InferConst, IntVid, OpaqueHiddenType, OpaqueTypeKey,
    PseudoCanonicalInput, Term, TermKind, Ty, TyCtxt, TyVid, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitable, TypeVisitableExt, TypingEnv, TypingMode, fold_regions,
};
use rustc_span::{Span, Symbol};
use snapshot::undo_log::InferCtxtUndoLogs;
use tracing::{debug, instrument};
use type_variable::TypeVariableOrigin;

use crate::infer::region_constraints::UndoLog;
use crate::infer::unify_key::{ConstVariableOrigin, ConstVariableValue, ConstVidKey};
use crate::traits::{
    self, ObligationCause, ObligationInspector, PredicateObligations, TraitEngine,
};

pub mod at;
pub mod canonical;
mod context;
mod free_regions;
mod freshen;
mod lexical_region_resolve;
mod opaque_types;
pub mod outlives;
mod projection;
pub mod region_constraints;
pub mod relate;
pub mod resolve;
pub(crate) mod snapshot;
mod type_variable;
mod unify_key;

/// `InferOk<'tcx, ()>` is used a lot. It may seem like a useless wrapper
/// around `PredicateObligations<'tcx>`, but it has one important property:
/// because `InferOk` is marked with `#[must_use]`, if you have a method
/// `InferCtxt::f` that returns `InferResult<'tcx, ()>` and you call it with
/// `infcx.f()?;` you'll get a warning about the obligations being discarded
/// without use, which is probably unintentional and has been a source of bugs
/// in the past.
#[must_use]
#[derive(Debug)]
pub struct InferOk<'tcx, T> {
    pub value: T,
    pub obligations: PredicateObligations<'tcx>,
}
pub type InferResult<'tcx, T> = Result<InferOk<'tcx, T>, TypeError<'tcx>>;

pub(crate) type FixupResult<T> = Result<T, FixupError>; // "fixup result"

pub(crate) type UnificationTable<'a, 'tcx, T> = ut::UnificationTable<
    ut::InPlace<T, &'a mut ut::UnificationStorage<T>, &'a mut InferCtxtUndoLogs<'tcx>>,
>;

/// This type contains all the things within `InferCtxt` that sit within a
/// `RefCell` and are involved with taking/rolling back snapshots. Snapshot
/// operations are hot enough that we want only one call to `borrow_mut` per
/// call to `start_snapshot` and `rollback_to`.
#[derive(Clone)]
pub struct InferCtxtInner<'tcx> {
    undo_log: InferCtxtUndoLogs<'tcx>,

    /// Cache for projections.
    ///
    /// This cache is snapshotted along with the infcx.
    projection_cache: traits::ProjectionCacheStorage<'tcx>,

    /// We instantiate `UnificationTable` with `bounds<Ty>` because the types
    /// that might instantiate a general type variable have an order,
    /// represented by its upper and lower bounds.
    type_variable_storage: type_variable::TypeVariableStorage<'tcx>,

    /// Map from const parameter variable to the kind of const it represents.
    const_unification_storage: ut::UnificationTableStorage<ConstVidKey<'tcx>>,

    /// Map from integral variable to the kind of integer it represents.
    int_unification_storage: ut::UnificationTableStorage<ty::IntVid>,

    /// Map from floating variable to the kind of float it represents.
    float_unification_storage: ut::UnificationTableStorage<ty::FloatVid>,

    /// Tracks the set of region variables and the constraints between them.
    ///
    /// This is initially `Some(_)` but when
    /// `resolve_regions_and_report_errors` is invoked, this gets set to `None`
    /// -- further attempts to perform unification, etc., may fail if new
    /// region constraints would've been added.
    region_constraint_storage: Option<RegionConstraintStorage<'tcx>>,

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
    region_obligations: Vec<TypeOutlivesConstraint<'tcx>>,

    /// Caches for opaque type inference.
    opaque_type_storage: OpaqueTypeStorage<'tcx>,
}

impl<'tcx> InferCtxtInner<'tcx> {
    fn new() -> InferCtxtInner<'tcx> {
        InferCtxtInner {
            undo_log: InferCtxtUndoLogs::default(),

            projection_cache: Default::default(),
            type_variable_storage: Default::default(),
            const_unification_storage: Default::default(),
            int_unification_storage: Default::default(),
            float_unification_storage: Default::default(),
            region_constraint_storage: Some(Default::default()),
            region_obligations: vec![],
            opaque_type_storage: Default::default(),
        }
    }

    #[inline]
    pub fn region_obligations(&self) -> &[TypeOutlivesConstraint<'tcx>] {
        &self.region_obligations
    }

    #[inline]
    pub fn projection_cache(&mut self) -> traits::ProjectionCache<'_, 'tcx> {
        self.projection_cache.with_log(&mut self.undo_log)
    }

    #[inline]
    fn try_type_variables_probe_ref(
        &self,
        vid: ty::TyVid,
    ) -> Option<&type_variable::TypeVariableValue<'tcx>> {
        // Uses a read-only view of the unification table, this way we don't
        // need an undo log.
        self.type_variable_storage.eq_relations_ref().try_probe_value(vid)
    }

    #[inline]
    fn type_variables(&mut self) -> type_variable::TypeVariableTable<'_, 'tcx> {
        self.type_variable_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    pub fn opaque_types(&mut self) -> opaque_types::OpaqueTypeTable<'_, 'tcx> {
        self.opaque_type_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    fn int_unification_table(&mut self) -> UnificationTable<'_, 'tcx, ty::IntVid> {
        self.int_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    fn float_unification_table(&mut self) -> UnificationTable<'_, 'tcx, ty::FloatVid> {
        self.float_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    fn const_unification_table(&mut self) -> UnificationTable<'_, 'tcx, ConstVidKey<'tcx>> {
        self.const_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    pub fn unwrap_region_constraints(&mut self) -> RegionConstraintCollector<'_, 'tcx> {
        self.region_constraint_storage
            .as_mut()
            .expect("region constraints already solved")
            .with_log(&mut self.undo_log)
    }
}

pub struct InferCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,

    /// The mode of this inference context, see the struct documentation
    /// for more details.
    typing_mode: TypingMode<'tcx>,

    /// Whether this inference context should care about region obligations in
    /// the root universe. Most notably, this is used during hir typeck as region
    /// solving is left to borrowck instead.
    pub considering_regions: bool,

    /// If set, this flag causes us to skip the 'leak check' during
    /// higher-ranked subtyping operations. This flag is a temporary one used
    /// to manage the removal of the leak-check: for the time being, we still run the
    /// leak-check, but we issue warnings.
    skip_leak_check: bool,

    pub inner: RefCell<InferCtxtInner<'tcx>>,

    /// Once region inference is done, the values for each variable.
    lexical_region_resolutions: RefCell<Option<LexicalRegionResolutions<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that depends on inference variables or placeholders.
    pub selection_cache: select::SelectionCache<'tcx, ty::ParamEnv<'tcx>>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that depends on inference variables or placeholders.
    pub evaluation_cache: select::EvaluationCache<'tcx, ty::ParamEnv<'tcx>>,

    /// The set of predicates on which errors have been reported, to
    /// avoid reporting the same error twice.
    pub reported_trait_errors:
        RefCell<FxIndexMap<Span, (Vec<Goal<'tcx, ty::Predicate<'tcx>>>, ErrorGuaranteed)>>,

    pub reported_signature_mismatch: RefCell<FxHashSet<(Span, Option<Span>)>>,

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
    universe: Cell<ty::UniverseIndex>,

    next_trait_solver: bool,

    pub obligation_inspector: Cell<Option<ObligationInspector<'tcx>>>,
}

/// See the `error_reporting` module for more details.
#[derive(Clone, Copy, Debug, PartialEq, Eq, TypeFoldable, TypeVisitable)]
pub enum ValuePairs<'tcx> {
    Regions(ExpectedFound<ty::Region<'tcx>>),
    Terms(ExpectedFound<ty::Term<'tcx>>),
    Aliases(ExpectedFound<ty::AliasTerm<'tcx>>),
    TraitRefs(ExpectedFound<ty::TraitRef<'tcx>>),
    PolySigs(ExpectedFound<ty::PolyFnSig<'tcx>>),
    ExistentialTraitRef(ExpectedFound<ty::PolyExistentialTraitRef<'tcx>>),
    ExistentialProjection(ExpectedFound<ty::PolyExistentialProjection<'tcx>>),
}

impl<'tcx> ValuePairs<'tcx> {
    pub fn ty(&self) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
        if let ValuePairs::Terms(ExpectedFound { expected, found }) = self
            && let Some(expected) = expected.as_type()
            && let Some(found) = found.as_type()
        {
            Some((expected, found))
        } else {
            None
        }
    }
}

/// The trace designates the path through inference that we took to
/// encounter an error or subtyping constraint.
///
/// See the `error_reporting` module for more details.
#[derive(Clone, Debug)]
pub struct TypeTrace<'tcx> {
    pub cause: ObligationCause<'tcx>,
    pub values: ValuePairs<'tcx>,
}

/// The origin of a `r1 <= r2` constraint.
///
/// See `error_reporting` module for more details
#[derive(Clone, Debug)]
pub enum SubregionOrigin<'tcx> {
    /// Arose from a subtyping relation
    Subtype(Box<TypeTrace<'tcx>>),

    /// When casting `&'a T` to an `&'b Trait` object,
    /// relating `'a` to `'b`.
    RelateObjectBound(Span),

    /// Some type parameter was instantiated with the given type,
    /// and that type must outlive some region.
    RelateParamBound(Span, Ty<'tcx>, Option<Span>),

    /// The given region parameter was instantiated with a region
    /// that must outlive some other region.
    RelateRegionParamBound(Span, Option<Ty<'tcx>>),

    /// Creating a pointer `b` to contents of another reference.
    Reborrow(Span),

    /// (&'a &'b T) where a >= b
    ReferenceOutlivesReferent(Ty<'tcx>, Span),

    /// Comparing the signature and requirements of an impl method against
    /// the containing trait.
    CompareImplItemObligation {
        span: Span,
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
    },

    /// Checking that the bounds of a trait's associated type hold for a given impl.
    CheckAssociatedTypeBounds {
        parent: Box<SubregionOrigin<'tcx>>,
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
    },

    AscribeUserTypeProvePredicate(Span),
}

// `SubregionOrigin` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(SubregionOrigin<'_>, 32);

impl<'tcx> SubregionOrigin<'tcx> {
    pub fn to_constraint_category(&self) -> ConstraintCategory<'tcx> {
        match self {
            Self::Subtype(type_trace) => type_trace.cause.to_constraint_category(),
            Self::AscribeUserTypeProvePredicate(span) => ConstraintCategory::Predicate(*span),
            _ => ConstraintCategory::BoringNoLocation,
        }
    }
}

/// Times when we replace bound regions with existentials:
#[derive(Clone, Copy, Debug)]
pub enum BoundRegionConversionTime {
    /// when a fn is called
    FnCall,

    /// when two higher-ranked types are compared
    HigherRankedType,

    /// when projecting an associated type
    AssocTypeProjection(DefId),
}

/// Reasons to create a region inference variable.
///
/// See `error_reporting` module for more details.
#[derive(Copy, Clone, Debug)]
pub enum RegionVariableOrigin {
    /// Region variables created for ill-categorized reasons.
    ///
    /// They mostly indicate places in need of refactoring.
    MiscVariable(Span),

    /// Regions created by a `&P` or `[...]` pattern.
    PatternRegion(Span),

    /// Regions created by `&` operator.
    BorrowRegion(Span),

    /// Regions created as part of an autoref of a method receiver.
    Autoref(Span),

    /// Regions created as part of an automatic coercion.
    Coercion(Span),

    /// Region variables created as the values for early-bound regions.
    ///
    /// FIXME(@lcnr): This should also store a `DefId`, similar to
    /// `TypeVariableOrigin`.
    RegionParameterDefinition(Span, Symbol),

    /// Region variables created when instantiating a binder with
    /// existential variables, e.g. when calling a function or method.
    BoundRegion(Span, ty::BoundRegionKind, BoundRegionConversionTime),

    UpvarRegion(ty::UpvarId, Span),

    /// This origin is used for the inference variables that we create
    /// during NLL region processing.
    Nll(NllRegionVariableOrigin),
}

#[derive(Copy, Clone, Debug)]
pub enum NllRegionVariableOrigin {
    /// During NLL region processing, we create variables for free
    /// regions that we encounter in the function signature and
    /// elsewhere. This origin indices we've got one of those.
    FreeRegion,

    /// "Universal" instantiation of a higher-ranked region (e.g.,
    /// from a `for<'a> T` binder). Meant to represent "any region".
    Placeholder(ty::PlaceholderRegion),

    Existential {
        /// If this is true, then this variable was created to represent a lifetime
        /// bound in a `for` binder. For example, it might have been created to
        /// represent the lifetime `'a` in a type like `for<'a> fn(&'a u32)`.
        /// Such variables are created when we are trying to figure out if there
        /// is any valid instantiation of `'a` that could fit into some scenario.
        ///
        /// This is used to inform error reporting: in the case that we are trying to
        /// determine whether there is any valid instantiation of a `'a` variable that meets
        /// some constraint C, we want to blame the "source" of that `for` type,
        /// rather than blaming the source of the constraint C.
        from_forall: bool,
    },
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
pub struct TypeOutlivesConstraint<'tcx> {
    pub sub_region: ty::Region<'tcx>,
    pub sup_type: Ty<'tcx>,
    pub origin: SubregionOrigin<'tcx>,
}

/// Used to configure inference contexts before their creation.
pub struct InferCtxtBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    considering_regions: bool,
    skip_leak_check: bool,
    /// Whether we should use the new trait solver in the local inference context,
    /// which affects things like which solver is used in `predicate_may_hold`.
    next_trait_solver: bool,
}

#[extension(pub trait TyCtxtInferExt<'tcx>)]
impl<'tcx> TyCtxt<'tcx> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'tcx> {
        InferCtxtBuilder {
            tcx: self,
            considering_regions: true,
            skip_leak_check: false,
            next_trait_solver: self.next_trait_solver_globally(),
        }
    }
}

impl<'tcx> InferCtxtBuilder<'tcx> {
    pub fn with_next_trait_solver(mut self, next_trait_solver: bool) -> Self {
        self.next_trait_solver = next_trait_solver;
        self
    }

    pub fn ignoring_regions(mut self) -> Self {
        self.considering_regions = false;
        self
    }

    pub fn skip_leak_check(mut self, skip_leak_check: bool) -> Self {
        self.skip_leak_check = skip_leak_check;
        self
    }

    /// Given a canonical value `C` as a starting point, create an
    /// inference context that contains each of the bound values
    /// within instantiated as a fresh variable. The `f` closure is
    /// invoked with the new infcx, along with the instantiated value
    /// `V` and a instantiation `S`. This instantiation `S` maps from
    /// the bound values in `C` to their instantiated values in `V`
    /// (in other words, `S(C) = V`).
    pub fn build_with_canonical<T>(
        mut self,
        span: Span,
        input: &CanonicalQueryInput<'tcx, T>,
    ) -> (InferCtxt<'tcx>, T, CanonicalVarValues<'tcx>)
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let infcx = self.build(input.typing_mode);
        let (value, args) = infcx.instantiate_canonical(span, &input.canonical);
        (infcx, value, args)
    }

    pub fn build_with_typing_env(
        mut self,
        TypingEnv { typing_mode, param_env }: TypingEnv<'tcx>,
    ) -> (InferCtxt<'tcx>, ty::ParamEnv<'tcx>) {
        (self.build(typing_mode), param_env)
    }

    pub fn build(&mut self, typing_mode: TypingMode<'tcx>) -> InferCtxt<'tcx> {
        let InferCtxtBuilder { tcx, considering_regions, skip_leak_check, next_trait_solver } =
            *self;
        InferCtxt {
            tcx,
            typing_mode,
            considering_regions,
            skip_leak_check,
            inner: RefCell::new(InferCtxtInner::new()),
            lexical_region_resolutions: RefCell::new(None),
            selection_cache: Default::default(),
            evaluation_cache: Default::default(),
            reported_trait_errors: Default::default(),
            reported_signature_mismatch: Default::default(),
            tainted_by_errors: Cell::new(None),
            universe: Cell::new(ty::UniverseIndex::ROOT),
            next_trait_solver,
            obligation_inspector: Cell::new(None),
        }
    }
}

impl<'tcx, T> InferOk<'tcx, T> {
    /// Extracts `value`, registering any obligations into `fulfill_cx`.
    pub fn into_value_registering_obligations<E: 'tcx>(
        self,
        infcx: &InferCtxt<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> T {
        let InferOk { value, obligations } = self;
        fulfill_cx.register_predicate_obligations(infcx, obligations);
        value
    }
}

impl<'tcx> InferOk<'tcx, ()> {
    pub fn into_obligations(self) -> PredicateObligations<'tcx> {
        self.obligations
    }
}

impl<'tcx> InferCtxt<'tcx> {
    pub fn dcx(&self) -> DiagCtxtHandle<'_> {
        self.tcx.dcx().taintable_handle(&self.tainted_by_errors)
    }

    pub fn next_trait_solver(&self) -> bool {
        self.next_trait_solver
    }

    #[inline(always)]
    pub fn typing_mode(&self) -> TypingMode<'tcx> {
        self.typing_mode
    }

    pub fn freshen<T: TypeFoldable<TyCtxt<'tcx>>>(&self, t: T) -> T {
        t.fold_with(&mut self.freshener())
    }

    /// Returns the origin of the type variable identified by `vid`.
    ///
    /// No attempt is made to resolve `vid` to its root variable.
    pub fn type_var_origin(&self, vid: TyVid) -> TypeVariableOrigin {
        self.inner.borrow_mut().type_variables().var_origin(vid)
    }

    /// Returns the origin of the const variable identified by `vid`
    // FIXME: We should store origins separately from the unification table
    // so this doesn't need to be optional.
    pub fn const_var_origin(&self, vid: ConstVid) -> Option<ConstVariableOrigin> {
        match self.inner.borrow_mut().const_unification_table().probe_value(vid) {
            ConstVariableValue::Known { .. } => None,
            ConstVariableValue::Unknown { origin, .. } => Some(origin),
        }
    }

    pub fn freshener<'b>(&'b self) -> TypeFreshener<'b, 'tcx> {
        freshen::TypeFreshener::new(self)
    }

    pub fn unresolved_variables(&self) -> Vec<Ty<'tcx>> {
        let mut inner = self.inner.borrow_mut();
        let mut vars: Vec<Ty<'_>> = inner
            .type_variables()
            .unresolved_variables()
            .into_iter()
            .map(|t| Ty::new_var(self.tcx, t))
            .collect();
        vars.extend(
            (0..inner.int_unification_table().len())
                .map(|i| ty::IntVid::from_usize(i))
                .filter(|&vid| inner.int_unification_table().probe_value(vid).is_unknown())
                .map(|v| Ty::new_int_var(self.tcx, v)),
        );
        vars.extend(
            (0..inner.float_unification_table().len())
                .map(|i| ty::FloatVid::from_usize(i))
                .filter(|&vid| inner.float_unification_table().probe_value(vid).is_unknown())
                .map(|v| Ty::new_float_var(self.tcx, v)),
        );
        vars
    }

    #[instrument(skip(self), level = "debug")]
    pub fn sub_regions(
        &self,
        origin: SubregionOrigin<'tcx>,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) {
        self.inner.borrow_mut().unwrap_region_constraints().make_subregion(origin, a, b);
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
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: ty::PolyCoercePredicate<'tcx>,
    ) -> Result<InferResult<'tcx, ()>, (TyVid, TyVid)> {
        let subtype_predicate = predicate.map_bound(|p| ty::SubtypePredicate {
            a_is_expected: false, // when coercing from `a` to `b`, `b` is expected
            a: p.a,
            b: p.b,
        });
        self.subtype_predicate(cause, param_env, subtype_predicate)
    }

    pub fn subtype_predicate(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: ty::PolySubtypePredicate<'tcx>,
    ) -> Result<InferResult<'tcx, ()>, (TyVid, TyVid)> {
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
            (&ty::Infer(ty::TyVar(a_vid)), &ty::Infer(ty::TyVar(b_vid))) => {
                return Err((a_vid, b_vid));
            }
            _ => {}
        }

        self.enter_forall(predicate, |ty::SubtypePredicate { a_is_expected, a, b }| {
            if a_is_expected {
                Ok(self.at(cause, param_env).sub(DefineOpaqueTypes::Yes, a, b))
            } else {
                Ok(self.at(cause, param_env).sup(DefineOpaqueTypes::Yes, b, a))
            }
        })
    }

    /// Number of type variables created so far.
    pub fn num_ty_vars(&self) -> usize {
        self.inner.borrow_mut().type_variables().num_vars()
    }

    pub fn next_ty_var(&self, span: Span) -> Ty<'tcx> {
        self.next_ty_var_with_origin(TypeVariableOrigin { span, param_def_id: None })
    }

    pub fn next_ty_var_with_origin(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        let vid = self.inner.borrow_mut().type_variables().new_var(self.universe(), origin);
        Ty::new_var(self.tcx, vid)
    }

    pub fn next_ty_var_id_in_universe(&self, span: Span, universe: ty::UniverseIndex) -> TyVid {
        let origin = TypeVariableOrigin { span, param_def_id: None };
        self.inner.borrow_mut().type_variables().new_var(universe, origin)
    }

    pub fn next_ty_var_in_universe(&self, span: Span, universe: ty::UniverseIndex) -> Ty<'tcx> {
        let vid = self.next_ty_var_id_in_universe(span, universe);
        Ty::new_var(self.tcx, vid)
    }

    pub fn next_const_var(&self, span: Span) -> ty::Const<'tcx> {
        self.next_const_var_with_origin(ConstVariableOrigin { span, param_def_id: None })
    }

    pub fn next_const_var_with_origin(&self, origin: ConstVariableOrigin) -> ty::Const<'tcx> {
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown { origin, universe: self.universe() })
            .vid;
        ty::Const::new_var(self.tcx, vid)
    }

    pub fn next_const_var_in_universe(
        &self,
        span: Span,
        universe: ty::UniverseIndex,
    ) -> ty::Const<'tcx> {
        let origin = ConstVariableOrigin { span, param_def_id: None };
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVariableValue::Unknown { origin, universe })
            .vid;
        ty::Const::new_var(self.tcx, vid)
    }

    pub fn next_int_var(&self) -> Ty<'tcx> {
        let next_int_var_id =
            self.inner.borrow_mut().int_unification_table().new_key(ty::IntVarValue::Unknown);
        Ty::new_int_var(self.tcx, next_int_var_id)
    }

    pub fn next_float_var(&self) -> Ty<'tcx> {
        let next_float_var_id =
            self.inner.borrow_mut().float_unification_table().new_key(ty::FloatVarValue::Unknown);
        Ty::new_float_var(self.tcx, next_float_var_id)
    }

    /// Creates a fresh region variable with the next available index.
    /// The variable will be created in the maximum universe created
    /// thus far, allowing it to name any region created thus far.
    pub fn next_region_var(&self, origin: RegionVariableOrigin) -> ty::Region<'tcx> {
        self.next_region_var_in_universe(origin, self.universe())
    }

    /// Creates a fresh region variable with the next available index
    /// in the given universe; typically, you can use
    /// `next_region_var` and just use the maximal universe.
    pub fn next_region_var_in_universe(
        &self,
        origin: RegionVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> ty::Region<'tcx> {
        let region_var =
            self.inner.borrow_mut().unwrap_region_constraints().new_region_var(universe, origin);
        ty::Region::new_var(self.tcx, region_var)
    }

    pub fn next_term_var_of_kind(&self, term: ty::Term<'tcx>, span: Span) -> ty::Term<'tcx> {
        match term.kind() {
            ty::TermKind::Ty(_) => self.next_ty_var(span).into(),
            ty::TermKind::Const(_) => self.next_const_var(span).into(),
        }
    }

    /// Return the universe that the region `r` was created in. For
    /// most regions (e.g., `'static`, named regions from the user,
    /// etc) this is the root universe U0. For inference variables or
    /// placeholders, however, it will return the universe which they
    /// are associated.
    pub fn universe_of_region(&self, r: ty::Region<'tcx>) -> ty::UniverseIndex {
        self.inner.borrow_mut().unwrap_region_constraints().universe(r)
    }

    /// Number of region variables created so far.
    pub fn num_region_vars(&self) -> usize {
        self.inner.borrow_mut().unwrap_region_constraints().num_region_vars()
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    #[instrument(skip(self), level = "debug")]
    pub fn next_nll_region_var(&self, origin: NllRegionVariableOrigin) -> ty::Region<'tcx> {
        self.next_region_var(RegionVariableOrigin::Nll(origin))
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    #[instrument(skip(self), level = "debug")]
    pub fn next_nll_region_var_in_universe(
        &self,
        origin: NllRegionVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> ty::Region<'tcx> {
        self.next_region_var_in_universe(RegionVariableOrigin::Nll(origin), universe)
    }

    pub fn var_for_def(&self, span: Span, param: &ty::GenericParamDef) -> GenericArg<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                // Create a region inference variable for the given
                // region parameter definition.
                self.next_region_var(RegionParameterDefinition(span, param.name)).into()
            }
            GenericParamDefKind::Type { .. } => {
                // Create a type inference variable for the given
                // type parameter definition. The generic parameters are
                // for actual parameters that may be referred to by
                // the default of this type parameter, if it exists.
                // e.g., `struct Foo<A, B, C = (A, B)>(...);` when
                // used in a path such as `Foo::<T, U>::new()` will
                // use an inference variable for `C` with `[T, U]`
                // as the generic parameters for the default, `(T, U)`.
                let ty_var_id = self.inner.borrow_mut().type_variables().new_var(
                    self.universe(),
                    TypeVariableOrigin { param_def_id: Some(param.def_id), span },
                );

                Ty::new_var(self.tcx, ty_var_id).into()
            }
            GenericParamDefKind::Const { .. } => {
                let origin = ConstVariableOrigin { param_def_id: Some(param.def_id), span };
                let const_var_id = self
                    .inner
                    .borrow_mut()
                    .const_unification_table()
                    .new_key(ConstVariableValue::Unknown { origin, universe: self.universe() })
                    .vid;
                ty::Const::new_var(self.tcx, const_var_id).into()
            }
        }
    }

    /// Given a set of generics defined on a type or impl, returns the generic parameters mapping
    /// each type/region parameter to a fresh inference variable.
    pub fn fresh_args_for_item(&self, span: Span, def_id: DefId) -> GenericArgsRef<'tcx> {
        GenericArgs::for_item(self.tcx, def_id, |param, _| self.var_for_def(span, param))
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

    pub fn region_var_origin(&self, vid: ty::RegionVid) -> RegionVariableOrigin {
        let mut inner = self.inner.borrow_mut();
        let inner = &mut *inner;
        inner.unwrap_region_constraints().var_origin(vid)
    }

    /// Clone the list of variable regions. This is used only during NLL processing
    /// to put the set of region variables into the NLL region context.
    pub fn get_region_var_infos(&self) -> VarInfos {
        let inner = self.inner.borrow();
        assert!(!UndoLogs::<UndoLog<'_>>::in_snapshot(&inner.undo_log));
        let storage = inner.region_constraint_storage.as_ref().expect("regions already resolved");
        assert!(storage.data.is_empty(), "{:#?}", storage.data);
        // We clone instead of taking because borrowck still wants to use the
        // inference context after calling this for diagnostics and the new
        // trait solver.
        storage.var_infos.clone()
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn take_opaque_types(&self) -> Vec<(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)> {
        self.inner.borrow_mut().opaque_type_storage.take_opaque_types().collect()
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn clone_opaque_types(&self) -> Vec<(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)> {
        self.inner.borrow_mut().opaque_type_storage.iter_opaque_types().collect()
    }

    #[inline(always)]
    pub fn can_define_opaque_ty(&self, id: impl Into<DefId>) -> bool {
        debug_assert!(!self.next_trait_solver());
        match self.typing_mode() {
            TypingMode::Analysis {
                defining_opaque_types_and_generators: defining_opaque_types,
            }
            | TypingMode::Borrowck { defining_opaque_types } => {
                id.into().as_local().is_some_and(|def_id| defining_opaque_types.contains(&def_id))
            }
            // FIXME(#132279): This function is quite weird in post-analysis
            // and post-borrowck analysis mode. We may need to modify its uses
            // to support PostBorrowckAnalysis in the old solver as well.
            TypingMode::Coherence
            | TypingMode::PostBorrowckAnalysis { .. }
            | TypingMode::PostAnalysis => false,
        }
    }

    pub fn ty_to_string(&self, t: Ty<'tcx>) -> String {
        self.resolve_vars_if_possible(t).to_string()
    }

    /// If `TyVar(vid)` resolves to a type, return that type. Else, return the
    /// universe index of `TyVar(vid)`.
    pub fn probe_ty_var(&self, vid: TyVid) -> Result<Ty<'tcx>, ty::UniverseIndex> {
        use self::type_variable::TypeVariableValue;

        match self.inner.borrow_mut().type_variables().probe(vid) {
            TypeVariableValue::Known { value } => Ok(value),
            TypeVariableValue::Unknown { universe } => Err(universe),
        }
    }

    pub fn shallow_resolve(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Infer(v) = *ty.kind() {
            match v {
                ty::TyVar(v) => {
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

                ty::IntVar(v) => {
                    match self.inner.borrow_mut().int_unification_table().probe_value(v) {
                        ty::IntVarValue::IntType(ty) => Ty::new_int(self.tcx, ty),
                        ty::IntVarValue::UintType(ty) => Ty::new_uint(self.tcx, ty),
                        ty::IntVarValue::Unknown => ty,
                    }
                }

                ty::FloatVar(v) => {
                    match self.inner.borrow_mut().float_unification_table().probe_value(v) {
                        ty::FloatVarValue::Known(ty) => Ty::new_float(self.tcx, ty),
                        ty::FloatVarValue::Unknown => ty,
                    }
                }

                ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => ty,
            }
        } else {
            ty
        }
    }

    pub fn shallow_resolve_const(&self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Infer(infer_ct) => match infer_ct {
                InferConst::Var(vid) => self
                    .inner
                    .borrow_mut()
                    .const_unification_table()
                    .probe_value(vid)
                    .known()
                    .unwrap_or(ct),
                InferConst::Fresh(_) => ct,
            },
            ty::ConstKind::Param(_)
            | ty::ConstKind::Bound(_, _)
            | ty::ConstKind::Placeholder(_)
            | ty::ConstKind::Unevaluated(_)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Error(_)
            | ty::ConstKind::Expr(_) => ct,
        }
    }

    pub fn root_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.inner.borrow_mut().type_variables().root_var(var)
    }

    pub fn root_const_var(&self, var: ty::ConstVid) -> ty::ConstVid {
        self.inner.borrow_mut().const_unification_table().find(var).vid
    }

    /// Resolves an int var to a rigid int type, if it was constrained to one,
    /// or else the root int var in the unification table.
    pub fn opportunistic_resolve_int_var(&self, vid: ty::IntVid) -> Ty<'tcx> {
        let mut inner = self.inner.borrow_mut();
        let value = inner.int_unification_table().probe_value(vid);
        match value {
            ty::IntVarValue::IntType(ty) => Ty::new_int(self.tcx, ty),
            ty::IntVarValue::UintType(ty) => Ty::new_uint(self.tcx, ty),
            ty::IntVarValue::Unknown => {
                Ty::new_int_var(self.tcx, inner.int_unification_table().find(vid))
            }
        }
    }

    /// Resolves a float var to a rigid int type, if it was constrained to one,
    /// or else the root float var in the unification table.
    pub fn opportunistic_resolve_float_var(&self, vid: ty::FloatVid) -> Ty<'tcx> {
        let mut inner = self.inner.borrow_mut();
        let value = inner.float_unification_table().probe_value(vid);
        match value {
            ty::FloatVarValue::Known(ty) => Ty::new_float(self.tcx, ty),
            ty::FloatVarValue::Unknown => {
                Ty::new_float_var(self.tcx, inner.float_unification_table().find(vid))
            }
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
        T: TypeFoldable<TyCtxt<'tcx>>,
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

    pub fn resolve_numeric_literals_with_default<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        if !value.has_infer() {
            return value; // Avoid duplicated type-folding.
        }
        let mut r = InferenceLiteralEraser { tcx: self.tcx };
        value.fold_with(&mut r)
    }

    pub fn probe_const_var(&self, vid: ty::ConstVid) -> Result<ty::Const<'tcx>, ty::UniverseIndex> {
        match self.inner.borrow_mut().const_unification_table().probe_value(vid) {
            ConstVariableValue::Known { value } => Ok(value),
            ConstVariableValue::Unknown { origin: _, universe } => Err(universe),
        }
    }

    /// Attempts to resolve all type/region/const variables in
    /// `value`. Region inference must have been run already (e.g.,
    /// by calling `resolve_regions_and_report_errors`). If some
    /// variable was never unified, an `Err` results.
    ///
    /// This method is idempotent, but it not typically not invoked
    /// except during the writeback phase.
    pub fn fully_resolve<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> FixupResult<T> {
        match resolve::fully_resolve(self, value) {
            Ok(value) => {
                if value.has_non_region_infer() {
                    bug!("`{value:?}` is not fully resolved");
                }
                if value.has_infer_regions() {
                    let guar = self.dcx().delayed_bug(format!("`{value:?}` is not fully resolved"));
                    Ok(fold_regions(self.tcx, value, |re, _| {
                        if re.is_var() { ty::Region::new_error(self.tcx, guar) } else { re }
                    }))
                } else {
                    Ok(value)
                }
            }
            Err(e) => Err(e),
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
        span: Span,
        lbrct: BoundRegionConversionTime,
        value: ty::Binder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        if let Some(inner) = value.no_bound_vars() {
            return inner;
        }

        let bound_vars = value.bound_vars();
        let mut args = Vec::with_capacity(bound_vars.len());

        for bound_var_kind in bound_vars {
            let arg: ty::GenericArg<'_> = match bound_var_kind {
                ty::BoundVariableKind::Ty(_) => self.next_ty_var(span).into(),
                ty::BoundVariableKind::Region(br) => {
                    self.next_region_var(BoundRegion(span, br, lbrct)).into()
                }
                ty::BoundVariableKind::Const => self.next_const_var(span).into(),
            };
            args.push(arg);
        }

        struct ToFreshVars<'tcx> {
            args: Vec<ty::GenericArg<'tcx>>,
        }

        impl<'tcx> BoundVarReplacerDelegate<'tcx> for ToFreshVars<'tcx> {
            fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx> {
                self.args[br.var.index()].expect_region()
            }
            fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx> {
                self.args[bt.var.index()].expect_ty()
            }
            fn replace_const(&mut self, bv: ty::BoundVar) -> ty::Const<'tcx> {
                self.args[bv.index()].expect_const()
            }
        }
        let delegate = ToFreshVars { args };
        self.tcx.replace_bound_vars_uncached(value, delegate)
    }

    /// See the [`region_constraints::RegionConstraintCollector::verify_generic_bound`] method.
    pub(crate) fn verify_generic_bound(
        &self,
        origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        a: ty::Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) {
        debug!("verify_generic_bound({:?}, {:?} <: {:?})", kind, a, bound);

        self.inner
            .borrow_mut()
            .unwrap_region_constraints()
            .verify_generic_bound(origin, kind, a, bound);
    }

    /// Obtains the latest type of the given closure; this may be a
    /// closure in the current function, in which case its
    /// `ClosureKind` may not yet be known.
    pub fn closure_kind(&self, closure_ty: Ty<'tcx>) -> Option<ty::ClosureKind> {
        let unresolved_kind_ty = match *closure_ty.kind() {
            ty::Closure(_, args) => args.as_closure().kind_ty(),
            ty::CoroutineClosure(_, args) => args.as_coroutine_closure().kind_ty(),
            _ => bug!("unexpected type {closure_ty}"),
        };
        let closure_kind_ty = self.shallow_resolve(unresolved_kind_ty);
        closure_kind_ty.to_opt_closure_kind()
    }

    pub fn universe(&self) -> ty::UniverseIndex {
        self.universe.get()
    }

    /// Creates and return a fresh universe that extends all previous
    /// universes. Updates `self.universe` to that new universe.
    pub fn create_next_universe(&self) -> ty::UniverseIndex {
        let u = self.universe.get().next_universe();
        debug!("create_next_universe {u:?}");
        self.universe.set(u);
        u
    }

    /// Extract [`ty::TypingMode`] of this inference context to get a `TypingEnv`
    /// which contains the necessary information to use the trait system without
    /// using canonicalization or carrying this inference context around.
    pub fn typing_env(&self, param_env: ty::ParamEnv<'tcx>) -> ty::TypingEnv<'tcx> {
        let typing_mode = match self.typing_mode() {
            // FIXME(#132279): This erases the `defining_opaque_types` as it isn't possible
            // to handle them without proper canonicalization. This means we may cause cycle
            // errors and fail to reveal opaques while inside of bodies. We should rename this
            // function and require explicit comments on all use-sites in the future.
            ty::TypingMode::Analysis { defining_opaque_types_and_generators: _ }
            | ty::TypingMode::Borrowck { defining_opaque_types: _ } => {
                TypingMode::non_body_analysis()
            }
            mode @ (ty::TypingMode::Coherence
            | ty::TypingMode::PostBorrowckAnalysis { .. }
            | ty::TypingMode::PostAnalysis) => mode,
        };
        ty::TypingEnv { typing_mode, param_env }
    }

    /// Similar to [`Self::canonicalize_query`], except that it returns
    /// a [`PseudoCanonicalInput`] and requires both the `value` and the
    /// `param_env` to not contain any inference variables or placeholders.
    pub fn pseudo_canonicalize_query<V>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        value: V,
    ) -> PseudoCanonicalInput<'tcx, V>
    where
        V: TypeVisitable<TyCtxt<'tcx>>,
    {
        debug_assert!(!value.has_infer());
        debug_assert!(!value.has_placeholders());
        debug_assert!(!param_env.has_infer());
        debug_assert!(!param_env.has_placeholders());
        self.typing_env(param_env).as_query_input(value)
    }

    /// The returned function is used in a fast path. If it returns `true` the variable is
    /// unchanged, `false` indicates that the status is unknown.
    #[inline]
    pub fn is_ty_infer_var_definitely_unchanged(&self) -> impl Fn(TyOrConstInferVar) -> bool {
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
    ///   * `shallow_resolve(ty) != ty` (where `ty.kind = ty::Infer(_)`)
    ///   * `shallow_resolve(ct) != ct` (where `ct.kind = ty::ConstKind::Infer(_)`)
    ///
    /// However, `ty_or_const_infer_var_changed` is more efficient. It's always
    /// inlined, despite being large, because it has only two call sites that
    /// are extremely hot (both in `traits::fulfill`'s checking of `stalled_on`
    /// inference variables), and it handles both `Ty` and `ty::Const` without
    /// having to resort to storing full `GenericArg`s in `stalled_on`.
    #[inline(always)]
    pub fn ty_or_const_infer_var_changed(&self, infer_var: TyOrConstInferVar) -> bool {
        match infer_var {
            TyOrConstInferVar::Ty(v) => {
                use self::type_variable::TypeVariableValue;

                // If `inlined_probe` returns a `Known` value, it never equals
                // `ty::Infer(ty::TyVar(v))`.
                match self.inner.borrow_mut().type_variables().inlined_probe(v) {
                    TypeVariableValue::Unknown { .. } => false,
                    TypeVariableValue::Known { .. } => true,
                }
            }

            TyOrConstInferVar::TyInt(v) => {
                // If `inlined_probe_value` returns a value it's always a
                // `ty::Int(_)` or `ty::UInt(_)`, which never matches a
                // `ty::Infer(_)`.
                self.inner.borrow_mut().int_unification_table().inlined_probe_value(v).is_known()
            }

            TyOrConstInferVar::TyFloat(v) => {
                // If `probe_value` returns a value it's always a
                // `ty::Float(_)`, which never matches a `ty::Infer(_)`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                self.inner.borrow_mut().float_unification_table().probe_value(v).is_known()
            }

            TyOrConstInferVar::Const(v) => {
                // If `probe_value` returns a `Known` value, it never equals
                // `ty::ConstKind::Infer(ty::InferConst::Var(v))`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                match self.inner.borrow_mut().const_unification_table().probe_value(v) {
                    ConstVariableValue::Unknown { .. } => false,
                    ConstVariableValue::Known { .. } => true,
                }
            }
        }
    }

    /// Attach a callback to be invoked on each root obligation evaluated in the new trait solver.
    pub fn attach_obligation_inspector(&self, inspector: ObligationInspector<'tcx>) {
        debug_assert!(
            self.obligation_inspector.get().is_none(),
            "shouldn't override a set obligation inspector"
        );
        self.obligation_inspector.set(Some(inspector));
    }
}

/// Helper for [InferCtxt::ty_or_const_infer_var_changed] (see comment on that), currently
/// used only for `traits::fulfill`'s list of `stalled_on` inference variables.
#[derive(Copy, Clone, Debug)]
pub enum TyOrConstInferVar {
    /// Equivalent to `ty::Infer(ty::TyVar(_))`.
    Ty(TyVid),
    /// Equivalent to `ty::Infer(ty::IntVar(_))`.
    TyInt(IntVid),
    /// Equivalent to `ty::Infer(ty::FloatVar(_))`.
    TyFloat(FloatVid),

    /// Equivalent to `ty::ConstKind::Infer(ty::InferConst::Var(_))`.
    Const(ConstVid),
}

impl<'tcx> TyOrConstInferVar {
    /// Tries to extract an inference variable from a type or a constant, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`) and
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    pub fn maybe_from_generic_arg(arg: GenericArg<'tcx>) -> Option<Self> {
        match arg.kind() {
            GenericArgKind::Type(ty) => Self::maybe_from_ty(ty),
            GenericArgKind::Const(ct) => Self::maybe_from_const(ct),
            GenericArgKind::Lifetime(_) => None,
        }
    }

    /// Tries to extract an inference variable from a type or a constant, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`) and
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    pub fn maybe_from_term(term: Term<'tcx>) -> Option<Self> {
        match term.kind() {
            TermKind::Ty(ty) => Self::maybe_from_ty(ty),
            TermKind::Const(ct) => Self::maybe_from_const(ct),
        }
    }

    /// Tries to extract an inference variable from a type, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`).
    fn maybe_from_ty(ty: Ty<'tcx>) -> Option<Self> {
        match *ty.kind() {
            ty::Infer(ty::TyVar(v)) => Some(TyOrConstInferVar::Ty(v)),
            ty::Infer(ty::IntVar(v)) => Some(TyOrConstInferVar::TyInt(v)),
            ty::Infer(ty::FloatVar(v)) => Some(TyOrConstInferVar::TyFloat(v)),
            _ => None,
        }
    }

    /// Tries to extract an inference variable from a constant, returns `None`
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    fn maybe_from_const(ct: ty::Const<'tcx>) -> Option<Self> {
        match ct.kind() {
            ty::ConstKind::Infer(InferConst::Var(v)) => Some(TyOrConstInferVar::Const(v)),
            _ => None,
        }
    }
}

/// Replace `{integer}` with `i32` and `{float}` with `f64`.
/// Used only for diagnostics.
struct InferenceLiteralEraser<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for InferenceLiteralEraser<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match ty.kind() {
            ty::Infer(ty::IntVar(_) | ty::FreshIntTy(_)) => self.tcx.types.i32,
            ty::Infer(ty::FloatVar(_) | ty::FreshFloatTy(_)) => self.tcx.types.f64,
            _ => ty.super_fold_with(self),
        }
    }
}

impl<'tcx> TypeTrace<'tcx> {
    pub fn span(&self) -> Span {
        self.cause.span
    }

    pub fn types(cause: &ObligationCause<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(a.into(), b.into())),
        }
    }

    pub fn trait_refs(
        cause: &ObligationCause<'tcx>,
        a: ty::TraitRef<'tcx>,
        b: ty::TraitRef<'tcx>,
    ) -> TypeTrace<'tcx> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::TraitRefs(ExpectedFound::new(a, b)) }
    }

    pub fn consts(
        cause: &ObligationCause<'tcx>,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(a.into(), b.into())),
        }
    }
}

impl<'tcx> SubregionOrigin<'tcx> {
    pub fn span(&self) -> Span {
        match *self {
            Subtype(ref a) => a.span(),
            RelateObjectBound(a) => a,
            RelateParamBound(a, ..) => a,
            RelateRegionParamBound(a, _) => a,
            Reborrow(a) => a,
            ReferenceOutlivesReferent(_, a) => a,
            CompareImplItemObligation { span, .. } => span,
            AscribeUserTypeProvePredicate(span) => span,
            CheckAssociatedTypeBounds { ref parent, .. } => parent.span(),
        }
    }

    pub fn from_obligation_cause<F>(cause: &traits::ObligationCause<'tcx>, default: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match *cause.code() {
            traits::ObligationCauseCode::ReferenceOutlivesReferent(ref_type) => {
                SubregionOrigin::ReferenceOutlivesReferent(ref_type, cause.span)
            }

            traits::ObligationCauseCode::CompareImplItem {
                impl_item_def_id,
                trait_item_def_id,
                kind: _,
            } => SubregionOrigin::CompareImplItemObligation {
                span: cause.span,
                impl_item_def_id,
                trait_item_def_id,
            },

            traits::ObligationCauseCode::CheckAssociatedTypeBounds {
                impl_item_def_id,
                trait_item_def_id,
            } => SubregionOrigin::CheckAssociatedTypeBounds {
                impl_item_def_id,
                trait_item_def_id,
                parent: Box::new(default()),
            },

            traits::ObligationCauseCode::AscribeUserTypeProvePredicate(span) => {
                SubregionOrigin::AscribeUserTypeProvePredicate(span)
            }

            traits::ObligationCauseCode::ObjectTypeBound(ty, _reg) => {
                SubregionOrigin::RelateRegionParamBound(cause.span, Some(ty))
            }

            _ => default(),
        }
    }
}

impl RegionVariableOrigin {
    pub fn span(&self) -> Span {
        match *self {
            MiscVariable(a)
            | PatternRegion(a)
            | BorrowRegion(a)
            | Autoref(a)
            | Coercion(a)
            | RegionParameterDefinition(a, ..)
            | BoundRegion(a, ..)
            | UpvarRegion(_, a) => a,
            Nll(..) => bug!("NLL variable used with `span`"),
        }
    }
}

impl<'tcx> InferCtxt<'tcx> {
    /// Given a [`hir::Block`], get the span of its last expression or
    /// statement, peeling off any inner blocks.
    pub fn find_block_span(&self, block: &'tcx hir::Block<'tcx>) -> Span {
        let block = block.innermost_block();
        if let Some(expr) = &block.expr {
            expr.span
        } else if let Some(stmt) = block.stmts.last() {
            // possibly incorrect trailing `;` in the else arm
            stmt.span
        } else {
            // empty block; point at its entirety
            block.span
        }
    }

    /// Given a [`hir::HirId`] for a block, get the span of its last expression
    /// or statement, peeling off any inner blocks.
    pub fn find_block_span_from_hir_id(&self, hir_id: hir::HirId) -> Span {
        match self.tcx.hir_node(hir_id) {
            hir::Node::Block(blk) => self.find_block_span(blk),
            // The parser was in a weird state if either of these happen, but
            // it's better not to panic.
            hir::Node::Expr(e) => e.span,
            _ => rustc_span::DUMMY_SP,
        }
    }
}
