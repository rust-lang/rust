pub use self::freshen::TypeFreshener;
pub use self::lexical_region_resolve::RegionResolutionError;
pub use self::LateBoundRegionConversionTime::*;
pub use self::RegionVariableOrigin::*;
pub use self::SubregionOrigin::*;
pub use self::ValuePairs::*;
pub use combine::ObligationEmittingRelation;

use self::opaque_types::OpaqueTypeStorage;
pub(crate) use self::undo_log::{InferCtxtUndoLogs, Snapshot, UndoLog};

use crate::traits::{self, ObligationCause, PredicateObligations, TraitEngine, TraitEngineExt};

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::undo_log::Rollback;
use rustc_data_structures::unify as ut;
use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::infer::canonical::{Canonical, CanonicalVarValues};
use rustc_middle::infer::unify_key::{ConstVarValue, ConstVariableValue};
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind, ToType};
use rustc_middle::mir::interpret::{ErrorHandled, EvalToValTreeResult};
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::traits::select;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::fold::BoundVarReplacerDelegate;
use rustc_middle::ty::fold::{ir::TypeFolder, TypeFoldable, TypeSuperFoldable};
use rustc_middle::ty::relate::RelateResult;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, InternalSubsts, SubstsRef};
use rustc_middle::ty::visit::TypeVisitable;
pub use rustc_middle::ty::IntVarValue;
use rustc_middle::ty::{self, GenericParamDefKind, InferConst, InferTy, Ty, TyCtxt};
use rustc_middle::ty::{ConstVid, FloatVid, IntVid, TyVid};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

use std::cell::{Cell, RefCell};
use std::fmt;

use self::combine::CombineFields;
use self::error_reporting::TypeErrCtxt;
use self::free_regions::RegionRelations;
use self::lexical_region_resolve::LexicalRegionResolutions;
use self::outlives::env::OutlivesEnvironment;
use self::region_constraints::{GenericKind, RegionConstraintData, VarInfos, VerifyBound};
use self::region_constraints::{
    RegionConstraintCollector, RegionConstraintStorage, RegionSnapshot,
};
use self::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};

pub mod at;
pub mod canonical;
mod combine;
mod equate;
pub mod error_reporting;
pub mod free_regions;
mod freshen;
mod fudge;
mod glb;
mod higher_ranked;
pub mod lattice;
mod lexical_region_resolve;
mod lub;
pub mod nll_relate;
pub mod opaque_types;
pub mod outlives;
mod projection;
pub mod region_constraints;
pub mod resolve;
mod sub;
pub mod type_variable;
mod undo_log;

#[must_use]
#[derive(Debug)]
pub struct InferOk<'tcx, T> {
    pub value: T,
    pub obligations: PredicateObligations<'tcx>,
}
pub type InferResult<'tcx, T> = Result<InferOk<'tcx, T>, TypeError<'tcx>>;

pub type UnitResult<'tcx> = RelateResult<'tcx, ()>; // "unify result"
pub type FixupResult<'tcx, T> = Result<T, FixupError<'tcx>>; // "fixup result"

pub(crate) type UnificationTable<'a, 'tcx, T> = ut::UnificationTable<
    ut::InPlace<T, &'a mut ut::UnificationStorage<T>, &'a mut InferCtxtUndoLogs<'tcx>>,
>;

/// This type contains all the things within `InferCtxt` that sit within a
/// `RefCell` and are involved with taking/rolling back snapshots. Snapshot
/// operations are hot enough that we want only one call to `borrow_mut` per
/// call to `start_snapshot` and `rollback_to`.
#[derive(Clone)]
pub struct InferCtxtInner<'tcx> {
    /// Cache for projections. This cache is snapshotted along with the infcx.
    ///
    /// Public so that `traits::project` can use it.
    pub projection_cache: traits::ProjectionCacheStorage<'tcx>,

    /// We instantiate `UnificationTable` with `bounds<Ty>` because the types
    /// that might instantiate a general type variable have an order,
    /// represented by its upper and lower bounds.
    type_variable_storage: type_variable::TypeVariableStorage<'tcx>,

    /// Map from const parameter variable to the kind of const it represents.
    const_unification_storage: ut::UnificationTableStorage<ty::ConstVid<'tcx>>,

    /// Map from integral variable to the kind of integer it represents.
    int_unification_storage: ut::UnificationTableStorage<ty::IntVid>,

    /// Map from floating variable to the kind of float it represents.
    float_unification_storage: ut::UnificationTableStorage<ty::FloatVid>,

    /// Tracks the set of region variables and the constraints between them.
    /// This is initially `Some(_)` but when
    /// `resolve_regions_and_report_errors` is invoked, this gets set to `None`
    /// -- further attempts to perform unification, etc., may fail if new
    /// region constraints would've been added.
    region_constraint_storage: Option<RegionConstraintStorage<'tcx>>,

    /// A set of constraints that regionck must validate. Each
    /// constraint has the form `T:'a`, meaning "some type `T` must
    /// outlive the lifetime 'a". These constraints derive from
    /// instantiated type parameters. So if you had a struct defined
    /// like
    /// ```ignore (illustrative)
    ///     struct Foo<T:'static> { ... }
    /// ```
    /// then in some expression `let x = Foo { ... }` it will
    /// instantiate the type parameter `T` with a fresh type `$0`. At
    /// the same time, it will record a region obligation of
    /// `$0:'static`. This will get checked later by regionck. (We
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
    region_obligations: Vec<RegionObligation<'tcx>>,

    undo_log: InferCtxtUndoLogs<'tcx>,

    /// Caches for opaque type inference.
    pub opaque_type_storage: OpaqueTypeStorage<'tcx>,
}

impl<'tcx> InferCtxtInner<'tcx> {
    fn new() -> InferCtxtInner<'tcx> {
        InferCtxtInner {
            projection_cache: Default::default(),
            type_variable_storage: type_variable::TypeVariableStorage::new(),
            undo_log: InferCtxtUndoLogs::default(),
            const_unification_storage: ut::UnificationTableStorage::new(),
            int_unification_storage: ut::UnificationTableStorage::new(),
            float_unification_storage: ut::UnificationTableStorage::new(),
            region_constraint_storage: Some(RegionConstraintStorage::new()),
            region_obligations: vec![],
            opaque_type_storage: Default::default(),
        }
    }

    #[inline]
    pub fn region_obligations(&self) -> &[RegionObligation<'tcx>] {
        &self.region_obligations
    }

    #[inline]
    pub fn projection_cache(&mut self) -> traits::ProjectionCache<'_, 'tcx> {
        self.projection_cache.with_log(&mut self.undo_log)
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
    fn int_unification_table(
        &mut self,
    ) -> ut::UnificationTable<
        ut::InPlace<
            ty::IntVid,
            &mut ut::UnificationStorage<ty::IntVid>,
            &mut InferCtxtUndoLogs<'tcx>,
        >,
    > {
        self.int_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    fn float_unification_table(
        &mut self,
    ) -> ut::UnificationTable<
        ut::InPlace<
            ty::FloatVid,
            &mut ut::UnificationStorage<ty::FloatVid>,
            &mut InferCtxtUndoLogs<'tcx>,
        >,
    > {
        self.float_unification_storage.with_log(&mut self.undo_log)
    }

    #[inline]
    fn const_unification_table(
        &mut self,
    ) -> ut::UnificationTable<
        ut::InPlace<
            ty::ConstVid<'tcx>,
            &mut ut::UnificationStorage<ty::ConstVid<'tcx>>,
            &mut InferCtxtUndoLogs<'tcx>,
        >,
    > {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefiningAnchor {
    /// `DefId` of the item.
    Bind(LocalDefId),
    /// When opaque types are not resolved, we `Bubble` up, meaning
    /// return the opaque/hidden type pair from query, for caller of query to handle it.
    Bubble,
    /// Used to catch type mismatch errors when handling opaque types.
    Error,
}

pub struct InferCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,

    /// The `DefId` of the item in whose context we are performing inference or typeck.
    /// It is used to check whether an opaque type use is a defining use.
    ///
    /// If it is `DefiningAnchor::Bubble`, we can't resolve opaque types here and need to bubble up
    /// the obligation. This frequently happens for
    /// short lived InferCtxt within queries. The opaque type obligations are forwarded
    /// to the outside until the end up in an `InferCtxt` for typeck or borrowck.
    ///
    /// It is default value is `DefiningAnchor::Error`, this way it is easier to catch errors that
    /// might come up during inference or typeck.
    pub defining_use_anchor: DefiningAnchor,

    /// Whether this inference context should care about region obligations in
    /// the root universe. Most notably, this is used during hir typeck as region
    /// solving is left to borrowck instead.
    pub considering_regions: bool,

    pub inner: RefCell<InferCtxtInner<'tcx>>,

    /// If set, this flag causes us to skip the 'leak check' during
    /// higher-ranked subtyping operations. This flag is a temporary one used
    /// to manage the removal of the leak-check: for the time being, we still run the
    /// leak-check, but we issue warnings. This flag can only be set to true
    /// when entering a snapshot.
    skip_leak_check: Cell<bool>,

    /// Once region inference is done, the values for each variable.
    lexical_region_resolutions: RefCell<Option<LexicalRegionResolutions<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that have to do with the parameters in scope.
    pub selection_cache: select::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation.
    pub evaluation_cache: select::EvaluationCache<'tcx>,

    /// the set of predicates on which errors have been reported, to
    /// avoid reporting the same error twice.
    pub reported_trait_errors: RefCell<FxIndexMap<Span, Vec<ty::Predicate<'tcx>>>>,

    pub reported_closure_mismatch: RefCell<FxHashSet<(Span, Option<Span>)>>,

    /// When an error occurs, we want to avoid reporting "derived"
    /// errors that are due to this original failure. Normally, we
    /// handle this with the `err_count_on_creation` count, which
    /// basically just tracks how many errors were reported when we
    /// started type-checking a fn and checks to see if any new errors
    /// have been reported since then. Not great, but it works.
    ///
    /// However, when errors originated in other passes -- notably
    /// resolve -- this heuristic breaks down. Therefore, we have this
    /// auxiliary flag that one can set whenever one creates a
    /// type-error that is due to an error in a prior pass.
    ///
    /// Don't read this flag directly, call `is_tainted_by_errors()`
    /// and `set_tainted_by_errors()`.
    tainted_by_errors: Cell<Option<ErrorGuaranteed>>,

    /// Track how many errors were reported when this infcx is created.
    /// If the number of errors increases, that's also a sign (line
    /// `tainted_by_errors`) to avoid reporting certain kinds of errors.
    // FIXME(matthewjasper) Merge into `tainted_by_errors`
    err_count_on_creation: usize,

    /// This flag is true while there is an active snapshot.
    in_snapshot: Cell<bool>,

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

    /// During coherence we have to assume that other crates may add
    /// additional impls which we currently don't know about.
    ///
    /// To deal with this evaluation should be conservative
    /// and consider the possibility of impls from outside this crate.
    /// This comes up primarily when resolving ambiguity. Imagine
    /// there is some trait reference `$0: Bar` where `$0` is an
    /// inference variable. If `intercrate` is true, then we can never
    /// say for sure that this reference is not implemented, even if
    /// there are *no impls at all for `Bar`*, because `$0` could be
    /// bound to some type that in a downstream crate that implements
    /// `Bar`.
    ///
    /// Outside of coherence we set this to false because we are only
    /// interested in types that the user could actually have written.
    /// In other words, we consider `$0: Bar` to be unimplemented if
    /// there is no type that the user could *actually name* that
    /// would satisfy it. This avoids crippling inference, basically.
    pub intercrate: bool,
}

/// See the `error_reporting` module for more details.
#[derive(Clone, Copy, Debug, PartialEq, Eq, TypeFoldable, TypeVisitable)]
pub enum ValuePairs<'tcx> {
    Regions(ExpectedFound<ty::Region<'tcx>>),
    Terms(ExpectedFound<ty::Term<'tcx>>),
    TraitRefs(ExpectedFound<ty::TraitRef<'tcx>>),
    PolyTraitRefs(ExpectedFound<ty::PolyTraitRef<'tcx>>),
    Sigs(ExpectedFound<ty::FnSig<'tcx>>),
}

impl<'tcx> ValuePairs<'tcx> {
    pub fn ty(&self) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
        if let ValuePairs::Terms(ExpectedFound { expected, found }) = self
            && let Some(expected) = expected.ty()
            && let Some(found) = found.ty()
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
    /// relating `'a` to `'b`
    RelateObjectBound(Span),

    /// Some type parameter was instantiated with the given type,
    /// and that type must outlive some region.
    RelateParamBound(Span, Ty<'tcx>, Option<Span>),

    /// The given region parameter was instantiated with a region
    /// that must outlive some other region.
    RelateRegionParamBound(Span),

    /// Creating a pointer `b` to contents of another reference
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

    /// Checking that the bounds of a trait's associated type hold for a given impl
    CheckAssociatedTypeBounds {
        parent: Box<SubregionOrigin<'tcx>>,
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
    },

    AscribeUserTypeProvePredicate(Span),
}

// `SubregionOrigin` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(SubregionOrigin<'_>, 32);

impl<'tcx> SubregionOrigin<'tcx> {
    pub fn to_constraint_category(&self) -> ConstraintCategory<'tcx> {
        match self {
            Self::Subtype(type_trace) => type_trace.cause.to_constraint_category(),
            Self::AscribeUserTypeProvePredicate(span) => ConstraintCategory::Predicate(*span),
            _ => ConstraintCategory::BoringNoLocation,
        }
    }
}

/// Times when we replace late-bound regions with variables:
#[derive(Clone, Copy, Debug)]
pub enum LateBoundRegionConversionTime {
    /// when a fn is called
    FnCall,

    /// when two higher-ranked types are compared
    HigherRankedType,

    /// when projecting an associated type
    AssocTypeProjection(DefId),
}

/// Reasons to create a region inference variable
///
/// See `error_reporting` module for more details
#[derive(Copy, Clone, Debug)]
pub enum RegionVariableOrigin {
    /// Region variables created for ill-categorized reasons,
    /// mostly indicates places in need of refactoring
    MiscVariable(Span),

    /// Regions created by a `&P` or `[...]` pattern
    PatternRegion(Span),

    /// Regions created by `&` operator
    AddrOfRegion(Span),

    /// Regions created as part of an autoref of a method receiver
    Autoref(Span),

    /// Regions created as part of an automatic coercion
    Coercion(Span),

    /// Region variables created as the values for early-bound regions
    EarlyBoundRegion(Span, Symbol),

    /// Region variables created for bound regions
    /// in a function or method that is called
    LateBoundRegion(Span, ty::BoundRegionKind, LateBoundRegionConversionTime),

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

// FIXME(eddyb) investigate overlap between this and `TyOrConstInferVar`.
#[derive(Copy, Clone, Debug)]
pub enum FixupError<'tcx> {
    UnresolvedIntTy(IntVid),
    UnresolvedFloatTy(FloatVid),
    UnresolvedTy(TyVid),
    UnresolvedConst(ConstVid<'tcx>),
}

/// See the `region_obligations` field for more information.
#[derive(Clone, Debug)]
pub struct RegionObligation<'tcx> {
    pub sub_region: ty::Region<'tcx>,
    pub sup_type: Ty<'tcx>,
    pub origin: SubregionOrigin<'tcx>,
}

impl<'tcx> fmt::Display for FixupError<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::FixupError::*;

        match *self {
            UnresolvedIntTy(_) => write!(
                f,
                "cannot determine the type of this integer; \
                 add a suffix to specify the type explicitly"
            ),
            UnresolvedFloatTy(_) => write!(
                f,
                "cannot determine the type of this number; \
                 add a suffix to specify the type explicitly"
            ),
            UnresolvedTy(_) => write!(f, "unconstrained type"),
            UnresolvedConst(_) => write!(f, "unconstrained const value"),
        }
    }
}

/// Used to configure inference contexts before their creation
pub struct InferCtxtBuilder<'tcx> {
    tcx: TyCtxt<'tcx>,
    defining_use_anchor: DefiningAnchor,
    considering_regions: bool,
    /// Whether we are in coherence mode.
    intercrate: bool,
}

pub trait TyCtxtInferExt<'tcx> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'tcx>;
}

impl<'tcx> TyCtxtInferExt<'tcx> for TyCtxt<'tcx> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'tcx> {
        InferCtxtBuilder {
            tcx: self,
            defining_use_anchor: DefiningAnchor::Error,
            considering_regions: true,
            intercrate: false,
        }
    }
}

impl<'tcx> InferCtxtBuilder<'tcx> {
    /// Whenever the `InferCtxt` should be able to handle defining uses of opaque types,
    /// you need to call this function. Otherwise the opaque type will be treated opaquely.
    ///
    /// It is only meant to be called in two places, for typeck
    /// (via `Inherited::build`) and for the inference context used
    /// in mir borrowck.
    pub fn with_opaque_type_inference(mut self, defining_use_anchor: DefiningAnchor) -> Self {
        self.defining_use_anchor = defining_use_anchor;
        self
    }

    pub fn intercrate(mut self) -> Self {
        self.intercrate = true;
        self
    }

    pub fn ignoring_regions(mut self) -> Self {
        self.considering_regions = false;
        self
    }

    /// Given a canonical value `C` as a starting point, create an
    /// inference context that contains each of the bound values
    /// within instantiated as a fresh variable. The `f` closure is
    /// invoked with the new infcx, along with the instantiated value
    /// `V` and a substitution `S`. This substitution `S` maps from
    /// the bound values in `C` to their instantiated values in `V`
    /// (in other words, `S(C) = V`).
    pub fn build_with_canonical<T>(
        &mut self,
        span: Span,
        canonical: &Canonical<'tcx, T>,
    ) -> (InferCtxt<'tcx>, T, CanonicalVarValues<'tcx>)
    where
        T: TypeFoldable<'tcx>,
    {
        let infcx = self.build();
        let (value, subst) = infcx.instantiate_canonical_with_fresh_inference_vars(span, canonical);
        (infcx, value, subst)
    }

    pub fn build(&mut self) -> InferCtxt<'tcx> {
        let InferCtxtBuilder { tcx, defining_use_anchor, considering_regions, intercrate } = *self;
        InferCtxt {
            tcx,
            defining_use_anchor,
            considering_regions,
            inner: RefCell::new(InferCtxtInner::new()),
            lexical_region_resolutions: RefCell::new(None),
            selection_cache: Default::default(),
            evaluation_cache: Default::default(),
            reported_trait_errors: Default::default(),
            reported_closure_mismatch: Default::default(),
            tainted_by_errors: Cell::new(None),
            err_count_on_creation: tcx.sess.err_count(),
            in_snapshot: Cell::new(false),
            skip_leak_check: Cell::new(false),
            universe: Cell::new(ty::UniverseIndex::ROOT),
            intercrate,
        }
    }
}

impl<'tcx, T> InferOk<'tcx, T> {
    pub fn unit(self) -> InferOk<'tcx, ()> {
        InferOk { value: (), obligations: self.obligations }
    }

    /// Extracts `value`, registering any obligations into `fulfill_cx`.
    pub fn into_value_registering_obligations(
        self,
        infcx: &InferCtxt<'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
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

#[must_use = "once you start a snapshot, you should always consume it"]
pub struct CombinedSnapshot<'tcx> {
    undo_snapshot: Snapshot<'tcx>,
    region_constraints_snapshot: RegionSnapshot,
    universe: ty::UniverseIndex,
    was_in_snapshot: bool,
}

impl<'tcx> InferCtxt<'tcx> {
    /// Creates a `TypeErrCtxt` for emitting various inference errors.
    /// During typeck, use `FnCtxt::err_ctxt` instead.
    pub fn err_ctxt(&self) -> TypeErrCtxt<'_, 'tcx> {
        TypeErrCtxt {
            infcx: self,
            typeck_results: None,
            fallback_has_occurred: false,
            normalize_fn_sig: Box::new(|fn_sig| fn_sig),
            autoderef_steps: Box::new(|ty| {
                debug_assert!(false, "shouldn't be using autoderef_steps outside of typeck");
                vec![(ty, vec![])]
            }),
        }
    }

    pub fn is_in_snapshot(&self) -> bool {
        self.in_snapshot.get()
    }

    pub fn freshen<T: TypeFoldable<'tcx>>(&self, t: T) -> T {
        t.fold_with(&mut self.freshener())
    }

    /// Returns the origin of the type variable identified by `vid`, or `None`
    /// if this is not a type variable.
    ///
    /// No attempt is made to resolve `ty`.
    pub fn type_var_origin(&self, ty: Ty<'tcx>) -> Option<TypeVariableOrigin> {
        match *ty.kind() {
            ty::Infer(ty::TyVar(vid)) => {
                Some(*self.inner.borrow_mut().type_variables().var_origin(vid))
            }
            _ => None,
        }
    }

    pub fn freshener<'b>(&'b self) -> TypeFreshener<'b, 'tcx> {
        freshen::TypeFreshener::new(self, false)
    }

    /// Like `freshener`, but does not replace `'static` regions.
    pub fn freshener_keep_static<'b>(&'b self) -> TypeFreshener<'b, 'tcx> {
        freshen::TypeFreshener::new(self, true)
    }

    pub fn unsolved_variables(&self) -> Vec<Ty<'tcx>> {
        let mut inner = self.inner.borrow_mut();
        let mut vars: Vec<Ty<'_>> = inner
            .type_variables()
            .unsolved_variables()
            .into_iter()
            .map(|t| self.tcx.mk_ty_var(t))
            .collect();
        vars.extend(
            (0..inner.int_unification_table().len())
                .map(|i| ty::IntVid { index: i as u32 })
                .filter(|&vid| inner.int_unification_table().probe_value(vid).is_none())
                .map(|v| self.tcx.mk_int_var(v)),
        );
        vars.extend(
            (0..inner.float_unification_table().len())
                .map(|i| ty::FloatVid { index: i as u32 })
                .filter(|&vid| inner.float_unification_table().probe_value(vid).is_none())
                .map(|v| self.tcx.mk_float_var(v)),
        );
        vars
    }

    fn combine_fields<'a>(
        &'a self,
        trace: TypeTrace<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        define_opaque_types: bool,
    ) -> CombineFields<'a, 'tcx> {
        CombineFields {
            infcx: self,
            trace,
            cause: None,
            param_env,
            obligations: PredicateObligations::new(),
            define_opaque_types,
        }
    }

    fn start_snapshot(&self) -> CombinedSnapshot<'tcx> {
        debug!("start_snapshot()");

        let in_snapshot = self.in_snapshot.replace(true);

        let mut inner = self.inner.borrow_mut();

        CombinedSnapshot {
            undo_snapshot: inner.undo_log.start_snapshot(),
            region_constraints_snapshot: inner.unwrap_region_constraints().start_snapshot(),
            universe: self.universe(),
            was_in_snapshot: in_snapshot,
        }
    }

    #[instrument(skip(self, snapshot), level = "debug")]
    fn rollback_to(&self, cause: &str, snapshot: CombinedSnapshot<'tcx>) {
        let CombinedSnapshot {
            undo_snapshot,
            region_constraints_snapshot,
            universe,
            was_in_snapshot,
        } = snapshot;

        self.in_snapshot.set(was_in_snapshot);
        self.universe.set(universe);

        let mut inner = self.inner.borrow_mut();
        inner.rollback_to(undo_snapshot);
        inner.unwrap_region_constraints().rollback_to(region_constraints_snapshot);
    }

    #[instrument(skip(self, snapshot), level = "debug")]
    fn commit_from(&self, snapshot: CombinedSnapshot<'tcx>) {
        let CombinedSnapshot {
            undo_snapshot,
            region_constraints_snapshot: _,
            universe: _,
            was_in_snapshot,
        } = snapshot;

        self.in_snapshot.set(was_in_snapshot);

        self.inner.borrow_mut().commit(undo_snapshot);
    }

    /// Execute `f` and commit the bindings if closure `f` returns `Ok(_)`.
    #[instrument(skip(self, f), level = "debug")]
    pub fn commit_if_ok<T, E, F>(&self, f: F) -> Result<T, E>
    where
        F: FnOnce(&CombinedSnapshot<'tcx>) -> Result<T, E>,
    {
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        debug!("commit_if_ok() -- r.is_ok() = {}", r.is_ok());
        match r {
            Ok(_) => {
                self.commit_from(snapshot);
            }
            Err(_) => {
                self.rollback_to("commit_if_ok -- error", snapshot);
            }
        }
        r
    }

    /// Execute `f` then unroll any bindings it creates.
    #[instrument(skip(self, f), level = "debug")]
    pub fn probe<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'tcx>) -> R,
    {
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        self.rollback_to("probe", snapshot);
        r
    }

    /// If `should_skip` is true, then execute `f` then unroll any bindings it creates.
    #[instrument(skip(self, f), level = "debug")]
    pub fn probe_maybe_skip_leak_check<R, F>(&self, should_skip: bool, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'tcx>) -> R,
    {
        let snapshot = self.start_snapshot();
        let was_skip_leak_check = self.skip_leak_check.get();
        if should_skip {
            self.skip_leak_check.set(true);
        }
        let r = f(&snapshot);
        self.rollback_to("probe", snapshot);
        self.skip_leak_check.set(was_skip_leak_check);
        r
    }

    /// Scan the constraints produced since `snapshot` began and returns:
    ///
    /// - `None` -- if none of them involve "region outlives" constraints
    /// - `Some(true)` -- if there are `'a: 'b` constraints where `'a` or `'b` is a placeholder
    /// - `Some(false)` -- if there are `'a: 'b` constraints but none involve placeholders
    pub fn region_constraints_added_in_snapshot(
        &self,
        snapshot: &CombinedSnapshot<'tcx>,
    ) -> Option<bool> {
        self.inner
            .borrow_mut()
            .unwrap_region_constraints()
            .region_constraints_added_in_snapshot(&snapshot.undo_snapshot)
    }

    pub fn opaque_types_added_in_snapshot(&self, snapshot: &CombinedSnapshot<'tcx>) -> bool {
        self.inner.borrow().undo_log.opaque_types_in_snapshot(&snapshot.undo_snapshot)
    }

    pub fn add_given(&self, sub: ty::Region<'tcx>, sup: ty::RegionVid) {
        self.inner.borrow_mut().unwrap_region_constraints().add_given(sub, sup);
    }

    pub fn can_sub<T>(&self, param_env: ty::ParamEnv<'tcx>, a: T, b: T) -> bool
    where
        T: at::ToTrace<'tcx>,
    {
        let origin = &ObligationCause::dummy();
        self.probe(|_| self.at(origin, param_env).sub(a, b).is_ok())
    }

    pub fn can_eq<T>(&self, param_env: ty::ParamEnv<'tcx>, a: T, b: T) -> bool
    where
        T: at::ToTrace<'tcx>,
    {
        let origin = &ObligationCause::dummy();
        self.probe(|_| self.at(origin, param_env).eq(a, b).is_ok())
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

    /// Require that the region `r` be equal to one of the regions in
    /// the set `regions`.
    #[instrument(skip(self), level = "debug")]
    pub fn member_constraint(
        &self,
        key: ty::OpaqueTypeKey<'tcx>,
        definition_span: Span,
        hidden_ty: Ty<'tcx>,
        region: ty::Region<'tcx>,
        in_regions: &Lrc<Vec<ty::Region<'tcx>>>,
    ) {
        self.inner.borrow_mut().unwrap_region_constraints().member_constraint(
            key,
            definition_span,
            hidden_ty,
            region,
            in_regions,
        );
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
                self.inner.borrow_mut().type_variables().sub(a_vid, b_vid);
                return Err((a_vid, b_vid));
            }
            _ => {}
        }

        Ok(self.commit_if_ok(|_snapshot| {
            let ty::SubtypePredicate { a_is_expected, a, b } =
                self.instantiate_binder_with_placeholders(predicate);

            let ok = self.at(cause, param_env).sub_exp(a_is_expected, a, b)?;

            Ok(ok.unit())
        }))
    }

    pub fn region_outlives_predicate(
        &self,
        cause: &traits::ObligationCause<'tcx>,
        predicate: ty::PolyRegionOutlivesPredicate<'tcx>,
    ) {
        let ty::OutlivesPredicate(r_a, r_b) = self.instantiate_binder_with_placeholders(predicate);
        let origin =
            SubregionOrigin::from_obligation_cause(cause, || RelateRegionParamBound(cause.span));
        self.sub_regions(origin, r_b, r_a); // `b : a` ==> `a <= b`
    }

    /// Number of type variables created so far.
    pub fn num_ty_vars(&self) -> usize {
        self.inner.borrow_mut().type_variables().num_vars()
    }

    pub fn next_ty_var_id(&self, origin: TypeVariableOrigin) -> TyVid {
        self.inner.borrow_mut().type_variables().new_var(self.universe(), origin)
    }

    pub fn next_ty_var(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        self.tcx.mk_ty_var(self.next_ty_var_id(origin))
    }

    pub fn next_ty_var_id_in_universe(
        &self,
        origin: TypeVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> TyVid {
        self.inner.borrow_mut().type_variables().new_var(universe, origin)
    }

    pub fn next_ty_var_in_universe(
        &self,
        origin: TypeVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> Ty<'tcx> {
        let vid = self.next_ty_var_id_in_universe(origin, universe);
        self.tcx.mk_ty_var(vid)
    }

    pub fn next_const_var(&self, ty: Ty<'tcx>, origin: ConstVariableOrigin) -> ty::Const<'tcx> {
        self.tcx.mk_const(self.next_const_var_id(origin), ty)
    }

    pub fn next_const_var_in_universe(
        &self,
        ty: Ty<'tcx>,
        origin: ConstVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> ty::Const<'tcx> {
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table()
            .new_key(ConstVarValue { origin, val: ConstVariableValue::Unknown { universe } });
        self.tcx.mk_const(vid, ty)
    }

    pub fn next_const_var_id(&self, origin: ConstVariableOrigin) -> ConstVid<'tcx> {
        self.inner.borrow_mut().const_unification_table().new_key(ConstVarValue {
            origin,
            val: ConstVariableValue::Unknown { universe: self.universe() },
        })
    }

    fn next_int_var_id(&self) -> IntVid {
        self.inner.borrow_mut().int_unification_table().new_key(None)
    }

    pub fn next_int_var(&self) -> Ty<'tcx> {
        self.tcx.mk_int_var(self.next_int_var_id())
    }

    fn next_float_var_id(&self) -> FloatVid {
        self.inner.borrow_mut().float_unification_table().new_key(None)
    }

    pub fn next_float_var(&self) -> Ty<'tcx> {
        self.tcx.mk_float_var(self.next_float_var_id())
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
        self.tcx.mk_re_var(region_var)
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
    pub fn next_nll_region_var(&self, origin: NllRegionVariableOrigin) -> ty::Region<'tcx> {
        self.next_region_var(RegionVariableOrigin::Nll(origin))
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
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
                self.next_region_var(EarlyBoundRegion(span, param.name)).into()
            }
            GenericParamDefKind::Type { .. } => {
                // Create a type inference variable for the given
                // type parameter definition. The substitutions are
                // for actual parameters that may be referred to by
                // the default of this type parameter, if it exists.
                // e.g., `struct Foo<A, B, C = (A, B)>(...);` when
                // used in a path such as `Foo::<T, U>::new()` will
                // use an inference variable for `C` with `[T, U]`
                // as the substitutions for the default, `(T, U)`.
                let ty_var_id = self.inner.borrow_mut().type_variables().new_var(
                    self.universe(),
                    TypeVariableOrigin {
                        kind: TypeVariableOriginKind::TypeParameterDefinition(
                            param.name,
                            Some(param.def_id),
                        ),
                        span,
                    },
                );

                self.tcx.mk_ty_var(ty_var_id).into()
            }
            GenericParamDefKind::Const { .. } => {
                let origin = ConstVariableOrigin {
                    kind: ConstVariableOriginKind::ConstParameterDefinition(
                        param.name,
                        param.def_id,
                    ),
                    span,
                };
                let const_var_id =
                    self.inner.borrow_mut().const_unification_table().new_key(ConstVarValue {
                        origin,
                        val: ConstVariableValue::Unknown { universe: self.universe() },
                    });
                self.tcx
                    .mk_const(
                        const_var_id,
                        self.tcx
                            .type_of(param.def_id)
                            .no_bound_vars()
                            .expect("const parameter types cannot be generic"),
                    )
                    .into()
            }
        }
    }

    /// Given a set of generics defined on a type or impl, returns a substitution mapping each
    /// type/region parameter to a fresh inference variable.
    pub fn fresh_substs_for_item(&self, span: Span, def_id: DefId) -> SubstsRef<'tcx> {
        InternalSubsts::for_item(self.tcx, def_id, |param, _| self.var_for_def(span, param))
    }

    /// Returns `true` if errors have been reported since this infcx was
    /// created. This is sometimes used as a heuristic to skip
    /// reporting errors that often occur as a result of earlier
    /// errors, but where it's hard to be 100% sure (e.g., unresolved
    /// inference variables, regionck errors).
    #[must_use = "this method does not have any side effects"]
    pub fn tainted_by_errors(&self) -> Option<ErrorGuaranteed> {
        debug!(
            "is_tainted_by_errors(err_count={}, err_count_on_creation={}, \
             tainted_by_errors={})",
            self.tcx.sess.err_count(),
            self.err_count_on_creation,
            self.tainted_by_errors.get().is_some()
        );

        if let Some(e) = self.tainted_by_errors.get() {
            return Some(e);
        }

        if self.tcx.sess.err_count() > self.err_count_on_creation {
            // errors reported since this infcx was made
            let e = self.tcx.sess.has_errors().unwrap();
            self.set_tainted_by_errors(e);
            return Some(e);
        }

        None
    }

    /// Set the "tainted by errors" flag to true. We call this when we
    /// observe an error from a prior pass.
    pub fn set_tainted_by_errors(&self, e: ErrorGuaranteed) {
        debug!("set_tainted_by_errors(ErrorGuaranteed)");
        self.tainted_by_errors.set(Some(e));
    }

    pub fn skip_region_resolution(&self) {
        let (var_infos, _) = {
            let mut inner = self.inner.borrow_mut();
            let inner = &mut *inner;
            // Note: `inner.region_obligations` may not be empty, because we
            // didn't necessarily call `process_registered_region_obligations`.
            // This is okay, because that doesn't introduce new vars.
            inner
                .region_constraint_storage
                .take()
                .expect("regions already resolved")
                .with_log(&mut inner.undo_log)
                .into_infos_and_data()
        };

        let lexical_region_resolutions = LexicalRegionResolutions {
            values: rustc_index::vec::IndexVec::from_elem_n(
                crate::infer::lexical_region_resolve::VarValue::Value(self.tcx.lifetimes.re_erased),
                var_infos.len(),
            ),
        };

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());
    }

    /// Process the region constraints and return any errors that
    /// result. After this, no more unification operations should be
    /// done -- or the compiler will panic -- but it is legal to use
    /// `resolve_vars_if_possible` as well as `fully_resolve`.
    pub fn resolve_regions(
        &self,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        let (var_infos, data) = {
            let mut inner = self.inner.borrow_mut();
            let inner = &mut *inner;
            assert!(
                self.tainted_by_errors().is_some() || inner.region_obligations.is_empty(),
                "region_obligations not empty: {:#?}",
                inner.region_obligations
            );
            inner
                .region_constraint_storage
                .take()
                .expect("regions already resolved")
                .with_log(&mut inner.undo_log)
                .into_infos_and_data()
        };

        let region_rels = &RegionRelations::new(self.tcx, outlives_env.free_region_map());

        let (lexical_region_resolutions, errors) =
            lexical_region_resolve::resolve(outlives_env.param_env, region_rels, var_infos, data);

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());

        errors
    }
    /// Obtains (and clears) the current set of region
    /// constraints. The inference context is still usable: further
    /// unifications will simply add new constraints.
    ///
    /// This method is not meant to be used with normal lexical region
    /// resolution. Rather, it is used in the NLL mode as a kind of
    /// interim hack: basically we run normal type-check and generate
    /// region constraints as normal, but then we take them and
    /// translate them into the form that the NLL solver
    /// understands. See the NLL module for mode details.
    pub fn take_and_reset_region_constraints(&self) -> RegionConstraintData<'tcx> {
        assert!(
            self.inner.borrow().region_obligations.is_empty(),
            "region_obligations not empty: {:#?}",
            self.inner.borrow().region_obligations
        );

        self.inner.borrow_mut().unwrap_region_constraints().take_and_reset_data()
    }

    /// Gives temporary access to the region constraint data.
    pub fn with_region_constraints<R>(
        &self,
        op: impl FnOnce(&RegionConstraintData<'tcx>) -> R,
    ) -> R {
        let mut inner = self.inner.borrow_mut();
        op(inner.unwrap_region_constraints().data())
    }

    pub fn region_var_origin(&self, vid: ty::RegionVid) -> RegionVariableOrigin {
        let mut inner = self.inner.borrow_mut();
        let inner = &mut *inner;
        inner
            .region_constraint_storage
            .as_mut()
            .expect("regions already resolved")
            .with_log(&mut inner.undo_log)
            .var_origin(vid)
    }

    /// Takes ownership of the list of variable regions. This implies
    /// that all the region constraints have already been taken, and
    /// hence that `resolve_regions_and_report_errors` can never be
    /// called. This is used only during NLL processing to "hand off" ownership
    /// of the set of region variables into the NLL region context.
    pub fn take_region_var_origins(&self) -> VarInfos {
        let mut inner = self.inner.borrow_mut();
        let (var_infos, data) = inner
            .region_constraint_storage
            .take()
            .expect("regions already resolved")
            .with_log(&mut inner.undo_log)
            .into_infos_and_data();
        assert!(data.is_empty());
        var_infos
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub fn take_opaque_types(&self) -> opaque_types::OpaqueTypeMap<'tcx> {
        debug_assert_ne!(self.defining_use_anchor, DefiningAnchor::Error);
        std::mem::take(&mut self.inner.borrow_mut().opaque_type_storage.opaque_types)
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

    /// Resolve any type variables found in `value` -- but only one
    /// level. So, if the variable `?X` is bound to some type
    /// `Foo<?Y>`, then this would return `Foo<?Y>` (but `?Y` may
    /// itself be bound to a type).
    ///
    /// Useful when you only need to inspect the outermost level of
    /// the type and don't care about nested types (or perhaps you
    /// will be resolving them as well, e.g. in a loop).
    pub fn shallow_resolve<T>(&self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        value.fold_with(&mut ShallowResolver { infcx: self })
    }

    pub fn root_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.inner.borrow_mut().type_variables().root_var(var)
    }

    /// Where possible, replaces type/const variables in
    /// `value` with their final value. Note that region variables
    /// are unaffected. If a type/const variable has not been unified, it
    /// is left as is. This is an idempotent operation that does
    /// not affect inference state in any way and so you can do it
    /// at will.
    pub fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        if !value.has_non_region_infer() {
            return value;
        }
        let mut r = resolve::OpportunisticVarResolver::new(self);
        value.fold_with(&mut r)
    }

    pub fn resolve_numeric_literals_with_default<T>(&self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        if !value.needs_infer() {
            return value; // Avoid duplicated subst-folding.
        }
        let mut r = InferenceLiteralEraser { tcx: self.tcx };
        value.fold_with(&mut r)
    }

    /// Returns the first unresolved type or const variable contained in `T`.
    pub fn first_unresolved_const_or_ty_var<T>(
        &self,
        value: &T,
    ) -> Option<(ty::Term<'tcx>, Option<Span>)>
    where
        T: TypeVisitable<'tcx>,
    {
        value.visit_with(&mut resolve::UnresolvedTypeOrConstFinder::new(self)).break_value()
    }

    pub fn probe_const_var(
        &self,
        vid: ty::ConstVid<'tcx>,
    ) -> Result<ty::Const<'tcx>, ty::UniverseIndex> {
        match self.inner.borrow_mut().const_unification_table().probe_value(vid).val {
            ConstVariableValue::Known { value } => Ok(value),
            ConstVariableValue::Unknown { universe } => Err(universe),
        }
    }

    pub fn fully_resolve<T: TypeFoldable<'tcx>>(&self, value: T) -> FixupResult<'tcx, T> {
        /*!
         * Attempts to resolve all type/region/const variables in
         * `value`. Region inference must have been run already (e.g.,
         * by calling `resolve_regions_and_report_errors`). If some
         * variable was never unified, an `Err` results.
         *
         * This method is idempotent, but it not typically not invoked
         * except during the writeback phase.
         */

        let value = resolve::fully_resolve(self, value);
        assert!(
            value.as_ref().map_or(true, |value| !value.needs_infer()),
            "`{value:?}` is not fully resolved"
        );
        value
    }

    // Instantiates the bound variables in a given binder with fresh inference
    // variables in the current universe.
    //
    // Use this method if you'd like to find some substitution of the binder's
    // variables (e.g. during a method call). If there isn't a [`LateBoundRegionConversionTime`]
    // that corresponds to your use case, consider whether or not you should
    // use [`InferCtxt::instantiate_binder_with_placeholders`] instead.
    pub fn instantiate_binder_with_fresh_vars<T>(
        &self,
        span: Span,
        lbrct: LateBoundRegionConversionTime,
        value: ty::Binder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<'tcx> + Copy,
    {
        if let Some(inner) = value.no_bound_vars() {
            return inner;
        }

        struct ToFreshVars<'a, 'tcx> {
            infcx: &'a InferCtxt<'tcx>,
            span: Span,
            lbrct: LateBoundRegionConversionTime,
            map: FxHashMap<ty::BoundVar, ty::GenericArg<'tcx>>,
        }

        impl<'tcx> BoundVarReplacerDelegate<'tcx> for ToFreshVars<'_, 'tcx> {
            fn replace_region(&mut self, br: ty::BoundRegion) -> ty::Region<'tcx> {
                self.map
                    .entry(br.var)
                    .or_insert_with(|| {
                        self.infcx
                            .next_region_var(LateBoundRegion(self.span, br.kind, self.lbrct))
                            .into()
                    })
                    .expect_region()
            }
            fn replace_ty(&mut self, bt: ty::BoundTy) -> Ty<'tcx> {
                self.map
                    .entry(bt.var)
                    .or_insert_with(|| {
                        self.infcx
                            .next_ty_var(TypeVariableOrigin {
                                kind: TypeVariableOriginKind::MiscVariable,
                                span: self.span,
                            })
                            .into()
                    })
                    .expect_ty()
            }
            fn replace_const(&mut self, bv: ty::BoundVar, ty: Ty<'tcx>) -> ty::Const<'tcx> {
                self.map
                    .entry(bv)
                    .or_insert_with(|| {
                        self.infcx
                            .next_const_var(
                                ty,
                                ConstVariableOrigin {
                                    kind: ConstVariableOriginKind::MiscVariable,
                                    span: self.span,
                                },
                            )
                            .into()
                    })
                    .expect_const()
            }
        }
        let delegate = ToFreshVars { infcx: self, span, lbrct, map: Default::default() };
        self.tcx.replace_bound_vars_uncached(value, delegate)
    }

    /// See the [`region_constraints::RegionConstraintCollector::verify_generic_bound`] method.
    pub fn verify_generic_bound(
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
    pub fn closure_kind(&self, closure_substs: SubstsRef<'tcx>) -> Option<ty::ClosureKind> {
        let closure_kind_ty = closure_substs.as_closure().kind_ty();
        let closure_kind_ty = self.shallow_resolve(closure_kind_ty);
        closure_kind_ty.to_opt_closure_kind()
    }

    /// Clears the selection, evaluation, and projection caches. This is useful when
    /// repeatedly attempting to select an `Obligation` while changing only
    /// its `ParamEnv`, since `FulfillmentContext` doesn't use probing.
    pub fn clear_caches(&self) {
        self.selection_cache.clear();
        self.evaluation_cache.clear();
        self.inner.borrow_mut().projection_cache().clear();
    }

    pub fn universe(&self) -> ty::UniverseIndex {
        self.universe.get()
    }

    /// Creates and return a fresh universe that extends all previous
    /// universes. Updates `self.universe` to that new universe.
    pub fn create_next_universe(&self) -> ty::UniverseIndex {
        let u = self.universe.get().next_universe();
        self.universe.set(u);
        u
    }

    pub fn try_const_eval_resolve(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        unevaluated: ty::UnevaluatedConst<'tcx>,
        ty: Ty<'tcx>,
        span: Option<Span>,
    ) -> Result<ty::Const<'tcx>, ErrorHandled> {
        match self.const_eval_resolve(param_env, unevaluated, span) {
            Ok(Some(val)) => Ok(self.tcx.mk_const(val, ty)),
            Ok(None) => {
                let tcx = self.tcx;
                let def_id = unevaluated.def.did;
                span_bug!(
                    tcx.def_span(def_id),
                    "unable to construct a constant value for the unevaluated constant {:?}",
                    unevaluated
                );
            }
            Err(err) => Err(err),
        }
    }

    /// Resolves and evaluates a constant.
    ///
    /// The constant can be located on a trait like `<A as B>::C`, in which case the given
    /// substitutions and environment are used to resolve the constant. Alternatively if the
    /// constant has generic parameters in scope the substitutions are used to evaluate the value of
    /// the constant. For example in `fn foo<T>() { let _ = [0; bar::<T>()]; }` the repeat count
    /// constant `bar::<T>()` requires a substitution for `T`, if the substitution for `T` is still
    /// too generic for the constant to be evaluated then `Err(ErrorHandled::TooGeneric)` is
    /// returned.
    ///
    /// This handles inferences variables within both `param_env` and `substs` by
    /// performing the operation on their respective canonical forms.
    #[instrument(skip(self), level = "debug")]
    pub fn const_eval_resolve(
        &self,
        mut param_env: ty::ParamEnv<'tcx>,
        unevaluated: ty::UnevaluatedConst<'tcx>,
        span: Option<Span>,
    ) -> EvalToValTreeResult<'tcx> {
        let mut substs = self.resolve_vars_if_possible(unevaluated.substs);
        debug!(?substs);

        // Postpone the evaluation of constants whose substs depend on inference
        // variables
        let tcx = self.tcx;
        if substs.has_non_region_infer() {
            if let Some(ct) = tcx.bound_abstract_const(unevaluated.def)? {
                let ct = tcx.expand_abstract_consts(ct.subst(tcx, substs));
                if let Err(e) = ct.error_reported() {
                    return Err(ErrorHandled::Reported(e));
                } else if ct.has_non_region_infer() || ct.has_non_region_param() {
                    return Err(ErrorHandled::TooGeneric);
                } else {
                    substs = replace_param_and_infer_substs_with_placeholder(tcx, substs);
                }
            } else {
                substs = InternalSubsts::identity_for_item(tcx, unevaluated.def.did);
                param_env = tcx.param_env(unevaluated.def.did);
            }
        }

        let param_env_erased = tcx.erase_regions(param_env);
        let substs_erased = tcx.erase_regions(substs);
        debug!(?param_env_erased);
        debug!(?substs_erased);

        let unevaluated = ty::UnevaluatedConst { def: unevaluated.def, substs: substs_erased };

        // The return value is the evaluated value which doesn't contain any reference to inference
        // variables, thus we don't need to substitute back the original values.
        tcx.const_eval_resolve_for_typeck(param_env_erased, unevaluated, span)
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
    pub fn ty_or_const_infer_var_changed(&self, infer_var: TyOrConstInferVar<'tcx>) -> bool {
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
                self.inner.borrow_mut().int_unification_table().inlined_probe_value(v).is_some()
            }

            TyOrConstInferVar::TyFloat(v) => {
                // If `probe_value` returns a value it's always a
                // `ty::Float(_)`, which never matches a `ty::Infer(_)`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                self.inner.borrow_mut().float_unification_table().probe_value(v).is_some()
            }

            TyOrConstInferVar::Const(v) => {
                // If `probe_value` returns a `Known` value, it never equals
                // `ty::ConstKind::Infer(ty::InferConst::Var(v))`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                match self.inner.borrow_mut().const_unification_table().probe_value(v).val {
                    ConstVariableValue::Unknown { .. } => false,
                    ConstVariableValue::Known { .. } => true,
                }
            }
        }
    }
}

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    /// Processes registered region obliations and resolves regions, reporting
    /// any errors if any were raised. Prefer using this function over manually
    /// calling `resolve_regions_and_report_errors`.
    pub fn check_region_obligations_and_report_errors(
        &self,
        generic_param_scope: LocalDefId,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        self.process_registered_region_obligations(
            outlives_env.region_bound_pairs(),
            outlives_env.param_env,
        );

        self.resolve_regions_and_report_errors(generic_param_scope, outlives_env)
    }

    /// Process the region constraints and report any errors that
    /// result. After this, no more unification operations should be
    /// done -- or the compiler will panic -- but it is legal to use
    /// `resolve_vars_if_possible` as well as `fully_resolve`.
    ///
    /// Make sure to call [`InferCtxt::process_registered_region_obligations`]
    /// first, or preferably use [`TypeErrCtxt::check_region_obligations_and_report_errors`]
    /// to do both of these operations together.
    pub fn resolve_regions_and_report_errors(
        &self,
        generic_param_scope: LocalDefId,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        let errors = self.resolve_regions(outlives_env);

        if let None = self.tainted_by_errors() {
            // As a heuristic, just skip reporting region errors
            // altogether if other errors have been reported while
            // this infcx was in use. This is totally hokey but
            // otherwise we have a hard time separating legit region
            // errors from silly ones.
            self.report_region_errors(generic_param_scope, &errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(self
                .tcx
                .sess
                .delay_span_bug(rustc_span::DUMMY_SP, "error should have been emitted"))
        }
    }

    // [Note-Type-error-reporting]
    // An invariant is that anytime the expected or actual type is Error (the special
    // error type, meaning that an error occurred when typechecking this expression),
    // this is a derived error. The error cascaded from another error (that was already
    // reported), so it's not useful to display it to the user.
    // The following methods implement this logic.
    // They check if either the actual or expected type is Error, and don't print the error
    // in this case. The typechecker should only ever report type errors involving mismatched
    // types using one of these methods, and should not call span_err directly for such
    // errors.

    pub fn type_error_struct_with_diag<M>(
        &self,
        sp: Span,
        mk_diag: M,
        actual_ty: Ty<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed>
    where
        M: FnOnce(String) -> DiagnosticBuilder<'tcx, ErrorGuaranteed>,
    {
        let actual_ty = self.resolve_vars_if_possible(actual_ty);
        debug!("type_error_struct_with_diag({:?}, {:?})", sp, actual_ty);

        let mut err = mk_diag(self.ty_to_string(actual_ty));

        // Don't report an error if actual type is `Error`.
        if actual_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        err
    }

    pub fn report_mismatched_types(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        err: TypeError<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        self.report_and_explain_type_error(TypeTrace::types(cause, true, expected, actual), err)
    }

    pub fn report_mismatched_consts(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: ty::Const<'tcx>,
        actual: ty::Const<'tcx>,
        err: TypeError<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        self.report_and_explain_type_error(TypeTrace::consts(cause, true, expected, actual), err)
    }
}

/// Helper for `ty_or_const_infer_var_changed` (see comment on that), currently
/// used only for `traits::fulfill`'s list of `stalled_on` inference variables.
#[derive(Copy, Clone, Debug)]
pub enum TyOrConstInferVar<'tcx> {
    /// Equivalent to `ty::Infer(ty::TyVar(_))`.
    Ty(TyVid),
    /// Equivalent to `ty::Infer(ty::IntVar(_))`.
    TyInt(IntVid),
    /// Equivalent to `ty::Infer(ty::FloatVar(_))`.
    TyFloat(FloatVid),

    /// Equivalent to `ty::ConstKind::Infer(ty::InferConst::Var(_))`.
    Const(ConstVid<'tcx>),
}

impl<'tcx> TyOrConstInferVar<'tcx> {
    /// Tries to extract an inference variable from a type or a constant, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`) and
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    pub fn maybe_from_generic_arg(arg: GenericArg<'tcx>) -> Option<Self> {
        match arg.unpack() {
            GenericArgKind::Type(ty) => Self::maybe_from_ty(ty),
            GenericArgKind::Const(ct) => Self::maybe_from_const(ct),
            GenericArgKind::Lifetime(_) => None,
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
    fn interner(&self) -> TyCtxt<'tcx> {
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

struct ShallowResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for ShallowResolver<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    /// If `ty` is a type variable of some kind, resolve it one level
    /// (but do not resolve types found in the result). If `typ` is
    /// not a type variable, just return it unmodified.
    #[inline]
    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Infer(v) = ty.kind() { self.fold_infer_ty(*v).unwrap_or(ty) } else { ty }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Infer(InferConst::Var(vid)) = ct.kind() {
            self.infcx
                .inner
                .borrow_mut()
                .const_unification_table()
                .probe_value(vid)
                .val
                .known()
                .unwrap_or(ct)
        } else {
            ct
        }
    }
}

impl<'a, 'tcx> ShallowResolver<'a, 'tcx> {
    // This is separate from `fold_ty` to keep that method small and inlinable.
    #[inline(never)]
    fn fold_infer_ty(&mut self, v: InferTy) -> Option<Ty<'tcx>> {
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
                let known = self.infcx.inner.borrow_mut().type_variables().probe(v).known();
                known.map(|t| self.fold_ty(t))
            }

            ty::IntVar(v) => self
                .infcx
                .inner
                .borrow_mut()
                .int_unification_table()
                .probe_value(v)
                .map(|v| v.to_type(self.infcx.tcx)),

            ty::FloatVar(v) => self
                .infcx
                .inner
                .borrow_mut()
                .float_unification_table()
                .probe_value(v)
                .map(|v| v.to_type(self.infcx.tcx)),

            ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_) => None,
        }
    }
}

impl<'tcx> TypeTrace<'tcx> {
    pub fn span(&self) -> Span {
        self.cause.span
    }

    pub fn types(
        cause: &ObligationCause<'tcx>,
        a_is_expected: bool,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
    ) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: Terms(ExpectedFound::new(a_is_expected, a.into(), b.into())),
        }
    }

    pub fn poly_trait_refs(
        cause: &ObligationCause<'tcx>,
        a_is_expected: bool,
        a: ty::PolyTraitRef<'tcx>,
        b: ty::PolyTraitRef<'tcx>,
    ) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: PolyTraitRefs(ExpectedFound::new(a_is_expected, a, b)),
        }
    }

    pub fn consts(
        cause: &ObligationCause<'tcx>,
        a_is_expected: bool,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: Terms(ExpectedFound::new(a_is_expected, a.into(), b.into())),
        }
    }
}

impl<'tcx> SubregionOrigin<'tcx> {
    pub fn span(&self) -> Span {
        match *self {
            Subtype(ref a) => a.span(),
            RelateObjectBound(a) => a,
            RelateParamBound(a, ..) => a,
            RelateRegionParamBound(a) => a,
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

            traits::ObligationCauseCode::CompareImplItemObligation {
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

            _ => default(),
        }
    }
}

impl RegionVariableOrigin {
    pub fn span(&self) -> Span {
        match *self {
            MiscVariable(a)
            | PatternRegion(a)
            | AddrOfRegion(a)
            | Autoref(a)
            | Coercion(a)
            | EarlyBoundRegion(a, ..)
            | LateBoundRegion(a, ..)
            | UpvarRegion(_, a) => a,
            Nll(..) => bug!("NLL variable used with `span`"),
        }
    }
}

/// Replaces substs that reference param or infer variables with suitable
/// placeholders. This function is meant to remove these param and infer
/// substs when they're not actually needed to evaluate a constant.
fn replace_param_and_infer_substs_with_placeholder<'tcx>(
    tcx: TyCtxt<'tcx>,
    substs: SubstsRef<'tcx>,
) -> SubstsRef<'tcx> {
    struct ReplaceParamAndInferWithPlaceholder<'tcx> {
        tcx: TyCtxt<'tcx>,
        idx: u32,
    }

    impl<'tcx> TypeFolder<TyCtxt<'tcx>> for ReplaceParamAndInferWithPlaceholder<'tcx> {
        fn interner(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
            if let ty::Infer(_) = t.kind() {
                self.tcx.mk_placeholder(ty::PlaceholderType {
                    universe: ty::UniverseIndex::ROOT,
                    name: ty::BoundTyKind::Anon({
                        let idx = self.idx;
                        self.idx += 1;
                        idx
                    }),
                })
            } else {
                t.super_fold_with(self)
            }
        }

        fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
            if let ty::ConstKind::Infer(_) = c.kind() {
                let ty = c.ty();
                // If the type references param or infer then ICE ICE ICE
                if ty.has_non_region_param() || ty.has_non_region_infer() {
                    bug!("const `{c}`'s type should not reference params or types");
                }
                self.tcx.mk_const(
                    ty::PlaceholderConst {
                        universe: ty::UniverseIndex::ROOT,
                        name: ty::BoundVar::from_u32({
                            let idx = self.idx;
                            self.idx += 1;
                            idx
                        }),
                    },
                    ty,
                )
            } else {
                c.super_fold_with(self)
            }
        }
    }

    substs.fold_with(&mut ReplaceParamAndInferWithPlaceholder { tcx, idx: 0 })
}
