//! See the Book for more information.

pub use self::freshen::TypeFreshener;
pub use self::LateBoundRegionConversionTime::*;
pub use self::RegionVariableOrigin::*;
pub use self::SubregionOrigin::*;
pub use self::ValuePairs::*;
pub use rustc::ty::IntVarValue;

use crate::traits::{self, ObligationCause, PredicateObligations, TraitEngine};

use rustc::infer::canonical::{Canonical, CanonicalVarValues};
use rustc::infer::unify_key::{ConstVarValue, ConstVariableValue};
use rustc::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind, ToType};
use rustc::middle::free_region::RegionRelations;
use rustc::middle::region;
use rustc::mir;
use rustc::mir::interpret::ConstEvalResult;
use rustc::session::config::BorrowckMode;
use rustc::traits::select;
use rustc::ty::error::{ExpectedFound, TypeError, UnconstrainedNumeric};
use rustc::ty::fold::{TypeFoldable, TypeFolder};
use rustc::ty::relate::RelateResult;
use rustc::ty::subst::{GenericArg, InternalSubsts, SubstsRef};
use rustc::ty::{self, GenericParamDefKind, InferConst, Ty, TyCtxt};
use rustc::ty::{ConstVid, FloatVid, IntVid, TyVid};

use rustc_ast::ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::unify as ut;
use rustc_errors::DiagnosticBuilder;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use std::cell::{Cell, Ref, RefCell};
use std::collections::BTreeMap;
use std::fmt;

use self::combine::CombineFields;
use self::lexical_region_resolve::LexicalRegionResolutions;
use self::outlives::env::OutlivesEnvironment;
use self::region_constraints::{GenericKind, RegionConstraintData, VarInfos, VerifyBound};
use self::region_constraints::{RegionConstraintCollector, RegionSnapshot};
use self::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};

pub mod at;
pub mod canonical;
mod combine;
mod equate;
pub mod error_reporting;
mod freshen;
mod fudge;
mod glb;
mod higher_ranked;
pub mod lattice;
mod lexical_region_resolve;
mod lub;
pub mod nll_relate;
pub mod outlives;
pub mod region_constraints;
pub mod resolve;
mod sub;
pub mod type_variable;

use crate::infer::canonical::OriginalQueryValues;
pub use rustc::infer::unify_key;

#[must_use]
#[derive(Debug)]
pub struct InferOk<'tcx, T> {
    pub value: T,
    pub obligations: PredicateObligations<'tcx>,
}
pub type InferResult<'tcx, T> = Result<InferOk<'tcx, T>, TypeError<'tcx>>;

pub type Bound<T> = Option<T>;
pub type UnitResult<'tcx> = RelateResult<'tcx, ()>; // "unify result"
pub type FixupResult<'tcx, T> = Result<T, FixupError<'tcx>>; // "fixup result"

/// A flag that is used to suppress region errors. This is normally
/// false, but sometimes -- when we are doing region checks that the
/// NLL borrow checker will also do -- it might be set to true.
#[derive(Copy, Clone, Default, Debug)]
pub struct SuppressRegionErrors {
    suppressed: bool,
}

impl SuppressRegionErrors {
    pub fn suppressed(self) -> bool {
        self.suppressed
    }

    /// Indicates that the MIR borrowck will repeat these region
    /// checks, so we should ignore errors if NLL is (unconditionally)
    /// enabled.
    pub fn when_nll_is_enabled(tcx: TyCtxt<'_>) -> Self {
        // FIXME(Centril): Once we actually remove `::Migrate` also make
        // this always `true` and then proceed to eliminate the dead code.
        match tcx.borrowck_mode() {
            // If we're on Migrate mode, report AST region errors
            BorrowckMode::Migrate => SuppressRegionErrors { suppressed: false },

            // If we're on MIR, don't report AST region errors as they should be reported by NLL
            BorrowckMode::Mir => SuppressRegionErrors { suppressed: true },
        }
    }
}

/// This type contains all the things within `InferCtxt` that sit within a
/// `RefCell` and are involved with taking/rolling back snapshots. Snapshot
/// operations are hot enough that we want only one call to `borrow_mut` per
/// call to `start_snapshot` and `rollback_to`.
pub struct InferCtxtInner<'tcx> {
    /// Cache for projections. This cache is snapshotted along with the infcx.
    ///
    /// Public so that `traits::project` can use it.
    pub projection_cache: traits::ProjectionCache<'tcx>,

    /// We instantiate `UnificationTable` with `bounds<Ty>` because the types
    /// that might instantiate a general type variable have an order,
    /// represented by its upper and lower bounds.
    type_variables: type_variable::TypeVariableTable<'tcx>,

    /// Map from const parameter variable to the kind of const it represents.
    const_unification_table: ut::UnificationTable<ut::InPlace<ty::ConstVid<'tcx>>>,

    /// Map from integral variable to the kind of integer it represents.
    int_unification_table: ut::UnificationTable<ut::InPlace<ty::IntVid>>,

    /// Map from floating variable to the kind of float it represents.
    float_unification_table: ut::UnificationTable<ut::InPlace<ty::FloatVid>>,

    /// Tracks the set of region variables and the constraints between them.
    /// This is initially `Some(_)` but when
    /// `resolve_regions_and_report_errors` is invoked, this gets set to `None`
    /// -- further attempts to perform unification, etc., may fail if new
    /// region constraints would've been added.
    region_constraints: Option<RegionConstraintCollector<'tcx>>,

    /// A set of constraints that regionck must validate. Each
    /// constraint has the form `T:'a`, meaning "some type `T` must
    /// outlive the lifetime 'a". These constraints derive from
    /// instantiated type parameters. So if you had a struct defined
    /// like
    ///
    ///     struct Foo<T:'static> { ... }
    ///
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
    /// `process_region_obligations` (defined in `self::region_obligations`)
    /// for each body-id in this map, which will process the
    /// obligations within. This is expected to be done 'late enough'
    /// that all type inference variables have been bound and so forth.
    pub region_obligations: Vec<(hir::HirId, RegionObligation<'tcx>)>,
}

impl<'tcx> InferCtxtInner<'tcx> {
    fn new() -> InferCtxtInner<'tcx> {
        InferCtxtInner {
            projection_cache: Default::default(),
            type_variables: type_variable::TypeVariableTable::new(),
            const_unification_table: ut::UnificationTable::new(),
            int_unification_table: ut::UnificationTable::new(),
            float_unification_table: ut::UnificationTable::new(),
            region_constraints: Some(RegionConstraintCollector::new()),
            region_obligations: vec![],
        }
    }

    pub fn unwrap_region_constraints(&mut self) -> &mut RegionConstraintCollector<'tcx> {
        self.region_constraints.as_mut().expect("region constraints already solved")
    }
}

pub struct InferCtxt<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,

    /// During type-checking/inference of a body, `in_progress_tables`
    /// contains a reference to the tables being built up, which are
    /// used for reading closure kinds/signatures as they are inferred,
    /// and for error reporting logic to read arbitrary node types.
    pub in_progress_tables: Option<&'a RefCell<ty::TypeckTables<'tcx>>>,

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
    pub reported_trait_errors: RefCell<FxHashMap<Span, Vec<ty::Predicate<'tcx>>>>,

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
    tainted_by_errors_flag: Cell<bool>,

    /// Track how many errors were reported when this infcx is created.
    /// If the number of errors increases, that's also a sign (line
    /// `tained_by_errors`) to avoid reporting certain kinds of errors.
    // FIXME(matthewjasper) Merge into `tainted_by_errors_flag`
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
}

/// A map returned by `replace_bound_vars_with_placeholders()`
/// indicating the placeholder region that each late-bound region was
/// replaced with.
pub type PlaceholderMap<'tcx> = BTreeMap<ty::BoundRegion, ty::Region<'tcx>>;

/// See the `error_reporting` module for more details.
#[derive(Clone, Debug, PartialEq, Eq, TypeFoldable)]
pub enum ValuePairs<'tcx> {
    Types(ExpectedFound<Ty<'tcx>>),
    Regions(ExpectedFound<ty::Region<'tcx>>),
    Consts(ExpectedFound<&'tcx ty::Const<'tcx>>),
    TraitRefs(ExpectedFound<ty::TraitRef<'tcx>>),
    PolyTraitRefs(ExpectedFound<ty::PolyTraitRef<'tcx>>),
}

/// The trace designates the path through inference that we took to
/// encounter an error or subtyping constraint.
///
/// See the `error_reporting` module for more details.
#[derive(Clone, Debug)]
pub struct TypeTrace<'tcx> {
    cause: ObligationCause<'tcx>,
    values: ValuePairs<'tcx>,
}

/// The origin of a `r1 <= r2` constraint.
///
/// See `error_reporting` module for more details
#[derive(Clone, Debug)]
pub enum SubregionOrigin<'tcx> {
    /// Arose from a subtyping relation
    Subtype(Box<TypeTrace<'tcx>>),

    /// Stack-allocated closures cannot outlive innermost loop
    /// or function so as to ensure we only require finite stack
    InfStackClosure(Span),

    /// Invocation of closure must be within its lifetime
    InvokeClosure(Span),

    /// Dereference of reference must be within its lifetime
    DerefPointer(Span),

    /// Closure bound must not outlive captured variables
    ClosureCapture(Span, hir::HirId),

    /// Index into slice must be within its lifetime
    IndexSlice(Span),

    /// When casting `&'a T` to an `&'b Trait` object,
    /// relating `'a` to `'b`
    RelateObjectBound(Span),

    /// Some type parameter was instantiated with the given type,
    /// and that type must outlive some region.
    RelateParamBound(Span, Ty<'tcx>),

    /// The given region parameter was instantiated with a region
    /// that must outlive some other region.
    RelateRegionParamBound(Span),

    /// A bound placed on type parameters that states that must outlive
    /// the moment of their instantiation.
    RelateDefaultParamBound(Span, Ty<'tcx>),

    /// Creating a pointer `b` to contents of another reference
    Reborrow(Span),

    /// Creating a pointer `b` to contents of an upvar
    ReborrowUpvar(Span, ty::UpvarId),

    /// Data with type `Ty<'tcx>` was borrowed
    DataBorrowed(Ty<'tcx>, Span),

    /// (&'a &'b T) where a >= b
    ReferenceOutlivesReferent(Ty<'tcx>, Span),

    /// Type or region parameters must be in scope.
    ParameterInScope(ParameterOrigin, Span),

    /// The type T of an expression E must outlive the lifetime for E.
    ExprTypeIsNotInScope(Ty<'tcx>, Span),

    /// A `ref b` whose region does not enclose the decl site
    BindingTypeIsNotValidAtDecl(Span),

    /// Regions appearing in a method receiver must outlive method call
    CallRcvr(Span),

    /// Regions appearing in a function argument must outlive func call
    CallArg(Span),

    /// Region in return type of invoked fn must enclose call
    CallReturn(Span),

    /// Operands must be in scope
    Operand(Span),

    /// Region resulting from a `&` expr must enclose the `&` expr
    AddrOf(Span),

    /// An auto-borrow that does not enclose the expr where it occurs
    AutoBorrow(Span),

    /// Region constraint arriving from destructor safety
    SafeDestructor(Span),

    /// Comparing the signature and requirements of an impl method against
    /// the containing trait.
    CompareImplMethodObligation {
        span: Span,
        item_name: ast::Name,
        impl_item_def_id: DefId,
        trait_item_def_id: DefId,
    },
}

// `SubregionOrigin` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(SubregionOrigin<'_>, 32);

/// Places that type/region parameters can appear.
#[derive(Clone, Copy, Debug)]
pub enum ParameterOrigin {
    Path,               // foo::bar
    MethodCall,         // foo.bar() <-- parameters on impl providing bar()
    OverloadedOperator, // a + b when overloaded
    OverloadedDeref,    // *a when overloaded
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
    LateBoundRegion(Span, ty::BoundRegion, LateBoundRegionConversionTime),

    UpvarRegion(ty::UpvarId, Span),

    BoundRegionInCoherence(ast::Name),

    /// This origin is used for the inference variables that we create
    /// during NLL region processing.
    NLL(NLLRegionVariableOrigin),
}

#[derive(Copy, Clone, Debug)]
pub enum NLLRegionVariableOrigin {
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

impl NLLRegionVariableOrigin {
    pub fn is_universal(self) -> bool {
        match self {
            NLLRegionVariableOrigin::FreeRegion => true,
            NLLRegionVariableOrigin::Placeholder(..) => true,
            NLLRegionVariableOrigin::Existential { .. } => false,
        }
    }

    pub fn is_existential(self) -> bool {
        !self.is_universal()
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FixupError<'tcx> {
    UnresolvedIntTy(IntVid),
    UnresolvedFloatTy(FloatVid),
    UnresolvedTy(TyVid),
    UnresolvedConst(ConstVid<'tcx>),
}

/// See the `region_obligations` field for more information.
#[derive(Clone)]
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

/// Helper type of a temporary returned by `tcx.infer_ctxt()`.
/// Necessary because we can't write the following bound:
/// `F: for<'b, 'tcx> where 'tcx FnOnce(InferCtxt<'b, 'tcx>)`.
pub struct InferCtxtBuilder<'tcx> {
    global_tcx: TyCtxt<'tcx>,
    fresh_tables: Option<RefCell<ty::TypeckTables<'tcx>>>,
}

pub trait TyCtxtInferExt<'tcx> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'tcx>;
}

impl TyCtxtInferExt<'tcx> for TyCtxt<'tcx> {
    fn infer_ctxt(self) -> InferCtxtBuilder<'tcx> {
        InferCtxtBuilder { global_tcx: self, fresh_tables: None }
    }
}

impl<'tcx> InferCtxtBuilder<'tcx> {
    /// Used only by `rustc_typeck` during body type-checking/inference,
    /// will initialize `in_progress_tables` with fresh `TypeckTables`.
    pub fn with_fresh_in_progress_tables(mut self, table_owner: DefId) -> Self {
        self.fresh_tables = Some(RefCell::new(ty::TypeckTables::empty(Some(table_owner))));
        self
    }

    /// Given a canonical value `C` as a starting point, create an
    /// inference context that contains each of the bound values
    /// within instantiated as a fresh variable. The `f` closure is
    /// invoked with the new infcx, along with the instantiated value
    /// `V` and a substitution `S`. This substitution `S` maps from
    /// the bound values in `C` to their instantiated values in `V`
    /// (in other words, `S(C) = V`).
    pub fn enter_with_canonical<T, R>(
        &mut self,
        span: Span,
        canonical: &Canonical<'tcx, T>,
        f: impl for<'a> FnOnce(InferCtxt<'a, 'tcx>, T, CanonicalVarValues<'tcx>) -> R,
    ) -> R
    where
        T: TypeFoldable<'tcx>,
    {
        self.enter(|infcx| {
            let (value, subst) =
                infcx.instantiate_canonical_with_fresh_inference_vars(span, canonical);
            f(infcx, value, subst)
        })
    }

    pub fn enter<R>(&mut self, f: impl for<'a> FnOnce(InferCtxt<'a, 'tcx>) -> R) -> R {
        let InferCtxtBuilder { global_tcx, ref fresh_tables } = *self;
        let in_progress_tables = fresh_tables.as_ref();
        global_tcx.enter_local(|tcx| {
            f(InferCtxt {
                tcx,
                in_progress_tables,
                inner: RefCell::new(InferCtxtInner::new()),
                lexical_region_resolutions: RefCell::new(None),
                selection_cache: Default::default(),
                evaluation_cache: Default::default(),
                reported_trait_errors: Default::default(),
                reported_closure_mismatch: Default::default(),
                tainted_by_errors_flag: Cell::new(false),
                err_count_on_creation: tcx.sess.err_count(),
                in_snapshot: Cell::new(false),
                skip_leak_check: Cell::new(false),
                universe: Cell::new(ty::UniverseIndex::ROOT),
            })
        })
    }
}

impl<'tcx, T> InferOk<'tcx, T> {
    pub fn unit(self) -> InferOk<'tcx, ()> {
        InferOk { value: (), obligations: self.obligations }
    }

    /// Extracts `value`, registering any obligations into `fulfill_cx`.
    pub fn into_value_registering_obligations(
        self,
        infcx: &InferCtxt<'_, 'tcx>,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> T {
        let InferOk { value, obligations } = self;
        for obligation in obligations {
            fulfill_cx.register_predicate_obligation(infcx, obligation);
        }
        value
    }
}

impl<'tcx> InferOk<'tcx, ()> {
    pub fn into_obligations(self) -> PredicateObligations<'tcx> {
        self.obligations
    }
}

#[must_use = "once you start a snapshot, you should always consume it"]
pub struct CombinedSnapshot<'a, 'tcx> {
    projection_cache_snapshot: traits::ProjectionCacheSnapshot,
    type_snapshot: type_variable::Snapshot<'tcx>,
    const_snapshot: ut::Snapshot<ut::InPlace<ty::ConstVid<'tcx>>>,
    int_snapshot: ut::Snapshot<ut::InPlace<ty::IntVid>>,
    float_snapshot: ut::Snapshot<ut::InPlace<ty::FloatVid>>,
    region_constraints_snapshot: RegionSnapshot,
    region_obligations_snapshot: usize,
    universe: ty::UniverseIndex,
    was_in_snapshot: bool,
    was_skip_leak_check: bool,
    _in_progress_tables: Option<Ref<'a, ty::TypeckTables<'tcx>>>,
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn is_in_snapshot(&self) -> bool {
        self.in_snapshot.get()
    }

    pub fn freshen<T: TypeFoldable<'tcx>>(&self, t: T) -> T {
        t.fold_with(&mut self.freshener())
    }

    pub fn type_var_diverges(&'a self, ty: Ty<'_>) -> bool {
        match ty.kind {
            ty::Infer(ty::TyVar(vid)) => self.inner.borrow().type_variables.var_diverges(vid),
            _ => false,
        }
    }

    pub fn freshener<'b>(&'b self) -> TypeFreshener<'b, 'tcx> {
        freshen::TypeFreshener::new(self)
    }

    pub fn type_is_unconstrained_numeric(&'a self, ty: Ty<'_>) -> UnconstrainedNumeric {
        use rustc::ty::error::UnconstrainedNumeric::Neither;
        use rustc::ty::error::UnconstrainedNumeric::{UnconstrainedFloat, UnconstrainedInt};
        match ty.kind {
            ty::Infer(ty::IntVar(vid)) => {
                if self.inner.borrow_mut().int_unification_table.probe_value(vid).is_some() {
                    Neither
                } else {
                    UnconstrainedInt
                }
            }
            ty::Infer(ty::FloatVar(vid)) => {
                if self.inner.borrow_mut().float_unification_table.probe_value(vid).is_some() {
                    Neither
                } else {
                    UnconstrainedFloat
                }
            }
            _ => Neither,
        }
    }

    pub fn unsolved_variables(&self) -> Vec<Ty<'tcx>> {
        let mut inner = self.inner.borrow_mut();
        // FIXME(const_generics): should there be an equivalent function for const variables?

        let mut vars: Vec<Ty<'_>> = inner
            .type_variables
            .unsolved_variables()
            .into_iter()
            .map(|t| self.tcx.mk_ty_var(t))
            .collect();
        vars.extend(
            (0..inner.int_unification_table.len())
                .map(|i| ty::IntVid { index: i as u32 })
                .filter(|&vid| inner.int_unification_table.probe_value(vid).is_none())
                .map(|v| self.tcx.mk_int_var(v)),
        );
        vars.extend(
            (0..inner.float_unification_table.len())
                .map(|i| ty::FloatVid { index: i as u32 })
                .filter(|&vid| inner.float_unification_table.probe_value(vid).is_none())
                .map(|v| self.tcx.mk_float_var(v)),
        );
        vars
    }

    fn combine_fields(
        &'a self,
        trace: TypeTrace<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> CombineFields<'a, 'tcx> {
        CombineFields {
            infcx: self,
            trace,
            cause: None,
            param_env,
            obligations: PredicateObligations::new(),
        }
    }

    /// Clear the "currently in a snapshot" flag, invoke the closure,
    /// then restore the flag to its original value. This flag is a
    /// debugging measure designed to detect cases where we start a
    /// snapshot, create type variables, and register obligations
    /// which may involve those type variables in the fulfillment cx,
    /// potentially leaving "dangling type variables" behind.
    /// In such cases, an assertion will fail when attempting to
    /// register obligations, within a snapshot. Very useful, much
    /// better than grovelling through megabytes of `RUSTC_LOG` output.
    ///
    /// HOWEVER, in some cases the flag is unhelpful. In particular, we
    /// sometimes create a "mini-fulfilment-cx" in which we enroll
    /// obligations. As long as this fulfillment cx is fully drained
    /// before we return, this is not a problem, as there won't be any
    /// escaping obligations in the main cx. In those cases, you can
    /// use this function.
    pub fn save_and_restore_in_snapshot_flag<F, R>(&self, func: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        let flag = self.in_snapshot.replace(false);
        let result = func(self);
        self.in_snapshot.set(flag);
        result
    }

    fn start_snapshot(&self) -> CombinedSnapshot<'a, 'tcx> {
        debug!("start_snapshot()");

        let in_snapshot = self.in_snapshot.replace(true);

        let mut inner = self.inner.borrow_mut();
        CombinedSnapshot {
            projection_cache_snapshot: inner.projection_cache.snapshot(),
            type_snapshot: inner.type_variables.snapshot(),
            const_snapshot: inner.const_unification_table.snapshot(),
            int_snapshot: inner.int_unification_table.snapshot(),
            float_snapshot: inner.float_unification_table.snapshot(),
            region_constraints_snapshot: inner.unwrap_region_constraints().start_snapshot(),
            region_obligations_snapshot: inner.region_obligations.len(),
            universe: self.universe(),
            was_in_snapshot: in_snapshot,
            was_skip_leak_check: self.skip_leak_check.get(),
            // Borrow tables "in progress" (i.e., during typeck)
            // to ban writes from within a snapshot to them.
            _in_progress_tables: self.in_progress_tables.map(|tables| tables.borrow()),
        }
    }

    fn rollback_to(&self, cause: &str, snapshot: CombinedSnapshot<'a, 'tcx>) {
        debug!("rollback_to(cause={})", cause);
        let CombinedSnapshot {
            projection_cache_snapshot,
            type_snapshot,
            const_snapshot,
            int_snapshot,
            float_snapshot,
            region_constraints_snapshot,
            region_obligations_snapshot,
            universe,
            was_in_snapshot,
            was_skip_leak_check,
            _in_progress_tables,
        } = snapshot;

        self.in_snapshot.set(was_in_snapshot);
        self.universe.set(universe);
        self.skip_leak_check.set(was_skip_leak_check);

        let mut inner = self.inner.borrow_mut();
        inner.projection_cache.rollback_to(projection_cache_snapshot);
        inner.type_variables.rollback_to(type_snapshot);
        inner.const_unification_table.rollback_to(const_snapshot);
        inner.int_unification_table.rollback_to(int_snapshot);
        inner.float_unification_table.rollback_to(float_snapshot);
        inner.unwrap_region_constraints().rollback_to(region_constraints_snapshot);
        inner.region_obligations.truncate(region_obligations_snapshot);
    }

    fn commit_from(&self, snapshot: CombinedSnapshot<'a, 'tcx>) {
        debug!("commit_from()");
        let CombinedSnapshot {
            projection_cache_snapshot,
            type_snapshot,
            const_snapshot,
            int_snapshot,
            float_snapshot,
            region_constraints_snapshot,
            region_obligations_snapshot: _,
            universe: _,
            was_in_snapshot,
            was_skip_leak_check,
            _in_progress_tables,
        } = snapshot;

        self.in_snapshot.set(was_in_snapshot);
        self.skip_leak_check.set(was_skip_leak_check);

        let mut inner = self.inner.borrow_mut();
        inner.projection_cache.commit(projection_cache_snapshot);
        inner.type_variables.commit(type_snapshot);
        inner.const_unification_table.commit(const_snapshot);
        inner.int_unification_table.commit(int_snapshot);
        inner.float_unification_table.commit(float_snapshot);
        inner.unwrap_region_constraints().commit(region_constraints_snapshot);
    }

    /// Executes `f` and commit the bindings.
    pub fn commit_unconditionally<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'a, 'tcx>) -> R,
    {
        debug!("commit_unconditionally()");
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        self.commit_from(snapshot);
        r
    }

    /// Execute `f` and commit the bindings if closure `f` returns `Ok(_)`.
    pub fn commit_if_ok<T, E, F>(&self, f: F) -> Result<T, E>
    where
        F: FnOnce(&CombinedSnapshot<'a, 'tcx>) -> Result<T, E>,
    {
        debug!("commit_if_ok()");
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
    pub fn probe<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'a, 'tcx>) -> R,
    {
        debug!("probe()");
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        self.rollback_to("probe", snapshot);
        r
    }

    /// If `should_skip` is true, then execute `f` then unroll any bindings it creates.
    pub fn probe_maybe_skip_leak_check<R, F>(&self, should_skip: bool, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'a, 'tcx>) -> R,
    {
        debug!("probe()");
        let snapshot = self.start_snapshot();
        let skip_leak_check = should_skip || self.skip_leak_check.get();
        self.skip_leak_check.set(skip_leak_check);
        let r = f(&snapshot);
        self.rollback_to("probe", snapshot);
        r
    }

    /// Scan the constraints produced since `snapshot` began and returns:
    ///
    /// - `None` -- if none of them involve "region outlives" constraints
    /// - `Some(true)` -- if there are `'a: 'b` constraints where `'a` or `'b` is a placeholder
    /// - `Some(false)` -- if there are `'a: 'b` constraints but none involve placeholders
    pub fn region_constraints_added_in_snapshot(
        &self,
        snapshot: &CombinedSnapshot<'a, 'tcx>,
    ) -> Option<bool> {
        self.inner
            .borrow_mut()
            .unwrap_region_constraints()
            .region_constraints_added_in_snapshot(&snapshot.region_constraints_snapshot)
    }

    pub fn add_given(&self, sub: ty::Region<'tcx>, sup: ty::RegionVid) {
        self.inner.borrow_mut().unwrap_region_constraints().add_given(sub, sup);
    }

    pub fn can_sub<T>(&self, param_env: ty::ParamEnv<'tcx>, a: T, b: T) -> UnitResult<'tcx>
    where
        T: at::ToTrace<'tcx>,
    {
        let origin = &ObligationCause::dummy();
        self.probe(|_| {
            self.at(origin, param_env).sub(a, b).map(|InferOk { obligations: _, .. }| {
                // Ignore obligations, since we are unrolling
                // everything anyway.
            })
        })
    }

    pub fn can_eq<T>(&self, param_env: ty::ParamEnv<'tcx>, a: T, b: T) -> UnitResult<'tcx>
    where
        T: at::ToTrace<'tcx>,
    {
        let origin = &ObligationCause::dummy();
        self.probe(|_| {
            self.at(origin, param_env).eq(a, b).map(|InferOk { obligations: _, .. }| {
                // Ignore obligations, since we are unrolling
                // everything anyway.
            })
        })
    }

    pub fn sub_regions(
        &self,
        origin: SubregionOrigin<'tcx>,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) {
        debug!("sub_regions({:?} <: {:?})", a, b);
        self.inner.borrow_mut().unwrap_region_constraints().make_subregion(origin, a, b);
    }

    /// Require that the region `r` be equal to one of the regions in
    /// the set `regions`.
    pub fn member_constraint(
        &self,
        opaque_type_def_id: DefId,
        definition_span: Span,
        hidden_ty: Ty<'tcx>,
        region: ty::Region<'tcx>,
        in_regions: &Lrc<Vec<ty::Region<'tcx>>>,
    ) {
        debug!("member_constraint({:?} <: {:?})", region, in_regions);
        self.inner.borrow_mut().unwrap_region_constraints().member_constraint(
            opaque_type_def_id,
            definition_span,
            hidden_ty,
            region,
            in_regions,
        );
    }

    pub fn subtype_predicate(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: &ty::PolySubtypePredicate<'tcx>,
    ) -> Option<InferResult<'tcx, ()>> {
        // Subtle: it's ok to skip the binder here and resolve because
        // `shallow_resolve` just ignores anything that is not a type
        // variable, and because type variable's can't (at present, at
        // least) capture any of the things bound by this binder.
        //
        // NOTE(nmatsakis): really, there is no *particular* reason to do this
        // `shallow_resolve` here except as a micro-optimization.
        // Naturally I could not resist.
        let two_unbound_type_vars = {
            let a = self.shallow_resolve(predicate.skip_binder().a);
            let b = self.shallow_resolve(predicate.skip_binder().b);
            a.is_ty_var() && b.is_ty_var()
        };

        if two_unbound_type_vars {
            // Two unbound type variables? Can't make progress.
            return None;
        }

        Some(self.commit_if_ok(|snapshot| {
            let (ty::SubtypePredicate { a_is_expected, a, b }, placeholder_map) =
                self.replace_bound_vars_with_placeholders(predicate);

            let ok = self.at(cause, param_env).sub_exp(a_is_expected, a, b)?;

            self.leak_check(false, &placeholder_map, snapshot)?;

            Ok(ok.unit())
        }))
    }

    pub fn region_outlives_predicate(
        &self,
        cause: &traits::ObligationCause<'tcx>,
        predicate: &ty::PolyRegionOutlivesPredicate<'tcx>,
    ) -> UnitResult<'tcx> {
        self.commit_if_ok(|snapshot| {
            let (ty::OutlivesPredicate(r_a, r_b), placeholder_map) =
                self.replace_bound_vars_with_placeholders(predicate);
            let origin = SubregionOrigin::from_obligation_cause(cause, || {
                RelateRegionParamBound(cause.span)
            });
            self.sub_regions(origin, r_b, r_a); // `b : a` ==> `a <= b`
            self.leak_check(false, &placeholder_map, snapshot)?;
            Ok(())
        })
    }

    pub fn next_ty_var_id(&self, diverging: bool, origin: TypeVariableOrigin) -> TyVid {
        self.inner.borrow_mut().type_variables.new_var(self.universe(), diverging, origin)
    }

    pub fn next_ty_var(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        self.tcx.mk_ty_var(self.next_ty_var_id(false, origin))
    }

    pub fn next_ty_var_in_universe(
        &self,
        origin: TypeVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> Ty<'tcx> {
        let vid = self.inner.borrow_mut().type_variables.new_var(universe, false, origin);
        self.tcx.mk_ty_var(vid)
    }

    pub fn next_diverging_ty_var(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        self.tcx.mk_ty_var(self.next_ty_var_id(true, origin))
    }

    pub fn next_const_var(
        &self,
        ty: Ty<'tcx>,
        origin: ConstVariableOrigin,
    ) -> &'tcx ty::Const<'tcx> {
        self.tcx.mk_const_var(self.next_const_var_id(origin), ty)
    }

    pub fn next_const_var_in_universe(
        &self,
        ty: Ty<'tcx>,
        origin: ConstVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> &'tcx ty::Const<'tcx> {
        let vid = self
            .inner
            .borrow_mut()
            .const_unification_table
            .new_key(ConstVarValue { origin, val: ConstVariableValue::Unknown { universe } });
        self.tcx.mk_const_var(vid, ty)
    }

    pub fn next_const_var_id(&self, origin: ConstVariableOrigin) -> ConstVid<'tcx> {
        self.inner.borrow_mut().const_unification_table.new_key(ConstVarValue {
            origin,
            val: ConstVariableValue::Unknown { universe: self.universe() },
        })
    }

    fn next_int_var_id(&self) -> IntVid {
        self.inner.borrow_mut().int_unification_table.new_key(None)
    }

    pub fn next_int_var(&self) -> Ty<'tcx> {
        self.tcx.mk_int_var(self.next_int_var_id())
    }

    fn next_float_var_id(&self) -> FloatVid {
        self.inner.borrow_mut().float_unification_table.new_key(None)
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
        self.tcx.mk_region(ty::ReVar(region_var))
    }

    /// Return the universe that the region `r` was created in.  For
    /// most regions (e.g., `'static`, named regions from the user,
    /// etc) this is the root universe U0. For inference variables or
    /// placeholders, however, it will return the universe which which
    /// they are associated.
    fn universe_of_region(&self, r: ty::Region<'tcx>) -> ty::UniverseIndex {
        self.inner.borrow_mut().unwrap_region_constraints().universe(r)
    }

    /// Number of region variables created so far.
    pub fn num_region_vars(&self) -> usize {
        self.inner.borrow_mut().unwrap_region_constraints().num_region_vars()
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    pub fn next_nll_region_var(&self, origin: NLLRegionVariableOrigin) -> ty::Region<'tcx> {
        self.next_region_var(RegionVariableOrigin::NLL(origin))
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    pub fn next_nll_region_var_in_universe(
        &self,
        origin: NLLRegionVariableOrigin,
        universe: ty::UniverseIndex,
    ) -> ty::Region<'tcx> {
        self.next_region_var_in_universe(RegionVariableOrigin::NLL(origin), universe)
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
                let ty_var_id = self.inner.borrow_mut().type_variables.new_var(
                    self.universe(),
                    false,
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
                    kind: ConstVariableOriginKind::ConstParameterDefinition(param.name),
                    span,
                };
                let const_var_id =
                    self.inner.borrow_mut().const_unification_table.new_key(ConstVarValue {
                        origin,
                        val: ConstVariableValue::Unknown { universe: self.universe() },
                    });
                self.tcx.mk_const_var(const_var_id, self.tcx.type_of(param.def_id)).into()
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
    pub fn is_tainted_by_errors(&self) -> bool {
        debug!(
            "is_tainted_by_errors(err_count={}, err_count_on_creation={}, \
             tainted_by_errors_flag={})",
            self.tcx.sess.err_count(),
            self.err_count_on_creation,
            self.tainted_by_errors_flag.get()
        );

        if self.tcx.sess.err_count() > self.err_count_on_creation {
            return true; // errors reported since this infcx was made
        }
        self.tainted_by_errors_flag.get()
    }

    /// Set the "tainted by errors" flag to true. We call this when we
    /// observe an error from a prior pass.
    pub fn set_tainted_by_errors(&self) {
        debug!("set_tainted_by_errors()");
        self.tainted_by_errors_flag.set(true)
    }

    /// Process the region constraints and report any errors that
    /// result. After this, no more unification operations should be
    /// done -- or the compiler will panic -- but it is legal to use
    /// `resolve_vars_if_possible` as well as `fully_resolve`.
    pub fn resolve_regions_and_report_errors(
        &self,
        region_context: DefId,
        region_map: &region::ScopeTree,
        outlives_env: &OutlivesEnvironment<'tcx>,
        suppress: SuppressRegionErrors,
    ) {
        assert!(
            self.is_tainted_by_errors() || self.inner.borrow().region_obligations.is_empty(),
            "region_obligations not empty: {:#?}",
            self.inner.borrow().region_obligations
        );

        let region_rels = &RegionRelations::new(
            self.tcx,
            region_context,
            region_map,
            outlives_env.free_region_map(),
        );
        let (var_infos, data) = self
            .inner
            .borrow_mut()
            .region_constraints
            .take()
            .expect("regions already resolved")
            .into_infos_and_data();
        let (lexical_region_resolutions, errors) =
            lexical_region_resolve::resolve(region_rels, var_infos, data);

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());

        if !self.is_tainted_by_errors() {
            // As a heuristic, just skip reporting region errors
            // altogether if other errors have been reported while
            // this infcx was in use.  This is totally hokey but
            // otherwise we have a hard time separating legit region
            // errors from silly ones.
            self.report_region_errors(region_map, &errors, suppress);
        }
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
    #[allow(non_camel_case_types)] // bug with impl trait
    pub fn with_region_constraints<R>(
        &self,
        op: impl FnOnce(&RegionConstraintData<'tcx>) -> R,
    ) -> R {
        let mut inner = self.inner.borrow_mut();
        op(inner.unwrap_region_constraints().data())
    }

    /// Takes ownership of the list of variable regions. This implies
    /// that all the region constraints have already been taken, and
    /// hence that `resolve_regions_and_report_errors` can never be
    /// called. This is used only during NLL processing to "hand off" ownership
    /// of the set of region variables into the NLL region context.
    pub fn take_region_var_origins(&self) -> VarInfos {
        let (var_infos, data) = self
            .inner
            .borrow_mut()
            .region_constraints
            .take()
            .expect("regions already resolved")
            .into_infos_and_data();
        assert!(data.is_empty());
        var_infos
    }

    pub fn ty_to_string(&self, t: Ty<'tcx>) -> String {
        self.resolve_vars_if_possible(&t).to_string()
    }

    pub fn tys_to_string(&self, ts: &[Ty<'tcx>]) -> String {
        let tstrs: Vec<String> = ts.iter().map(|t| self.ty_to_string(*t)).collect();
        format!("({})", tstrs.join(", "))
    }

    pub fn trait_ref_to_string(&self, t: &ty::TraitRef<'tcx>) -> String {
        self.resolve_vars_if_possible(t).print_only_trait_path().to_string()
    }

    /// If `TyVar(vid)` resolves to a type, return that type. Else, return the
    /// universe index of `TyVar(vid)`.
    pub fn probe_ty_var(&self, vid: TyVid) -> Result<Ty<'tcx>, ty::UniverseIndex> {
        use self::type_variable::TypeVariableValue;

        match self.inner.borrow_mut().type_variables.probe(vid) {
            TypeVariableValue::Known { value } => Ok(value),
            TypeVariableValue::Unknown { universe } => Err(universe),
        }
    }

    /// Resolve any type variables found in `value` -- but only one
    /// level.  So, if the variable `?X` is bound to some type
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
        let mut r = ShallowResolver::new(self);
        value.fold_with(&mut r)
    }

    pub fn root_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.inner.borrow_mut().type_variables.root_var(var)
    }

    /// Where possible, replaces type/const variables in
    /// `value` with their final value. Note that region variables
    /// are unaffected. If a type/const variable has not been unified, it
    /// is left as is. This is an idempotent operation that does
    /// not affect inference state in any way and so you can do it
    /// at will.
    pub fn resolve_vars_if_possible<T>(&self, value: &T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        if !value.needs_infer() {
            return value.clone(); // Avoid duplicated subst-folding.
        }
        let mut r = resolve::OpportunisticVarResolver::new(self);
        value.fold_with(&mut r)
    }

    /// Returns the first unresolved variable contained in `T`. In the
    /// process of visiting `T`, this will resolve (where possible)
    /// type variables in `T`, but it never constructs the final,
    /// resolved type, so it's more efficient than
    /// `resolve_vars_if_possible()`.
    pub fn unresolved_type_vars<T>(&self, value: &T) -> Option<(Ty<'tcx>, Option<Span>)>
    where
        T: TypeFoldable<'tcx>,
    {
        let mut r = resolve::UnresolvedTypeFinder::new(self);
        value.visit_with(&mut r);
        r.first_unresolved
    }

    pub fn probe_const_var(
        &self,
        vid: ty::ConstVid<'tcx>,
    ) -> Result<&'tcx ty::Const<'tcx>, ty::UniverseIndex> {
        match self.inner.borrow_mut().const_unification_table.probe_value(vid).val {
            ConstVariableValue::Known { value } => Ok(value),
            ConstVariableValue::Unknown { universe } => Err(universe),
        }
    }

    pub fn fully_resolve<T: TypeFoldable<'tcx>>(&self, value: &T) -> FixupResult<'tcx, T> {
        /*!
         * Attempts to resolve all type/region/const variables in
         * `value`. Region inference must have been run already (e.g.,
         * by calling `resolve_regions_and_report_errors`). If some
         * variable was never unified, an `Err` results.
         *
         * This method is idempotent, but it not typically not invoked
         * except during the writeback phase.
         */

        resolve::fully_resolve(self, value)
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
    ) -> DiagnosticBuilder<'tcx>
    where
        M: FnOnce(String) -> DiagnosticBuilder<'tcx>,
    {
        let actual_ty = self.resolve_vars_if_possible(&actual_ty);
        debug!("type_error_struct_with_diag({:?}, {:?})", sp, actual_ty);

        // Don't report an error if actual type is `Error`.
        if actual_ty.references_error() {
            return self.tcx.sess.diagnostic().struct_dummy();
        }

        mk_diag(self.ty_to_string(actual_ty))
    }

    pub fn report_mismatched_types(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        err: TypeError<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        let trace = TypeTrace::types(cause, true, expected, actual);
        self.report_and_explain_type_error(trace, &err)
    }

    pub fn replace_bound_vars_with_fresh_vars<T>(
        &self,
        span: Span,
        lbrct: LateBoundRegionConversionTime,
        value: &ty::Binder<T>,
    ) -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
    where
        T: TypeFoldable<'tcx>,
    {
        let fld_r = |br| self.next_region_var(LateBoundRegion(span, br, lbrct));
        let fld_t = |_| {
            self.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::MiscVariable,
                span,
            })
        };
        let fld_c = |_, ty| {
            self.next_const_var(
                ty,
                ConstVariableOrigin { kind: ConstVariableOriginKind::MiscVariable, span },
            )
        };
        self.tcx.replace_bound_vars(value, fld_r, fld_t, fld_c)
    }

    /// See the [`region_constraints::verify_generic_bound`] method.
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
    pub fn closure_kind(
        &self,
        closure_def_id: DefId,
        closure_substs: SubstsRef<'tcx>,
    ) -> Option<ty::ClosureKind> {
        let closure_kind_ty = closure_substs.as_closure().kind_ty(closure_def_id, self.tcx);
        let closure_kind_ty = self.shallow_resolve(closure_kind_ty);
        closure_kind_ty.to_opt_closure_kind()
    }

    /// Obtains the signature of a closure. For closures, unlike
    /// `tcx.fn_sig(def_id)`, this method will work during the
    /// type-checking of the enclosing function and return the closure
    /// signature in its partially inferred state.
    pub fn closure_sig(&self, def_id: DefId, substs: SubstsRef<'tcx>) -> ty::PolyFnSig<'tcx> {
        let closure_sig_ty = substs.as_closure().sig_ty(def_id, self.tcx);
        let closure_sig_ty = self.shallow_resolve(closure_sig_ty);
        closure_sig_ty.fn_sig(self.tcx)
    }

    /// Clears the selection, evaluation, and projection caches. This is useful when
    /// repeatedly attempting to select an `Obligation` while changing only
    /// its `ParamEnv`, since `FulfillmentContext` doesn't use probing.
    pub fn clear_caches(&self) {
        self.selection_cache.clear();
        self.evaluation_cache.clear();
        self.inner.borrow_mut().projection_cache.clear();
    }

    fn universe(&self) -> ty::UniverseIndex {
        self.universe.get()
    }

    /// Creates and return a fresh universe that extends all previous
    /// universes. Updates `self.universe` to that new universe.
    pub fn create_next_universe(&self) -> ty::UniverseIndex {
        let u = self.universe.get().next_universe();
        self.universe.set(u);
        u
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
    pub fn const_eval_resolve(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        promoted: Option<mir::Promoted>,
        span: Option<Span>,
    ) -> ConstEvalResult<'tcx> {
        let mut original_values = OriginalQueryValues::default();
        let canonical = self.canonicalize_query(&(param_env, substs), &mut original_values);

        let (param_env, substs) = canonical.value;
        // The return value is the evaluated value which doesn't contain any reference to inference
        // variables, thus we don't need to substitute back the original values.
        self.tcx.const_eval_resolve(param_env, def_id, substs, promoted, span)
    }
}

pub struct ShallowResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> ShallowResolver<'a, 'tcx> {
    #[inline(always)]
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
        ShallowResolver { infcx }
    }

    /// If `typ` is a type variable of some kind, resolve it one level
    /// (but do not resolve types found in the result). If `typ` is
    /// not a type variable, just return it unmodified.
    pub fn shallow_resolve(&mut self, typ: Ty<'tcx>) -> Ty<'tcx> {
        match typ.kind {
            ty::Infer(ty::TyVar(v)) => {
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
                // dynamic borrow errors on `self.infcx.inner`.
                let known = self.infcx.inner.borrow_mut().type_variables.probe(v).known();
                known.map(|t| self.fold_ty(t)).unwrap_or(typ)
            }

            ty::Infer(ty::IntVar(v)) => self
                .infcx
                .inner
                .borrow_mut()
                .int_unification_table
                .probe_value(v)
                .map(|v| v.to_type(self.infcx.tcx))
                .unwrap_or(typ),

            ty::Infer(ty::FloatVar(v)) => self
                .infcx
                .inner
                .borrow_mut()
                .float_unification_table
                .probe_value(v)
                .map(|v| v.to_type(self.infcx.tcx))
                .unwrap_or(typ),

            _ => typ,
        }
    }

    // `resolver.shallow_resolve_changed(ty)` is equivalent to
    // `resolver.shallow_resolve(ty) != ty`, but more efficient. It's always
    // inlined, despite being large, because it has only two call sites that
    // are extremely hot.
    #[inline(always)]
    pub fn shallow_resolve_changed(&self, infer: ty::InferTy) -> bool {
        match infer {
            ty::TyVar(v) => {
                use self::type_variable::TypeVariableValue;

                // If `inlined_probe` returns a `Known` value its `kind` never
                // matches `infer`.
                match self.infcx.inner.borrow_mut().type_variables.inlined_probe(v) {
                    TypeVariableValue::Unknown { .. } => false,
                    TypeVariableValue::Known { .. } => true,
                }
            }

            ty::IntVar(v) => {
                // If inlined_probe_value returns a value it's always a
                // `ty::Int(_)` or `ty::UInt(_)`, which never matches a
                // `ty::Infer(_)`.
                self.infcx.inner.borrow_mut().int_unification_table.inlined_probe_value(v).is_some()
            }

            ty::FloatVar(v) => {
                // If inlined_probe_value returns a value it's always a
                // `ty::Float(_)`, which never matches a `ty::Infer(_)`.
                //
                // Not `inlined_probe_value(v)` because this call site is colder.
                self.infcx.inner.borrow_mut().float_unification_table.probe_value(v).is_some()
            }

            _ => unreachable!(),
        }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for ShallowResolver<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.shallow_resolve(ty)
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if let ty::Const { val: ty::ConstKind::Infer(InferConst::Var(vid)), .. } = ct {
            self.infcx
                .inner
                .borrow_mut()
                .const_unification_table
                .probe_value(*vid)
                .val
                .known()
                .unwrap_or(ct)
        } else {
            ct
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
        TypeTrace { cause: cause.clone(), values: Types(ExpectedFound::new(a_is_expected, a, b)) }
    }

    pub fn dummy(tcx: TyCtxt<'tcx>) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: ObligationCause::dummy(),
            values: Types(ExpectedFound { expected: tcx.types.err, found: tcx.types.err }),
        }
    }
}

impl<'tcx> SubregionOrigin<'tcx> {
    pub fn span(&self) -> Span {
        match *self {
            Subtype(ref a) => a.span(),
            InfStackClosure(a) => a,
            InvokeClosure(a) => a,
            DerefPointer(a) => a,
            ClosureCapture(a, _) => a,
            IndexSlice(a) => a,
            RelateObjectBound(a) => a,
            RelateParamBound(a, _) => a,
            RelateRegionParamBound(a) => a,
            RelateDefaultParamBound(a, _) => a,
            Reborrow(a) => a,
            ReborrowUpvar(a, _) => a,
            DataBorrowed(_, a) => a,
            ReferenceOutlivesReferent(_, a) => a,
            ParameterInScope(_, a) => a,
            ExprTypeIsNotInScope(_, a) => a,
            BindingTypeIsNotValidAtDecl(a) => a,
            CallRcvr(a) => a,
            CallArg(a) => a,
            CallReturn(a) => a,
            Operand(a) => a,
            AddrOf(a) => a,
            AutoBorrow(a) => a,
            SafeDestructor(a) => a,
            CompareImplMethodObligation { span, .. } => span,
        }
    }

    pub fn from_obligation_cause<F>(cause: &traits::ObligationCause<'tcx>, default: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match cause.code {
            traits::ObligationCauseCode::ReferenceOutlivesReferent(ref_type) => {
                SubregionOrigin::ReferenceOutlivesReferent(ref_type, cause.span)
            }

            traits::ObligationCauseCode::CompareImplMethodObligation {
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            } => SubregionOrigin::CompareImplMethodObligation {
                span: cause.span,
                item_name,
                impl_item_def_id,
                trait_item_def_id,
            },

            _ => default(),
        }
    }
}

impl RegionVariableOrigin {
    pub fn span(&self) -> Span {
        match *self {
            MiscVariable(a) => a,
            PatternRegion(a) => a,
            AddrOfRegion(a) => a,
            Autoref(a) => a,
            Coercion(a) => a,
            EarlyBoundRegion(a, ..) => a,
            LateBoundRegion(a, ..) => a,
            BoundRegionInCoherence(_) => rustc_span::DUMMY_SP,
            UpvarRegion(_, a) => a,
            NLL(..) => bug!("NLL variable used with `span`"),
        }
    }
}

impl<'tcx> fmt::Debug for RegionObligation<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RegionObligation(sub_region={:?}, sup_type={:?})",
            self.sub_region, self.sup_type
        )
    }
}
