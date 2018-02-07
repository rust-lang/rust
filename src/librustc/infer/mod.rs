// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See the Book for more information.

pub use self::LateBoundRegionConversionTime::*;
pub use self::RegionVariableOrigin::*;
pub use self::SubregionOrigin::*;
pub use self::ValuePairs::*;
pub use ty::IntVarValue;
pub use self::freshen::TypeFreshener;

use hir::def_id::DefId;
use middle::free_region::RegionRelations;
use middle::region;
use middle::lang_items;
use mir::tcx::PlaceTy;
use ty::subst::{Kind, Subst, Substs};
use ty::{TyVid, IntVid, FloatVid};
use ty::{self, Ty, TyCtxt};
use ty::error::{ExpectedFound, TypeError, UnconstrainedNumeric};
use ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use ty::relate::RelateResult;
use traits::{self, ObligationCause, PredicateObligations, Reveal};
use rustc_data_structures::unify::{self, UnificationTable};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::BTreeMap;
use std::fmt;
use syntax::ast;
use errors::DiagnosticBuilder;
use syntax_pos::{self, Span, DUMMY_SP};
use util::nodemap::FxHashMap;
use arena::DroplessArena;

use self::combine::CombineFields;
use self::higher_ranked::HrMatchResult;
use self::region_constraints::{RegionConstraintCollector, RegionSnapshot};
use self::region_constraints::{GenericKind, VerifyBound, RegionConstraintData, VarOrigins};
use self::lexical_region_resolve::LexicalRegionResolutions;
use self::outlives::env::OutlivesEnvironment;
use self::type_variable::TypeVariableOrigin;
use self::unify_key::ToType;

pub mod anon_types;
pub mod at;
mod combine;
mod equate;
pub mod error_reporting;
mod fudge;
mod glb;
mod higher_ranked;
pub mod lattice;
mod lub;
pub mod region_constraints;
mod lexical_region_resolve;
pub mod outlives;
pub mod resolve;
mod freshen;
mod sub;
pub mod type_variable;
pub mod unify_key;

#[must_use]
pub struct InferOk<'tcx, T> {
    pub value: T,
    pub obligations: PredicateObligations<'tcx>,
}
pub type InferResult<'tcx, T> = Result<InferOk<'tcx, T>, TypeError<'tcx>>;

pub type Bound<T> = Option<T>;
pub type UnitResult<'tcx> = RelateResult<'tcx, ()>; // "unify result"
pub type FixupResult<T> = Result<T, FixupError>; // "fixup result"

pub struct InferCtxt<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,

    /// During type-checking/inference of a body, `in_progress_tables`
    /// contains a reference to the tables being built up, which are
    /// used for reading closure kinds/signatures as they are inferred,
    /// and for error reporting logic to read arbitrary node types.
    pub in_progress_tables: Option<&'a RefCell<ty::TypeckTables<'tcx>>>,

    // Cache for projections. This cache is snapshotted along with the
    // infcx.
    //
    // Public so that `traits::project` can use it.
    pub projection_cache: RefCell<traits::ProjectionCache<'tcx>>,

    // We instantiate UnificationTable with bounds<Ty> because the
    // types that might instantiate a general type variable have an
    // order, represented by its upper and lower bounds.
    pub type_variables: RefCell<type_variable::TypeVariableTable<'tcx>>,

    // Map from integral variable to the kind of integer it represents
    int_unification_table: RefCell<UnificationTable<ty::IntVid>>,

    // Map from floating variable to the kind of float it represents
    float_unification_table: RefCell<UnificationTable<ty::FloatVid>>,

    // Tracks the set of region variables and the constraints between
    // them.  This is initially `Some(_)` but when
    // `resolve_regions_and_report_errors` is invoked, this gets set
    // to `None` -- further attempts to perform unification etc may
    // fail if new region constraints would've been added.
    region_constraints: RefCell<Option<RegionConstraintCollector<'tcx>>>,

    // Once region inference is done, the values for each variable.
    lexical_region_resolutions: RefCell<Option<LexicalRegionResolutions<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx>,

    /// Caches the results of trait evaluation.
    pub evaluation_cache: traits::EvaluationCache<'tcx>,

    // the set of predicates on which errors have been reported, to
    // avoid reporting the same error twice.
    pub reported_trait_errors: RefCell<FxHashMap<Span, Vec<ty::Predicate<'tcx>>>>,

    // When an error occurs, we want to avoid reporting "derived"
    // errors that are due to this original failure. Normally, we
    // handle this with the `err_count_on_creation` count, which
    // basically just tracks how many errors were reported when we
    // started type-checking a fn and checks to see if any new errors
    // have been reported since then. Not great, but it works.
    //
    // However, when errors originated in other passes -- notably
    // resolve -- this heuristic breaks down. Therefore, we have this
    // auxiliary flag that one can set whenever one creates a
    // type-error that is due to an error in a prior pass.
    //
    // Don't read this flag directly, call `is_tainted_by_errors()`
    // and `set_tainted_by_errors()`.
    tainted_by_errors_flag: Cell<bool>,

    // Track how many errors were reported when this infcx is created.
    // If the number of errors increases, that's also a sign (line
    // `tained_by_errors`) to avoid reporting certain kinds of errors.
    err_count_on_creation: usize,

    // This flag is true while there is an active snapshot.
    in_snapshot: Cell<bool>,

    // A set of constraints that regionck must validate. Each
    // constraint has the form `T:'a`, meaning "some type `T` must
    // outlive the lifetime 'a". These constraints derive from
    // instantiated type parameters. So if you had a struct defined
    // like
    //
    //     struct Foo<T:'static> { ... }
    //
    // then in some expression `let x = Foo { ... }` it will
    // instantiate the type parameter `T` with a fresh type `$0`. At
    // the same time, it will record a region obligation of
    // `$0:'static`. This will get checked later by regionck. (We
    // can't generally check these things right away because we have
    // to wait until types are resolved.)
    //
    // These are stored in a map keyed to the id of the innermost
    // enclosing fn body / static initializer expression. This is
    // because the location where the obligation was incurred can be
    // relevant with respect to which sublifetime assumptions are in
    // place. The reason that we store under the fn-id, and not
    // something more fine-grained, is so that it is easier for
    // regionck to be sure that it has found *all* the region
    // obligations (otherwise, it's easy to fail to walk to a
    // particular node-id).
    //
    // Before running `resolve_regions_and_report_errors`, the creator
    // of the inference context is expected to invoke
    // `process_region_obligations` (defined in `self::region_obligations`)
    // for each body-id in this map, which will process the
    // obligations within. This is expected to be done 'late enough'
    // that all type inference variables have been bound and so forth.
    region_obligations: RefCell<Vec<(ast::NodeId, RegionObligation<'tcx>)>>,
}

/// A map returned by `skolemize_late_bound_regions()` indicating the skolemized
/// region that each late-bound region was replaced with.
pub type SkolemizationMap<'tcx> = BTreeMap<ty::BoundRegion, ty::Region<'tcx>>;

/// See `error_reporting` module for more details
#[derive(Clone, Debug)]
pub enum ValuePairs<'tcx> {
    Types(ExpectedFound<Ty<'tcx>>),
    TraitRefs(ExpectedFound<ty::TraitRef<'tcx>>),
    PolyTraitRefs(ExpectedFound<ty::PolyTraitRef<'tcx>>),
}

/// The trace designates the path through inference that we took to
/// encounter an error or subtyping constraint.
///
/// See `error_reporting` module for more details.
#[derive(Clone)]
pub struct TypeTrace<'tcx> {
    cause: ObligationCause<'tcx>,
    values: ValuePairs<'tcx>,
}

/// The origin of a `r1 <= r2` constraint.
///
/// See `error_reporting` module for more details
#[derive(Clone, Debug)]
pub enum SubregionOrigin<'tcx> {
    // Arose from a subtyping relation
    Subtype(TypeTrace<'tcx>),

    // Stack-allocated closures cannot outlive innermost loop
    // or function so as to ensure we only require finite stack
    InfStackClosure(Span),

    // Invocation of closure must be within its lifetime
    InvokeClosure(Span),

    // Dereference of reference must be within its lifetime
    DerefPointer(Span),

    // Closure bound must not outlive captured free variables
    FreeVariable(Span, ast::NodeId),

    // Index into slice must be within its lifetime
    IndexSlice(Span),

    // When casting `&'a T` to an `&'b Trait` object,
    // relating `'a` to `'b`
    RelateObjectBound(Span),

    // Some type parameter was instantiated with the given type,
    // and that type must outlive some region.
    RelateParamBound(Span, Ty<'tcx>),

    // The given region parameter was instantiated with a region
    // that must outlive some other region.
    RelateRegionParamBound(Span),

    // A bound placed on type parameters that states that must outlive
    // the moment of their instantiation.
    RelateDefaultParamBound(Span, Ty<'tcx>),

    // Creating a pointer `b` to contents of another reference
    Reborrow(Span),

    // Creating a pointer `b` to contents of an upvar
    ReborrowUpvar(Span, ty::UpvarId),

    // Data with type `Ty<'tcx>` was borrowed
    DataBorrowed(Ty<'tcx>, Span),

    // (&'a &'b T) where a >= b
    ReferenceOutlivesReferent(Ty<'tcx>, Span),

    // Type or region parameters must be in scope.
    ParameterInScope(ParameterOrigin, Span),

    // The type T of an expression E must outlive the lifetime for E.
    ExprTypeIsNotInScope(Ty<'tcx>, Span),

    // A `ref b` whose region does not enclose the decl site
    BindingTypeIsNotValidAtDecl(Span),

    // Regions appearing in a method receiver must outlive method call
    CallRcvr(Span),

    // Regions appearing in a function argument must outlive func call
    CallArg(Span),

    // Region in return type of invoked fn must enclose call
    CallReturn(Span),

    // Operands must be in scope
    Operand(Span),

    // Region resulting from a `&` expr must enclose the `&` expr
    AddrOf(Span),

    // An auto-borrow that does not enclose the expr where it occurs
    AutoBorrow(Span),

    // Region constraint arriving from destructor safety
    SafeDestructor(Span),

    // Comparing the signature and requirements of an impl method against
    // the containing trait.
    CompareImplMethodObligation {
        span: Span,
        item_name: ast::Name,
        impl_item_def_id: DefId,
        trait_item_def_id: DefId,
    },
}

/// Places that type/region parameters can appear.
#[derive(Clone, Copy, Debug)]
pub enum ParameterOrigin {
    Path, // foo::bar
    MethodCall, // foo.bar() <-- parameters on impl providing bar()
    OverloadedOperator, // a + b when overloaded
    OverloadedDeref, // *a when overloaded
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
    // Region variables created for ill-categorized reasons,
    // mostly indicates places in need of refactoring
    MiscVariable(Span),

    // Regions created by a `&P` or `[...]` pattern
    PatternRegion(Span),

    // Regions created by `&` operator
    AddrOfRegion(Span),

    // Regions created as part of an autoref of a method receiver
    Autoref(Span),

    // Regions created as part of an automatic coercion
    Coercion(Span),

    // Region variables created as the values for early-bound regions
    EarlyBoundRegion(Span, ast::Name),

    // Region variables created for bound regions
    // in a function or method that is called
    LateBoundRegion(Span, ty::BoundRegion, LateBoundRegionConversionTime),

    UpvarRegion(ty::UpvarId, Span),

    BoundRegionInCoherence(ast::Name),

    // This origin is used for the inference variables that we create
    // during NLL region processing.
    NLL(NLLRegionVariableOrigin),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NLLRegionVariableOrigin {
    // During NLL region processing, we create variables for free
    // regions that we encounter in the function signature and
    // elsewhere. This origin indices we've got one of those.
    FreeRegion,

    Inferred(::mir::visit::TyContext),
}

#[derive(Copy, Clone, Debug)]
pub enum FixupError {
    UnresolvedIntTy(IntVid),
    UnresolvedFloatTy(FloatVid),
    UnresolvedTy(TyVid)
}

/// See the `region_obligations` field for more information.
#[derive(Clone)]
pub struct RegionObligation<'tcx> {
    pub sub_region: ty::Region<'tcx>,
    pub sup_type: Ty<'tcx>,
    pub cause: ObligationCause<'tcx>,
}

impl fmt::Display for FixupError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::FixupError::*;

        match *self {
            UnresolvedIntTy(_) => {
                write!(f, "cannot determine the type of this integer; \
                           add a suffix to specify the type explicitly")
            }
            UnresolvedFloatTy(_) => {
                write!(f, "cannot determine the type of this number; \
                           add a suffix to specify the type explicitly")
            }
            UnresolvedTy(_) => write!(f, "unconstrained type")
        }
    }
}

/// Helper type of a temporary returned by tcx.infer_ctxt().
/// Necessary because we can't write the following bound:
/// F: for<'b, 'tcx> where 'gcx: 'tcx FnOnce(InferCtxt<'b, 'gcx, 'tcx>).
pub struct InferCtxtBuilder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    global_tcx: TyCtxt<'a, 'gcx, 'gcx>,
    arena: DroplessArena,
    fresh_tables: Option<RefCell<ty::TypeckTables<'tcx>>>,
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'gcx> {
    pub fn infer_ctxt(self) -> InferCtxtBuilder<'a, 'gcx, 'tcx> {
        InferCtxtBuilder {
            global_tcx: self,
            arena: DroplessArena::new(),
            fresh_tables: None,

        }
    }
}

impl<'a, 'gcx, 'tcx> InferCtxtBuilder<'a, 'gcx, 'tcx> {
    /// Used only by `rustc_typeck` during body type-checking/inference,
    /// will initialize `in_progress_tables` with fresh `TypeckTables`.
    pub fn with_fresh_in_progress_tables(mut self, table_owner: DefId) -> Self {
        self.fresh_tables = Some(RefCell::new(ty::TypeckTables::empty(Some(table_owner))));
        self
    }

    pub fn enter<F, R>(&'tcx mut self, f: F) -> R
        where F: for<'b> FnOnce(InferCtxt<'b, 'gcx, 'tcx>) -> R
    {
        let InferCtxtBuilder {
            global_tcx,
            ref arena,
            ref fresh_tables,
        } = *self;
        let in_progress_tables = fresh_tables.as_ref();
        global_tcx.enter_local(arena, |tcx| f(InferCtxt {
            tcx,
            in_progress_tables,
            projection_cache: RefCell::new(traits::ProjectionCache::new()),
            type_variables: RefCell::new(type_variable::TypeVariableTable::new()),
            int_unification_table: RefCell::new(UnificationTable::new()),
            float_unification_table: RefCell::new(UnificationTable::new()),
            region_constraints: RefCell::new(Some(RegionConstraintCollector::new())),
            lexical_region_resolutions: RefCell::new(None),
            selection_cache: traits::SelectionCache::new(),
            evaluation_cache: traits::EvaluationCache::new(),
            reported_trait_errors: RefCell::new(FxHashMap()),
            tainted_by_errors_flag: Cell::new(false),
            err_count_on_creation: tcx.sess.err_count(),
            in_snapshot: Cell::new(false),
            region_obligations: RefCell::new(vec![]),
        }))
    }
}

impl<T> ExpectedFound<T> {
    pub fn new(a_is_expected: bool, a: T, b: T) -> Self {
        if a_is_expected {
            ExpectedFound {expected: a, found: b}
        } else {
            ExpectedFound {expected: b, found: a}
        }
    }
}

impl<'tcx, T> InferOk<'tcx, T> {
    pub fn unit(self) -> InferOk<'tcx, ()> {
        InferOk { value: (), obligations: self.obligations }
    }
}

#[must_use = "once you start a snapshot, you should always consume it"]
pub struct CombinedSnapshot<'a, 'tcx:'a> {
    projection_cache_snapshot: traits::ProjectionCacheSnapshot,
    type_snapshot: type_variable::Snapshot,
    int_snapshot: unify::Snapshot<ty::IntVid>,
    float_snapshot: unify::Snapshot<ty::FloatVid>,
    region_constraints_snapshot: RegionSnapshot,
    region_obligations_snapshot: usize,
    was_in_snapshot: bool,
    _in_progress_tables: Option<Ref<'a, ty::TypeckTables<'tcx>>>,
}

/// Helper trait for shortening the lifetimes inside a
/// value for post-type-checking normalization.
pub trait TransNormalize<'gcx>: TypeFoldable<'gcx> {
    fn trans_normalize<'a, 'tcx>(&self,
                                 infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                 param_env: ty::ParamEnv<'tcx>)
                                 -> Self;
}

macro_rules! items { ($($item:item)+) => ($($item)+) }
macro_rules! impl_trans_normalize {
    ($lt_gcx:tt, $($ty:ty),+) => {
        items!($(impl<$lt_gcx> TransNormalize<$lt_gcx> for $ty {
            fn trans_normalize<'a, 'tcx>(&self,
                                         infcx: &InferCtxt<'a, $lt_gcx, 'tcx>,
                                         param_env: ty::ParamEnv<'tcx>)
                                         -> Self {
                infcx.normalize_projections_in(param_env, self)
            }
        })+);
    }
}

impl_trans_normalize!('gcx,
    Ty<'gcx>,
    &'gcx ty::Const<'gcx>,
    &'gcx Substs<'gcx>,
    ty::FnSig<'gcx>,
    ty::PolyFnSig<'gcx>,
    ty::ClosureSubsts<'gcx>,
    ty::PolyTraitRef<'gcx>,
    ty::ExistentialTraitRef<'gcx>
);

impl<'gcx> TransNormalize<'gcx> for PlaceTy<'gcx> {
    fn trans_normalize<'a, 'tcx>(&self,
                                 infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                 param_env: ty::ParamEnv<'tcx>)
                                 -> Self {
        match *self {
            PlaceTy::Ty { ty } => PlaceTy::Ty { ty: ty.trans_normalize(infcx, param_env) },
            PlaceTy::Downcast { adt_def, substs, variant_index } => {
                PlaceTy::Downcast {
                    adt_def,
                    substs: substs.trans_normalize(infcx, param_env),
                    variant_index,
                }
            }
        }
    }
}

// NOTE: Callable from trans only!
impl<'a, 'tcx> TyCtxt<'a, 'tcx, 'tcx> {
    /// Currently, higher-ranked type bounds inhibit normalization. Therefore,
    /// each time we erase them in translation, we need to normalize
    /// the contents.
    pub fn erase_late_bound_regions_and_normalize<T>(self, value: &ty::Binder<T>)
        -> T
        where T: TransNormalize<'tcx>
    {
        assert!(!value.needs_subst());
        let value = self.erase_late_bound_regions(value);
        self.fully_normalize_associated_types_in(&value)
    }

    /// Fully normalizes any associated types in `value`, using an
    /// empty environment and `Reveal::All` mode (therefore, suitable
    /// only for monomorphized code during trans, basically).
    pub fn fully_normalize_associated_types_in<T>(self, value: &T) -> T
        where T: TransNormalize<'tcx>
    {
        debug!("fully_normalize_associated_types_in(t={:?})", value);

        let param_env = ty::ParamEnv::empty(Reveal::All);
        let value = self.erase_regions(value);

        if !value.has_projections() {
            return value;
        }

        self.infer_ctxt().enter(|infcx| {
            value.trans_normalize(&infcx, param_env)
        })
    }

    /// Does a best-effort to normalize any associated types in
    /// `value`; this includes revealing specializable types, so this
    /// should be not be used during type-checking, but only during
    /// optimization and code generation.
    pub fn normalize_associated_type_in_env<T>(
        self, value: &T, env: ty::ParamEnv<'tcx>
    ) -> T
        where T: TransNormalize<'tcx>
    {
        debug!("normalize_associated_type_in_env(t={:?})", value);

        let value = self.erase_regions(value);

        if !value.has_projections() {
            return value;
        }

        self.infer_ctxt().enter(|infcx| {
            value.trans_normalize(&infcx, env.reveal_all())
       })
    }
}

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    fn normalize_projections_in<T>(&self, param_env: ty::ParamEnv<'tcx>, value: &T) -> T::Lifted
        where T: TypeFoldable<'tcx> + ty::Lift<'gcx>
    {
        let mut selcx = traits::SelectionContext::new(self);
        let cause = traits::ObligationCause::dummy();
        let traits::Normalized { value: result, obligations } =
            traits::normalize(&mut selcx, param_env, cause, value);

        debug!("normalize_projections_in: result={:?} obligations={:?}",
                result, obligations);

        let mut fulfill_cx = traits::FulfillmentContext::new();

        for obligation in obligations {
            fulfill_cx.register_predicate_obligation(self, obligation);
        }

        self.drain_fulfillment_cx_or_panic(DUMMY_SP, &mut fulfill_cx, &result)
    }

    /// Finishes processes any obligations that remain in the
    /// fulfillment context, and then returns the result with all type
    /// variables removed and regions erased. Because this is intended
    /// for use after type-check has completed, if any errors occur,
    /// it will panic. It is used during normalization and other cases
    /// where processing the obligations in `fulfill_cx` may cause
    /// type inference variables that appear in `result` to be
    /// unified, and hence we need to process those obligations to get
    /// the complete picture of the type.
    pub fn drain_fulfillment_cx_or_panic<T>(&self,
                                            span: Span,
                                            fulfill_cx: &mut traits::FulfillmentContext<'tcx>,
                                            result: &T)
                                            -> T::Lifted
        where T: TypeFoldable<'tcx> + ty::Lift<'gcx>
    {
        debug!("drain_fulfillment_cx_or_panic()");

        // In principle, we only need to do this so long as `result`
        // contains unbound type parameters. It could be a slight
        // optimization to stop iterating early.
        match fulfill_cx.select_all_or_error(self) {
            Ok(()) => { }
            Err(errors) => {
                span_bug!(span, "Encountered errors `{:?}` resolving bounds after type-checking",
                          errors);
            }
        }

        let result = self.resolve_type_vars_if_possible(result);
        let result = self.tcx.erase_regions(&result);

        match self.tcx.lift_to_global(&result) {
            Some(result) => result,
            None => {
                span_bug!(span, "Uninferred types/regions in `{:?}`", result);
            }
        }
    }

    pub fn is_in_snapshot(&self) -> bool {
        self.in_snapshot.get()
    }

    pub fn freshen<T:TypeFoldable<'tcx>>(&self, t: T) -> T {
        t.fold_with(&mut self.freshener())
    }

    pub fn type_var_diverges(&'a self, ty: Ty) -> bool {
        match ty.sty {
            ty::TyInfer(ty::TyVar(vid)) => self.type_variables.borrow().var_diverges(vid),
            _ => false
        }
    }

    pub fn freshener<'b>(&'b self) -> TypeFreshener<'b, 'gcx, 'tcx> {
        freshen::TypeFreshener::new(self)
    }

    pub fn type_is_unconstrained_numeric(&'a self, ty: Ty) -> UnconstrainedNumeric {
        use ty::error::UnconstrainedNumeric::Neither;
        use ty::error::UnconstrainedNumeric::{UnconstrainedInt, UnconstrainedFloat};
        match ty.sty {
            ty::TyInfer(ty::IntVar(vid)) => {
                if self.int_unification_table.borrow_mut().has_value(vid) {
                    Neither
                } else {
                    UnconstrainedInt
                }
            },
            ty::TyInfer(ty::FloatVar(vid)) => {
                if self.float_unification_table.borrow_mut().has_value(vid) {
                    Neither
                } else {
                    UnconstrainedFloat
                }
            },
            _ => Neither,
        }
    }

    /// Returns a type variable's default fallback if any exists. A default
    /// must be attached to the variable when created, if it is created
    /// without a default, this will return None.
    ///
    /// This code does not apply to integral or floating point variables,
    /// only to use declared defaults.
    ///
    /// See `new_ty_var_with_default` to create a type variable with a default.
    /// See `type_variable::Default` for details about what a default entails.
    pub fn default(&self, ty: Ty<'tcx>) -> Option<type_variable::Default<'tcx>> {
        match ty.sty {
            ty::TyInfer(ty::TyVar(vid)) => self.type_variables.borrow().default(vid),
            _ => None
        }
    }

    pub fn unsolved_variables(&self) -> Vec<Ty<'tcx>> {
        let mut variables = Vec::new();

        let unbound_ty_vars = self.type_variables
                                  .borrow_mut()
                                  .unsolved_variables()
                                  .into_iter()
                                  .map(|t| self.tcx.mk_var(t));

        let unbound_int_vars = self.int_unification_table
                                   .borrow_mut()
                                   .unsolved_variables()
                                   .into_iter()
                                   .map(|v| self.tcx.mk_int_var(v));

        let unbound_float_vars = self.float_unification_table
                                     .borrow_mut()
                                     .unsolved_variables()
                                     .into_iter()
                                     .map(|v| self.tcx.mk_float_var(v));

        variables.extend(unbound_ty_vars);
        variables.extend(unbound_int_vars);
        variables.extend(unbound_float_vars);

        return variables;
    }

    fn combine_fields(&'a self, trace: TypeTrace<'tcx>, param_env: ty::ParamEnv<'tcx>)
                      -> CombineFields<'a, 'gcx, 'tcx> {
        CombineFields {
            infcx: self,
            trace,
            cause: None,
            param_env,
            obligations: PredicateObligations::new(),
        }
    }

    // Clear the "currently in a snapshot" flag, invoke the closure,
    // then restore the flag to its original value. This flag is a
    // debugging measure designed to detect cases where we start a
    // snapshot, create type variables, and register obligations
    // which may involve those type variables in the fulfillment cx,
    // potentially leaving "dangling type variables" behind.
    // In such cases, an assertion will fail when attempting to
    // register obligations, within a snapshot. Very useful, much
    // better than grovelling through megabytes of RUST_LOG output.
    //
    // HOWEVER, in some cases the flag is unhelpful. In particular, we
    // sometimes create a "mini-fulfilment-cx" in which we enroll
    // obligations. As long as this fulfillment cx is fully drained
    // before we return, this is not a problem, as there won't be any
    // escaping obligations in the main cx. In those cases, you can
    // use this function.
    pub fn save_and_restore_in_snapshot_flag<F, R>(&self, func: F) -> R
        where F: FnOnce(&Self) -> R
    {
        let flag = self.in_snapshot.get();
        self.in_snapshot.set(false);
        let result = func(self);
        self.in_snapshot.set(flag);
        result
    }

    fn start_snapshot<'b>(&'b self) -> CombinedSnapshot<'b, 'tcx> {
        debug!("start_snapshot()");

        let in_snapshot = self.in_snapshot.get();
        self.in_snapshot.set(true);

        CombinedSnapshot {
            projection_cache_snapshot: self.projection_cache.borrow_mut().snapshot(),
            type_snapshot: self.type_variables.borrow_mut().snapshot(),
            int_snapshot: self.int_unification_table.borrow_mut().snapshot(),
            float_snapshot: self.float_unification_table.borrow_mut().snapshot(),
            region_constraints_snapshot: self.borrow_region_constraints().start_snapshot(),
            region_obligations_snapshot: self.region_obligations.borrow().len(),
            was_in_snapshot: in_snapshot,
            // Borrow tables "in progress" (i.e. during typeck)
            // to ban writes from within a snapshot to them.
            _in_progress_tables: self.in_progress_tables.map(|tables| {
                tables.borrow()
            })
        }
    }

    fn rollback_to(&self, cause: &str, snapshot: CombinedSnapshot) {
        debug!("rollback_to(cause={})", cause);
        let CombinedSnapshot { projection_cache_snapshot,
                               type_snapshot,
                               int_snapshot,
                               float_snapshot,
                               region_constraints_snapshot,
                               region_obligations_snapshot,
                               was_in_snapshot,
                               _in_progress_tables } = snapshot;

        self.in_snapshot.set(was_in_snapshot);

        self.projection_cache
            .borrow_mut()
            .rollback_to(projection_cache_snapshot);
        self.type_variables
            .borrow_mut()
            .rollback_to(type_snapshot);
        self.int_unification_table
            .borrow_mut()
            .rollback_to(int_snapshot);
        self.float_unification_table
            .borrow_mut()
            .rollback_to(float_snapshot);
        self.region_obligations
            .borrow_mut()
            .truncate(region_obligations_snapshot);
        self.borrow_region_constraints()
            .rollback_to(region_constraints_snapshot);
    }

    fn commit_from(&self, snapshot: CombinedSnapshot) {
        debug!("commit_from()");
        let CombinedSnapshot { projection_cache_snapshot,
                               type_snapshot,
                               int_snapshot,
                               float_snapshot,
                               region_constraints_snapshot,
                               region_obligations_snapshot: _,
                               was_in_snapshot,
                               _in_progress_tables } = snapshot;

        self.in_snapshot.set(was_in_snapshot);

        self.projection_cache
            .borrow_mut()
            .commit(projection_cache_snapshot);
        self.type_variables
            .borrow_mut()
            .commit(type_snapshot);
        self.int_unification_table
            .borrow_mut()
            .commit(int_snapshot);
        self.float_unification_table
            .borrow_mut()
            .commit(float_snapshot);
        self.borrow_region_constraints()
            .commit(region_constraints_snapshot);
    }

    /// Execute `f` and commit the bindings
    pub fn commit_unconditionally<R, F>(&self, f: F) -> R where
        F: FnOnce() -> R,
    {
        debug!("commit()");
        let snapshot = self.start_snapshot();
        let r = f();
        self.commit_from(snapshot);
        r
    }

    /// Execute `f` and commit the bindings if closure `f` returns `Ok(_)`
    pub fn commit_if_ok<T, E, F>(&self, f: F) -> Result<T, E> where
        F: FnOnce(&CombinedSnapshot) -> Result<T, E>
    {
        debug!("commit_if_ok()");
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        debug!("commit_if_ok() -- r.is_ok() = {}", r.is_ok());
        match r {
            Ok(_) => { self.commit_from(snapshot); }
            Err(_) => { self.rollback_to("commit_if_ok -- error", snapshot); }
        }
        r
    }

    // Execute `f` in a snapshot, and commit the bindings it creates
    pub fn in_snapshot<T, F>(&self, f: F) -> T where
        F: FnOnce(&CombinedSnapshot) -> T
    {
        debug!("in_snapshot()");
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        self.commit_from(snapshot);
        r
    }

    /// Execute `f` then unroll any bindings it creates
    pub fn probe<R, F>(&self, f: F) -> R where
        F: FnOnce(&CombinedSnapshot) -> R,
    {
        debug!("probe()");
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        self.rollback_to("probe", snapshot);
        r
    }

    pub fn add_given(&self,
                     sub: ty::Region<'tcx>,
                     sup: ty::RegionVid)
    {
        self.borrow_region_constraints().add_given(sub, sup);
    }

    pub fn can_sub<T>(&self,
                      param_env: ty::ParamEnv<'tcx>,
                      a: T,
                      b: T)
                      -> UnitResult<'tcx>
        where T: at::ToTrace<'tcx>
    {
        let origin = &ObligationCause::dummy();
        self.probe(|_| {
            self.at(origin, param_env).sub(a, b).map(|InferOk { obligations: _, .. }| {
                // Ignore obligations, since we are unrolling
                // everything anyway.
            })
        })
    }

    pub fn can_eq<T>(&self,
                      param_env: ty::ParamEnv<'tcx>,
                      a: T,
                      b: T)
                      -> UnitResult<'tcx>
        where T: at::ToTrace<'tcx>
    {
        let origin = &ObligationCause::dummy();
        self.probe(|_| {
            self.at(origin, param_env).eq(a, b).map(|InferOk { obligations: _, .. }| {
                // Ignore obligations, since we are unrolling
                // everything anyway.
            })
        })
    }

    pub fn sub_regions(&self,
                       origin: SubregionOrigin<'tcx>,
                       a: ty::Region<'tcx>,
                       b: ty::Region<'tcx>) {
        debug!("sub_regions({:?} <: {:?})", a, b);
        self.borrow_region_constraints().make_subregion(origin, a, b);
    }

    pub fn equality_predicate(&self,
                              cause: &ObligationCause<'tcx>,
                              param_env: ty::ParamEnv<'tcx>,
                              predicate: &ty::PolyEquatePredicate<'tcx>)
        -> InferResult<'tcx, ()>
    {
        self.commit_if_ok(|snapshot| {
            let (ty::EquatePredicate(a, b), skol_map) =
                self.skolemize_late_bound_regions(predicate, snapshot);
            let cause_span = cause.span;
            let eqty_ok = self.at(cause, param_env).eq(b, a)?;
            self.leak_check(false, cause_span, &skol_map, snapshot)?;
            self.pop_skolemized(skol_map, snapshot);
            Ok(eqty_ok.unit())
        })
    }

    pub fn subtype_predicate(&self,
                             cause: &ObligationCause<'tcx>,
                             param_env: ty::ParamEnv<'tcx>,
                             predicate: &ty::PolySubtypePredicate<'tcx>)
        -> Option<InferResult<'tcx, ()>>
    {
        // Subtle: it's ok to skip the binder here and resolve because
        // `shallow_resolve` just ignores anything that is not a type
        // variable, and because type variable's can't (at present, at
        // least) capture any of the things bound by this binder.
        //
        // Really, there is no *particular* reason to do this
        // `shallow_resolve` here except as a
        // micro-optimization. Naturally I could not
        // resist. -nmatsakis
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
            let (ty::SubtypePredicate { a_is_expected, a, b}, skol_map) =
                self.skolemize_late_bound_regions(predicate, snapshot);

            let cause_span = cause.span;
            let ok = self.at(cause, param_env).sub_exp(a_is_expected, a, b)?;
            self.leak_check(false, cause_span, &skol_map, snapshot)?;
            self.pop_skolemized(skol_map, snapshot);
            Ok(ok.unit())
        }))
    }

    pub fn region_outlives_predicate(&self,
                                     cause: &traits::ObligationCause<'tcx>,
                                     predicate: &ty::PolyRegionOutlivesPredicate<'tcx>)
        -> UnitResult<'tcx>
    {
        self.commit_if_ok(|snapshot| {
            let (ty::OutlivesPredicate(r_a, r_b), skol_map) =
                self.skolemize_late_bound_regions(predicate, snapshot);
            let origin =
                SubregionOrigin::from_obligation_cause(cause,
                                                       || RelateRegionParamBound(cause.span));
            self.sub_regions(origin, r_b, r_a); // `b : a` ==> `a <= b`
            self.leak_check(false, cause.span, &skol_map, snapshot)?;
            Ok(self.pop_skolemized(skol_map, snapshot))
        })
    }

    pub fn next_ty_var_id(&self, diverging: bool, origin: TypeVariableOrigin) -> TyVid {
        self.type_variables
            .borrow_mut()
            .new_var(diverging, origin, None)
    }

    pub fn next_ty_var(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        self.tcx.mk_var(self.next_ty_var_id(false, origin))
    }

    pub fn next_diverging_ty_var(&self, origin: TypeVariableOrigin) -> Ty<'tcx> {
        self.tcx.mk_var(self.next_ty_var_id(true, origin))
    }

    pub fn next_int_var_id(&self) -> IntVid {
        self.int_unification_table
            .borrow_mut()
            .new_key(None)
    }

    pub fn next_float_var_id(&self) -> FloatVid {
        self.float_unification_table
            .borrow_mut()
            .new_key(None)
    }

    /// Create a fresh region variable with the next available index.
    ///
    /// # Parameters
    ///
    /// - `origin`: information about why we created this variable, for use
    ///   during diagnostics / error-reporting.
    pub fn next_region_var(&self, origin: RegionVariableOrigin)
                           -> ty::Region<'tcx> {
        self.tcx.mk_region(ty::ReVar(self.borrow_region_constraints().new_region_var(origin)))
    }

    /// Number of region variables created so far.
    pub fn num_region_vars(&self) -> usize {
        self.borrow_region_constraints().var_origins().len()
    }

    /// Just a convenient wrapper of `next_region_var` for using during NLL.
    pub fn next_nll_region_var(&self, origin: NLLRegionVariableOrigin)
                               -> ty::Region<'tcx> {
        self.next_region_var(RegionVariableOrigin::NLL(origin))
    }

    /// Create a region inference variable for the given
    /// region parameter definition.
    pub fn region_var_for_def(&self,
                              span: Span,
                              def: &ty::RegionParameterDef)
                              -> ty::Region<'tcx> {
        self.next_region_var(EarlyBoundRegion(span, def.name))
    }

    /// Create a type inference variable for the given
    /// type parameter definition. The substitutions are
    /// for actual parameters that may be referred to by
    /// the default of this type parameter, if it exists.
    /// E.g. `struct Foo<A, B, C = (A, B)>(...);` when
    /// used in a path such as `Foo::<T, U>::new()` will
    /// use an inference variable for `C` with `[T, U]`
    /// as the substitutions for the default, `(T, U)`.
    pub fn type_var_for_def(&self,
                            span: Span,
                            def: &ty::TypeParameterDef,
                            substs: &[Kind<'tcx>])
                            -> Ty<'tcx> {
        let default = if def.has_default {
            let default = self.tcx.type_of(def.def_id);
            Some(type_variable::Default {
                ty: default.subst_spanned(self.tcx, substs, Some(span)),
                origin_span: span,
                def_id: def.def_id
            })
        } else {
            None
        };


        let ty_var_id = self.type_variables
                            .borrow_mut()
                            .new_var(false,
                                     TypeVariableOrigin::TypeParameterDefinition(span, def.name),
                                     default);

        self.tcx.mk_var(ty_var_id)
    }

    /// Given a set of generics defined on a type or impl, returns a substitution mapping each
    /// type/region parameter to a fresh inference variable.
    pub fn fresh_substs_for_item(&self,
                                 span: Span,
                                 def_id: DefId)
                                 -> &'tcx Substs<'tcx> {
        Substs::for_item(self.tcx, def_id, |def, _| {
            self.region_var_for_def(span, def)
        }, |def, substs| {
            self.type_var_for_def(span, def, substs)
        })
    }

    /// True if errors have been reported since this infcx was
    /// created.  This is sometimes used as a heuristic to skip
    /// reporting errors that often occur as a result of earlier
    /// errors, but where it's hard to be 100% sure (e.g., unresolved
    /// inference variables, regionck errors).
    pub fn is_tainted_by_errors(&self) -> bool {
        debug!("is_tainted_by_errors(err_count={}, err_count_on_creation={}, \
                tainted_by_errors_flag={})",
               self.tcx.sess.err_count(),
               self.err_count_on_creation,
               self.tainted_by_errors_flag.get());

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
    /// `resolve_type_vars_if_possible` as well as `fully_resolve`.
    pub fn resolve_regions_and_report_errors(
        &self,
        region_context: DefId,
        region_map: &region::ScopeTree,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) {
        self.resolve_regions_and_report_errors_inner(
            region_context,
            region_map,
            outlives_env,
            false,
        )
    }

    /// Like `resolve_regions_and_report_errors`, but skips error
    /// reporting if NLL is enabled.  This is used for fn bodies where
    /// the same error may later be reported by the NLL-based
    /// inference.
    pub fn resolve_regions_and_report_errors_unless_nll(
        &self,
        region_context: DefId,
        region_map: &region::ScopeTree,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) {
        self.resolve_regions_and_report_errors_inner(
            region_context,
            region_map,
            outlives_env,
            true,
        )
    }

    fn resolve_regions_and_report_errors_inner(
        &self,
        region_context: DefId,
        region_map: &region::ScopeTree,
        outlives_env: &OutlivesEnvironment<'tcx>,
        will_later_be_reported_by_nll: bool,
    ) {
        assert!(self.is_tainted_by_errors() || self.region_obligations.borrow().is_empty(),
                "region_obligations not empty: {:#?}",
                self.region_obligations.borrow());

        let region_rels = &RegionRelations::new(self.tcx,
                                                region_context,
                                                region_map,
                                                outlives_env.free_region_map());
        let (var_origins, data) = self.region_constraints.borrow_mut()
                                                         .take()
                                                         .expect("regions already resolved")
                                                         .into_origins_and_data();
        let (lexical_region_resolutions, errors) =
            lexical_region_resolve::resolve(region_rels, var_origins, data);

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());

        if !self.is_tainted_by_errors() {
            // As a heuristic, just skip reporting region errors
            // altogether if other errors have been reported while
            // this infcx was in use.  This is totally hokey but
            // otherwise we have a hard time separating legit region
            // errors from silly ones.
            self.report_region_errors(region_map, &errors, will_later_be_reported_by_nll);
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
        assert!(self.region_obligations.borrow().is_empty(),
                "region_obligations not empty: {:#?}",
                self.region_obligations.borrow());

        self.borrow_region_constraints().take_and_reset_data()
    }

    /// Takes ownership of the list of variable regions. This implies
    /// that all the region constriants have already been taken, and
    /// hence that `resolve_regions_and_report_errors` can never be
    /// called. This is used only during NLL processing to "hand off" ownership
    /// of the set of region vairables into the NLL region context.
    pub fn take_region_var_origins(&self) -> VarOrigins {
        let (var_origins, data) = self.region_constraints.borrow_mut()
                                                         .take()
                                                         .expect("regions already resolved")
                                                         .into_origins_and_data();
        assert!(data.is_empty());
        var_origins
    }

    pub fn ty_to_string(&self, t: Ty<'tcx>) -> String {
        self.resolve_type_vars_if_possible(&t).to_string()
    }

    pub fn tys_to_string(&self, ts: &[Ty<'tcx>]) -> String {
        let tstrs: Vec<String> = ts.iter().map(|t| self.ty_to_string(*t)).collect();
        format!("({})", tstrs.join(", "))
    }

    pub fn trait_ref_to_string(&self, t: &ty::TraitRef<'tcx>) -> String {
        self.resolve_type_vars_if_possible(t).to_string()
    }

    pub fn shallow_resolve(&self, typ: Ty<'tcx>) -> Ty<'tcx> {
        match typ.sty {
            ty::TyInfer(ty::TyVar(v)) => {
                // Not entirely obvious: if `typ` is a type variable,
                // it can be resolved to an int/float variable, which
                // can then be recursively resolved, hence the
                // recursion. Note though that we prevent type
                // variables from unifying to other type variables
                // directly (though they may be embedded
                // structurally), and we prevent cycles in any case,
                // so this recursion should always be of very limited
                // depth.
                self.type_variables.borrow_mut()
                    .probe(v)
                    .map(|t| self.shallow_resolve(t))
                    .unwrap_or(typ)
            }

            ty::TyInfer(ty::IntVar(v)) => {
                self.int_unification_table
                    .borrow_mut()
                    .probe(v)
                    .map(|v| v.to_type(self.tcx))
                    .unwrap_or(typ)
            }

            ty::TyInfer(ty::FloatVar(v)) => {
                self.float_unification_table
                    .borrow_mut()
                    .probe(v)
                    .map(|v| v.to_type(self.tcx))
                    .unwrap_or(typ)
            }

            _ => {
                typ
            }
        }
    }

    pub fn resolve_type_vars_if_possible<T>(&self, value: &T) -> T
        where T: TypeFoldable<'tcx>
    {
        /*!
         * Where possible, replaces type/int/float variables in
         * `value` with their final value. Note that region variables
         * are unaffected. If a type variable has not been unified, it
         * is left as is.  This is an idempotent operation that does
         * not affect inference state in any way and so you can do it
         * at will.
         */

        if !value.needs_infer() {
            return value.clone(); // avoid duplicated subst-folding
        }
        let mut r = resolve::OpportunisticTypeResolver::new(self);
        value.fold_with(&mut r)
    }

    /// Returns true if `T` contains unresolved type variables. In the
    /// process of visiting `T`, this will resolve (where possible)
    /// type variables in `T`, but it never constructs the final,
    /// resolved type, so it's more efficient than
    /// `resolve_type_vars_if_possible()`.
    pub fn any_unresolved_type_vars<T>(&self, value: &T) -> bool
        where T: TypeFoldable<'tcx>
    {
        let mut r = resolve::UnresolvedTypeFinder::new(self);
        value.visit_with(&mut r)
    }

    pub fn resolve_type_and_region_vars_if_possible<T>(&self, value: &T) -> T
        where T: TypeFoldable<'tcx>
    {
        let mut r = resolve::OpportunisticTypeAndRegionResolver::new(self);
        value.fold_with(&mut r)
    }

    pub fn fully_resolve<T:TypeFoldable<'tcx>>(&self, value: &T) -> FixupResult<T> {
        /*!
         * Attempts to resolve all type/region variables in
         * `value`. Region inference must have been run already (e.g.,
         * by calling `resolve_regions_and_report_errors`).  If some
         * variable was never unified, an `Err` results.
         *
         * This method is idempotent, but it not typically not invoked
         * except during the writeback phase.
         */

        resolve::fully_resolve(self, value)
    }

    // [Note-Type-error-reporting]
    // An invariant is that anytime the expected or actual type is TyError (the special
    // error type, meaning that an error occurred when typechecking this expression),
    // this is a derived error. The error cascaded from another error (that was already
    // reported), so it's not useful to display it to the user.
    // The following methods implement this logic.
    // They check if either the actual or expected type is TyError, and don't print the error
    // in this case. The typechecker should only ever report type errors involving mismatched
    // types using one of these methods, and should not call span_err directly for such
    // errors.

    pub fn type_error_struct_with_diag<M>(&self,
                                          sp: Span,
                                          mk_diag: M,
                                          actual_ty: Ty<'tcx>)
                                          -> DiagnosticBuilder<'tcx>
        where M: FnOnce(String) -> DiagnosticBuilder<'tcx>,
    {
        let actual_ty = self.resolve_type_vars_if_possible(&actual_ty);
        debug!("type_error_struct_with_diag({:?}, {:?})", sp, actual_ty);

        // Don't report an error if actual type is TyError.
        if actual_ty.references_error() {
            return self.tcx.sess.diagnostic().struct_dummy();
        }

        mk_diag(self.ty_to_string(actual_ty))
    }

    pub fn report_mismatched_types(&self,
                                   cause: &ObligationCause<'tcx>,
                                   expected: Ty<'tcx>,
                                   actual: Ty<'tcx>,
                                   err: TypeError<'tcx>)
                                   -> DiagnosticBuilder<'tcx> {
        let trace = TypeTrace::types(cause, true, expected, actual);
        self.report_and_explain_type_error(trace, &err)
    }

    pub fn report_conflicting_default_types(&self,
                                            span: Span,
                                            body_id: ast::NodeId,
                                            expected: type_variable::Default<'tcx>,
                                            actual: type_variable::Default<'tcx>) {
        let trace = TypeTrace {
            cause: ObligationCause::misc(span, body_id),
            values: Types(ExpectedFound {
                expected: expected.ty,
                found: actual.ty
            })
        };

        self.report_and_explain_type_error(
            trace,
            &TypeError::TyParamDefaultMismatch(ExpectedFound {
                expected,
                found: actual
            }))
            .emit();
    }

    pub fn replace_late_bound_regions_with_fresh_var<T>(
        &self,
        span: Span,
        lbrct: LateBoundRegionConversionTime,
        value: &ty::Binder<T>)
        -> (T, BTreeMap<ty::BoundRegion, ty::Region<'tcx>>)
        where T : TypeFoldable<'tcx>
    {
        self.tcx.replace_late_bound_regions(
            value,
            |br| self.next_region_var(LateBoundRegion(span, br, lbrct)))
    }

    /// Given a higher-ranked projection predicate like:
    ///
    ///     for<'a> <T as Fn<&'a u32>>::Output = &'a u32
    ///
    /// and a target trait-ref like:
    ///
    ///     <T as Fn<&'x u32>>
    ///
    /// find a substitution `S` for the higher-ranked regions (here,
    /// `['a => 'x]`) such that the predicate matches the trait-ref,
    /// and then return the value (here, `&'a u32`) but with the
    /// substitution applied (hence, `&'x u32`).
    ///
    /// See `higher_ranked_match` in `higher_ranked/mod.rs` for more
    /// details.
    pub fn match_poly_projection_predicate(&self,
                                           cause: ObligationCause<'tcx>,
                                           param_env: ty::ParamEnv<'tcx>,
                                           match_a: ty::PolyProjectionPredicate<'tcx>,
                                           match_b: ty::TraitRef<'tcx>)
                                           -> InferResult<'tcx, HrMatchResult<Ty<'tcx>>>
    {
        let match_pair = match_a.map_bound(|p| (p.projection_ty.trait_ref(self.tcx), p.ty));
        let trace = TypeTrace {
            cause,
            values: TraitRefs(ExpectedFound::new(true, match_pair.skip_binder().0, match_b))
        };

        let mut combine = self.combine_fields(trace, param_env);
        let result = combine.higher_ranked_match(&match_pair, &match_b, true)?;
        Ok(InferOk { value: result, obligations: combine.obligations })
    }

    /// See `verify_generic_bound` method in `region_constraints`
    pub fn verify_generic_bound(&self,
                                origin: SubregionOrigin<'tcx>,
                                kind: GenericKind<'tcx>,
                                a: ty::Region<'tcx>,
                                bound: VerifyBound<'tcx>) {
        debug!("verify_generic_bound({:?}, {:?} <: {:?})",
               kind,
               a,
               bound);

        self.borrow_region_constraints().verify_generic_bound(origin, kind, a, bound);
    }

    pub fn type_moves_by_default(&self,
                                 param_env: ty::ParamEnv<'tcx>,
                                 ty: Ty<'tcx>,
                                 span: Span)
                                 -> bool {
        let ty = self.resolve_type_vars_if_possible(&ty);
        // Even if the type may have no inference variables, during
        // type-checking closure types are in local tables only.
        if !self.in_progress_tables.is_some() || !ty.has_closure_types() {
            if let Some((param_env, ty)) = self.tcx.lift_to_global(&(param_env, ty)) {
                return ty.moves_by_default(self.tcx.global_tcx(), param_env, span);
            }
        }

        let copy_def_id = self.tcx.require_lang_item(lang_items::CopyTraitLangItem);

        // this can get called from typeck (by euv), and moves_by_default
        // rightly refuses to work with inference variables, but
        // moves_by_default has a cache, which we want to use in other
        // cases.
        !traits::type_known_to_meet_bound(self, param_env, ty, copy_def_id, span)
    }

    /// Obtains the latest type of the given closure; this may be a
    /// closure in the current function, in which case its
    /// `ClosureKind` may not yet be known.
    pub fn closure_kind(&self,
                        closure_def_id: DefId,
                        closure_substs: ty::ClosureSubsts<'tcx>)
                        -> Option<ty::ClosureKind>
    {
        let closure_kind_ty = closure_substs.closure_kind_ty(closure_def_id, self.tcx);
        let closure_kind_ty = self.shallow_resolve(&closure_kind_ty);
        closure_kind_ty.to_opt_closure_kind()
    }

    /// Obtain the signature of a closure.  For closures, unlike
    /// `tcx.fn_sig(def_id)`, this method will work during the
    /// type-checking of the enclosing function and return the closure
    /// signature in its partially inferred state.
    pub fn closure_sig(
        &self,
        def_id: DefId,
        substs: ty::ClosureSubsts<'tcx>
    ) -> ty::PolyFnSig<'tcx> {
        let closure_sig_ty = substs.closure_sig_ty(def_id, self.tcx);
        let closure_sig_ty = self.shallow_resolve(&closure_sig_ty);
        closure_sig_ty.fn_sig(self.tcx)
    }

    /// Normalizes associated types in `value`, potentially returning
    /// new obligations that must further be processed.
    pub fn partially_normalize_associated_types_in<T>(&self,
                                                      span: Span,
                                                      body_id: ast::NodeId,
                                                      param_env: ty::ParamEnv<'tcx>,
                                                      value: &T)
                                                      -> InferOk<'tcx, T>
        where T : TypeFoldable<'tcx>
    {
        debug!("partially_normalize_associated_types_in(value={:?})", value);
        let mut selcx = traits::SelectionContext::new(self);
        let cause = ObligationCause::misc(span, body_id);
        let traits::Normalized { value, obligations } =
            traits::normalize(&mut selcx, param_env, cause, value);
        debug!("partially_normalize_associated_types_in: result={:?} predicates={:?}",
            value,
            obligations);
        InferOk { value, obligations }
    }

    fn borrow_region_constraints(&self) -> RefMut<'_, RegionConstraintCollector<'tcx>> {
        RefMut::map(
            self.region_constraints.borrow_mut(),
            |c| c.as_mut().expect("region constraints already solved"))
    }
}

impl<'a, 'gcx, 'tcx> TypeTrace<'tcx> {
    pub fn span(&self) -> Span {
        self.cause.span
    }

    pub fn types(cause: &ObligationCause<'tcx>,
                 a_is_expected: bool,
                 a: Ty<'tcx>,
                 b: Ty<'tcx>)
                 -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: Types(ExpectedFound::new(a_is_expected, a, b))
        }
    }

    pub fn dummy(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: ObligationCause::dummy(),
            values: Types(ExpectedFound {
                expected: tcx.types.err,
                found: tcx.types.err,
            })
        }
    }
}

impl<'tcx> fmt::Debug for TypeTrace<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypeTrace({:?})", self.cause)
    }
}

impl<'tcx> SubregionOrigin<'tcx> {
    pub fn span(&self) -> Span {
        match *self {
            Subtype(ref a) => a.span(),
            InfStackClosure(a) => a,
            InvokeClosure(a) => a,
            DerefPointer(a) => a,
            FreeVariable(a, _) => a,
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

    pub fn from_obligation_cause<F>(cause: &traits::ObligationCause<'tcx>,
                                    default: F)
                                    -> Self
        where F: FnOnce() -> Self
    {
        match cause.code {
            traits::ObligationCauseCode::ReferenceOutlivesReferent(ref_type) =>
                SubregionOrigin::ReferenceOutlivesReferent(ref_type, cause.span),

            traits::ObligationCauseCode::CompareImplMethodObligation { item_name,
                                                                       impl_item_def_id,
                                                                       trait_item_def_id, } =>
                SubregionOrigin::CompareImplMethodObligation {
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
            BoundRegionInCoherence(_) => syntax_pos::DUMMY_SP,
            UpvarRegion(_, a) => a,
            NLL(..) => bug!("NLL variable used with `span`"),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for ValuePairs<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        match *self {
            ValuePairs::Types(ref ef) => {
                ValuePairs::Types(ef.fold_with(folder))
            }
            ValuePairs::TraitRefs(ref ef) => {
                ValuePairs::TraitRefs(ef.fold_with(folder))
            }
            ValuePairs::PolyTraitRefs(ref ef) => {
                ValuePairs::PolyTraitRefs(ef.fold_with(folder))
            }
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        match *self {
            ValuePairs::Types(ref ef) => ef.visit_with(visitor),
            ValuePairs::TraitRefs(ref ef) => ef.visit_with(visitor),
            ValuePairs::PolyTraitRefs(ref ef) => ef.visit_with(visitor),
        }
    }
}

impl<'tcx> TypeFoldable<'tcx> for TypeTrace<'tcx> {
    fn super_fold_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(&self, folder: &mut F) -> Self {
        TypeTrace {
            cause: self.cause.fold_with(folder),
            values: self.values.fold_with(folder)
        }
    }

    fn super_visit_with<V: TypeVisitor<'tcx>>(&self, visitor: &mut V) -> bool {
        self.cause.visit_with(visitor) || self.values.visit_with(visitor)
    }
}

impl<'tcx> fmt::Debug for RegionObligation<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionObligation(sub_region={:?}, sup_type={:?})",
               self.sub_region,
               self.sup_type)
    }
}

