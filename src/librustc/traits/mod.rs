//! Trait Resolution. See the [rustc guide] for more information on how this works.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/traits/resolution.html

pub mod query;
pub mod select;
pub mod specialization_graph;
mod structural_impls;

use crate::mir::interpret::ErrorHandled;
use crate::ty::subst::SubstsRef;
use crate::ty::{self, AdtKind, List, Ty, TyCtxt};

use rustc_ast::ast;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_span::{Span, DUMMY_SP};
use smallvec::SmallVec;

use std::borrow::Cow;
use std::fmt::Debug;
use std::rc::Rc;

pub use self::select::{EvaluationCache, EvaluationResult, OverflowError, SelectionCache};

pub use self::ObligationCauseCode::*;
pub use self::SelectionError::*;
pub use self::Vtable::*;

/// Depending on the stage of compilation, we want projection to be
/// more or less conservative.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, HashStable)]
pub enum Reveal {
    /// At type-checking time, we refuse to project any associated
    /// type that is marked `default`. Non-`default` ("final") types
    /// are always projected. This is necessary in general for
    /// soundness of specialization. However, we *could* allow
    /// projections in fully-monomorphic cases. We choose not to,
    /// because we prefer for `default type` to force the type
    /// definition to be treated abstractly by any consumers of the
    /// impl. Concretely, that means that the following example will
    /// fail to compile:
    ///
    /// ```
    /// trait Assoc {
    ///     type Output;
    /// }
    ///
    /// impl<T> Assoc for T {
    ///     default type Output = bool;
    /// }
    ///
    /// fn main() {
    ///     let <() as Assoc>::Output = true;
    /// }
    /// ```
    UserFacing,

    /// At codegen time, all monomorphic projections will succeed.
    /// Also, `impl Trait` is normalized to the concrete type,
    /// which has to be already collected by type-checking.
    ///
    /// NOTE: as `impl Trait`'s concrete type should *never*
    /// be observable directly by the user, `Reveal::All`
    /// should not be used by checks which may expose
    /// type equality or type contents to the user.
    /// There are some exceptions, e.g., around OIBITS and
    /// transmute-checking, which expose some details, but
    /// not the whole concrete type of the `impl Trait`.
    All,
}

/// The reason why we incurred this obligation; used for error reporting.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ObligationCause<'tcx> {
    pub span: Span,

    /// The ID of the fn body that triggered this obligation. This is
    /// used for region obligations to determine the precise
    /// environment in which the region obligation should be evaluated
    /// (in particular, closures can add new assumptions). See the
    /// field `region_obligations` of the `FulfillmentContext` for more
    /// information.
    pub body_id: hir::HirId,

    pub code: ObligationCauseCode<'tcx>,
}

impl<'tcx> ObligationCause<'tcx> {
    #[inline]
    pub fn new(
        span: Span,
        body_id: hir::HirId,
        code: ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        ObligationCause { span, body_id, code }
    }

    pub fn misc(span: Span, body_id: hir::HirId) -> ObligationCause<'tcx> {
        ObligationCause { span, body_id, code: MiscObligation }
    }

    pub fn dummy() -> ObligationCause<'tcx> {
        ObligationCause { span: DUMMY_SP, body_id: hir::CRATE_HIR_ID, code: MiscObligation }
    }

    pub fn span(&self, tcx: TyCtxt<'tcx>) -> Span {
        match self.code {
            ObligationCauseCode::CompareImplMethodObligation { .. }
            | ObligationCauseCode::MainFunctionType
            | ObligationCauseCode::StartFunctionType => tcx.sess.source_map().def_span(self.span),
            ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                arm_span,
                ..
            }) => arm_span,
            _ => self.span,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ObligationCauseCode<'tcx> {
    /// Not well classified or should be obvious from the span.
    MiscObligation,

    /// A slice or array is WF only if `T: Sized`.
    SliceOrArrayElem,

    /// A tuple is WF only if its middle elements are `Sized`.
    TupleElem,

    /// This is the trait reference from the given projection.
    ProjectionWf(ty::ProjectionTy<'tcx>),

    /// In an impl of trait `X` for type `Y`, type `Y` must
    /// also implement all supertraits of `X`.
    ItemObligation(DefId),

    /// Like `ItemObligation`, but with extra detail on the source of the obligation.
    BindingObligation(DefId, Span),

    /// A type like `&'a T` is WF only if `T: 'a`.
    ReferenceOutlivesReferent(Ty<'tcx>),

    /// A type like `Box<Foo<'a> + 'b>` is WF only if `'b: 'a`.
    ObjectTypeBound(Ty<'tcx>, ty::Region<'tcx>),

    /// Obligation incurred due to an object cast.
    ObjectCastObligation(/* Object type */ Ty<'tcx>),

    /// Obligation incurred due to a coercion.
    Coercion {
        source: Ty<'tcx>,
        target: Ty<'tcx>,
    },

    /// Various cases where expressions must be `Sized` / `Copy` / etc.
    /// `L = X` implies that `L` is `Sized`.
    AssignmentLhsSized,
    /// `(x1, .., xn)` must be `Sized`.
    TupleInitializerSized,
    /// `S { ... }` must be `Sized`.
    StructInitializerSized,
    /// Type of each variable must be `Sized`.
    VariableType(hir::HirId),
    /// Argument type must be `Sized`.
    SizedArgumentType,
    /// Return type must be `Sized`.
    SizedReturnType,
    /// Yield type must be `Sized`.
    SizedYieldType,
    /// `[T, ..n]` implies that `T` must be `Copy`.
    /// If `true`, suggest `const_in_array_repeat_expressions` feature flag.
    RepeatVec(bool),

    /// Types of fields (other than the last, except for packed structs) in a struct must be sized.
    FieldSized {
        adt_kind: AdtKind,
        last: bool,
    },

    /// Constant expressions must be sized.
    ConstSized,

    /// `static` items must have `Sync` type.
    SharedStatic,

    BuiltinDerivedObligation(DerivedObligationCause<'tcx>),

    ImplDerivedObligation(DerivedObligationCause<'tcx>),

    /// Error derived when matching traits/impls; see ObligationCause for more details
    CompareImplMethodObligation {
        item_name: ast::Name,
        impl_item_def_id: DefId,
        trait_item_def_id: DefId,
    },

    /// Error derived when matching traits/impls; see ObligationCause for more details
    CompareImplTypeObligation {
        item_name: ast::Name,
        impl_item_def_id: DefId,
        trait_item_def_id: DefId,
    },

    /// Checking that this expression can be assigned where it needs to be
    // FIXME(eddyb) #11161 is the original Expr required?
    ExprAssignable,

    /// Computing common supertype in the arms of a match expression
    MatchExpressionArm(Box<MatchExpressionArmCause<'tcx>>),

    /// Type error arising from type checking a pattern against an expected type.
    Pattern {
        /// The span of the scrutinee or type expression which caused the `root_ty` type.
        span: Option<Span>,
        /// The root expected type induced by a scrutinee or type expression.
        root_ty: Ty<'tcx>,
        /// Whether the `Span` came from an expression or a type expression.
        origin_expr: bool,
    },

    /// Constants in patterns must have `Structural` type.
    ConstPatternStructural,

    /// Computing common supertype in an if expression
    IfExpression(Box<IfExpressionCause>),

    /// Computing common supertype of an if expression with no else counter-part
    IfExpressionWithNoElse,

    /// `main` has wrong type
    MainFunctionType,

    /// `start` has wrong type
    StartFunctionType,

    /// Intrinsic has wrong type
    IntrinsicType,

    /// Method receiver
    MethodReceiver,

    /// `return` with no expression
    ReturnNoExpression,

    /// `return` with an expression
    ReturnValue(hir::HirId),

    /// Return type of this function
    ReturnType,

    /// Block implicit return
    BlockTailExpression(hir::HirId),

    /// #[feature(trivial_bounds)] is not enabled
    TrivialBound,

    AssocTypeBound(Box<AssocTypeBoundData>),
}

impl ObligationCauseCode<'_> {
    // Return the base obligation, ignoring derived obligations.
    pub fn peel_derives(&self) -> &Self {
        let mut base_cause = self;
        while let BuiltinDerivedObligation(cause) | ImplDerivedObligation(cause) = base_cause {
            base_cause = &cause.parent_code;
        }
        base_cause
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AssocTypeBoundData {
    pub impl_span: Option<Span>,
    pub original: Span,
    pub bounds: Vec<Span>,
}

// `ObligationCauseCode` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(ObligationCauseCode<'_>, 32);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatchExpressionArmCause<'tcx> {
    pub arm_span: Span,
    pub source: hir::MatchSource,
    pub prior_arms: Vec<Span>,
    pub last_ty: Ty<'tcx>,
    pub scrut_hir_id: hir::HirId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IfExpressionCause {
    pub then: Span,
    pub outer: Option<Span>,
    pub semicolon: Option<Span>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DerivedObligationCause<'tcx> {
    /// The trait reference of the parent obligation that led to the
    /// current obligation. Note that only trait obligations lead to
    /// derived obligations, so we just store the trait reference here
    /// directly.
    pub parent_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The parent trait had this cause.
    pub parent_code: Rc<ObligationCauseCode<'tcx>>,
}

/// The following types:
/// * `WhereClause`,
/// * `WellFormed`,
/// * `FromEnv`,
/// * `DomainGoal`,
/// * `Goal`,
/// * `Clause`,
/// * `Environment`,
/// * `InEnvironment`,
/// are used for representing the trait system in the form of
/// logic programming clauses. They are part of the interface
/// for the chalk SLG solver.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable, Lift)]
pub enum WhereClause<'tcx> {
    Implemented(ty::TraitPredicate<'tcx>),
    ProjectionEq(ty::ProjectionPredicate<'tcx>),
    RegionOutlives(ty::RegionOutlivesPredicate<'tcx>),
    TypeOutlives(ty::TypeOutlivesPredicate<'tcx>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable, Lift)]
pub enum WellFormed<'tcx> {
    Trait(ty::TraitPredicate<'tcx>),
    Ty(Ty<'tcx>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable, Lift)]
pub enum FromEnv<'tcx> {
    Trait(ty::TraitPredicate<'tcx>),
    Ty(Ty<'tcx>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable, Lift)]
pub enum DomainGoal<'tcx> {
    Holds(WhereClause<'tcx>),
    WellFormed(WellFormed<'tcx>),
    FromEnv(FromEnv<'tcx>),
    Normalize(ty::ProjectionPredicate<'tcx>),
}

pub type PolyDomainGoal<'tcx> = ty::Binder<DomainGoal<'tcx>>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable)]
pub enum QuantifierKind {
    Universal,
    Existential,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable, Lift)]
pub enum GoalKind<'tcx> {
    Implies(Clauses<'tcx>, Goal<'tcx>),
    And(Goal<'tcx>, Goal<'tcx>),
    Not(Goal<'tcx>),
    DomainGoal(DomainGoal<'tcx>),
    Quantified(QuantifierKind, ty::Binder<Goal<'tcx>>),
    Subtype(Ty<'tcx>, Ty<'tcx>),
    CannotProve,
}

pub type Goal<'tcx> = &'tcx GoalKind<'tcx>;

pub type Goals<'tcx> = &'tcx List<Goal<'tcx>>;

impl<'tcx> DomainGoal<'tcx> {
    pub fn into_goal(self) -> GoalKind<'tcx> {
        GoalKind::DomainGoal(self)
    }

    pub fn into_program_clause(self) -> ProgramClause<'tcx> {
        ProgramClause {
            goal: self,
            hypotheses: ty::List::empty(),
            category: ProgramClauseCategory::Other,
        }
    }
}

impl<'tcx> GoalKind<'tcx> {
    pub fn from_poly_domain_goal(
        domain_goal: PolyDomainGoal<'tcx>,
        tcx: TyCtxt<'tcx>,
    ) -> GoalKind<'tcx> {
        match domain_goal.no_bound_vars() {
            Some(p) => p.into_goal(),
            None => GoalKind::Quantified(
                QuantifierKind::Universal,
                domain_goal.map_bound(|p| tcx.mk_goal(p.into_goal())),
            ),
        }
    }
}

/// This matches the definition from Page 7 of "A Proof Procedure for the Logic of Hereditary
/// Harrop Formulas".
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable)]
pub enum Clause<'tcx> {
    Implies(ProgramClause<'tcx>),
    ForAll(ty::Binder<ProgramClause<'tcx>>),
}

impl Clause<'tcx> {
    pub fn category(self) -> ProgramClauseCategory {
        match self {
            Clause::Implies(clause) => clause.category,
            Clause::ForAll(clause) => clause.skip_binder().category,
        }
    }
}

/// Multiple clauses.
pub type Clauses<'tcx> = &'tcx List<Clause<'tcx>>;

/// A "program clause" has the form `D :- G1, ..., Gn`. It is saying
/// that the domain goal `D` is true if `G1...Gn` are provable. This
/// is equivalent to the implication `G1..Gn => D`; we usually write
/// it with the reverse implication operator `:-` to emphasize the way
/// that programs are actually solved (via backchaining, which starts
/// with the goal to solve and proceeds from there).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable)]
pub struct ProgramClause<'tcx> {
    /// This goal will be considered true ...
    pub goal: DomainGoal<'tcx>,

    /// ... if we can prove these hypotheses (there may be no hypotheses at all):
    pub hypotheses: Goals<'tcx>,

    /// Useful for filtering clauses.
    pub category: ProgramClauseCategory,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable)]
pub enum ProgramClauseCategory {
    ImpliedBound,
    WellFormed,
    Other,
}

/// A set of clauses that we assume to be true.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable)]
pub struct Environment<'tcx> {
    pub clauses: Clauses<'tcx>,
}

impl Environment<'tcx> {
    pub fn with<G>(self, goal: G) -> InEnvironment<'tcx, G> {
        InEnvironment { environment: self, goal }
    }
}

/// Something (usually a goal), along with an environment.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable)]
pub struct InEnvironment<'tcx, G> {
    pub environment: Environment<'tcx>,
    pub goal: G,
}

#[derive(Clone, Debug, TypeFoldable)]
pub enum SelectionError<'tcx> {
    Unimplemented,
    OutputTypeParameterMismatch(
        ty::PolyTraitRef<'tcx>,
        ty::PolyTraitRef<'tcx>,
        ty::error::TypeError<'tcx>,
    ),
    TraitNotObjectSafe(DefId),
    ConstEvalFailure(ErrorHandled),
    Overflow,
}

/// When performing resolution, it is typically the case that there
/// can be one of three outcomes:
///
/// - `Ok(Some(r))`: success occurred with result `r`
/// - `Ok(None)`: could not definitely determine anything, usually due
///   to inconclusive type inference.
/// - `Err(e)`: error `e` occurred
pub type SelectionResult<'tcx, T> = Result<Option<T>, SelectionError<'tcx>>;

/// Given the successful resolution of an obligation, the `Vtable`
/// indicates where the vtable comes from. Note that while we call this
/// a "vtable", it does not necessarily indicate dynamic dispatch at
/// runtime. `Vtable` instances just tell the compiler where to find
/// methods, but in generic code those methods are typically statically
/// dispatched -- only when an object is constructed is a `Vtable`
/// instance reified into an actual vtable.
///
/// For example, the vtable may be tied to a specific impl (case A),
/// or it may be relative to some bound that is in scope (case B).
///
/// ```
/// impl<T:Clone> Clone<T> for Option<T> { ... } // Impl_1
/// impl<T:Clone> Clone<T> for Box<T> { ... }    // Impl_2
/// impl Clone for int { ... }             // Impl_3
///
/// fn foo<T:Clone>(concrete: Option<Box<int>>,
///                 param: T,
///                 mixed: Option<T>) {
///
///    // Case A: Vtable points at a specific impl. Only possible when
///    // type is concretely known. If the impl itself has bounded
///    // type parameters, Vtable will carry resolutions for those as well:
///    concrete.clone(); // Vtable(Impl_1, [Vtable(Impl_2, [Vtable(Impl_3)])])
///
///    // Case B: Vtable must be provided by caller. This applies when
///    // type is a type parameter.
///    param.clone();    // VtableParam
///
///    // Case C: A mix of cases A and B.
///    mixed.clone();    // Vtable(Impl_1, [VtableParam])
/// }
/// ```
///
/// ### The type parameter `N`
///
/// See explanation on `VtableImplData`.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub enum Vtable<'tcx, N> {
    /// Vtable identifying a particular impl.
    VtableImpl(VtableImplData<'tcx, N>),

    /// Vtable for auto trait implementations.
    /// This carries the information and nested obligations with regards
    /// to an auto implementation for a trait `Trait`. The nested obligations
    /// ensure the trait implementation holds for all the constituent types.
    VtableAutoImpl(VtableAutoImplData<N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter. The `Vec<N>` represents the
    /// obligations incurred from normalizing the where-clause (if
    /// any).
    VtableParam(Vec<N>),

    /// Virtual calls through an object.
    VtableObject(VtableObjectData<'tcx, N>),

    /// Successful resolution for a builtin trait.
    VtableBuiltin(VtableBuiltinData<N>),

    /// Vtable automatically generated for a closure. The `DefId` is the ID
    /// of the closure expression. This is a `VtableImpl` in spirit, but the
    /// impl is generated by the compiler and does not appear in the source.
    VtableClosure(VtableClosureData<'tcx, N>),

    /// Same as above, but for a function pointer type with the given signature.
    VtableFnPointer(VtableFnPointerData<'tcx, N>),

    /// Vtable automatically generated for a generator.
    VtableGenerator(VtableGeneratorData<'tcx, N>),

    /// Vtable for a trait alias.
    VtableTraitAlias(VtableTraitAliasData<'tcx, N>),
}

impl<'tcx, N> Vtable<'tcx, N> {
    pub fn nested_obligations(self) -> Vec<N> {
        match self {
            VtableImpl(i) => i.nested,
            VtableParam(n) => n,
            VtableBuiltin(i) => i.nested,
            VtableAutoImpl(d) => d.nested,
            VtableClosure(c) => c.nested,
            VtableGenerator(c) => c.nested,
            VtableObject(d) => d.nested,
            VtableFnPointer(d) => d.nested,
            VtableTraitAlias(d) => d.nested,
        }
    }

    pub fn borrow_nested_obligations(&self) -> &[N] {
        match &self {
            VtableImpl(i) => &i.nested[..],
            VtableParam(n) => &n[..],
            VtableBuiltin(i) => &i.nested[..],
            VtableAutoImpl(d) => &d.nested[..],
            VtableClosure(c) => &c.nested[..],
            VtableGenerator(c) => &c.nested[..],
            VtableObject(d) => &d.nested[..],
            VtableFnPointer(d) => &d.nested[..],
            VtableTraitAlias(d) => &d.nested[..],
        }
    }

    pub fn map<M, F>(self, f: F) -> Vtable<'tcx, M>
    where
        F: FnMut(N) -> M,
    {
        match self {
            VtableImpl(i) => VtableImpl(VtableImplData {
                impl_def_id: i.impl_def_id,
                substs: i.substs,
                nested: i.nested.into_iter().map(f).collect(),
            }),
            VtableParam(n) => VtableParam(n.into_iter().map(f).collect()),
            VtableBuiltin(i) => {
                VtableBuiltin(VtableBuiltinData { nested: i.nested.into_iter().map(f).collect() })
            }
            VtableObject(o) => VtableObject(VtableObjectData {
                upcast_trait_ref: o.upcast_trait_ref,
                vtable_base: o.vtable_base,
                nested: o.nested.into_iter().map(f).collect(),
            }),
            VtableAutoImpl(d) => VtableAutoImpl(VtableAutoImplData {
                trait_def_id: d.trait_def_id,
                nested: d.nested.into_iter().map(f).collect(),
            }),
            VtableClosure(c) => VtableClosure(VtableClosureData {
                closure_def_id: c.closure_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            }),
            VtableGenerator(c) => VtableGenerator(VtableGeneratorData {
                generator_def_id: c.generator_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            }),
            VtableFnPointer(p) => VtableFnPointer(VtableFnPointerData {
                fn_ty: p.fn_ty,
                nested: p.nested.into_iter().map(f).collect(),
            }),
            VtableTraitAlias(d) => VtableTraitAlias(VtableTraitAliasData {
                alias_def_id: d.alias_def_id,
                substs: d.substs,
                nested: d.nested.into_iter().map(f).collect(),
            }),
        }
    }
}

/// Identifies a particular impl in the source, along with a set of
/// substitutions from the impl's type/lifetime parameters. The
/// `nested` vector corresponds to the nested obligations attached to
/// the impl's type parameters.
///
/// The type parameter `N` indicates the type used for "nested
/// obligations" that are required by the impl. During type-check, this
/// is `Obligation`, as one might expect. During codegen, however, this
/// is `()`, because codegen only requires a shallow resolution of an
/// impl, and nested obligations are satisfied later.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableImplData<'tcx, N> {
    pub impl_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableGeneratorData<'tcx, N> {
    pub generator_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    /// Nested obligations. This can be non-empty if the generator
    /// signature contains associated types.
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableClosureData<'tcx, N> {
    pub closure_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    /// Nested obligations. This can be non-empty if the closure
    /// signature contains associated types.
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableAutoImplData<N> {
    pub trait_def_id: DefId,
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableBuiltinData<N> {
    pub nested: Vec<N>,
}

/// A vtable for some object-safe trait `Foo` automatically derived
/// for the object type `Foo`.
#[derive(PartialEq, Eq, Clone, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableObjectData<'tcx, N> {
    /// `Foo` upcast to the obligation trait. This will be some supertrait of `Foo`.
    pub upcast_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The vtable is formed by concatenating together the method lists of
    /// the base object trait and all supertraits; this is the start of
    /// `upcast_trait_ref`'s methods in that vtable.
    pub vtable_base: usize,

    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableFnPointerData<'tcx, N> {
    pub fn_ty: Ty<'tcx>,
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable, TypeFoldable)]
pub struct VtableTraitAliasData<'tcx, N> {
    pub alias_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    pub nested: Vec<N>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, HashStable)]
pub enum ObjectSafetyViolation {
    /// `Self: Sized` declared on the trait.
    SizedSelf(SmallVec<[Span; 1]>),

    /// Supertrait reference references `Self` an in illegal location
    /// (e.g., `trait Foo : Bar<Self>`).
    SupertraitSelf(SmallVec<[Span; 1]>),

    /// Method has something illegal.
    Method(ast::Name, MethodViolationCode, Span),

    /// Associated const.
    AssocConst(ast::Name, Span),
}

impl ObjectSafetyViolation {
    pub fn error_msg(&self) -> Cow<'static, str> {
        match *self {
            ObjectSafetyViolation::SizedSelf(_) => "it requires `Self: Sized`".into(),
            ObjectSafetyViolation::SupertraitSelf(ref spans) => {
                if spans.iter().any(|sp| *sp != DUMMY_SP) {
                    "it uses `Self` as a type parameter in this".into()
                } else {
                    "it cannot use `Self` as a type parameter in a supertrait or `where`-clause"
                        .into()
                }
            }
            ObjectSafetyViolation::Method(name, MethodViolationCode::StaticMethod(_), _) => {
                format!("associated function `{}` has no `self` parameter", name).into()
            }
            ObjectSafetyViolation::Method(
                name,
                MethodViolationCode::ReferencesSelfInput(_),
                DUMMY_SP,
            ) => format!("method `{}` references the `Self` type in its parameters", name).into(),
            ObjectSafetyViolation::Method(name, MethodViolationCode::ReferencesSelfInput(_), _) => {
                format!("method `{}` references the `Self` type in this parameter", name).into()
            }
            ObjectSafetyViolation::Method(name, MethodViolationCode::ReferencesSelfOutput, _) => {
                format!("method `{}` references the `Self` type in its return type", name).into()
            }
            ObjectSafetyViolation::Method(
                name,
                MethodViolationCode::WhereClauseReferencesSelf,
                _,
            ) => {
                format!("method `{}` references the `Self` type in its `where` clause", name).into()
            }
            ObjectSafetyViolation::Method(name, MethodViolationCode::Generic, _) => {
                format!("method `{}` has generic type parameters", name).into()
            }
            ObjectSafetyViolation::Method(name, MethodViolationCode::UndispatchableReceiver, _) => {
                format!("method `{}`'s `self` parameter cannot be dispatched on", name).into()
            }
            ObjectSafetyViolation::AssocConst(name, DUMMY_SP) => {
                format!("it contains associated `const` `{}`", name).into()
            }
            ObjectSafetyViolation::AssocConst(..) => "it contains this associated `const`".into(),
        }
    }

    pub fn solution(&self) -> Option<(String, Option<(String, Span)>)> {
        Some(match *self {
            ObjectSafetyViolation::SizedSelf(_) | ObjectSafetyViolation::SupertraitSelf(_) => {
                return None;
            }
            ObjectSafetyViolation::Method(name, MethodViolationCode::StaticMethod(sugg), _) => (
                format!(
                    "consider turning `{}` into a method by giving it a `&self` argument or \
                     constraining it so it does not apply to trait objects",
                    name
                ),
                sugg.map(|(sugg, sp)| (sugg.to_string(), sp)),
            ),
            ObjectSafetyViolation::Method(
                name,
                MethodViolationCode::UndispatchableReceiver,
                span,
            ) => (
                format!("consider changing method `{}`'s `self` parameter to be `&self`", name),
                Some(("&Self".to_string(), span)),
            ),
            ObjectSafetyViolation::AssocConst(name, _)
            | ObjectSafetyViolation::Method(name, ..) => {
                (format!("consider moving `{}` to another trait", name), None)
            }
        })
    }

    pub fn spans(&self) -> SmallVec<[Span; 1]> {
        // When `span` comes from a separate crate, it'll be `DUMMY_SP`. Treat it as `None` so
        // diagnostics use a `note` instead of a `span_label`.
        match self {
            ObjectSafetyViolation::SupertraitSelf(spans)
            | ObjectSafetyViolation::SizedSelf(spans) => spans.clone(),
            ObjectSafetyViolation::AssocConst(_, span)
            | ObjectSafetyViolation::Method(_, _, span)
                if *span != DUMMY_SP =>
            {
                smallvec![*span]
            }
            _ => smallvec![],
        }
    }
}

/// Reasons a method might not be object-safe.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
pub enum MethodViolationCode {
    /// e.g., `fn foo()`
    StaticMethod(Option<(&'static str, Span)>),

    /// e.g., `fn foo(&self, x: Self)`
    ReferencesSelfInput(usize),

    /// e.g., `fn foo(&self) -> Self`
    ReferencesSelfOutput,

    /// e.g., `fn foo(&self) where Self: Clone`
    WhereClauseReferencesSelf,

    /// e.g., `fn foo<A>()`
    Generic,

    /// the method's receiver (`self` argument) can't be dispatched on
    UndispatchableReceiver,
}
