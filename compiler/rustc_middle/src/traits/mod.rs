//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

mod chalk;
pub mod query;
pub mod select;
pub mod solve;
pub mod specialization_graph;
mod structural_impls;
pub mod util;

use crate::infer::canonical::Canonical;
use crate::mir::ConstraintCategory;
use crate::ty::abstract_const::NotConstEvaluatable;
use crate::ty::subst::SubstsRef;
use crate::ty::{self, AdtKind, Ty, TyCtxt};

use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_span::def_id::{LocalDefId, CRATE_DEF_ID};
use rustc_span::symbol::Symbol;
use rustc_span::{Span, DUMMY_SP};
use smallvec::SmallVec;

use std::borrow::Cow;
use std::hash::{Hash, Hasher};

pub use self::select::{EvaluationCache, EvaluationResult, OverflowError, SelectionCache};

pub type CanonicalChalkEnvironmentAndGoal<'tcx> = Canonical<'tcx, ChalkEnvironmentAndGoal<'tcx>>;

pub use self::ObligationCauseCode::*;

pub use self::chalk::{ChalkEnvironmentAndGoal, RustInterner as ChalkRustInterner};

/// Depending on the stage of compilation, we want projection to be
/// more or less conservative.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, HashStable, Encodable, Decodable)]
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
    /// ```compile_fail,E0308
    /// #![feature(specialization)]
    /// trait Assoc {
    ///     type Output;
    /// }
    ///
    /// impl<T> Assoc for T {
    ///     default type Output = bool;
    /// }
    ///
    /// fn main() {
    ///     let x: <() as Assoc>::Output = true;
    /// }
    /// ```
    ///
    /// We also do not reveal the hidden type of opaque types during
    /// type-checking.
    UserFacing,

    /// At codegen time, all monomorphic projections will succeed.
    /// Also, `impl Trait` is normalized to the concrete type,
    /// which has to be already collected by type-checking.
    ///
    /// NOTE: as `impl Trait`'s concrete type should *never*
    /// be observable directly by the user, `Reveal::All`
    /// should not be used by checks which may expose
    /// type equality or type contents to the user.
    /// There are some exceptions, e.g., around auto traits and
    /// transmute-checking, which expose some details, but
    /// not the whole concrete type of the `impl Trait`.
    All,
}

/// The reason why we incurred this obligation; used for error reporting.
///
/// Non-misc `ObligationCauseCode`s are stored on the heap. This gives the
/// best trade-off between keeping the type small (which makes copies cheaper)
/// while not doing too many heap allocations.
///
/// We do not want to intern this as there are a lot of obligation causes which
/// only live for a short period of time.
#[derive(Clone, Debug, PartialEq, Eq, Lift, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct ObligationCause<'tcx> {
    pub span: Span,

    /// The ID of the fn body that triggered this obligation. This is
    /// used for region obligations to determine the precise
    /// environment in which the region obligation should be evaluated
    /// (in particular, closures can add new assumptions). See the
    /// field `region_obligations` of the `FulfillmentContext` for more
    /// information.
    pub body_id: LocalDefId,

    code: InternedObligationCauseCode<'tcx>,
}

// This custom hash function speeds up hashing for `Obligation` deduplication
// greatly by skipping the `code` field, which can be large and complex. That
// shouldn't affect hash quality much since there are several other fields in
// `Obligation` which should be unique enough, especially the predicate itself
// which is hashed as an interned pointer. See #90996.
impl Hash for ObligationCause<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.body_id.hash(state);
        self.span.hash(state);
    }
}

impl<'tcx> ObligationCause<'tcx> {
    #[inline]
    pub fn new(
        span: Span,
        body_id: LocalDefId,
        code: ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        ObligationCause { span, body_id, code: code.into() }
    }

    pub fn misc(span: Span, body_id: LocalDefId) -> ObligationCause<'tcx> {
        ObligationCause::new(span, body_id, MiscObligation)
    }

    #[inline(always)]
    pub fn dummy() -> ObligationCause<'tcx> {
        ObligationCause::dummy_with_span(DUMMY_SP)
    }

    #[inline(always)]
    pub fn dummy_with_span(span: Span) -> ObligationCause<'tcx> {
        ObligationCause { span, body_id: CRATE_DEF_ID, code: Default::default() }
    }

    pub fn span(&self) -> Span {
        match *self.code() {
            ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                arm_span,
                ..
            }) => arm_span,
            _ => self.span,
        }
    }

    #[inline]
    pub fn code(&self) -> &ObligationCauseCode<'tcx> {
        &self.code
    }

    pub fn map_code(
        &mut self,
        f: impl FnOnce(InternedObligationCauseCode<'tcx>) -> ObligationCauseCode<'tcx>,
    ) {
        self.code = f(std::mem::take(&mut self.code)).into();
    }

    pub fn derived_cause(
        mut self,
        parent_trait_pred: ty::PolyTraitPredicate<'tcx>,
        variant: impl FnOnce(DerivedObligationCause<'tcx>) -> ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        /*!
         * Creates a cause for obligations that are derived from
         * `obligation` by a recursive search (e.g., for a builtin
         * bound, or eventually a `auto trait Foo`). If `obligation`
         * is itself a derived obligation, this is just a clone, but
         * otherwise we create a "derived obligation" cause so as to
         * keep track of the original root obligation for error
         * reporting.
         */

        // NOTE(flaper87): As of now, it keeps track of the whole error
        // chain. Ideally, we should have a way to configure this either
        // by using -Z verbose or just a CLI argument.
        self.code =
            variant(DerivedObligationCause { parent_trait_pred, parent_code: self.code }).into();
        self
    }

    pub fn to_constraint_category(&self) -> ConstraintCategory<'tcx> {
        match self.code() {
            MatchImpl(cause, _) => cause.to_constraint_category(),
            AscribeUserTypeProvePredicate(predicate_span) => {
                ConstraintCategory::Predicate(*predicate_span)
            }
            _ => ConstraintCategory::BoringNoLocation,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Lift, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct UnifyReceiverContext<'tcx> {
    pub assoc_item: ty::AssocItem,
    pub param_env: ty::ParamEnv<'tcx>,
    pub substs: SubstsRef<'tcx>,
}

#[derive(Clone, PartialEq, Eq, Hash, Lift, Default, HashStable)]
#[derive(TypeVisitable, TypeFoldable, TyEncodable, TyDecodable)]
pub struct InternedObligationCauseCode<'tcx> {
    /// `None` for `ObligationCauseCode::MiscObligation` (a common case, occurs ~60% of
    /// the time). `Some` otherwise.
    code: Option<Lrc<ObligationCauseCode<'tcx>>>,
}

impl<'tcx> std::fmt::Debug for InternedObligationCauseCode<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cause: &ObligationCauseCode<'_> = self;
        cause.fmt(f)
    }
}

impl<'tcx> ObligationCauseCode<'tcx> {
    #[inline(always)]
    fn into(self) -> InternedObligationCauseCode<'tcx> {
        InternedObligationCauseCode {
            code: if let ObligationCauseCode::MiscObligation = self {
                None
            } else {
                Some(Lrc::new(self))
            },
        }
    }
}

impl<'tcx> std::ops::Deref for InternedObligationCauseCode<'tcx> {
    type Target = ObligationCauseCode<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.code.as_deref().unwrap_or(&ObligationCauseCode::MiscObligation)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Lift, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub enum ObligationCauseCode<'tcx> {
    /// Not well classified or should be obvious from the span.
    MiscObligation,

    /// A slice or array is WF only if `T: Sized`.
    SliceOrArrayElem,

    /// A tuple is WF only if its middle elements are `Sized`.
    TupleElem,

    /// This is the trait reference from the given projection.
    ProjectionWf(ty::AliasTy<'tcx>),

    /// Must satisfy all of the where-clause predicates of the
    /// given item.
    ItemObligation(DefId),

    /// Like `ItemObligation`, but carries the span of the
    /// predicate when it can be identified.
    BindingObligation(DefId, Span),

    /// Like `ItemObligation`, but carries the `HirId` of the
    /// expression that caused the obligation, and the `usize`
    /// indicates exactly which predicate it is in the list of
    /// instantiated predicates.
    ExprItemObligation(DefId, rustc_hir::HirId, usize),

    /// Combines `ExprItemObligation` and `BindingObligation`.
    ExprBindingObligation(DefId, Span, rustc_hir::HirId, usize),

    /// A type like `&'a T` is WF only if `T: 'a`.
    ReferenceOutlivesReferent(Ty<'tcx>),

    /// A type like `Box<Foo<'a> + 'b>` is WF only if `'b: 'a`.
    ObjectTypeBound(Ty<'tcx>, ty::Region<'tcx>),

    /// Obligation incurred due to an object cast.
    ObjectCastObligation(/* Concrete type */ Ty<'tcx>, /* Object type */ Ty<'tcx>),

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
    SizedArgumentType(Option<Span>),
    /// Return type must be `Sized`.
    SizedReturnType,
    /// Yield type must be `Sized`.
    SizedYieldType,
    /// Box expression result type must be `Sized`.
    SizedBoxType,
    /// Inline asm operand type must be `Sized`.
    InlineAsmSized,
    /// `[expr; N]` requires `type_of(expr): Copy`.
    RepeatElementCopy {
        /// If element is a `const fn` we display a help message suggesting to move the
        /// function call to a new `const` item while saying that `T` doesn't implement `Copy`.
        is_const_fn: bool,
    },

    /// Types of fields (other than the last, except for packed structs) in a struct must be sized.
    FieldSized {
        adt_kind: AdtKind,
        span: Span,
        last: bool,
    },

    /// Constant expressions must be sized.
    ConstSized,

    /// `static` items must have `Sync` type.
    SharedStatic,

    BuiltinDerivedObligation(DerivedObligationCause<'tcx>),

    ImplDerivedObligation(Box<ImplDerivedObligationCause<'tcx>>),

    DerivedObligation(DerivedObligationCause<'tcx>),

    FunctionArgumentObligation {
        /// The node of the relevant argument in the function call.
        arg_hir_id: hir::HirId,
        /// The node of the function call.
        call_hir_id: hir::HirId,
        /// The obligation introduced by this argument.
        parent_code: InternedObligationCauseCode<'tcx>,
    },

    /// Error derived when matching traits/impls; see ObligationCause for more details
    CompareImplItemObligation {
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
        kind: ty::AssocKind,
    },

    /// Checking that the bounds of a trait's associated type hold for a given impl
    CheckAssociatedTypeBounds {
        impl_item_def_id: LocalDefId,
        trait_item_def_id: DefId,
    },

    /// Checking that this expression can be assigned to its target.
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
    IfExpression(Box<IfExpressionCause<'tcx>>),

    /// Computing common supertype of an if expression with no else counter-part
    IfExpressionWithNoElse,

    /// `main` has wrong type
    MainFunctionType,

    /// `start` has wrong type
    StartFunctionType,

    /// Intrinsic has wrong type
    IntrinsicType,

    /// A let else block does not diverge
    LetElse,

    /// Method receiver
    MethodReceiver,

    UnifyReceiver(Box<UnifyReceiverContext<'tcx>>),

    /// `return` with no expression
    ReturnNoExpression,

    /// `return` with an expression
    ReturnValue(hir::HirId),

    /// Return type of this function
    ReturnType,

    /// Opaque return type of this function
    OpaqueReturnType(Option<(Ty<'tcx>, Span)>),

    /// Block implicit return
    BlockTailExpression(hir::HirId),

    /// #[feature(trivial_bounds)] is not enabled
    TrivialBound,

    /// If `X` is the concrete type of an opaque type `impl Y`, then `X` must implement `Y`
    OpaqueType,

    AwaitableExpr(Option<hir::HirId>),

    ForLoopIterator,

    QuestionMark,

    /// Well-formed checking. If a `WellFormedLoc` is provided,
    /// then it will be used to perform HIR-based wf checking
    /// after an error occurs, in order to generate a more precise error span.
    /// This is purely for diagnostic purposes - it is always
    /// correct to use `MiscObligation` instead, or to specify
    /// `WellFormed(None)`
    WellFormed(Option<WellFormedLoc>),

    /// From `match_impl`. The cause for us having to match an impl, and the DefId we are matching against.
    MatchImpl(ObligationCause<'tcx>, DefId),

    BinOp {
        rhs_span: Option<Span>,
        is_lit: bool,
        output_ty: Option<Ty<'tcx>>,
    },

    AscribeUserTypeProvePredicate(Span),

    RustCall,
}

/// The 'location' at which we try to perform HIR-based wf checking.
/// This information is used to obtain an `hir::Ty`, which
/// we can walk in order to obtain precise spans for any
/// 'nested' types (e.g. `Foo` in `Option<Foo>`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, Encodable, Decodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub enum WellFormedLoc {
    /// Use the type of the provided definition.
    Ty(LocalDefId),
    /// Use the type of the parameter of the provided function.
    /// We cannot use `hir::Param`, since the function may
    /// not have a body (e.g. a trait method definition)
    Param {
        /// The function to lookup the parameter in
        function: LocalDefId,
        /// The index of the parameter to use.
        /// Parameters are indexed from 0, with the return type
        /// being the last 'parameter'
        param_idx: u16,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Lift, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct ImplDerivedObligationCause<'tcx> {
    pub derived: DerivedObligationCause<'tcx>,
    pub impl_def_id: DefId,
    /// The index of the derived predicate in the parent impl's predicates.
    pub impl_def_predicate_index: Option<usize>,
    pub span: Span,
}

impl<'tcx> ObligationCauseCode<'tcx> {
    /// Returns the base obligation, ignoring derived obligations.
    pub fn peel_derives(&self) -> &Self {
        let mut base_cause = self;
        while let Some((parent_code, _)) = base_cause.parent() {
            base_cause = parent_code;
        }
        base_cause
    }

    pub fn parent(&self) -> Option<(&Self, Option<ty::PolyTraitPredicate<'tcx>>)> {
        match self {
            FunctionArgumentObligation { parent_code, .. } => Some((parent_code, None)),
            BuiltinDerivedObligation(derived)
            | DerivedObligation(derived)
            | ImplDerivedObligation(box ImplDerivedObligationCause { derived, .. }) => {
                Some((&derived.parent_code, Some(derived.parent_trait_pred)))
            }
            _ => None,
        }
    }

    pub fn peel_match_impls(&self) -> &Self {
        match self {
            MatchImpl(cause, _) => cause.code(),
            _ => self,
        }
    }
}

// `ObligationCauseCode` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ObligationCauseCode<'_>, 48);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StatementAsExpression {
    CorrectType,
    NeedsBoxing,
}

impl<'tcx> ty::Lift<'tcx> for StatementAsExpression {
    type Lifted = StatementAsExpression;
    fn lift_to_tcx(self, _tcx: TyCtxt<'tcx>) -> Option<StatementAsExpression> {
        Some(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Lift, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct MatchExpressionArmCause<'tcx> {
    pub arm_block_id: Option<hir::HirId>,
    pub arm_ty: Ty<'tcx>,
    pub arm_span: Span,
    pub prior_arm_block_id: Option<hir::HirId>,
    pub prior_arm_ty: Ty<'tcx>,
    pub prior_arm_span: Span,
    pub scrut_span: Span,
    pub source: hir::MatchSource,
    pub prior_arms: Vec<Span>,
    pub scrut_hir_id: hir::HirId,
    pub opt_suggest_box_span: Option<Span>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(Lift, TypeFoldable, TypeVisitable, HashStable, TyEncodable, TyDecodable)]
pub struct IfExpressionCause<'tcx> {
    pub then_id: hir::HirId,
    pub else_id: hir::HirId,
    pub then_ty: Ty<'tcx>,
    pub else_ty: Ty<'tcx>,
    pub outer_span: Option<Span>,
    pub opt_suggest_box_span: Option<Span>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Lift, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct DerivedObligationCause<'tcx> {
    /// The trait predicate of the parent obligation that led to the
    /// current obligation. Note that only trait obligations lead to
    /// derived obligations, so we just store the trait predicate here
    /// directly.
    pub parent_trait_pred: ty::PolyTraitPredicate<'tcx>,

    /// The parent trait had this cause.
    pub parent_code: InternedObligationCauseCode<'tcx>,
}

#[derive(Clone, Debug, TypeFoldable, TypeVisitable, Lift)]
pub enum SelectionError<'tcx> {
    /// The trait is not implemented.
    Unimplemented,
    /// After a closure impl has selected, its "outputs" were evaluated
    /// (which for closures includes the "input" type params) and they
    /// didn't resolve. See `confirm_poly_trait_refs` for more.
    OutputTypeParameterMismatch(
        ty::PolyTraitRef<'tcx>,
        ty::PolyTraitRef<'tcx>,
        ty::error::TypeError<'tcx>,
    ),
    /// The trait pointed by `DefId` is not object safe.
    TraitNotObjectSafe(DefId),
    /// A given constant couldn't be evaluated.
    NotConstEvaluatable(NotConstEvaluatable),
    /// Exceeded the recursion depth during type projection.
    Overflow(OverflowError),
    /// Signaling that an error has already been emitted, to avoid
    /// multiple errors being shown.
    ErrorReporting,
}

/// When performing resolution, it is typically the case that there
/// can be one of three outcomes:
///
/// - `Ok(Some(r))`: success occurred with result `r`
/// - `Ok(None)`: could not definitely determine anything, usually due
///   to inconclusive type inference.
/// - `Err(e)`: error `e` occurred
pub type SelectionResult<'tcx, T> = Result<Option<T>, SelectionError<'tcx>>;

/// Given the successful resolution of an obligation, the `ImplSource`
/// indicates where the impl comes from.
///
/// For example, the obligation may be satisfied by a specific impl (case A),
/// or it may be relative to some bound that is in scope (case B).
///
/// ```ignore (illustrative)
/// impl<T:Clone> Clone<T> for Option<T> { ... } // Impl_1
/// impl<T:Clone> Clone<T> for Box<T> { ... }    // Impl_2
/// impl Clone for i32 { ... }                   // Impl_3
///
/// fn foo<T: Clone>(concrete: Option<Box<i32>>, param: T, mixed: Option<T>) {
///     // Case A: ImplSource points at a specific impl. Only possible when
///     // type is concretely known. If the impl itself has bounded
///     // type parameters, ImplSource will carry resolutions for those as well:
///     concrete.clone(); // ImplSource(Impl_1, [ImplSource(Impl_2, [ImplSource(Impl_3)])])
///
///     // Case B: ImplSource must be provided by caller. This applies when
///     // type is a type parameter.
///     param.clone();    // ImplSource::Param
///
///     // Case C: A mix of cases A and B.
///     mixed.clone();    // ImplSource(Impl_1, [ImplSource::Param])
/// }
/// ```
///
/// ### The type parameter `N`
///
/// See explanation on `ImplSourceUserDefinedData`.
#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum ImplSource<'tcx, N> {
    /// ImplSource identifying a particular impl.
    UserDefined(ImplSourceUserDefinedData<'tcx, N>),

    /// ImplSource for auto trait implementations.
    /// This carries the information and nested obligations with regards
    /// to an auto implementation for a trait `Trait`. The nested obligations
    /// ensure the trait implementation holds for all the constituent types.
    AutoImpl(ImplSourceAutoImplData<N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter. The `Vec<N>` represents the
    /// obligations incurred from normalizing the where-clause (if
    /// any).
    Param(Vec<N>, ty::BoundConstness),

    /// Virtual calls through an object.
    Object(ImplSourceObjectData<'tcx, N>),

    /// Successful resolution for a builtin trait.
    Builtin(ImplSourceBuiltinData<N>),

    /// ImplSource for trait upcasting coercion
    TraitUpcasting(ImplSourceTraitUpcastingData<'tcx, N>),

    /// ImplSource automatically generated for a closure. The `DefId` is the ID
    /// of the closure expression. This is an `ImplSource::UserDefined` in spirit, but the
    /// impl is generated by the compiler and does not appear in the source.
    Closure(ImplSourceClosureData<'tcx, N>),

    /// Same as above, but for a function pointer type with the given signature.
    FnPointer(ImplSourceFnPointerData<'tcx, N>),

    /// ImplSource automatically generated for a generator.
    Generator(ImplSourceGeneratorData<'tcx, N>),

    /// ImplSource automatically generated for a generator backing an async future.
    Future(ImplSourceFutureData<'tcx, N>),

    /// ImplSource for a trait alias.
    TraitAlias(ImplSourceTraitAliasData<'tcx, N>),

    /// ImplSource for a `const Drop` implementation.
    ConstDestruct(ImplSourceConstDestructData<N>),
}

impl<'tcx, N> ImplSource<'tcx, N> {
    pub fn nested_obligations(self) -> Vec<N> {
        match self {
            ImplSource::UserDefined(i) => i.nested,
            ImplSource::Param(n, _) => n,
            ImplSource::Builtin(i) => i.nested,
            ImplSource::AutoImpl(d) => d.nested,
            ImplSource::Closure(c) => c.nested,
            ImplSource::Generator(c) => c.nested,
            ImplSource::Future(c) => c.nested,
            ImplSource::Object(d) => d.nested,
            ImplSource::FnPointer(d) => d.nested,
            ImplSource::TraitAlias(d) => d.nested,
            ImplSource::TraitUpcasting(d) => d.nested,
            ImplSource::ConstDestruct(i) => i.nested,
        }
    }

    pub fn borrow_nested_obligations(&self) -> &[N] {
        match &self {
            ImplSource::UserDefined(i) => &i.nested[..],
            ImplSource::Param(n, _) => &n,
            ImplSource::Builtin(i) => &i.nested,
            ImplSource::AutoImpl(d) => &d.nested,
            ImplSource::Closure(c) => &c.nested,
            ImplSource::Generator(c) => &c.nested,
            ImplSource::Future(c) => &c.nested,
            ImplSource::Object(d) => &d.nested,
            ImplSource::FnPointer(d) => &d.nested,
            ImplSource::TraitAlias(d) => &d.nested,
            ImplSource::TraitUpcasting(d) => &d.nested,
            ImplSource::ConstDestruct(i) => &i.nested,
        }
    }

    pub fn map<M, F>(self, f: F) -> ImplSource<'tcx, M>
    where
        F: FnMut(N) -> M,
    {
        match self {
            ImplSource::UserDefined(i) => ImplSource::UserDefined(ImplSourceUserDefinedData {
                impl_def_id: i.impl_def_id,
                substs: i.substs,
                nested: i.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Param(n, ct) => ImplSource::Param(n.into_iter().map(f).collect(), ct),
            ImplSource::Builtin(i) => ImplSource::Builtin(ImplSourceBuiltinData {
                nested: i.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Object(o) => ImplSource::Object(ImplSourceObjectData {
                upcast_trait_ref: o.upcast_trait_ref,
                vtable_base: o.vtable_base,
                nested: o.nested.into_iter().map(f).collect(),
            }),
            ImplSource::AutoImpl(d) => ImplSource::AutoImpl(ImplSourceAutoImplData {
                trait_def_id: d.trait_def_id,
                nested: d.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Closure(c) => ImplSource::Closure(ImplSourceClosureData {
                closure_def_id: c.closure_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Generator(c) => ImplSource::Generator(ImplSourceGeneratorData {
                generator_def_id: c.generator_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Future(c) => ImplSource::Future(ImplSourceFutureData {
                generator_def_id: c.generator_def_id,
                substs: c.substs,
                nested: c.nested.into_iter().map(f).collect(),
            }),
            ImplSource::FnPointer(p) => ImplSource::FnPointer(ImplSourceFnPointerData {
                fn_ty: p.fn_ty,
                nested: p.nested.into_iter().map(f).collect(),
            }),
            ImplSource::TraitAlias(d) => ImplSource::TraitAlias(ImplSourceTraitAliasData {
                alias_def_id: d.alias_def_id,
                substs: d.substs,
                nested: d.nested.into_iter().map(f).collect(),
            }),
            ImplSource::TraitUpcasting(d) => {
                ImplSource::TraitUpcasting(ImplSourceTraitUpcastingData {
                    upcast_trait_ref: d.upcast_trait_ref,
                    vtable_vptr_slot: d.vtable_vptr_slot,
                    nested: d.nested.into_iter().map(f).collect(),
                })
            }
            ImplSource::ConstDestruct(i) => {
                ImplSource::ConstDestruct(ImplSourceConstDestructData {
                    nested: i.nested.into_iter().map(f).collect(),
                })
            }
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
#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceUserDefinedData<'tcx, N> {
    pub impl_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceGeneratorData<'tcx, N> {
    pub generator_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    /// Nested obligations. This can be non-empty if the generator
    /// signature contains associated types.
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceFutureData<'tcx, N> {
    pub generator_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    /// Nested obligations. This can be non-empty if the generator
    /// signature contains associated types.
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceClosureData<'tcx, N> {
    pub closure_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    /// Nested obligations. This can be non-empty if the closure
    /// signature contains associated types.
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceAutoImplData<N> {
    pub trait_def_id: DefId,
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceTraitUpcastingData<'tcx, N> {
    /// `Foo` upcast to the obligation trait. This will be some supertrait of `Foo`.
    pub upcast_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The vtable is formed by concatenating together the method lists of
    /// the base object trait and all supertraits, pointers to supertrait vtable will
    /// be provided when necessary; this is the position of `upcast_trait_ref`'s vtable
    /// within that vtable.
    pub vtable_vptr_slot: Option<usize>,

    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceBuiltinData<N> {
    pub nested: Vec<N>,
}

#[derive(PartialEq, Eq, Clone, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceObjectData<'tcx, N> {
    /// `Foo` upcast to the obligation trait. This will be some supertrait of `Foo`.
    pub upcast_trait_ref: ty::PolyTraitRef<'tcx>,

    /// The vtable is formed by concatenating together the method lists of
    /// the base object trait and all supertraits, pointers to supertrait vtable will
    /// be provided when necessary; this is the start of `upcast_trait_ref`'s methods
    /// in that vtable.
    pub vtable_base: usize,

    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceFnPointerData<'tcx, N> {
    pub fn_ty: Ty<'tcx>,
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceConstDestructData<N> {
    pub nested: Vec<N>,
}

#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceTraitAliasData<'tcx, N> {
    pub alias_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    pub nested: Vec<N>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, HashStable, PartialOrd, Ord)]
pub enum ObjectSafetyViolation {
    /// `Self: Sized` declared on the trait.
    SizedSelf(SmallVec<[Span; 1]>),

    /// Supertrait reference references `Self` an in illegal location
    /// (e.g., `trait Foo : Bar<Self>`).
    SupertraitSelf(SmallVec<[Span; 1]>),

    /// Method has something illegal.
    Method(Symbol, MethodViolationCode, Span),

    /// Associated const.
    AssocConst(Symbol, Span),

    /// GAT
    GAT(Symbol, Span),
}

impl ObjectSafetyViolation {
    pub fn error_msg(&self) -> Cow<'static, str> {
        match self {
            ObjectSafetyViolation::SizedSelf(_) => "it requires `Self: Sized`".into(),
            ObjectSafetyViolation::SupertraitSelf(ref spans) => {
                if spans.iter().any(|sp| *sp != DUMMY_SP) {
                    "it uses `Self` as a type parameter".into()
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
                MethodViolationCode::ReferencesImplTraitInTrait(_),
                _,
            ) => format!("method `{}` references an `impl Trait` type in its return type", name)
                .into(),
            ObjectSafetyViolation::Method(name, MethodViolationCode::AsyncFn, _) => {
                format!("method `{}` is `async`", name).into()
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
            ObjectSafetyViolation::Method(
                name,
                MethodViolationCode::UndispatchableReceiver(_),
                _,
            ) => format!("method `{}`'s `self` parameter cannot be dispatched on", name).into(),
            ObjectSafetyViolation::AssocConst(name, DUMMY_SP) => {
                format!("it contains associated `const` `{}`", name).into()
            }
            ObjectSafetyViolation::AssocConst(..) => "it contains this associated `const`".into(),
            ObjectSafetyViolation::GAT(name, _) => {
                format!("it contains the generic associated type `{}`", name).into()
            }
        }
    }

    pub fn solution(&self, err: &mut Diagnostic) {
        match self {
            ObjectSafetyViolation::SizedSelf(_) | ObjectSafetyViolation::SupertraitSelf(_) => {}
            ObjectSafetyViolation::Method(
                name,
                MethodViolationCode::StaticMethod(Some((add_self_sugg, make_sized_sugg))),
                _,
            ) => {
                err.span_suggestion(
                    add_self_sugg.1,
                    format!(
                        "consider turning `{}` into a method by giving it a `&self` argument",
                        name
                    ),
                    add_self_sugg.0.to_string(),
                    Applicability::MaybeIncorrect,
                );
                err.span_suggestion(
                    make_sized_sugg.1,
                    format!(
                        "alternatively, consider constraining `{}` so it does not apply to \
                             trait objects",
                        name
                    ),
                    make_sized_sugg.0.to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            ObjectSafetyViolation::Method(
                name,
                MethodViolationCode::UndispatchableReceiver(Some(span)),
                _,
            ) => {
                err.span_suggestion(
                    *span,
                    &format!(
                        "consider changing method `{}`'s `self` parameter to be `&self`",
                        name
                    ),
                    "&Self",
                    Applicability::MachineApplicable,
                );
            }
            ObjectSafetyViolation::AssocConst(name, _)
            | ObjectSafetyViolation::GAT(name, _)
            | ObjectSafetyViolation::Method(name, ..) => {
                err.help(&format!("consider moving `{}` to another trait", name));
            }
        }
    }

    pub fn spans(&self) -> SmallVec<[Span; 1]> {
        // When `span` comes from a separate crate, it'll be `DUMMY_SP`. Treat it as `None` so
        // diagnostics use a `note` instead of a `span_label`.
        match self {
            ObjectSafetyViolation::SupertraitSelf(spans)
            | ObjectSafetyViolation::SizedSelf(spans) => spans.clone(),
            ObjectSafetyViolation::AssocConst(_, span)
            | ObjectSafetyViolation::GAT(_, span)
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, HashStable, PartialOrd, Ord)]
pub enum MethodViolationCode {
    /// e.g., `fn foo()`
    StaticMethod(Option<(/* add &self */ (String, Span), /* add Self: Sized */ (String, Span))>),

    /// e.g., `fn foo(&self, x: Self)`
    ReferencesSelfInput(Option<Span>),

    /// e.g., `fn foo(&self) -> Self`
    ReferencesSelfOutput,

    /// e.g., `fn foo(&self) -> impl Sized`
    ReferencesImplTraitInTrait(Span),

    /// e.g., `async fn foo(&self)`
    AsyncFn,

    /// e.g., `fn foo(&self) where Self: Clone`
    WhereClauseReferencesSelf,

    /// e.g., `fn foo<A>()`
    Generic,

    /// the method's receiver (`self` argument) can't be dispatched on
    UndispatchableReceiver(Option<Span>),
}

/// These are the error cases for `codegen_select_candidate`.
#[derive(Copy, Clone, Debug, Hash, HashStable, Encodable, Decodable)]
pub enum CodegenObligationError {
    /// Ambiguity can happen when monomorphizing during trans
    /// expands to some humongous type that never occurred
    /// statically -- this humongous type can then overflow,
    /// leading to an ambiguous result. So report this as an
    /// overflow bug, since I believe this is the only case
    /// where ambiguity can result.
    Ambiguity,
    /// This can trigger when we probe for the source of a `'static` lifetime requirement
    /// on a trait object: `impl Foo for dyn Trait {}` has an implicit `'static` bound.
    /// This can also trigger when we have a global bound that is not actually satisfied,
    /// but was included during typeck due to the trivial_bounds feature.
    Unimplemented,
    FulfillmentError,
}
