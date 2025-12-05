//! Trait Resolution. See the [rustc dev guide] for more information on how this works.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

pub mod query;
pub mod select;
pub mod solve;
pub mod specialization_graph;
mod structural_impls;

use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rustc_errors::{Applicability, Diag, EmissionGuarantee, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::def_id::DefId;
use rustc_macros::{
    Decodable, Encodable, HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable,
};
use rustc_span::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_span::{DUMMY_SP, Span, Symbol};
use smallvec::{SmallVec, smallvec};
use thin_vec::ThinVec;

pub use self::select::{EvaluationCache, EvaluationResult, OverflowError, SelectionCache};
use crate::mir::ConstraintCategory;
pub use crate::traits::solve::BuiltinImplSource;
use crate::ty::abstract_const::NotConstEvaluatable;
use crate::ty::{self, AdtKind, GenericArgsRef, Ty};

/// The reason why we incurred this obligation; used for error reporting.
///
/// Non-misc `ObligationCauseCode`s are stored on the heap. This gives the
/// best trade-off between keeping the type small (which makes copies cheaper)
/// while not doing too many heap allocations.
///
/// We do not want to intern this as there are a lot of obligation causes which
/// only live for a short period of time.
#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
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

    code: ObligationCauseCodeHandle<'tcx>,
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
        ObligationCause::new(span, body_id, ObligationCauseCode::Misc)
    }

    #[inline(always)]
    pub fn dummy() -> ObligationCause<'tcx> {
        ObligationCause::dummy_with_span(DUMMY_SP)
    }

    #[inline(always)]
    pub fn dummy_with_span(span: Span) -> ObligationCause<'tcx> {
        ObligationCause { span, body_id: CRATE_DEF_ID, code: Default::default() }
    }

    #[inline]
    pub fn code(&self) -> &ObligationCauseCode<'tcx> {
        &self.code
    }

    pub fn map_code(
        &mut self,
        f: impl FnOnce(ObligationCauseCodeHandle<'tcx>) -> ObligationCauseCode<'tcx>,
    ) {
        self.code = f(std::mem::take(&mut self.code)).into();
    }

    pub fn derived_cause(
        mut self,
        parent_trait_pred: ty::PolyTraitPredicate<'tcx>,
        variant: impl FnOnce(DerivedCause<'tcx>) -> ObligationCauseCode<'tcx>,
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
        // by using -Z verbose-internals or just a CLI argument.
        self.code = variant(DerivedCause { parent_trait_pred, parent_code: self.code }).into();
        self
    }

    pub fn derived_host_cause(
        mut self,
        parent_host_pred: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,
        variant: impl FnOnce(DerivedHostCause<'tcx>) -> ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        self.code = variant(DerivedHostCause { parent_host_pred, parent_code: self.code }).into();
        self
    }

    pub fn to_constraint_category(&self) -> ConstraintCategory<'tcx> {
        match self.code() {
            ObligationCauseCode::MatchImpl(cause, _) => cause.to_constraint_category(),
            ObligationCauseCode::AscribeUserTypeProvePredicate(predicate_span) => {
                ConstraintCategory::Predicate(*predicate_span)
            }
            _ => ConstraintCategory::BoringNoLocation,
        }
    }
}

/// A compact form of `ObligationCauseCode`.
#[derive(Clone, PartialEq, Eq, Default, HashStable)]
#[derive(TypeVisitable, TypeFoldable, TyEncodable, TyDecodable)]
pub struct ObligationCauseCodeHandle<'tcx> {
    /// `None` for `ObligationCauseCode::Misc` (a common case, occurs ~60% of
    /// the time). `Some` otherwise.
    code: Option<Arc<ObligationCauseCode<'tcx>>>,
}

impl<'tcx> std::fmt::Debug for ObligationCauseCodeHandle<'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cause: &ObligationCauseCode<'_> = self;
        cause.fmt(f)
    }
}

impl<'tcx> ObligationCauseCode<'tcx> {
    #[inline(always)]
    fn into(self) -> ObligationCauseCodeHandle<'tcx> {
        ObligationCauseCodeHandle {
            code: if let ObligationCauseCode::Misc = self { None } else { Some(Arc::new(self)) },
        }
    }
}

impl<'tcx> std::ops::Deref for ObligationCauseCodeHandle<'tcx> {
    type Target = ObligationCauseCode<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.code.as_deref().unwrap_or(&ObligationCauseCode::Misc)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub enum ObligationCauseCode<'tcx> {
    /// Not well classified or should be obvious from the span.
    Misc,

    /// A slice or array is WF only if `T: Sized`.
    SliceOrArrayElem,

    /// An array `[T; N]` can only be indexed (and is only well-formed if) `N` has type usize.
    ArrayLen(Ty<'tcx>),

    /// A tuple is WF only if its middle elements are `Sized`.
    TupleElem,

    /// Represents a clause that comes from a specific item.
    /// The span corresponds to the clause.
    WhereClause(DefId, Span),

    /// Represents a bound for an opaque we are checking the well-formedness of.
    /// The def-id corresponds to a specific definition site that we found the
    /// hidden type from, if any.
    OpaqueTypeBound(Span, Option<LocalDefId>),

    /// Like `WhereClause`, but also identifies the expression
    /// which requires the `where` clause to be proven, and also
    /// identifies the index of the predicate in the `predicates_of`
    /// list of the item.
    WhereClauseInExpr(DefId, Span, HirId, usize),

    /// Like `WhereClauseinExpr`, but indexes into the `const_conditions`
    /// rather than the `predicates_of`.
    HostEffectInExpr(DefId, Span, HirId, usize),

    /// A type like `&'a T` is WF only if `T: 'a`.
    ReferenceOutlivesReferent(Ty<'tcx>),

    /// A type like `Box<Foo<'a> + 'b>` is WF only if `'b: 'a`.
    ObjectTypeBound(Ty<'tcx>, ty::Region<'tcx>),

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
    VariableType(HirId),
    /// Argument type must be `Sized`.
    SizedArgumentType(Option<HirId>),
    /// Return type must be `Sized`.
    SizedReturnType,
    /// Return type of a call expression must be `Sized`.
    SizedCallReturnType,
    /// Yield type must be `Sized`.
    SizedYieldType,
    /// Inline asm operand type must be `Sized`.
    InlineAsmSized,
    /// Captured closure type must be `Sized`.
    SizedClosureCapture(LocalDefId),
    /// Types live across coroutine yields must be `Sized`.
    SizedCoroutineInterior(LocalDefId),
    /// `[expr; N]` requires `type_of(expr): Copy`.
    RepeatElementCopy {
        /// If element is a `const fn` or const ctor we display a help message suggesting
        /// to move it to a new `const` item while saying that `T` doesn't implement `Copy`.
        is_constable: IsConstable,

        /// Span of the repeat element.
        ///
        /// This is used to suggest wrapping it in a `const { ... }` block.
        elt_span: Span,
    },

    /// Types of fields (other than the last, except for packed structs) in a struct must be sized.
    FieldSized {
        adt_kind: AdtKind,
        span: Span,
        last: bool,
    },

    /// Constant expressions must be sized.
    SizedConstOrStatic,

    /// `static` items must have `Sync` type.
    SharedStatic,

    /// Derived obligation (i.e. theoretical `where` clause) on a built-in
    /// implementation like `Copy` or `Sized`.
    BuiltinDerived(DerivedCause<'tcx>),

    /// Derived obligation (i.e. `where` clause) on an user-provided impl
    /// or a trait alias.
    ImplDerived(Box<ImplDerivedCause<'tcx>>),

    /// Derived obligation for WF goals.
    WellFormedDerived(DerivedCause<'tcx>),

    /// Derived obligation (i.e. `where` clause) on an user-provided impl
    /// or a trait alias.
    ImplDerivedHost(Box<ImplDerivedHostCause<'tcx>>),

    /// Derived obligation (i.e. `where` clause) on an user-provided impl
    /// or a trait alias.
    BuiltinDerivedHost(DerivedHostCause<'tcx>),

    /// Derived obligation refined to point at a specific argument in
    /// a call or method expression.
    FunctionArg {
        /// The node of the relevant argument in the function call.
        arg_hir_id: HirId,
        /// The node of the function call.
        call_hir_id: HirId,
        /// The obligation introduced by this argument.
        parent_code: ObligationCauseCodeHandle<'tcx>,
    },

    /// Error derived when checking an impl item is compatible with
    /// its corresponding trait item's definition
    CompareImplItem {
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
        /// Information about the `Span`, if it came from an expression, otherwise `None`.
        origin_expr: Option<PatternOriginExpr>,
    },

    /// Computing common supertype in an if expression
    IfExpression {
        expr_id: HirId,
        // Is the expectation of this match expression an RPIT?
        tail_defines_return_position_impl_trait: Option<LocalDefId>,
    },

    /// Computing common supertype of an if expression with no else counter-part
    IfExpressionWithNoElse,

    /// `main` has wrong type
    MainFunctionType,

    /// language function has wrong type
    LangFunctionType(Symbol),

    /// Intrinsic has wrong type
    IntrinsicType,

    /// A let else block does not diverge
    LetElse,

    /// Method receiver
    MethodReceiver,

    /// `return` with no expression
    ReturnNoExpression,

    /// `return` with an expression
    ReturnValue(HirId),

    /// Opaque return type of this function
    OpaqueReturnType(Option<(Ty<'tcx>, HirId)>),

    /// Block implicit return
    BlockTailExpression(HirId, hir::MatchSource),

    /// #[feature(trivial_bounds)] is not enabled
    TrivialBound,

    AwaitableExpr(HirId),

    ForLoopIterator,

    QuestionMark,

    /// Well-formed checking. If a `WellFormedLoc` is provided,
    /// then it will be used to perform HIR-based wf checking
    /// after an error occurs, in order to generate a more precise error span.
    /// This is purely for diagnostic purposes - it is always
    /// correct to use `Misc` instead, or to specify
    /// `WellFormed(None)`.
    WellFormed(Option<WellFormedLoc>),

    /// From `match_impl`. The cause for us having to match an impl, and the DefId we are matching
    /// against.
    MatchImpl(ObligationCause<'tcx>, DefId),

    UnOp {
        hir_id: HirId,
    },

    BinOp {
        lhs_hir_id: HirId,
        rhs_hir_id: HirId,
        rhs_span: Span,
        rhs_is_lit: bool,
        output_ty: Option<Ty<'tcx>>,
    },

    AscribeUserTypeProvePredicate(Span),

    RustCall,

    DynCompatible(Span),

    /// Obligations to prove that a `Drop` or negative auto trait impl is not stronger than
    /// the ADT it's being implemented for.
    AlwaysApplicableImpl,

    /// Requirement for a `const N: Ty` to implement `Ty: ConstParamTy`
    ConstParam(Ty<'tcx>),

    /// Obligations emitted during the normalization of a free type alias.
    TypeAlias(ObligationCauseCodeHandle<'tcx>, Span, DefId),

    /// Only reachable if the `unsized_fn_params` feature is used. Unsized function arguments must
    /// be place expressions because we can't store them in MIR locals as temporaries.
    UnsizedNonPlaceExpr(Span),
}

/// Whether a value can be extracted into a const.
/// Used for diagnostics around array repeat expressions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub enum IsConstable {
    No,
    /// Call to a const fn
    Fn,
    /// Use of a const ctor
    Ctor,
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
        param_idx: usize,
    },
}

impl<'tcx> ObligationCauseCode<'tcx> {
    /// Returns the base obligation, ignoring derived obligations.
    pub fn peel_derives(&self) -> &Self {
        let mut base_cause = self;
        while let Some(parent_code) = base_cause.parent() {
            base_cause = parent_code;
        }
        base_cause
    }

    pub fn parent(&self) -> Option<&Self> {
        match self {
            ObligationCauseCode::FunctionArg { parent_code, .. } => Some(parent_code),
            ObligationCauseCode::BuiltinDerived(derived)
            | ObligationCauseCode::WellFormedDerived(derived)
            | ObligationCauseCode::ImplDerived(box ImplDerivedCause { derived, .. }) => {
                Some(&derived.parent_code)
            }
            ObligationCauseCode::BuiltinDerivedHost(derived)
            | ObligationCauseCode::ImplDerivedHost(box ImplDerivedHostCause { derived, .. }) => {
                Some(&derived.parent_code)
            }
            _ => None,
        }
    }

    /// Returns the base obligation and the base trait predicate, if any, ignoring
    /// derived obligations.
    pub fn peel_derives_with_predicate(&self) -> (&Self, Option<ty::PolyTraitPredicate<'tcx>>) {
        let mut base_cause = self;
        let mut base_trait_pred = None;
        while let Some((parent_code, parent_pred)) = base_cause.parent_with_predicate() {
            base_cause = parent_code;
            if let Some(parent_pred) = parent_pred {
                base_trait_pred = Some(parent_pred);
            }
        }

        (base_cause, base_trait_pred)
    }

    pub fn parent_with_predicate(&self) -> Option<(&Self, Option<ty::PolyTraitPredicate<'tcx>>)> {
        match self {
            ObligationCauseCode::FunctionArg { parent_code, .. } => Some((parent_code, None)),
            ObligationCauseCode::BuiltinDerived(derived)
            | ObligationCauseCode::WellFormedDerived(derived)
            | ObligationCauseCode::ImplDerived(box ImplDerivedCause { derived, .. }) => {
                Some((&derived.parent_code, Some(derived.parent_trait_pred)))
            }
            _ => None,
        }
    }

    pub fn peel_match_impls(&self) -> &Self {
        match self {
            ObligationCauseCode::MatchImpl(cause, _) => cause.code(),
            _ => self,
        }
    }
}

// `ObligationCauseCode` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(ObligationCauseCode<'_>, 48);

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct MatchExpressionArmCause<'tcx> {
    pub arm_block_id: Option<HirId>,
    pub arm_ty: Ty<'tcx>,
    pub arm_span: Span,
    pub prior_arm_block_id: Option<HirId>,
    pub prior_arm_ty: Ty<'tcx>,
    pub prior_arm_span: Span,
    /// Span of the scrutinee of the match (the matched value).
    pub scrut_span: Span,
    /// Source of the match, i.e. `match` or a desugaring.
    pub source: hir::MatchSource,
    /// Span of the *whole* match expr.
    pub expr_span: Span,
    /// Spans of the previous arms except for those that diverge (i.e. evaluate to `!`).
    ///
    /// These are used for pointing out errors that may affect several arms.
    pub prior_non_diverging_arms: Vec<Span>,
    /// Is the expectation of this match expression an RPIT?
    pub tail_defines_return_position_impl_trait: Option<LocalDefId>,
}

/// Information about the origin expression of a pattern, relevant to diagnostics.
/// Fields here refer to the scrutinee of a pattern.
/// If the scrutinee isn't given in the diagnostic, then this won't exist.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[derive(TypeFoldable, TypeVisitable, HashStable, TyEncodable, TyDecodable)]
pub struct PatternOriginExpr {
    /// A span representing the scrutinee expression, with all leading references
    /// peeled from the expression.
    /// Only references in the expression are peeled - if the expression refers to a variable
    /// whose type is a reference, then that reference is kept because it wasn't created
    /// in the expression.
    pub peeled_span: Span,
    /// The number of references that were peeled to produce `peeled_span`.
    pub peeled_count: usize,
    /// Does the peeled expression need to be wrapped in parentheses for
    /// a prefix suggestion (i.e., dereference) to be valid.
    pub peeled_prefix_suggestion_parentheses: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct DerivedCause<'tcx> {
    /// The trait predicate of the parent obligation that led to the
    /// current obligation. Note that only trait obligations lead to
    /// derived obligations, so we just store the trait predicate here
    /// directly.
    pub parent_trait_pred: ty::PolyTraitPredicate<'tcx>,

    /// The parent trait had this cause.
    pub parent_code: ObligationCauseCodeHandle<'tcx>,
}

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct ImplDerivedCause<'tcx> {
    pub derived: DerivedCause<'tcx>,
    /// The `DefId` of the `impl` that gave rise to the `derived` obligation.
    /// If the `derived` obligation arose from a trait alias, which conceptually has a synthetic
    /// impl, then this will be the `DefId` of that trait alias. Care should therefore be taken to
    /// handle that exceptional case where appropriate.
    pub impl_or_alias_def_id: DefId,
    /// The index of the derived predicate in the parent impl's predicates.
    pub impl_def_predicate_index: Option<usize>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct DerivedHostCause<'tcx> {
    /// The trait predicate of the parent obligation that led to the
    /// current obligation. Note that only trait obligations lead to
    /// derived obligations, so we just store the trait predicate here
    /// directly.
    pub parent_host_pred: ty::Binder<'tcx, ty::HostEffectPredicate<'tcx>>,

    /// The parent trait had this cause.
    pub parent_code: ObligationCauseCodeHandle<'tcx>,
}

#[derive(Clone, Debug, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct ImplDerivedHostCause<'tcx> {
    pub derived: DerivedHostCause<'tcx>,
    /// The `DefId` of the `impl` that gave rise to the `derived` obligation.
    pub impl_def_id: DefId,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, TypeVisitable)]
pub enum SelectionError<'tcx> {
    /// The trait is not implemented.
    Unimplemented,
    /// After a closure impl has selected, its "outputs" were evaluated
    /// (which for closures includes the "input" type params) and they
    /// didn't resolve. See `confirm_poly_trait_refs` for more.
    SignatureMismatch(Box<SignatureMismatchData<'tcx>>),
    /// The trait pointed by `DefId` is dyn-incompatible.
    TraitDynIncompatible(DefId),
    /// A given constant couldn't be evaluated.
    NotConstEvaluatable(NotConstEvaluatable),
    /// Exceeded the recursion depth during type projection.
    Overflow(OverflowError),
    /// Computing an opaque type's hidden type caused an error (e.g. a cycle error).
    /// We can thus not know whether the hidden type implements an auto trait, so
    /// we should not presume anything about it.
    OpaqueTypeAutoTraitLeakageUnknown(DefId),
    /// Error for a `ConstArgHasType` goal
    ConstArgHasWrongType { ct: ty::Const<'tcx>, ct_ty: Ty<'tcx>, expected_ty: Ty<'tcx> },
}

#[derive(Clone, Debug, PartialEq, Eq, TypeVisitable)]
pub struct SignatureMismatchData<'tcx> {
    pub found_trait_ref: ty::TraitRef<'tcx>,
    pub expected_trait_ref: ty::TraitRef<'tcx>,
    pub terr: ty::error::TypeError<'tcx>,
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
#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum ImplSource<'tcx, N> {
    /// ImplSource identifying a particular impl.
    UserDefined(ImplSourceUserDefinedData<'tcx, N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter. The `Vec<N>` represents the
    /// obligations incurred from normalizing the where-clause (if
    /// any).
    Param(ThinVec<N>),

    /// Successful resolution for a builtin impl.
    Builtin(BuiltinImplSource, ThinVec<N>),
}

impl<'tcx, N> ImplSource<'tcx, N> {
    pub fn nested_obligations(self) -> ThinVec<N> {
        match self {
            ImplSource::UserDefined(i) => i.nested,
            ImplSource::Param(n) | ImplSource::Builtin(_, n) => n,
        }
    }

    pub fn borrow_nested_obligations(&self) -> &[N] {
        match self {
            ImplSource::UserDefined(i) => &i.nested,
            ImplSource::Param(n) | ImplSource::Builtin(_, n) => n,
        }
    }

    pub fn borrow_nested_obligations_mut(&mut self) -> &mut [N] {
        match self {
            ImplSource::UserDefined(i) => &mut i.nested,
            ImplSource::Param(n) | ImplSource::Builtin(_, n) => n,
        }
    }

    pub fn map<M, F>(self, f: F) -> ImplSource<'tcx, M>
    where
        F: FnMut(N) -> M,
    {
        match self {
            ImplSource::UserDefined(i) => ImplSource::UserDefined(ImplSourceUserDefinedData {
                impl_def_id: i.impl_def_id,
                args: i.args,
                nested: i.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Param(n) => ImplSource::Param(n.into_iter().map(f).collect()),
            ImplSource::Builtin(source, n) => {
                ImplSource::Builtin(source, n.into_iter().map(f).collect())
            }
        }
    }
}

/// Identifies a particular impl in the source, along with a set of
/// generic parameters from the impl's type/lifetime parameters. The
/// `nested` vector corresponds to the nested obligations attached to
/// the impl's type parameters.
///
/// The type parameter `N` indicates the type used for "nested
/// obligations" that are required by the impl. During type-check, this
/// is `Obligation`, as one might expect. During codegen, however, this
/// is `()`, because codegen only requires a shallow resolution of an
/// impl, and nested obligations are satisfied later.
#[derive(Clone, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ImplSourceUserDefinedData<'tcx, N> {
    pub impl_def_id: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub nested: ThinVec<N>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, HashStable, PartialOrd, Ord)]
pub enum DynCompatibilityViolation {
    /// `Self: Sized` declared on the trait.
    SizedSelf(SmallVec<[Span; 1]>),

    /// Supertrait reference references `Self` an in illegal location
    /// (e.g., `trait Foo : Bar<Self>`).
    SupertraitSelf(SmallVec<[Span; 1]>),

    // Supertrait has a non-lifetime `for<T>` binder.
    SupertraitNonLifetimeBinder(SmallVec<[Span; 1]>),

    // Trait has a `const Trait` supertrait.
    SupertraitConst(SmallVec<[Span; 1]>),

    /// Method has something illegal.
    Method(Symbol, MethodViolationCode, Span),

    /// Associated const.
    AssocConst(Symbol, Span),

    /// GAT
    GAT(Symbol, Span),
}

impl DynCompatibilityViolation {
    pub fn error_msg(&self) -> Cow<'static, str> {
        match self {
            DynCompatibilityViolation::SizedSelf(_) => "it requires `Self: Sized`".into(),
            DynCompatibilityViolation::SupertraitSelf(spans) => {
                if spans.iter().any(|sp| *sp != DUMMY_SP) {
                    "it uses `Self` as a type parameter".into()
                } else {
                    "it cannot use `Self` as a type parameter in a supertrait or `where`-clause"
                        .into()
                }
            }
            DynCompatibilityViolation::SupertraitNonLifetimeBinder(_) => {
                "where clause cannot reference non-lifetime `for<...>` variables".into()
            }
            DynCompatibilityViolation::SupertraitConst(_) => {
                "it cannot have a `const` supertrait".into()
            }
            DynCompatibilityViolation::Method(name, MethodViolationCode::StaticMethod(_), _) => {
                format!("associated function `{name}` has no `self` parameter").into()
            }
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::ReferencesSelfInput(_),
                DUMMY_SP,
            ) => format!("method `{name}` references the `Self` type in its parameters").into(),
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::ReferencesSelfInput(_),
                _,
            ) => format!("method `{name}` references the `Self` type in this parameter").into(),
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::ReferencesSelfOutput,
                _,
            ) => format!("method `{name}` references the `Self` type in its return type").into(),
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::ReferencesImplTraitInTrait(_),
                _,
            ) => {
                format!("method `{name}` references an `impl Trait` type in its return type").into()
            }
            DynCompatibilityViolation::Method(name, MethodViolationCode::AsyncFn, _) => {
                format!("method `{name}` is `async`").into()
            }
            DynCompatibilityViolation::Method(name, MethodViolationCode::CVariadic, _) => {
                format!("method `{name}` is C-variadic").into()
            }
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::WhereClauseReferencesSelf,
                _,
            ) => format!("method `{name}` references the `Self` type in its `where` clause").into(),
            DynCompatibilityViolation::Method(name, MethodViolationCode::Generic, _) => {
                format!("method `{name}` has generic type parameters").into()
            }
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::UndispatchableReceiver(_),
                _,
            ) => format!("method `{name}`'s `self` parameter cannot be dispatched on").into(),
            DynCompatibilityViolation::AssocConst(name, DUMMY_SP) => {
                format!("it contains associated `const` `{name}`").into()
            }
            DynCompatibilityViolation::AssocConst(..) => {
                "it contains this associated `const`".into()
            }
            DynCompatibilityViolation::GAT(name, _) => {
                format!("it contains the generic associated type `{name}`").into()
            }
        }
    }

    pub fn solution(&self) -> DynCompatibilityViolationSolution {
        match self {
            DynCompatibilityViolation::SizedSelf(_)
            | DynCompatibilityViolation::SupertraitSelf(_)
            | DynCompatibilityViolation::SupertraitNonLifetimeBinder(..)
            | DynCompatibilityViolation::SupertraitConst(_) => {
                DynCompatibilityViolationSolution::None
            }
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::StaticMethod(Some((add_self_sugg, make_sized_sugg))),
                _,
            ) => DynCompatibilityViolationSolution::AddSelfOrMakeSized {
                name: *name,
                add_self_sugg: add_self_sugg.clone(),
                make_sized_sugg: make_sized_sugg.clone(),
            },
            DynCompatibilityViolation::Method(
                name,
                MethodViolationCode::UndispatchableReceiver(Some(span)),
                _,
            ) => DynCompatibilityViolationSolution::ChangeToRefSelf(*name, *span),
            DynCompatibilityViolation::AssocConst(name, _)
            | DynCompatibilityViolation::GAT(name, _)
            | DynCompatibilityViolation::Method(name, ..) => {
                DynCompatibilityViolationSolution::MoveToAnotherTrait(*name)
            }
        }
    }

    pub fn spans(&self) -> SmallVec<[Span; 1]> {
        // When `span` comes from a separate crate, it'll be `DUMMY_SP`. Treat it as `None` so
        // diagnostics use a `note` instead of a `span_label`.
        match self {
            DynCompatibilityViolation::SupertraitSelf(spans)
            | DynCompatibilityViolation::SizedSelf(spans)
            | DynCompatibilityViolation::SupertraitNonLifetimeBinder(spans)
            | DynCompatibilityViolation::SupertraitConst(spans) => spans.clone(),
            DynCompatibilityViolation::AssocConst(_, span)
            | DynCompatibilityViolation::GAT(_, span)
            | DynCompatibilityViolation::Method(_, _, span) => {
                if *span != DUMMY_SP {
                    smallvec![*span]
                } else {
                    smallvec![]
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DynCompatibilityViolationSolution {
    None,
    AddSelfOrMakeSized {
        name: Symbol,
        add_self_sugg: (String, Span),
        make_sized_sugg: (String, Span),
    },
    ChangeToRefSelf(Symbol, Span),
    MoveToAnotherTrait(Symbol),
}

impl DynCompatibilityViolationSolution {
    pub fn add_to<G: EmissionGuarantee>(self, err: &mut Diag<'_, G>) {
        match self {
            DynCompatibilityViolationSolution::None => {}
            DynCompatibilityViolationSolution::AddSelfOrMakeSized {
                name,
                add_self_sugg,
                make_sized_sugg,
            } => {
                err.span_suggestion(
                    add_self_sugg.1,
                    format!(
                        "consider turning `{name}` into a method by giving it a `&self` argument"
                    ),
                    add_self_sugg.0,
                    Applicability::MaybeIncorrect,
                );
                err.span_suggestion(
                    make_sized_sugg.1,
                    format!(
                        "alternatively, consider constraining `{name}` so it does not apply to \
                             trait objects"
                    ),
                    make_sized_sugg.0,
                    Applicability::MaybeIncorrect,
                );
            }
            DynCompatibilityViolationSolution::ChangeToRefSelf(name, span) => {
                err.span_suggestion(
                    span,
                    format!("consider changing method `{name}`'s `self` parameter to be `&self`"),
                    "&Self",
                    Applicability::MachineApplicable,
                );
            }
            DynCompatibilityViolationSolution::MoveToAnotherTrait(name) => {
                err.help(format!("consider moving `{name}` to another trait"));
            }
        }
    }
}

/// Reasons a method might not be dyn-compatible.
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

    /// e.g., `fn (mut ap: ...)`
    CVariadic,

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
    /// This can trigger when we have a global bound that is not actually satisfied
    /// due to trivial bounds.
    Unimplemented,
    /// The selected impl has unconstrained generic parameters. This will emit an error
    /// during impl WF checking.
    UnconstrainedParam(ErrorGuaranteed),
}
