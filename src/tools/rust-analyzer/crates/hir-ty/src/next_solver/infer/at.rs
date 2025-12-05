//! A nice interface for working with the infcx. The basic idea is to
//! do `infcx.at(cause, param_env)`, which sets the "cause" of the
//! operation as well as the surrounding parameter environment. Then
//! you can do something like `.sub(a, b)` or `.eq(a, b)` to create a
//! subtype or equality relationship respectively. The first argument
//! is always the "expected" output from the POV of diagnostics.
//!
//! Examples:
//! ```ignore (fragment)
//!     infcx.at(cause, param_env).sub(a, b)
//!     // requires that `a <: b`, with `a` considered the "expected" type
//!
//!     infcx.at(cause, param_env).sup(a, b)
//!     // requires that `b <: a`, with `a` considered the "expected" type
//!
//!     infcx.at(cause, param_env).eq(a, b)
//!     // requires that `a == b`, with `a` considered the "expected" type
//! ```
//! For finer-grained control, you can also do use `trace`:
//! ```ignore (fragment)
//!     infcx.at(...).trace(a, b).sub(&c, &d)
//! ```
//! This will set `a` and `b` as the "root" values for
//! error-reporting, but actually operate on `c` and `d`. This is
//! sometimes useful when the types of `c` and `d` are not traceable
//! things. (That system should probably be refactored.)

use rustc_type_ir::{
    FnSig, GenericArgKind, TypeFoldable, TypingMode, Variance,
    error::ExpectedFound,
    inherent::{IntoKind, Span as _},
    relate::{Relate, TypeRelation, solver_relating::RelateExt},
};

use crate::next_solver::{
    AliasTerm, AliasTy, Binder, Const, DbInterner, GenericArg, Goal, ParamEnv,
    PolyExistentialProjection, PolyExistentialTraitRef, PolyFnSig, Predicate, Region, Span, Term,
    TraitRef, Ty,
    fulfill::NextSolverError,
    infer::relate::lattice::{LatticeOp, LatticeOpKind},
};

use super::{
    InferCtxt, InferOk, InferResult, TypeTrace, ValuePairs,
    traits::{Obligation, ObligationCause},
};

#[derive(Clone, Copy)]
pub struct At<'a, 'db> {
    pub infcx: &'a InferCtxt<'db>,
    pub cause: &'a ObligationCause,
    pub param_env: ParamEnv<'db>,
}

impl<'db> InferCtxt<'db> {
    #[inline]
    pub fn at<'a>(&'a self, cause: &'a ObligationCause, param_env: ParamEnv<'db>) -> At<'a, 'db> {
        At { infcx: self, cause, param_env }
    }

    /// Forks the inference context, creating a new inference context with the same inference
    /// variables in the same state. This can be used to "branch off" many tests from the same
    /// common state.
    pub fn fork(&self) -> Self {
        Self {
            interner: self.interner,
            typing_mode: self.typing_mode,
            inner: self.inner.clone(),
            tainted_by_errors: self.tainted_by_errors.clone(),
            universe: self.universe.clone(),
        }
    }

    /// Forks the inference context, creating a new inference context with the same inference
    /// variables in the same state, except possibly changing the intercrate mode. This can be
    /// used to "branch off" many tests from the same common state. Used in negative coherence.
    pub fn fork_with_typing_mode(&self, typing_mode: TypingMode<DbInterner<'db>>) -> Self {
        // Unlike `fork`, this invalidates all cache entries as they may depend on the
        // typing mode.

        Self {
            interner: self.interner,
            typing_mode,
            inner: self.inner.clone(),
            tainted_by_errors: self.tainted_by_errors.clone(),
            universe: self.universe.clone(),
        }
    }
}

pub trait ToTrace<'db>: Relate<DbInterner<'db>> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db>;
}

impl<'a, 'db> At<'a, 'db> {
    /// Makes `actual <: expected`. For example, if type-checking a
    /// call like `foo(x)`, where `foo: fn(i32)`, you might have
    /// `sup(i32, x)`, since the "expected" type is the type that
    /// appears in the signature.
    pub fn sup<T>(self, expected: T, actual: T) -> InferResult<'db, ()>
    where
        T: ToTrace<'db>,
    {
        RelateExt::relate(
            self.infcx,
            self.param_env,
            expected,
            Variance::Contravariant,
            actual,
            Span::dummy(),
        )
        .map(|goals| self.goals_to_obligations(goals))
    }

    /// Makes `expected <: actual`.
    pub fn sub<T>(self, expected: T, actual: T) -> InferResult<'db, ()>
    where
        T: ToTrace<'db>,
    {
        RelateExt::relate(
            self.infcx,
            self.param_env,
            expected,
            Variance::Covariant,
            actual,
            Span::dummy(),
        )
        .map(|goals| self.goals_to_obligations(goals))
    }

    /// Makes `expected == actual`.
    pub fn eq<T>(self, expected: T, actual: T) -> InferResult<'db, ()>
    where
        T: Relate<DbInterner<'db>>,
    {
        RelateExt::relate(
            self.infcx,
            self.param_env,
            expected,
            Variance::Invariant,
            actual,
            Span::dummy(),
        )
        .map(|goals| self.goals_to_obligations(goals))
    }

    pub fn relate<T>(self, expected: T, variance: Variance, actual: T) -> InferResult<'db, ()>
    where
        T: ToTrace<'db>,
    {
        match variance {
            Variance::Covariant => self.sub(expected, actual),
            Variance::Invariant => self.eq(expected, actual),
            Variance::Contravariant => self.sup(expected, actual),

            // We could make this make sense but it's not readily
            // exposed and I don't feel like dealing with it. Note
            // that bivariance in general does a bit more than just
            // *nothing*, it checks that the types are the same
            // "modulo variance" basically.
            Variance::Bivariant => panic!("Bivariant given to `relate()`"),
        }
    }

    /// Deeply normalizes `value`, replacing all aliases which can by normalized in
    /// the current environment. This errors in case normalization fails or is ambiguous.
    pub fn deeply_normalize<T>(self, value: T) -> Result<T, Vec<NextSolverError<'db>>>
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        crate::next_solver::normalize::deeply_normalize(self, value)
    }

    /// Computes the least-upper-bound, or mutual supertype, of two
    /// values. The order of the arguments doesn't matter, but since
    /// this can result in an error (e.g., if asked to compute LUB of
    /// u32 and i32), it is meaningful to call one of them the
    /// "expected type".
    pub fn lub<T>(self, expected: T, actual: T) -> InferResult<'db, T>
    where
        T: ToTrace<'db>,
    {
        let mut op = LatticeOp::new(
            self.infcx,
            ToTrace::to_trace(self.cause, expected, actual),
            self.param_env,
            LatticeOpKind::Lub,
        );
        let value = op.relate(expected, actual)?;
        Ok(InferOk { value, obligations: op.into_obligations() })
    }

    fn goals_to_obligations(&self, goals: Vec<Goal<'db, Predicate<'db>>>) -> InferOk<'db, ()> {
        InferOk {
            value: (),
            obligations: goals
                .into_iter()
                .map(|goal| {
                    Obligation::new(
                        self.infcx.interner,
                        self.cause.clone(),
                        goal.param_env,
                        goal.predicate,
                    )
                })
                .collect(),
        }
    }
}

impl<'db> ToTrace<'db> for Ty<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(a.into(), b.into())),
        }
    }
}

impl<'db> ToTrace<'db> for Region<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::Regions(ExpectedFound::new(a, b)) }
    }
}

impl<'db> ToTrace<'db> for Const<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(a.into(), b.into())),
        }
    }
}

impl<'db> ToTrace<'db> for GenericArg<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: match (a.kind(), b.kind()) {
                (GenericArgKind::Lifetime(a), GenericArgKind::Lifetime(b)) => {
                    ValuePairs::Regions(ExpectedFound::new(a, b))
                }
                (GenericArgKind::Type(a), GenericArgKind::Type(b)) => {
                    ValuePairs::Terms(ExpectedFound::new(a.into(), b.into()))
                }
                (GenericArgKind::Const(a), GenericArgKind::Const(b)) => {
                    ValuePairs::Terms(ExpectedFound::new(a.into(), b.into()))
                }
                _ => panic!("relating different kinds: {a:?} {b:?}"),
            },
        }
    }
}

impl<'db> ToTrace<'db> for Term<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::Terms(ExpectedFound::new(a, b)) }
    }
}

impl<'db> ToTrace<'db> for TraitRef<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::TraitRefs(ExpectedFound::new(a, b)) }
    }
}

impl<'db> ToTrace<'db> for AliasTy<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Aliases(ExpectedFound::new(a.into(), b.into())),
        }
    }
}

impl<'db> ToTrace<'db> for AliasTerm<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::Aliases(ExpectedFound::new(a, b)) }
    }
}

impl<'db> ToTrace<'db> for FnSig<DbInterner<'db>> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::PolySigs(ExpectedFound::new(Binder::dummy(a), Binder::dummy(b))),
        }
    }
}

impl<'db> ToTrace<'db> for PolyFnSig<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace { cause: cause.clone(), values: ValuePairs::PolySigs(ExpectedFound::new(a, b)) }
    }
}

impl<'db> ToTrace<'db> for PolyExistentialTraitRef<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::ExistentialTraitRef(ExpectedFound::new(a, b)),
        }
    }
}

impl<'db> ToTrace<'db> for PolyExistentialProjection<'db> {
    fn to_trace(cause: &ObligationCause, a: Self, b: Self) -> TypeTrace<'db> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::ExistentialProjection(ExpectedFound::new(a, b)),
        }
    }
}
