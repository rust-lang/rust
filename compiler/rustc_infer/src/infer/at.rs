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

use relate::lattice::{LatticeOp, LatticeOpKind};
use rustc_middle::bug;
use rustc_middle::ty::relate::solver_relating::RelateExt as NextSolverRelate;
use rustc_middle::ty::{Const, ImplSubject};

use super::*;
use crate::infer::relate::type_relating::TypeRelating;
use crate::infer::relate::{Relate, TypeRelation};
use crate::traits::Obligation;
use crate::traits::solve::Goal;

/// Whether we should define opaque types or just treat them opaquely.
///
/// Currently only used to prevent predicate matching from matching anything
/// against opaque types.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DefineOpaqueTypes {
    Yes,
    No,
}

#[derive(Clone, Copy)]
pub struct At<'a, 'tcx> {
    pub infcx: &'a InferCtxt<'tcx>,
    pub cause: &'a ObligationCause<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> InferCtxt<'tcx> {
    #[inline]
    pub fn at<'a>(
        &'a self,
        cause: &'a ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> At<'a, 'tcx> {
        At { infcx: self, cause, param_env }
    }

    /// Forks the inference context, creating a new inference context with the same inference
    /// variables in the same state. This can be used to "branch off" many tests from the same
    /// common state.
    pub fn fork(&self) -> Self {
        self.fork_with_intercrate(self.intercrate)
    }

    /// Forks the inference context, creating a new inference context with the same inference
    /// variables in the same state, except possibly changing the intercrate mode. This can be
    /// used to "branch off" many tests from the same common state. Used in negative coherence.
    pub fn fork_with_intercrate(&self, intercrate: bool) -> Self {
        Self {
            tcx: self.tcx,
            defining_opaque_types: self.defining_opaque_types,
            considering_regions: self.considering_regions,
            skip_leak_check: self.skip_leak_check,
            inner: self.inner.clone(),
            lexical_region_resolutions: self.lexical_region_resolutions.clone(),
            selection_cache: self.selection_cache.clone(),
            evaluation_cache: self.evaluation_cache.clone(),
            reported_trait_errors: self.reported_trait_errors.clone(),
            reported_signature_mismatch: self.reported_signature_mismatch.clone(),
            tainted_by_errors: self.tainted_by_errors.clone(),
            universe: self.universe.clone(),
            intercrate,
            next_trait_solver: self.next_trait_solver,
            obligation_inspector: self.obligation_inspector.clone(),
        }
    }
}

pub trait ToTrace<'tcx>: Relate<TyCtxt<'tcx>> + Copy {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx>;
}

impl<'a, 'tcx> At<'a, 'tcx> {
    /// Makes `actual <: expected`. For example, if type-checking a
    /// call like `foo(x)`, where `foo: fn(i32)`, you might have
    /// `sup(i32, x)`, since the "expected" type is the type that
    /// appears in the signature.
    pub fn sup<T>(
        self,
        define_opaque_types: DefineOpaqueTypes,
        expected: T,
        actual: T,
    ) -> InferResult<'tcx, ()>
    where
        T: ToTrace<'tcx>,
    {
        if self.infcx.next_trait_solver {
            NextSolverRelate::relate(
                self.infcx,
                self.param_env,
                expected,
                ty::Contravariant,
                actual,
            )
            .map(|goals| self.goals_to_obligations(goals))
        } else {
            let mut op = TypeRelating::new(
                self.infcx,
                ToTrace::to_trace(self.cause, expected, actual),
                self.param_env,
                define_opaque_types,
                ty::Contravariant,
            );
            op.relate(expected, actual)?;
            Ok(InferOk { value: (), obligations: op.into_obligations() })
        }
    }

    /// Makes `expected <: actual`.
    pub fn sub<T>(
        self,
        define_opaque_types: DefineOpaqueTypes,
        expected: T,
        actual: T,
    ) -> InferResult<'tcx, ()>
    where
        T: ToTrace<'tcx>,
    {
        if self.infcx.next_trait_solver {
            NextSolverRelate::relate(self.infcx, self.param_env, expected, ty::Covariant, actual)
                .map(|goals| self.goals_to_obligations(goals))
        } else {
            let mut op = TypeRelating::new(
                self.infcx,
                ToTrace::to_trace(self.cause, expected, actual),
                self.param_env,
                define_opaque_types,
                ty::Covariant,
            );
            op.relate(expected, actual)?;
            Ok(InferOk { value: (), obligations: op.into_obligations() })
        }
    }

    /// Makes `expected == actual`.
    pub fn eq<T>(
        self,
        define_opaque_types: DefineOpaqueTypes,
        expected: T,
        actual: T,
    ) -> InferResult<'tcx, ()>
    where
        T: ToTrace<'tcx>,
    {
        self.eq_trace(
            define_opaque_types,
            ToTrace::to_trace(self.cause, expected, actual),
            expected,
            actual,
        )
    }

    /// Makes `expected == actual`.
    pub fn eq_trace<T>(
        self,
        define_opaque_types: DefineOpaqueTypes,
        trace: TypeTrace<'tcx>,
        expected: T,
        actual: T,
    ) -> InferResult<'tcx, ()>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        if self.infcx.next_trait_solver {
            NextSolverRelate::relate(self.infcx, self.param_env, expected, ty::Invariant, actual)
                .map(|goals| self.goals_to_obligations(goals))
        } else {
            let mut op = TypeRelating::new(
                self.infcx,
                trace,
                self.param_env,
                define_opaque_types,
                ty::Invariant,
            );
            op.relate(expected, actual)?;
            Ok(InferOk { value: (), obligations: op.into_obligations() })
        }
    }

    pub fn relate<T>(
        self,
        define_opaque_types: DefineOpaqueTypes,
        expected: T,
        variance: ty::Variance,
        actual: T,
    ) -> InferResult<'tcx, ()>
    where
        T: ToTrace<'tcx>,
    {
        match variance {
            ty::Covariant => self.sub(define_opaque_types, expected, actual),
            ty::Invariant => self.eq(define_opaque_types, expected, actual),
            ty::Contravariant => self.sup(define_opaque_types, expected, actual),

            // We could make this make sense but it's not readily
            // exposed and I don't feel like dealing with it. Note
            // that bivariance in general does a bit more than just
            // *nothing*, it checks that the types are the same
            // "modulo variance" basically.
            ty::Bivariant => panic!("Bivariant given to `relate()`"),
        }
    }

    /// Computes the least-upper-bound, or mutual supertype, of two
    /// values. The order of the arguments doesn't matter, but since
    /// this can result in an error (e.g., if asked to compute LUB of
    /// u32 and i32), it is meaningful to call one of them the
    /// "expected type".
    pub fn lub<T>(self, expected: T, actual: T) -> InferResult<'tcx, T>
    where
        T: ToTrace<'tcx>,
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

    fn goals_to_obligations(
        &self,
        goals: Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> InferOk<'tcx, ()> {
        InferOk {
            value: (),
            obligations: goals
                .into_iter()
                .map(|goal| {
                    Obligation::new(
                        self.infcx.tcx,
                        self.cause.clone(),
                        goal.param_env,
                        goal.predicate,
                    )
                })
                .collect(),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ImplSubject<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        match (a, b) {
            (ImplSubject::Trait(trait_ref_a), ImplSubject::Trait(trait_ref_b)) => {
                ToTrace::to_trace(cause, trait_ref_a, trait_ref_b)
            }
            (ImplSubject::Inherent(ty_a), ImplSubject::Inherent(ty_b)) => {
                ToTrace::to_trace(cause, ty_a, ty_b)
            }
            (ImplSubject::Trait(_), ImplSubject::Inherent(_))
            | (ImplSubject::Inherent(_), ImplSubject::Trait(_)) => {
                bug!("can not trace TraitRef and Ty");
            }
        }
    }
}

impl<'tcx> ToTrace<'tcx> for Ty<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(true, a.into(), b.into())),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::Region<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Regions(ExpectedFound::new(true, a, b)),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for Const<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(true, a.into(), b.into())),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::GenericArg<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: match (a.unpack(), b.unpack()) {
                (GenericArgKind::Lifetime(a), GenericArgKind::Lifetime(b)) => {
                    ValuePairs::Regions(ExpectedFound::new(true, a, b))
                }
                (GenericArgKind::Type(a), GenericArgKind::Type(b)) => {
                    ValuePairs::Terms(ExpectedFound::new(true, a.into(), b.into()))
                }
                (GenericArgKind::Const(a), GenericArgKind::Const(b)) => {
                    ValuePairs::Terms(ExpectedFound::new(true, a.into(), b.into()))
                }
                _ => bug!("relating different kinds: {a:?} {b:?}"),
            },
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::Term<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Terms(ExpectedFound::new(true, a, b)),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::TraitRef<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::TraitRefs(ExpectedFound::new(true, a, b)),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::AliasTy<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Aliases(ExpectedFound::new(true, a.into(), b.into())),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::AliasTerm<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::Aliases(ExpectedFound::new(true, a, b)),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::FnSig<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::PolySigs(ExpectedFound::new(
                true,
                ty::Binder::dummy(a),
                ty::Binder::dummy(b),
            )),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::PolyFnSig<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::PolySigs(ExpectedFound::new(true, a, b)),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::PolyExistentialTraitRef<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::ExistentialTraitRef(ExpectedFound::new(true, a, b)),
        }
    }
}

impl<'tcx> ToTrace<'tcx> for ty::PolyExistentialProjection<'tcx> {
    fn to_trace(cause: &ObligationCause<'tcx>, a: Self, b: Self) -> TypeTrace<'tcx> {
        TypeTrace {
            cause: cause.clone(),
            values: ValuePairs::ExistentialProjection(ExpectedFound::new(true, a, b)),
        }
    }
}
