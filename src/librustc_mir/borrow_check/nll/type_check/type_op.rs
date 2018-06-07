// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::type_check::TypeChecker;
use rustc::infer::{InferCtxt, InferOk, InferResult};
use rustc::traits::query::NoSolution;
use rustc::traits::{Normalized, Obligation, ObligationCause, PredicateObligation};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{ParamEnv, Predicate, Ty};
use std::fmt;

pub(super) trait TypeOp<'gcx, 'tcx>: Sized {
    type Output;

    /// Micro-optimization: returns `Ok(x)` if we can trivially
    /// produce the output, else returns `Err(self)` back.
    fn trivial_noop(self) -> Result<Self::Output, Self>;

    /// Produce a description of the operation for the debug logs.
    fn perform(
        self,
        type_checker: &mut TypeChecker<'_, 'gcx, 'tcx>,
    ) -> InferResult<'tcx, Self::Output>;
}

pub(super) struct CustomTypeOp<F> {
    closure: F,
}

impl<F> CustomTypeOp<F> {
    pub(super) fn new<'gcx, 'tcx, R>(closure: F) -> Self
    where
        F: FnOnce(&mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R>,
    {
        CustomTypeOp { closure }
    }
}

impl<'gcx, 'tcx, F, R> TypeOp<'gcx, 'tcx> for CustomTypeOp<F>
where
    F: FnOnce(&mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R>,
{
    type Output = R;

    fn trivial_noop(self) -> Result<Self::Output, Self> {
        Err(self)
    }

    fn perform(self, type_checker: &mut TypeChecker<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R> {
        (self.closure)(type_checker)
    }
}

pub(super) trait InfcxTypeOp<'gcx, 'tcx>: Sized {
    type Output;

    /// Micro-optimization: returns `Ok(x)` if we can trivially
    /// produce the output, else returns `Err(self)` back.
    fn trivial_noop(self) -> Result<Self::Output, Self>;

    /// Produce a description of the operation for the debug logs.
    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output>;
}

impl<'gcx, 'tcx, OP> TypeOp<'gcx, 'tcx> for OP
where
    OP: InfcxTypeOp<'gcx, 'tcx>,
{
    type Output = OP::Output;

    fn trivial_noop(self) -> Result<Self::Output, Self> {
        InfcxTypeOp::trivial_noop(self)
    }

    fn perform(
        self,
        type_checker: &mut TypeChecker<'_, 'gcx, 'tcx>,
    ) -> InferResult<'tcx, OP::Output> {
        InfcxTypeOp::perform(self, type_checker.infcx)
    }
}

pub(super) struct Subtype<'tcx> {
    param_env: ParamEnv<'tcx>,
    sub: Ty<'tcx>,
    sup: Ty<'tcx>,
}

impl<'tcx> Subtype<'tcx> {
    pub(super) fn new(param_env: ParamEnv<'tcx>, sub: Ty<'tcx>, sup: Ty<'tcx>) -> Self {
        Self {
            param_env,
            sub,
            sup,
        }
    }
}

impl<'gcx, 'tcx> InfcxTypeOp<'gcx, 'tcx> for Subtype<'tcx> {
    type Output = ();

    fn trivial_noop(self) -> Result<Self::Output, Self> {
        if self.sub == self.sup {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .sup(self.sup, self.sub)
    }
}

pub(super) struct Eq<'tcx> {
    param_env: ParamEnv<'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
}

impl<'tcx> Eq<'tcx> {
    pub(super) fn new(param_env: ParamEnv<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> Self {
        Self { param_env, a, b }
    }
}

impl<'gcx, 'tcx> InfcxTypeOp<'gcx, 'tcx> for Eq<'tcx> {
    type Output = ();

    fn trivial_noop(self) -> Result<Self::Output, Self> {
        if self.a == self.b {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .eq(self.a, self.b)
    }
}

pub(super) struct ProvePredicates<'tcx> {
    obligations: Vec<PredicateObligation<'tcx>>,
}

impl<'tcx> ProvePredicates<'tcx> {
    pub(super) fn new(
        param_env: ParamEnv<'tcx>,
        predicates: impl IntoIterator<Item = Predicate<'tcx>>,
    ) -> Self {
        ProvePredicates {
            obligations: predicates
                .into_iter()
                .map(|p| Obligation::new(ObligationCause::dummy(), param_env, p))
                .collect(),
        }
    }
}

impl<'gcx, 'tcx> InfcxTypeOp<'gcx, 'tcx> for ProvePredicates<'tcx> {
    type Output = ();

    fn trivial_noop(self) -> Result<Self::Output, Self> {
        if self.obligations.is_empty() {
            Ok(())
        } else {
            Err(self)
        }
    }

    fn perform(self, _infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        Ok(InferOk {
            value: (),
            obligations: self.obligations,
        })
    }
}

pub(super) struct Normalize<'tcx, T> {
    param_env: ParamEnv<'tcx>,
    value: T,
}

impl<'tcx, T> Normalize<'tcx, T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    pub(super) fn new(param_env: ParamEnv<'tcx>, value: T) -> Self {
        Self { param_env, value }
    }
}

impl<'gcx, 'tcx, T> InfcxTypeOp<'gcx, 'tcx> for Normalize<'tcx, T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    type Output = T;

    fn trivial_noop(self) -> Result<Self::Output, Self> {
        if !self.value.has_projections() {
            Ok(self.value)
        } else {
            Err(self)
        }
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        let Normalized { value, obligations } = infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .normalize(&self.value)
            .unwrap_or_else(|NoSolution| {
                bug!(
                    "normalization of `{:?}` failed",
                    self.value,
                );
            });
        Ok(InferOk { value, obligations })
    }
}
