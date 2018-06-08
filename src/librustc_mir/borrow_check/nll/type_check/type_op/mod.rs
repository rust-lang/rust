// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::canonical::query_result;
use rustc::infer::canonical::QueryRegionConstraint;
use rustc::infer::{InferCtxt, InferOk, InferResult};
use rustc::traits::query::dropck_outlives::trivial_dropck_outlives;
use rustc::traits::query::NoSolution;
use rustc::traits::{Normalized, Obligation, ObligationCause, PredicateObligation, TraitEngine};
use rustc::ty::error::TypeError;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::Kind;
use rustc::ty::{ParamEnv, Predicate, Ty, TyCtxt};
use std::fmt;
use std::rc::Rc;
use syntax::codemap::DUMMY_SP;

pub(super) trait TypeOp<'gcx, 'tcx>: Sized + fmt::Debug {
    type Output;

    /// Micro-optimization: returns `Ok(x)` if we can trivially
    /// produce the output, else returns `Err(self)` back.
    fn trivial_noop(self, tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self>;

    /// Given an infcx, performs **the kernel** of the operation: this does the
    /// key action and then, optionally, returns a set of obligations which must be proven.
    ///
    /// This method is not meant to be invoked directly: instead, one
    /// should use `fully_perform`, which will take those resulting
    /// obligations and prove them, and then process the combined
    /// results into region obligations which are returned.
    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output>;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<(Self::Output, Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>), TypeError<'tcx>> {
        match self.trivial_noop(infcx.tcx) {
            Ok(r) => Ok((r, None)),
            Err(op) => op.fully_perform_nontrivial(infcx),
        }
    }

    /// Helper for `fully_perform` that handles the nontrivial cases.
    #[inline(never)] // just to help with profiling
    fn fully_perform_nontrivial(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<(Self::Output, Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>), TypeError<'tcx>> {
        if cfg!(debug_assertions) {
            info!("fully_perform_op_and_get_region_constraint_data({:?})", self);
        }

        let mut fulfill_cx = TraitEngine::new(infcx.tcx);
        let dummy_body_id = ObligationCause::dummy().body_id;
        let InferOk { value, obligations } = infcx.commit_if_ok(|_| self.perform(infcx))?;
        debug_assert!(obligations.iter().all(|o| o.cause.body_id == dummy_body_id));
        fulfill_cx.register_predicate_obligations(infcx, obligations);
        if let Err(e) = fulfill_cx.select_all_or_error(infcx) {
            infcx.tcx.sess.diagnostic().delay_span_bug(
                DUMMY_SP,
                &format!("errors selecting obligation during MIR typeck: {:?}", e),
            );
        }

        let region_obligations = infcx.take_registered_region_obligations();

        let region_constraint_data = infcx.take_and_reset_region_constraints();

        let outlives = query_result::make_query_outlives(
            infcx.tcx,
            region_obligations,
            &region_constraint_data,
        );

        if outlives.is_empty() {
            Ok((value, None))
        } else {
            Ok((value, Some(Rc::new(outlives))))
        }
    }
}

pub(super) struct CustomTypeOp<F, G> {
    closure: F,
    description: G,
}

impl<F, G> CustomTypeOp<F, G> {
    pub(super) fn new<'gcx, 'tcx, R>(closure: F, description: G) -> Self
    where
        F: FnOnce(&InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R>,
        G: Fn() -> String,
    {
        CustomTypeOp {
            closure,
            description,
        }
    }
}

impl<'gcx, 'tcx, F, R, G> TypeOp<'gcx, 'tcx> for CustomTypeOp<F, G>
where
    F: FnOnce(&InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R>,
    G: Fn() -> String,
{
    type Output = R;

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        Err(self)
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, R> {
        (self.closure)(infcx)
    }
}

impl<F, G> fmt::Debug for CustomTypeOp<F, G>
where
    G: Fn() -> String,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", (self.description)())
    }
}

#[derive(Debug)]
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

impl<'gcx, 'tcx> TypeOp<'gcx, 'tcx> for Subtype<'tcx> {
    type Output = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
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

#[derive(Debug)]
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

impl<'gcx, 'tcx> TypeOp<'gcx, 'tcx> for Eq<'tcx> {
    type Output = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
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

#[derive(Debug)]
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

impl<'gcx, 'tcx> TypeOp<'gcx, 'tcx> for ProvePredicates<'tcx> {
    type Output = ();

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
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

#[derive(Debug)]
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

impl<'gcx, 'tcx, T> TypeOp<'gcx, 'tcx> for Normalize<'tcx, T>
where
    T: fmt::Debug + TypeFoldable<'tcx>,
{
    type Output = T;

    fn trivial_noop(self, _tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
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
                bug!("normalization of `{:?}` failed", self.value,);
            });
        Ok(InferOk { value, obligations })
    }
}

#[derive(Debug)]
pub(super) struct DropckOutlives<'tcx> {
    param_env: ParamEnv<'tcx>,
    dropped_ty: Ty<'tcx>,
}

impl<'tcx> DropckOutlives<'tcx> {
    pub(super) fn new(param_env: ParamEnv<'tcx>, dropped_ty: Ty<'tcx>) -> Self {
        DropckOutlives {
            param_env,
            dropped_ty,
        }
    }
}

impl<'gcx, 'tcx> TypeOp<'gcx, 'tcx> for DropckOutlives<'tcx> {
    type Output = Vec<Kind<'tcx>>;

    fn trivial_noop(self, tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        if trivial_dropck_outlives(tcx, self.dropped_ty) {
            Ok(vec![])
        } else {
            Err(self)
        }
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        Ok(infcx
            .at(&ObligationCause::dummy(), self.param_env)
            .dropck_outlives(self.dropped_ty))
    }
}
