// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::InferCtxt;
use ty::{self, Ty, TyCtxt};
use hir::def_id::DefId;

use super::{FulfillmentContext, FulfillmentError};
use super::{ObligationCause, PredicateObligation, PendingPredicateObligation};

pub trait TraitEngine<'tcx> {
    /// "Normalize" a projection type `<SomeType as SomeTrait>::X` by
    /// creating a fresh type variable `$0` as well as a projection
    /// predicate `<SomeType as SomeTrait>::X == $0`. When the
    /// inference engine runs, it will attempt to find an impl of
    /// `SomeTrait` or a where clause that lets us unify `$0` with
    /// something concrete. If this fails, we'll unify `$0` with
    /// `projection_ty` again.
    fn normalize_projection_type<'a, 'gcx>(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        cause: ObligationCause<'tcx>,
    ) -> Ty<'tcx>;

    /// Requires that `ty` must implement the trait with `def_id` in
    /// the given environment. This trait must not have any type
    /// parameters (except for `Self`).
    fn register_bound<'a, 'gcx>(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        def_id: DefId,
        cause: ObligationCause<'tcx>,
    );

    fn register_predicate_obligation<'a, 'gcx>(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    );

    fn select_all_or_error<'a, 'gcx>(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    fn select_where_possible<'a, 'gcx>(
        &mut self,
        infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    fn pending_obligations(&self) -> Vec<PendingPredicateObligation<'tcx>>;
}

impl<'tcx> dyn TraitEngine<'tcx> {
    pub fn new(_tcx: TyCtxt<'_, '_, 'tcx>) -> Box<Self> {
        Box::new(FulfillmentContext::new())
    }
}
