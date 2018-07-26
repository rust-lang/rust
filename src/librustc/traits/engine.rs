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
use super::{ObligationCause, PredicateObligation};

pub trait TraitEngine<'tcx>: 'tcx {
    fn normalize_projection_type(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        cause: ObligationCause<'tcx>,
    ) -> Ty<'tcx>;

    fn register_bound(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
        def_id: DefId,
        cause: ObligationCause<'tcx>,
    );

    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    );

    fn select_all_or_error(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>>;

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>>;
}

pub trait TraitEngineExt<'tcx> {
    fn register_predicate_obligations(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
    );
}

impl<T: ?Sized + TraitEngine<'tcx>> TraitEngineExt<'tcx> for T {
    fn register_predicate_obligations(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        obligations: impl IntoIterator<Item = PredicateObligation<'tcx>>,
    ) {
        for obligation in obligations {
            self.register_predicate_obligation(infcx, obligation);
        }
    }
}

impl dyn TraitEngine<'tcx> {
    pub fn new(_tcx: TyCtxt<'_, '_, 'tcx>) -> Box<Self> {
        Box::new(FulfillmentContext::new())
    }
}
