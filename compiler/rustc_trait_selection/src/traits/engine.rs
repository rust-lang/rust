use rustc_middle::ty::{Ty, TyCtxt};

use super::TraitEngine;
use super::{ChalkFulfillmentContext, FulfillmentContext};

pub trait TraitEngineExt<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.debugging_opts.chalk_migration {
            Box::new(ChalkMigration {
                chalk: ChalkFulfillmentContext::new(),
                legacy: FulfillmentContext::new(),
            })
        } else if tcx.sess.opts.debugging_opts.chalk {
            Box::new(ChalkFulfillmentContext::new())
        } else {
            Box::new(FulfillmentContext::new())
        }
    }
}

struct ChalkMigration<'tcx> {
    chalk: ChalkFulfillmentContext<'tcx>,
    legacy: FulfillmentContext<'tcx>,
}

macro_rules! try_chalk {
    ($chalk:expr, $legacy:expr, $method:ident, $($args:expr),*) => {{
        let chalk = $chalk.$method($($args.clone()),*);
        let legacy = $legacy.$method($($args),*);
        if chalk != legacy {
            warn!(
                ?chalk,
                ?legacy,
                "`{}` yielded different results, falling back to legacy value",
                stringify!($method),
            );
        }
        legacy
    }};
}

impl<'tcx> TraitEngine<'tcx> for ChalkMigration<'tcx> {
    fn normalize_projection_type(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
        param_env: rustc_middle::ty::ParamEnv<'tcx>,
        projection_ty: rustc_middle::ty::ProjectionTy<'tcx>,
        cause: rustc_infer::traits::ObligationCause<'tcx>,
    ) -> Ty<'tcx> {
        try_chalk!(
            self.chalk,
            self.legacy,
            normalize_projection_type,
            infcx,
            param_env,
            projection_ty,
            cause
        )
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
        obligation: rustc_infer::traits::PredicateObligation<'tcx>,
    ) {
        try_chalk!(self.chalk, self.legacy, register_predicate_obligation, infcx, obligation)
    }

    fn select_all_or_error(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
    ) -> Vec<rustc_infer::traits::FulfillmentError<'tcx>> {
        try_chalk!(self.chalk, self.legacy, select_all_or_error, infcx)
    }

    fn select_where_possible(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
    ) -> Vec<rustc_infer::traits::FulfillmentError<'tcx>> {
        try_chalk!(self.chalk, self.legacy, select_where_possible, infcx)
    }

    fn pending_obligations(&self) -> Vec<rustc_infer::traits::PredicateObligation<'tcx>> {
        try_chalk!(self.chalk, self.legacy, pending_obligations,)
    }

    fn relationships(
        &mut self,
    ) -> &mut rustc_data_structures::fx::FxHashMap<
        rustc_middle::ty::TyVid,
        rustc_middle::ty::FoundRelationships,
    > {
        try_chalk!(self.chalk, self.legacy, relationships,)
    }
}
