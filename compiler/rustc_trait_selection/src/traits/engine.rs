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

impl<'tcx> TraitEngine<'tcx> for ChalkMigration<'tcx> {
    fn normalize_projection_type(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
        param_env: rustc_middle::ty::ParamEnv<'tcx>,
        projection_ty: rustc_middle::ty::ProjectionTy<'tcx>,
        cause: rustc_infer::traits::ObligationCause<'tcx>,
    ) -> Ty<'tcx> {
        let chalk =
            self.chalk.normalize_projection_type(infcx, param_env, projection_ty, cause.clone());
        let legacy = self.legacy.normalize_projection_type(infcx, param_env, projection_ty, cause);
        if chalk != legacy {
            warn!(
                ?chalk,
                ?legacy,
                "normalization yielded different types, falling back to legacy value"
            );
        }
        legacy
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
        obligation: rustc_infer::traits::PredicateObligation<'tcx>,
    ) {
        self.chalk.register_predicate_obligation(infcx, obligation.clone());
        self.legacy.register_predicate_obligation(infcx, obligation);
    }

    fn select_all_or_error(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
    ) -> Vec<rustc_infer::traits::FulfillmentError<'tcx>> {
        let chalk = self.chalk.select_all_or_error(infcx);
        let legacy = self.legacy.select_all_or_error(infcx);
        if chalk != legacy {
            warn!(
                ?chalk,
                ?legacy,
                "select_all_or_error yielded different errors, falling back to legacy value"
            );
        }
        legacy
    }

    fn select_where_possible(
        &mut self,
        infcx: &rustc_infer::infer::InferCtxt<'_, 'tcx>,
    ) -> Vec<rustc_infer::traits::FulfillmentError<'tcx>> {
        let chalk = self.chalk.select_where_possible(infcx);
        let legacy = self.legacy.select_where_possible(infcx);
        if chalk != legacy {
            warn!(
                ?chalk,
                ?legacy,
                "select_where_possible yielded different errors, falling back to legacy value"
            );
        }
        legacy
    }

    fn pending_obligations(&self) -> Vec<rustc_infer::traits::PredicateObligation<'tcx>> {
        let chalk = self.chalk.pending_obligations();
        let legacy = self.legacy.pending_obligations();
        if chalk != legacy {
            warn!(
                ?chalk,
                ?legacy,
                "pending_obligations yielded different errors, falling back to legacy value"
            );
        }
        legacy
    }

    fn relationships(
        &mut self,
    ) -> &mut rustc_data_structures::fx::FxHashMap<
        rustc_middle::ty::TyVid,
        rustc_middle::ty::FoundRelationships,
    > {
        let chalk = self.chalk.relationships();
        let legacy = self.legacy.relationships();
        if chalk.len() != legacy.len() || chalk.iter().all(|(k, v)| &legacy[k] == v) {
            warn!(
                ?chalk,
                ?legacy,
                "relationships yielded different errors, falling back to legacy value"
            );
        }
        legacy
    }
}
