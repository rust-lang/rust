use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::{FoundRelationships, Ty, TyCtxt, TyVid};

use super::TraitEngine;
use super::{ChalkFulfillmentContext, FulfillmentContext};

use std::panic;

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
        let legacy = $legacy.$method($($args.clone()),*);
        let hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {
            // report nothing
        }));
        match panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let chalk = $chalk.$method($($args),*);
            if chalk != legacy {
                chalk.report(&legacy);
            }
        })) {
            Ok(()) => {},
            Err(_) => eprintln!("chalk panicked, rerun with `-Zchalk` if you are interested in the ICE itself."),
        }
        panic::set_hook(hook);
        legacy
    }};
}

trait ChalkDiff {
    fn report(&self, legacy: &Self);
}

impl<'tcx> ChalkDiff for Ty<'tcx> {
    fn report(&self, legacy: &Self) {
        eprintln!();
        eprintln!("chalk normalized projection to `{}`", self);
        eprintln!("but legacy mode normalized to  `{}`", legacy);
        eprintln!();
    }
}

impl ChalkDiff for () {
    fn report(&self, _: &Self) {
        unreachable!()
    }
}

impl<T: std::fmt::Debug + PartialEq> ChalkDiff for Vec<T> {
    fn report(&self, legacy: &Self) {
        for chalk in self {
            if !legacy.contains(chalk) {
                eprintln!();
                eprintln!("chalk yielded item that legacy mode did not return: {:#?}", chalk);
                eprintln!();
            }
        }
        for legacy in legacy {
            if !self.contains(legacy) {
                eprintln!();
                eprintln!("legacy mode yielded item that chalk did not return: {:#?}", legacy);
                eprintln!();
            }
        }
    }
}

impl ChalkDiff for &mut FxHashMap<TyVid, FoundRelationships> {
    fn report(&self, legacy: &Self) {
        for (key, chalk) in self.iter() {
            if let Some(legacy) = legacy.get(key) {
                if chalk != legacy {
                    eprintln!();
                    eprintln!("item for key {:?} differs between chalk and legacy mode:", key);
                    eprintln!("chalk:  {:?}", chalk);
                    eprintln!("legacy: {:?}", legacy);
                    eprintln!();
                }
            } else {
                eprintln!();
                eprintln!(
                    "chalk had entry for key {:?} that legacy mode did not return: {:#?}",
                    key, chalk
                );
                eprintln!();
            }
        }
        for (key, legacy) in legacy.iter() {
            if !self.contains_key(key) {
                eprintln!();
                eprintln!(
                    "legacy mode had entry for key {:?} that chalk did not return: {:#?}",
                    key, legacy
                );
                eprintln!();
            }
        }
    }
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

    fn relationships(&mut self) -> &mut FxHashMap<TyVid, FoundRelationships> {
        try_chalk!(self.chalk, self.legacy, relationships,)
    }
}
