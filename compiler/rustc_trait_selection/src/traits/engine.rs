use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::TyCtxt;

use super::TraitEngine;
use super::{ChalkFulfillmentContext, FulfillmentContext};

pub trait TraitEngineExt<'tcx>: TraitEngine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self>;
    fn new_with_deregister<'cx>(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
    ) -> DeregisterOnDropEngine<'cx, 'tcx, Box<Self>>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> + 'tcx {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.debugging_opts.chalk {
            Box::new(ChalkFulfillmentContext::new())
        } else {
            Box::new(FulfillmentContext::new())
        }
    }
    fn new_with_deregister<'cx>(
        infcx: &'cx InferCtxt<'cx, 'tcx>,
    ) -> DeregisterOnDropEngine<'cx, 'tcx, Box<Self>> {
        DeregisterOnDropEngine { engine: Self::new(infcx.tcx), infcx }
    }
}

/// Deregisters any variable watches on drop automatically
pub struct DeregisterOnDropEngine<'cx, 'tcx, T>
where
    T: TraitEngine<'tcx>,
{
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    engine: T,
}

impl<'tcx, T> Drop for DeregisterOnDropEngine<'_, 'tcx, T>
where
    T: TraitEngine<'tcx>,
{
    fn drop(&mut self) {
        self.engine.deregister(self.infcx)
    }
}

impl<'tcx, T> std::ops::Deref for DeregisterOnDropEngine<'_, 'tcx, T>
where
    T: TraitEngine<'tcx>,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl<'tcx, T> std::ops::DerefMut for DeregisterOnDropEngine<'_, 'tcx, T>
where
    T: TraitEngine<'tcx>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}
