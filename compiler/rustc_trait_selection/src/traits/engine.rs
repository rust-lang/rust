use rustc_middle::ty::TyCtxt;

use super::TraitEngine;
use super::{ChalkFulfillmentContext, FulfillmentContext};

pub trait TraitEngineExt<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self> {
        if tcx.sess.opts.debugging_opts.chalk {
            Box::new(ChalkFulfillmentContext::new())
        } else {
            Box::new(FulfillmentContext::new())
        }
    }
}
