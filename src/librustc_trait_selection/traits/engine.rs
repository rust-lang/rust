use rustc::ty::TyCtxt;

use super::FulfillmentContext;
use super::TraitEngine;

pub trait TraitEngineExt<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Box<Self>;
}

impl<'tcx> TraitEngineExt<'tcx> for dyn TraitEngine<'tcx> {
    fn new(_tcx: TyCtxt<'tcx>) -> Box<Self> {
        Box::new(FulfillmentContext::new())
    }
}
