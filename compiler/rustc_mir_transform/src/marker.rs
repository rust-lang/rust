use std::borrow::Cow;

use crate::MirPass;
use rustc_middle::mir::{Body, MirPhase};
use rustc_middle::ty::TyCtxt;

/// Changes the MIR phase without changing the MIR itself.
pub struct PhaseChange(pub MirPhase);

impl<'tcx> MirPass<'tcx> for PhaseChange {
    fn phase_change(&self) -> Option<MirPhase> {
        Some(self.0)
    }

    fn name(&self) -> Cow<'_, str> {
        Cow::from(format!("PhaseChange-{:?}", self.0))
    }

    fn run_pass(&self, _: TyCtxt<'tcx>, _body: &mut Body<'tcx>) {}
}
