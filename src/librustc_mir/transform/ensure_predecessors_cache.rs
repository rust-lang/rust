use rustc::mir::Body;
use rustc::ty::TyCtxt;
use crate::transform::{MirPass, MirSource};

pub struct EnsurePredecessorsCache {
    label: String,
}

impl EnsurePredecessorsCache {
    pub fn new<S: Into<String>>(label: S) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl<'tcx> MirPass<'tcx> for EnsurePredecessorsCache {
    fn run_pass(&self, _: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        debug!("{}: Ensure predecessors cache: {:?}", self.label, body.span.data());
        // predecessors is lazily calculated. We want to ensure that the cache is properly filled
        // before the next stages of compilation, since these following stages will only be allowed
        // to read the cache and not generate it. If the cache is already up to date, this line is
        // a nop.
        body.ensure_predecessors();
    }
}
