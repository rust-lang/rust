//! This pass replaces a drop of a type that does not need dropping, with a goto

use crate::transform::{MirPass, MirSource};
use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct RemoveUnneededDrops {
    def_id: LocalDefId,
}

impl RemoveUnneededDrops {
    pub fn new(def_id: LocalDefId) -> Self {
        Self { def_id }
    }
}

impl<'tcx> MirPass<'tcx> for RemoveUnneededDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running RemoveUnneededDrops on {:?}", source);
        let mut opt_finder = RemoveUnneededDropsOptimizationFinder {
            tcx,
            body,
            optimizations: vec![],
            def_id: self.def_id,
        };
        opt_finder.visit_body(body);
        for (loc, target) in opt_finder.optimizations {
            let terminator = body.basic_blocks_mut()[loc.block].terminator_mut();
            debug!("SUCCESS: replacing `drop` with goto({:?})", target);
            terminator.kind = TerminatorKind::Goto { target };
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RemoveUnneededDropsOptimizationFinder<'a, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match terminator.kind {
            TerminatorKind::Drop { place, target, .. } => {
                let ty = place.ty(self.body, self.tcx);
                let needs_drop = ty.ty.needs_drop(self.tcx, self.tcx.param_env(self.def_id));
                if !needs_drop {
                    self.optimizations.push((location, target));
                }
            }
            _ => {}
        }
        self.super_terminator(terminator, location);
    }
}
pub struct RemoveUnneededDropsOptimizationFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    optimizations: Vec<(Location, BasicBlock)>,
    def_id: LocalDefId,
}
