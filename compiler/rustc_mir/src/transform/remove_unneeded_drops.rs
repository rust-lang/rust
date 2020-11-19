//! This pass replaces a drop of a type that does not need dropping, with a goto

use crate::transform::MirPass;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, TyCtxt};

use super::simplify::simplify_cfg;

pub struct RemoveUnneededDrops;

impl<'tcx> MirPass<'tcx> for RemoveUnneededDrops {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running RemoveUnneededDrops on {:?}", body.source);
        let mut opt_finder = RemoveUnneededDropsOptimizationFinder {
            tcx,
            body,
            param_env: tcx.param_env(body.source.def_id()),
            optimizations: vec![],
        };
        opt_finder.visit_body(body);
        let should_simplify = !opt_finder.optimizations.is_empty();
        for (loc, target) in opt_finder.optimizations {
            if !tcx
                .consider_optimizing(|| format!("RemoveUnneededDrops {:?} ", body.source.def_id()))
            {
                break;
            }

            let terminator = body.basic_blocks_mut()[loc.block].terminator_mut();
            debug!("SUCCESS: replacing `drop` with goto({:?})", target);
            terminator.kind = TerminatorKind::Goto { target };
        }

        // if we applied optimizations, we potentially have some cfg to cleanup to
        // make it easier for further passes
        if should_simplify {
            simplify_cfg(body);
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for RemoveUnneededDropsOptimizationFinder<'a, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match terminator.kind {
            TerminatorKind::Drop { place, target, .. } => {
                let ty = place.ty(self.body, self.tcx);
                let needs_drop = ty.ty.needs_drop(self.tcx, self.param_env);
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
    param_env: ParamEnv<'tcx>,
}
