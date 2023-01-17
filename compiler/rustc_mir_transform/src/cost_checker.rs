use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, ParamEnv, TyCtxt};

const INSTR_COST: usize = 5;
const CALL_PENALTY: usize = 25;
const LANDINGPAD_PENALTY: usize = 50;
const RESUME_PENALTY: usize = 45;

/// Verify that the callee body is compatible with the caller.
#[derive(Clone)]
pub(crate) struct CostChecker<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    cost: usize,
    callee_body: &'b Body<'tcx>,
    instance: ty::Instance<'tcx>,
}

impl<'b, 'tcx> CostChecker<'b, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        instance: ty::Instance<'tcx>,
        callee_body: &'b Body<'tcx>,
    ) -> CostChecker<'b, 'tcx> {
        CostChecker { tcx, param_env, callee_body, instance, cost: 0 }
    }

    pub fn cost(&self) -> usize {
        self.cost
    }
}

impl<'tcx> Visitor<'tcx> for CostChecker<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, _: Location) {
        // Don't count StorageLive/StorageDead in the inlining cost.
        match statement.kind {
            StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Deinit(_)
            | StatementKind::Nop => {}
            _ => self.cost += INSTR_COST,
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, _: Location) {
        let tcx = self.tcx;
        match terminator.kind {
            TerminatorKind::Drop { ref place, unwind, .. } => {
                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = self.instance.instantiate_mir(
                    tcx,
                    ty::EarlyBinder::bind(&place.ty(self.callee_body, tcx).ty),
                );
                if ty.needs_drop(tcx, self.param_env) {
                    self.cost += CALL_PENALTY;
                    if let UnwindAction::Cleanup(_) = unwind {
                        self.cost += LANDINGPAD_PENALTY;
                    }
                } else {
                    self.cost += INSTR_COST;
                }
            }
            TerminatorKind::Call { func: Operand::Constant(ref f), unwind, .. } => {
                let fn_ty =
                    self.instance.instantiate_mir(tcx, ty::EarlyBinder::bind(&f.const_.ty()));
                self.cost += if let ty::FnDef(def_id, _) = *fn_ty.kind() && tcx.is_intrinsic(def_id) {
                    // Don't give intrinsics the extra penalty for calls
                    INSTR_COST
                } else {
                    CALL_PENALTY
                };
                if let UnwindAction::Cleanup(_) = unwind {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Assert { unwind, .. } => {
                self.cost += CALL_PENALTY;
                if let UnwindAction::Cleanup(_) = unwind {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::UnwindResume => self.cost += RESUME_PENALTY,
            TerminatorKind::InlineAsm { unwind, .. } => {
                self.cost += INSTR_COST;
                if let UnwindAction::Cleanup(_) = unwind {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            _ => self.cost += INSTR_COST,
        }
    }
}
