use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};

// Even if they're zero-cost at runtime, everything has *some* cost to inline
// in terms of copying them into the MIR caller, processing them in codegen, etc.
// These baseline costs give a simple usually-too-low estimate of the cost,
// which will be updated afterwards to account for the "real" costs.
const STMT_BASELINE_COST: usize = 1;
const BLOCK_BASELINE_COST: usize = 3;
const DEBUG_BASELINE_COST: usize = 1;
const LOCAL_BASELINE_COST: usize = 1;

// These penalties represent the cost above baseline for those things which
// have substantially more cost than is typical for their kind.
const CALL_PENALTY: usize = 22;
const LANDINGPAD_PENALTY: usize = 47;
const RESUME_PENALTY: usize = 42;
const DEREF_PENALTY: usize = 4;
const CHECKED_OP_PENALTY: usize = 2;
const THREAD_LOCAL_PENALTY: usize = 20;
const SMALL_SWITCH_PENALTY: usize = 3;
const LARGE_SWITCH_PENALTY: usize = 20;

// Passing arguments isn't free, so give a bonus to functions with lots of them:
// if the body is small despite lots of arguments, some are probably unused.
const EXTRA_ARG_BONUS: usize = 4;
const MAX_ARG_BONUS: usize = CALL_PENALTY;

/// Verify that the callee body is compatible with the caller.
#[derive(Clone)]
pub(crate) struct CostChecker<'b, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    cost: usize,
    callee_body: &'b Body<'tcx>,
    instance: Option<ty::Instance<'tcx>>,
}

impl<'b, 'tcx> CostChecker<'b, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        instance: Option<ty::Instance<'tcx>>,
        callee_body: &'b Body<'tcx>,
    ) -> CostChecker<'b, 'tcx> {
        CostChecker { tcx, param_env, callee_body, instance, cost: 0 }
    }

    // `Inline` doesn't call `visit_body`, so this is separate from the visitor.
    pub fn before_body(&mut self, body: &Body<'tcx>) {
        self.cost += BLOCK_BASELINE_COST * body.basic_blocks.len();
        self.cost += DEBUG_BASELINE_COST * body.var_debug_info.len();
        self.cost += LOCAL_BASELINE_COST * body.local_decls.len();

        let total_statements = body.basic_blocks.iter().map(|x| x.statements.len()).sum::<usize>();
        self.cost += STMT_BASELINE_COST * total_statements;

        if let Some(extra_args) = body.arg_count.checked_sub(2) {
            self.cost = self.cost.saturating_sub((EXTRA_ARG_BONUS * extra_args).min(MAX_ARG_BONUS));
        }
    }

    pub fn cost(&self) -> usize {
        self.cost
    }

    fn instantiate_ty(&self, v: Ty<'tcx>) -> Ty<'tcx> {
        if let Some(instance) = self.instance {
            instance.instantiate_mir(self.tcx, ty::EarlyBinder::bind(&v))
        } else {
            v
        }
    }
}

impl<'tcx> Visitor<'tcx> for CostChecker<'_, 'tcx> {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(place_and_rvalue) => {
                if place_and_rvalue.0.is_indirect_first_projection() {
                    self.cost += DEREF_PENALTY;
                }
                self.visit_rvalue(&place_and_rvalue.1, location);
            }
            StatementKind::Intrinsic(intr) => match &**intr {
                NonDivergingIntrinsic::Assume(..) => {}
                NonDivergingIntrinsic::CopyNonOverlapping(_cno) => {
                    self.cost += CALL_PENALTY;
                }
            },
            StatementKind::FakeRead(..)
            | StatementKind::SetDiscriminant { .. }
            | StatementKind::StorageLive(_)
            | StatementKind::StorageDead(_)
            | StatementKind::Retag(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::Deinit(_)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => {
                // No extra cost for these
            }
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, _location: Location) {
        match rvalue {
            Rvalue::Use(operand) => {
                if let Some(place) = operand.place()
                    && place.is_indirect_first_projection()
                {
                    self.cost += DEREF_PENALTY;
                }
            }
            Rvalue::Repeat(_item, count) => {
                let count = count.try_to_target_usize(self.tcx).unwrap_or(u64::MAX);
                self.cost += (STMT_BASELINE_COST * count as usize).min(CALL_PENALTY);
            }
            Rvalue::Aggregate(_kind, fields) => {
                self.cost += STMT_BASELINE_COST * fields.len();
            }
            Rvalue::CheckedBinaryOp(..) => {
                self.cost += CHECKED_OP_PENALTY;
            }
            Rvalue::ThreadLocalRef(..) => {
                self.cost += THREAD_LOCAL_PENALTY;
            }
            Rvalue::Ref(..)
            | Rvalue::AddressOf(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::CopyForDeref(..) => {
                // No extra cost for these
            }
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, _: Location) {
        let tcx = self.tcx;
        match terminator.kind {
            TerminatorKind::Drop { ref place, unwind, .. } => {
                // If the place doesn't actually need dropping, treat it like a regular goto.
                let ty = self.instantiate_ty(place.ty(self.callee_body, tcx).ty);
                if ty.needs_drop(tcx, self.param_env) {
                    self.cost += CALL_PENALTY;
                    if let UnwindAction::Cleanup(_) = unwind {
                        self.cost += LANDINGPAD_PENALTY;
                    }
                }
            }
            TerminatorKind::Call { ref func, unwind, .. } => {
                if let Some(f) = func.constant()
                    && let fn_ty = self.instantiate_ty(f.ty())
                    && let ty::FnDef(def_id, _) = *fn_ty.kind()
                    && tcx.intrinsic(def_id).is_some()
                {
                    // Don't give intrinsics the extra penalty for calls
                } else {
                    self.cost += CALL_PENALTY;
                };
                if let UnwindAction::Cleanup(_) = unwind {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::SwitchInt { ref discr, ref targets } => {
                if let Operand::Constant(..) = discr {
                    // This'll be a goto once we're monomorphizing
                } else {
                    // 0/1/unreachable is extremely common (bool, Option, Result, ...)
                    // but once there's more this can be a fair bit of work.
                    self.cost += if targets.all_targets().len() <= 3 {
                        SMALL_SWITCH_PENALTY
                    } else {
                        LARGE_SWITCH_PENALTY
                    };
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
                if let UnwindAction::Cleanup(_) = unwind {
                    self.cost += LANDINGPAD_PENALTY;
                }
            }
            TerminatorKind::Goto { .. }
            | TerminatorKind::UnwindTerminate(..)
            | TerminatorKind::Return
            | TerminatorKind::Yield { .. }
            | TerminatorKind::CoroutineDrop
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::Unreachable => {
                // No extra cost for these
            }
        }
    }
}
