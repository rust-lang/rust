use genmc_sys::{ActionKind, ExecutionState};
use rustc_data_structures::either::Either;
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::{self, Ty};

use super::GenmcCtx;
use crate::{
    InterpCx, InterpResult, MiriMachine, TerminationInfo, ThreadId, interp_ok, throw_machine_stop,
};

enum NextInstrKind {
    MaybeAtomic(ActionKind),
    NonAtomic,
}

/// Check if a call or tail-call could have atomic load semantics.
fn get_next_instruction_kind<'tcx>(
    ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
) -> InterpResult<'tcx, NextInstrKind> {
    use NextInstrKind::*;

    let thread_manager = &ecx.machine.threads;

    // Determine whether the next instruction in the current thread might be a load.
    // This is used for the "writes-first" scheduling in GenMC.
    // Scheduling writes before reads can be beneficial for verification performance.
    // `Load` is a safe default for the next instruction type if we cannot guarantee that it isn't a load.
    if !thread_manager.active_thread_ref().is_enabled() {
        // The current thread can get blocked (e.g., due to a thread join, `Mutex::lock`, assume statement, ...), then we need to ask GenMC for another thread to schedule.
        // Most to all blocking operations have load semantics, since they wait on something to change in another thread,
        // e.g., a thread join waiting on another thread to finish (join loads the return value(s) of the other thread),
        // or a thread waiting for another thread to unlock a `Mutex`, which loads the mutex state (Locked, Unlocked).
        // `Load` is a safe default for the next instruction type, since we may not know what the next instruction is.
        return interp_ok(MaybeAtomic(ActionKind::Load));
    }
    // This thread is still enabled. If it executes a terminator next, we consider yielding,
    // but in all other cases we just keep running this thread since it never makes sense
    // to yield before a non-atomic operation.
    let Some(frame) = thread_manager.active_thread_stack().last() else {
        return interp_ok(NonAtomic);
    };
    let Either::Left(loc) = frame.current_loc() else {
        // We are unwinding, so the next step is definitely not atomic.
        return interp_ok(NonAtomic);
    };
    let basic_block = &frame.body().basic_blocks[loc.block];
    if let Some(_statement) = basic_block.statements.get(loc.statement_index) {
        // Statements can't be atomic.
        return interp_ok(NonAtomic);
    }
    match &basic_block.terminator().kind {
        // All atomics are modeled as function calls to intrinsic functions.
        // The one exception is thread joining, but those are also calls.
        TerminatorKind::Call { func, .. } | TerminatorKind::TailCall { func, .. } =>
            get_function_kind(ecx, func.ty(&frame.body().local_decls, *ecx.tcx)),
        // Non-call terminators are not atomic.
        _ => interp_ok(NonAtomic),
    }
}

fn get_function_kind<'tcx>(
    ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    func_ty: Ty<'tcx>,
) -> InterpResult<'tcx, NextInstrKind> {
    use NextInstrKind::*;
    let callee_def_id = match func_ty.kind() {
        ty::FnDef(def_id, _args) => *def_id,
        _ => return interp_ok(MaybeAtomic(ActionKind::Load)), // we don't know the callee, might be pthread_join
    };
    let Some(intrinsic_def) = ecx.tcx.intrinsic(callee_def_id) else {
        if ecx.tcx.is_foreign_item(callee_def_id) {
            // Some shims, like pthread_join, must be considered loads. So just consider them all loads,
            // these calls are not *that* common.
            return interp_ok(MaybeAtomic(ActionKind::Load));
        }
        // NOTE: Functions intercepted by Miri in `concurrency/genmc/intercep.rs` must also be added here.
        // Such intercepted functions, like `sys::Mutex::lock`, should be treated as atomics to ensure we call the scheduler when we encounter one of them.
        // These functions must also be classified whether they may have load semantics.
        if ecx.tcx.is_diagnostic_item(rustc_span::sym::sys_mutex_lock, callee_def_id)
            || ecx.tcx.is_diagnostic_item(rustc_span::sym::sys_mutex_try_lock, callee_def_id)
        {
            return interp_ok(MaybeAtomic(ActionKind::Load));
        } else if ecx.tcx.is_diagnostic_item(rustc_span::sym::sys_mutex_unlock, callee_def_id) {
            return interp_ok(MaybeAtomic(ActionKind::NonLoad));
        }
        // The next step is a call to a regular Rust function.
        return interp_ok(NonAtomic);
    };
    let intrinsic_name = intrinsic_def.name.as_str();
    let Some(suffix) = intrinsic_name.strip_prefix("atomic_") else {
        return interp_ok(NonAtomic); // Non-atomic intrinsic, so guaranteed not an atomic load
    };
    // `atomic_store`, `atomic_fence` and `atomic_singlethreadfence` are not considered loads.
    // Any future `atomic_*` intrinsics may have load semantics, so we err on the side of caution and classify them as "maybe loads".
    interp_ok(MaybeAtomic(if matches!(suffix, "store" | "fence" | "singlethreadfence") {
        ActionKind::NonLoad
    } else {
        ActionKind::Load
    }))
}

impl GenmcCtx {
    /// Returns the thread ID of the next thread to schedule, or `None` to continue with the current thread.
    pub(crate) fn schedule_thread<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> InterpResult<'tcx, Option<ThreadId>> {
        let atomic_kind = match get_next_instruction_kind(ecx)? {
            NextInstrKind::MaybeAtomic(atomic_kind) => atomic_kind,
            NextInstrKind::NonAtomic => return interp_ok(None), // No need to reschedule on a non-atomic.
        };

        let active_thread_id = ecx.machine.threads.active_thread();
        let thread_infos = self.exec_state.thread_id_manager.borrow();
        let genmc_tid = thread_infos.get_genmc_tid(active_thread_id);

        let result = self.handle.borrow_mut().pin_mut().schedule_next(genmc_tid, atomic_kind);
        // Depending on the exec_state, we either schedule the given thread, or we are finished with this execution.
        match result.exec_state {
            ExecutionState::Ok => interp_ok(Some(thread_infos.get_miri_tid(result.next_thread))),
            ExecutionState::Blocked => throw_machine_stop!(TerminationInfo::GenmcBlockedExecution),
            ExecutionState::Finished => {
                let exit_status = self.exec_state.exit_status.get().expect(
                    "If the execution is finished, we should have a return value from the program.",
                );
                throw_machine_stop!(TerminationInfo::Exit {
                    code: exit_status.exit_code,
                    leak_check: exit_status.do_leak_check()
                });
            }
            ExecutionState::Error => {
                // GenMC found an error in one of the `handle_*` functions, but didn't return the detected error from the function immediately.
                // This is still an bug in the user program, so we print the error string.
                panic!(
                    "GenMC found an error ({:?}), but didn't report it immediately, so we cannot provide an appropriate source code location for where it happened.",
                    self.try_get_error().unwrap()
                );
            }
            _ => unreachable!(),
        }
    }
}
