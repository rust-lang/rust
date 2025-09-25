use genmc_sys::{ActionKind, ExecutionState};

use super::GenmcCtx;
use crate::{
    InterpCx, InterpResult, MiriMachine, TerminationInfo, ThreadId, interp_ok, throw_machine_stop,
};

impl GenmcCtx {
    pub(crate) fn schedule_thread<'tcx>(
        &self,
        ecx: &InterpCx<'tcx, MiriMachine<'tcx>>,
    ) -> InterpResult<'tcx, ThreadId> {
        let thread_manager = &ecx.machine.threads;
        let active_thread_id = thread_manager.active_thread();

        // Determine whether the next instruction in the current thread might be a load.
        // This is used for the "writes-first" scheduling in GenMC.
        // Scheduling writes before reads can be beneficial for verification performance.
        // `Load` is a safe default for the next instruction type if we cannot guarantee that it isn't a load.
        let curr_thread_next_instr_kind = if !thread_manager.active_thread_ref().is_enabled() {
            // The current thread can get blocked (e.g., due to a thread join, `Mutex::lock`, assume statement, ...), then we need to ask GenMC for another thread to schedule.
            // Most to all blocking operations have load semantics, since they wait on something to change in another thread,
            // e.g., a thread join waiting on another thread to finish (join loads the return value(s) of the other thread),
            // or a thread waiting for another thread to unlock a `Mutex`, which loads the mutex state (Locked, Unlocked).
            ActionKind::Load
        } else {
            // This thread is still enabled. If it executes a terminator next, we consider yielding,
            // but in all other cases we just keep running this thread since it never makes sense
            // to yield before a non-atomic operation.
            let Some(frame) = thread_manager.active_thread_stack().last() else {
                return interp_ok(active_thread_id);
            };
            let either::Either::Left(loc) = frame.current_loc() else {
                // We are unwinding, so the next step is definitely not atomic.
                return interp_ok(active_thread_id);
            };
            let basic_block = &frame.body().basic_blocks[loc.block];
            if let Some(_statement) = basic_block.statements.get(loc.statement_index) {
                // Statements can't be atomic.
                return interp_ok(active_thread_id);
            }

            // FIXME(genmc): determine terminator kind.
            ActionKind::Load
        };

        let thread_infos = self.exec_state.thread_id_manager.borrow();
        let genmc_tid = thread_infos.get_genmc_tid(active_thread_id);

        let mut mc = self.handle.borrow_mut();
        let pinned_mc = mc.as_mut().unwrap();
        let result = pinned_mc.schedule_next(genmc_tid, curr_thread_next_instr_kind);
        // Depending on the exec_state, we either schedule the given thread, or we are finished with this execution.
        match result.exec_state {
            ExecutionState::Ok => interp_ok(thread_infos.get_miri_tid(result.next_thread)),
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
            _ => unreachable!(),
        }
    }
}
