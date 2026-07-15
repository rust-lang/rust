//! This file contains Miri's scheduler, which is in charge of picking the next thread to run when
//! the current thread blocks/yields. It also manages timeouts and runs an async event loop
//! to deal with host I/O. In GenMC mode, it handles delegating schedule control to GenMC.

use std::sync::atomic::Ordering;
use std::task::Poll;
use std::time::Duration;

use rand::seq::IteratorRandom;
use rustc_const_eval::CTRL_C_RECEIVED;
use rustc_index::Idx;
use rustc_span::DUMMY_SP;

use crate::shims::readiness::DelayedReadinessUpdates;
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum SchedulingAction {
    /// Execute step on the active thread.
    ExecuteStep,
    /// Wait for a bit, but at most as long as the duration specified.
    /// We wake up early if an I/O event happened.
    /// If the duration is [`None`], we sleep indefinitely. This is
    /// only allowed when isolation is disabled and there are threads waiting for I/O!
    SleepAndWaitForIo(Option<Duration>),
}

impl<'tcx> EvalContextPrivExt<'tcx> for MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: MiriInterpCxExt<'tcx> {
    #[inline]
    fn run_on_stack_empty(&mut self) -> InterpResult<'tcx, Poll<()>> {
        let this = self.eval_context_mut();
        let active_thread = this.active_thread_mut();
        active_thread.origin_span = DUMMY_SP; // reset, the old value no longer applied
        let mut callback = active_thread
            .on_stack_empty
            .take()
            .expect("`on_stack_empty` not set up, or already running");
        let res = callback(this)?;
        this.active_thread_mut().on_stack_empty = Some(callback);
        interp_ok(res)
    }

    /// Decide which action to take next and on which thread.
    ///
    /// The currently implemented scheduling policy is the one that is commonly
    /// used in stateless model checkers such as Loom: run the active thread as
    /// long as we can and switch only when we have to (the active thread was
    /// blocked, terminated, or has explicitly asked to be preempted).
    ///
    /// If GenMC mode is active, the scheduling is instead handled by GenMC.
    fn schedule(&mut self) -> InterpResult<'tcx, SchedulingAction> {
        let this = self.eval_context_mut();

        // In GenMC mode, we let GenMC do the scheduling.
        if this.machine.data_race.as_genmc_ref().is_some() {
            loop {
                let genmc_ctx = this.machine.data_race.as_genmc_ref().unwrap();
                let Some(next_thread_id) = genmc_ctx.schedule_thread(this)? else {
                    return interp_ok(SchedulingAction::ExecuteStep);
                };
                // If a thread is blocked on GenMC, we have to implicitly unblock it when it gets scheduled again.
                if this.machine.threads.thread_ref(next_thread_id).is_blocked_on(BlockReason::Genmc)
                {
                    info!(
                        "GenMC: scheduling blocked thread {next_thread_id:?}, so we unblock it now."
                    );
                    this.unblock_thread(next_thread_id, BlockReason::Genmc)?;
                }
                // The thread we just unblocked may have been blocked again during the unblocking callback.
                // In that case, we need to ask for a different thread to run next.
                let thread_manager = &mut this.machine.threads;
                if thread_manager.thread_ref(next_thread_id).is_enabled() {
                    // Set the new active thread.
                    thread_manager.set_active_thread(next_thread_id);
                    return interp_ok(SchedulingAction::ExecuteStep);
                }
            }
        }

        // We are not in GenMC mode, so we control the scheduling.
        let thread_manager = &this.machine.threads;
        // This thread and the program can keep going.
        if thread_manager.active_thread_ref().is_enabled() && !thread_manager.yield_active_thread {
            // The currently active thread is still enabled, just continue with it.
            return interp_ok(SchedulingAction::ExecuteStep);
        }

        // The active thread yielded or got terminated. Let's see if there are any I/O events
        // or timeouts to take care of.

        // There may be delayed readiness updates we have to process.
        DelayedReadinessUpdates::process(this)?;

        if this.machine.communicate() {
            // When isolation is disabled we need to check for events for threads
            // which are blocked on host I/O. Unlike the `poll_and_unblock` before
            // any foreign item, the call here is needed to ensure that threads which
            // are blocked on host I/O are woken up even if no shimmed functions are
            // executed afterwards.
            // We do this before running any other threads such that the threads
            // which received events are available for scheduling afterwards.

            // Perform a non-blocking poll for newly available I/O events from the OS.
            this.poll_and_unblock(Some(Duration::ZERO))?;
        }

        // We also check timeouts before running any other thread, to ensure that timeouts
        // "in the past" fire before any other thread can take an action. This ensures that for
        // `pthread_cond_timedwait`, "an error is returned if [...] the absolute time specified by
        // abstime has already been passed at the time of the call".
        // <https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_cond_timedwait.html>
        let potential_sleep_time = this.unblock_expired_deadlines()?;

        let thread_manager = &mut this.machine.threads;
        let rng = this.machine.rng.get_mut();

        // No callbacks immediately scheduled, pick a regular thread to execute.
        // The active thread blocked or yielded. So we go search for another enabled thread.
        // We build the list of threads by starting with the threads after the current one, followed by
        // the threads before the current one and then the current thread itself (i.e., this iterator acts
        // like `threads.rotate_left(self.active_thread.index() + 1)`. This ensures that if we pick the first
        // eligible thread, we do regular round-robin scheduling, and all threads get a chance to take a step.
        let mut threads_iter = thread_manager
            .all_threads()
            .skip(thread_manager.active_thread().index() + 1)
            .chain(thread_manager.all_threads().take(thread_manager.active_thread().index() + 1))
            .filter(|(_id, thread)| thread.is_enabled());
        // Pick a new thread, and switch to it.
        let new_thread = if thread_manager.fixed_scheduling() {
            let next = threads_iter.next();
            drop(threads_iter);
            next
        } else {
            threads_iter.choose(rng)
        };

        if let Some((id, _thread)) = new_thread {
            if thread_manager.active_thread() != id {
                thread_manager.set_active_thread(id);
            }
        }
        // This completes the `yield`, if any was requested.
        thread_manager.yield_active_thread = false;

        if thread_manager.active_thread_ref().is_enabled() {
            return interp_ok(SchedulingAction::ExecuteStep);
        }

        // We have not found a thread to execute.
        if this.machine.threads.all_threads().all(|(_id, thread)| thread.is_terminated()) {
            unreachable!("all threads terminated without the main thread terminating?!");
        } else if let Some(sleep_time) = potential_sleep_time {
            // All threads are currently blocked, but we have unexecuted
            // timeout_callbacks, which may unblock some of the threads. Hence,
            // sleep until the first callback.
            interp_ok(SchedulingAction::SleepAndWaitForIo(Some(sleep_time)))
        } else if this.any_thread_blocked_on_host() {
            // At least one thread doesn't have a timeout set, and is blocked on host I/O or is waiting on an
            // epoll instance which contains a host source interest. Hence, we sleep indefinitely in the
            // hope that eventually an I/O event happens.
            interp_ok(SchedulingAction::SleepAndWaitForIo(None))
        } else {
            throw_machine_stop!(TerminationInfo::GlobalDeadlock);
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Public because this is used by Priroda.
    fn step_current_thread(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        if !this.step()? {
            // See if this thread can do something else.
            match this.run_on_stack_empty()? {
                Poll::Pending => {} // keep going
                Poll::Ready(()) => {
                    this.terminate_active_thread(TlsAllocAction::Deallocate)?;
                }
            }
        }

        interp_ok(())
    }

    /// Run the core interpreter loop. Returns only when an interrupt occurs (an error or program
    /// termination).
    fn run_threads(&mut self) -> InterpResult<'tcx, !> {
        let this = self.eval_context_mut();
        loop {
            if CTRL_C_RECEIVED.load(Ordering::Relaxed) {
                this.machine.handle_abnormal_termination();
                throw_machine_stop!(TerminationInfo::Interrupted);
            }
            match this.schedule()? {
                SchedulingAction::ExecuteStep => {
                    this.step_current_thread()?;
                }
                SchedulingAction::SleepAndWaitForIo(duration) => {
                    if this.machine.communicate() {
                        // When we're running with isolation disabled, instead of
                        // strictly sleeping the duration we allow waking up
                        // early for I/O events from the OS.

                        this.poll_and_unblock(duration)?;
                    } else {
                        let duration = duration.expect(
                            "Infinite sleep should not be triggered when isolation is enabled",
                        );
                        this.machine.monotonic_clock.sleep(duration);
                    }
                }
            }
        }
    }
}
