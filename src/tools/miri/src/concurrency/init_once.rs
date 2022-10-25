use std::collections::VecDeque;
use std::num::NonZeroU32;

use rustc_index::vec::Idx;

use super::sync::EvalContextExtPriv;
use super::thread::MachineCallback;
use super::vector_clock::VClock;
use crate::*;

declare_id!(InitOnceId);

/// A thread waiting on an InitOnce object.
struct InitOnceWaiter<'mir, 'tcx> {
    /// The thread that is waiting.
    thread: ThreadId,
    /// The callback that should be executed, after the thread has been woken up.
    callback: Box<dyn MachineCallback<'mir, 'tcx> + 'tcx>,
}

impl<'mir, 'tcx> std::fmt::Debug for InitOnceWaiter<'mir, 'tcx> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitOnce")
            .field("thread", &self.thread)
            .field("callback", &"dyn MachineCallback")
            .finish()
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
/// The current status of a one time initialization.
pub enum InitOnceStatus {
    #[default]
    Uninitialized,
    Begun,
    Complete,
}

/// The one time initialization state.
#[derive(Default, Debug)]
pub(super) struct InitOnce<'mir, 'tcx> {
    status: InitOnceStatus,
    waiters: VecDeque<InitOnceWaiter<'mir, 'tcx>>,
    data_race: VClock,
}

impl<'mir, 'tcx> VisitTags for InitOnce<'mir, 'tcx> {
    fn visit_tags(&self, visit: &mut dyn FnMut(SbTag)) {
        for waiter in self.waiters.iter() {
            waiter.callback.visit_tags(visit);
        }
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn init_once_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
        offset: u64,
    ) -> InterpResult<'tcx, InitOnceId> {
        let this = self.eval_context_mut();
        this.init_once_get_or_create(|ecx, next_id| ecx.get_or_create_id(next_id, lock_op, offset))
    }

    /// Provides the closure with the next InitOnceId. Creates that InitOnce if the closure returns None,
    /// otherwise returns the value from the closure.
    #[inline]
    fn init_once_get_or_create<F>(&mut self, existing: F) -> InterpResult<'tcx, InitOnceId>
    where
        F: FnOnce(
            &mut MiriInterpCx<'mir, 'tcx>,
            InitOnceId,
        ) -> InterpResult<'tcx, Option<InitOnceId>>,
    {
        let this = self.eval_context_mut();
        let next_index = this.machine.threads.sync.init_onces.next_index();
        if let Some(old) = existing(this, next_index)? {
            Ok(old)
        } else {
            let new_index = this.machine.threads.sync.init_onces.push(Default::default());
            assert_eq!(next_index, new_index);
            Ok(new_index)
        }
    }

    #[inline]
    fn init_once_status(&mut self, id: InitOnceId) -> InitOnceStatus {
        let this = self.eval_context_ref();
        this.machine.threads.sync.init_onces[id].status
    }

    /// Put the thread into the queue waiting for the initialization.
    #[inline]
    fn init_once_enqueue_and_block(
        &mut self,
        id: InitOnceId,
        thread: ThreadId,
        callback: Box<dyn MachineCallback<'mir, 'tcx> + 'tcx>,
    ) {
        let this = self.eval_context_mut();
        let init_once = &mut this.machine.threads.sync.init_onces[id];
        assert_ne!(init_once.status, InitOnceStatus::Complete, "queueing on complete init once");
        init_once.waiters.push_back(InitOnceWaiter { thread, callback });
        this.block_thread(thread);
    }

    /// Begin initializing this InitOnce. Must only be called after checking that it is currently
    /// uninitialized.
    #[inline]
    fn init_once_begin(&mut self, id: InitOnceId) {
        let this = self.eval_context_mut();
        let init_once = &mut this.machine.threads.sync.init_onces[id];
        assert_eq!(
            init_once.status,
            InitOnceStatus::Uninitialized,
            "begining already begun or complete init once"
        );
        init_once.status = InitOnceStatus::Begun;
    }

    #[inline]
    fn init_once_complete(&mut self, id: InitOnceId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let current_thread = this.get_active_thread();
        let init_once = &mut this.machine.threads.sync.init_onces[id];

        assert_eq!(
            init_once.status,
            InitOnceStatus::Begun,
            "completing already complete or uninit init once"
        );

        init_once.status = InitOnceStatus::Complete;

        // Each complete happens-before the end of the wait
        if let Some(data_race) = &this.machine.data_race {
            data_race.validate_lock_release(&mut init_once.data_race, current_thread);
        }

        // Wake up everyone.
        // need to take the queue to avoid having `this` be borrowed multiple times
        for waiter in std::mem::take(&mut init_once.waiters) {
            // End of the wait happens-before woken-up thread.
            if let Some(data_race) = &this.machine.data_race {
                data_race.validate_lock_acquire(
                    &this.machine.threads.sync.init_onces[id].data_race,
                    waiter.thread,
                );
            }

            this.unblock_thread(waiter.thread);

            // Call callback, with the woken-up thread as `current`.
            this.set_active_thread(waiter.thread);
            waiter.callback.call(this)?;
            this.set_active_thread(current_thread);
        }

        Ok(())
    }

    #[inline]
    fn init_once_fail(&mut self, id: InitOnceId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let current_thread = this.get_active_thread();
        let init_once = &mut this.machine.threads.sync.init_onces[id];
        assert_eq!(
            init_once.status,
            InitOnceStatus::Begun,
            "failing already completed or uninit init once"
        );

        // Each complete happens-before the end of the wait
        // FIXME: should this really induce synchronization? If we think of it as a lock, then yes,
        // but the docs don't talk about such details.
        if let Some(data_race) = &this.machine.data_race {
            data_race.validate_lock_release(&mut init_once.data_race, current_thread);
        }

        // Wake up one waiting thread, so they can go ahead and try to init this.
        if let Some(waiter) = init_once.waiters.pop_front() {
            // End of the wait happens-before woken-up thread.
            if let Some(data_race) = &this.machine.data_race {
                data_race.validate_lock_acquire(
                    &this.machine.threads.sync.init_onces[id].data_race,
                    waiter.thread,
                );
            }

            this.unblock_thread(waiter.thread);

            // Call callback, with the woken-up thread as `current`.
            this.set_active_thread(waiter.thread);
            waiter.callback.call(this)?;
            this.set_active_thread(current_thread);
        } else {
            // Nobody there to take this, so go back to 'uninit'
            init_once.status = InitOnceStatus::Uninitialized;
        }

        Ok(())
    }
}
