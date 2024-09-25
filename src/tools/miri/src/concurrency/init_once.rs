use std::collections::VecDeque;

use rustc_index::Idx;

use super::sync::EvalContextExtPriv as _;
use super::vector_clock::VClock;
use crate::*;

super::sync::declare_id!(InitOnceId);

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
pub(super) struct InitOnce {
    status: InitOnceStatus,
    waiters: VecDeque<ThreadId>,
    clock: VClock,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn init_once_get_or_create_id(
        &mut self,
        lock: &MPlaceTy<'tcx>,
        offset: u64,
    ) -> InterpResult<'tcx, InitOnceId> {
        let this = self.eval_context_mut();
        this.get_or_create_id(
            lock,
            offset,
            |ecx| &mut ecx.machine.sync.init_onces,
            |_| Ok(Default::default()),
        )?
        .ok_or_else(|| err_ub_format!("init_once has invalid ID").into())
    }

    #[inline]
    fn init_once_status(&mut self, id: InitOnceId) -> InitOnceStatus {
        let this = self.eval_context_ref();
        this.machine.sync.init_onces[id].status
    }

    /// Put the thread into the queue waiting for the initialization.
    #[inline]
    fn init_once_enqueue_and_block(
        &mut self,
        id: InitOnceId,
        callback: impl UnblockCallback<'tcx> + 'tcx,
    ) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let init_once = &mut this.machine.sync.init_onces[id];
        assert_ne!(init_once.status, InitOnceStatus::Complete, "queueing on complete init once");
        init_once.waiters.push_back(thread);
        this.block_thread(BlockReason::InitOnce(id), None, callback);
    }

    /// Begin initializing this InitOnce. Must only be called after checking that it is currently
    /// uninitialized.
    #[inline]
    fn init_once_begin(&mut self, id: InitOnceId) {
        let this = self.eval_context_mut();
        let init_once = &mut this.machine.sync.init_onces[id];
        assert_eq!(
            init_once.status,
            InitOnceStatus::Uninitialized,
            "beginning already begun or complete init once"
        );
        init_once.status = InitOnceStatus::Begun;
    }

    #[inline]
    fn init_once_complete(&mut self, id: InitOnceId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let init_once = &mut this.machine.sync.init_onces[id];

        assert_eq!(
            init_once.status,
            InitOnceStatus::Begun,
            "completing already complete or uninit init once"
        );

        init_once.status = InitOnceStatus::Complete;

        // Each complete happens-before the end of the wait
        if let Some(data_race) = &this.machine.data_race {
            init_once.clock.clone_from(&data_race.release_clock(&this.machine.threads));
        }

        // Wake up everyone.
        // need to take the queue to avoid having `this` be borrowed multiple times
        for waiter in std::mem::take(&mut init_once.waiters) {
            this.unblock_thread(waiter, BlockReason::InitOnce(id))?;
        }

        Ok(())
    }

    #[inline]
    fn init_once_fail(&mut self, id: InitOnceId) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let init_once = &mut this.machine.sync.init_onces[id];
        assert_eq!(
            init_once.status,
            InitOnceStatus::Begun,
            "failing already completed or uninit init once"
        );
        // This is again uninitialized.
        init_once.status = InitOnceStatus::Uninitialized;

        // Each complete happens-before the end of the wait
        if let Some(data_race) = &this.machine.data_race {
            init_once.clock.clone_from(&data_race.release_clock(&this.machine.threads));
        }

        // Wake up one waiting thread, so they can go ahead and try to init this.
        if let Some(waiter) = init_once.waiters.pop_front() {
            this.unblock_thread(waiter, BlockReason::InitOnce(id))?;
        }

        Ok(())
    }

    /// Synchronize with the previous completion of an InitOnce.
    /// Must only be called after checking that it is complete.
    #[inline]
    fn init_once_observe_completed(&mut self, id: InitOnceId) {
        let this = self.eval_context_mut();

        assert_eq!(
            this.init_once_status(id),
            InitOnceStatus::Complete,
            "observing the completion of incomplete init once"
        );

        this.acquire_clock(&this.machine.sync.init_onces[id].clock);
    }
}
