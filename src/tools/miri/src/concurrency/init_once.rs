use std::collections::VecDeque;

use rustc_index::Idx;
use rustc_middle::ty::layout::TyAndLayout;

use super::sync::EvalContextExtPriv as _;
use super::vector_clock::VClock;
use crate::*;

declare_id!(InitOnceId);

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

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Provides the closure with the next InitOnceId. Creates that InitOnce if the closure returns None,
    /// otherwise returns the value from the closure.
    #[inline]
    fn init_once_get_or_create<F>(&mut self, existing: F) -> InterpResult<'tcx, InitOnceId>
    where
        F: FnOnce(
            &mut MiriInterpCx<'tcx>,
            InitOnceId,
        ) -> InterpResult<'tcx, Option<InitOnceId>>,
    {
        let this = self.eval_context_mut();
        let next_index = this.machine.sync.init_onces.next_index();
        if let Some(old) = existing(this, next_index)? {
            Ok(old)
        } else {
            let new_index = this.machine.sync.init_onces.push(Default::default());
            assert_eq!(next_index, new_index);
            Ok(new_index)
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn init_once_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
        lock_layout: TyAndLayout<'tcx>,
        offset: u64,
    ) -> InterpResult<'tcx, InitOnceId> {
        let this = self.eval_context_mut();
        this.init_once_get_or_create(|ecx, next_id| {
            ecx.get_or_create_id(next_id, lock_op, lock_layout, offset)
        })
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
