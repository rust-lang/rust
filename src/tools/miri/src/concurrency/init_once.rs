use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use super::thread::DynUnblockCallback;
use super::vector_clock::VClock;
use crate::*;

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

impl InitOnce {
    #[inline]
    pub fn status(&self) -> InitOnceStatus {
        self.status
    }

    /// Begin initializing this InitOnce. Must only be called after checking that it is currently
    /// uninitialized.
    #[inline]
    pub fn begin(&mut self) {
        assert_eq!(
            self.status(),
            InitOnceStatus::Uninitialized,
            "beginning already begun or complete init once"
        );
        self.status = InitOnceStatus::Begun;
    }
}

#[derive(Default, Clone, Debug)]
pub struct InitOnceRef(Rc<RefCell<InitOnce>>);

impl InitOnceRef {
    pub fn new() -> Self {
        Self(Default::default())
    }

    pub fn status(&self) -> InitOnceStatus {
        self.0.borrow().status()
    }

    pub fn begin(&self) {
        self.0.borrow_mut().begin();
    }
}

impl VisitProvenance for InitOnceRef {
    // InitOnce contains no provenance.
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Put the thread into the queue waiting for the initialization.
    #[inline]
    fn init_once_enqueue_and_block(
        &mut self,
        init_once_ref: InitOnceRef,
        callback: DynUnblockCallback<'tcx>,
    ) {
        let this = self.eval_context_mut();
        let thread = this.active_thread();
        let mut init_once = init_once_ref.0.borrow_mut();
        assert_ne!(init_once.status, InitOnceStatus::Complete, "queueing on complete init once");

        init_once.waiters.push_back(thread);
        this.block_thread(BlockReason::InitOnce, None, callback);
    }

    #[inline]
    fn init_once_complete(&mut self, init_once_ref: &InitOnceRef) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let mut init_once = init_once_ref.0.borrow_mut();
        assert_eq!(
            init_once.status,
            InitOnceStatus::Begun,
            "completing already complete or uninit init once"
        );

        init_once.status = InitOnceStatus::Complete;

        // Each complete happens-before the end of the wait
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race
                .release_clock(&this.machine.threads, |clock| init_once.clock.clone_from(clock));
        }

        // Wake up everyone.
        // need to take the queue to avoid having `this` be borrowed multiple times
        let waiters = std::mem::take(&mut init_once.waiters);
        drop(init_once);
        for waiter in waiters {
            this.unblock_thread(waiter, BlockReason::InitOnce)?;
        }

        interp_ok(())
    }

    #[inline]
    fn init_once_fail(&mut self, init_once_ref: &InitOnceRef) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let mut init_once = init_once_ref.0.borrow_mut();
        assert_eq!(
            init_once.status,
            InitOnceStatus::Begun,
            "failing already completed or uninit init once"
        );
        // This is again uninitialized.
        init_once.status = InitOnceStatus::Uninitialized;

        // Each complete happens-before the end of the wait
        if let Some(data_race) = this.machine.data_race.as_vclocks_ref() {
            data_race
                .release_clock(&this.machine.threads, |clock| init_once.clock.clone_from(clock));
        }

        // Wake up one waiting thread, so they can go ahead and try to init this.
        if let Some(waiter) = init_once.waiters.pop_front() {
            drop(init_once);
            this.unblock_thread(waiter, BlockReason::InitOnce)?;
        }

        interp_ok(())
    }

    /// Synchronize with the previous completion of an InitOnce.
    /// Must only be called after checking that it is complete.
    #[inline]
    fn init_once_observe_completed(&mut self, init_once_ref: &InitOnceRef) {
        let this = self.eval_context_mut();
        let init_once = init_once_ref.0.borrow();

        assert_eq!(
            init_once.status,
            InitOnceStatus::Complete,
            "observing the completion of incomplete init once"
        );

        this.acquire_clock(&init_once.clock);
    }
}
