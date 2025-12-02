use genmc_sys::AssumeType;
use rustc_middle::ty;
use tracing::debug;

use crate::concurrency::genmc::MAX_ACCESS_SIZE;
use crate::concurrency::thread::EvalContextExt as _;
use crate::*;

impl GenmcCtx {
    /// Handle a user thread getting blocked.
    /// This may happen due to an manual `assume` statement added by a user
    /// or added by some automated program transformation, e.g., for spinloops.
    fn handle_assume_block<'tcx>(
        &self,
        machine: &MiriMachine<'tcx>,
        assume_type: AssumeType,
    ) -> InterpResult<'tcx> {
        debug!("GenMC: assume statement, blocking active thread.");
        self.handle
            .borrow_mut()
            .pin_mut()
            .handle_assume_block(self.active_thread_genmc_tid(machine), assume_type);
        interp_ok(())
    }
}

// Handling of code intercepted by Miri in GenMC mode, such as assume statement or `std::sync::Mutex`.

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Small helper to get the arguments of an intercepted function call.
    fn get_fn_args<const N: usize>(
        &self,
        instance: ty::Instance<'tcx>,
        args: &[FnArg<'tcx>],
    ) -> InterpResult<'tcx, [OpTy<'tcx>; N]> {
        let this = self.eval_context_ref();
        let args = this.copy_fn_args(args); // FIXME: Should `InPlace` arguments be reset to uninit?
        if let Ok(ops) = args.try_into() {
            return interp_ok(ops);
        }
        panic!("{} is a diagnostic item expected to have {} arguments", instance, N);
    }

    /**** Blocking functionality ****/

    /// Handle a thread getting blocked by a user assume (not an automatically generated assume).
    /// Unblocking this thread in the current execution will cause a panic.
    /// Miri does not provide GenMC with the annotations to determine when to unblock the thread, so it should never be unblocked.
    fn handle_user_assume_block(&mut self) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        debug!(
            "GenMC: block thread {:?} due to failing assume statement.",
            this.machine.threads.active_thread()
        );
        assert!(this.machine.threads.active_thread_ref().is_enabled());
        // Block the thread on the GenMC side.
        let genmc_ctx = this.machine.data_race.as_genmc_ref().unwrap();
        genmc_ctx.handle_assume_block(&this.machine, AssumeType::User)?;
        // Block the thread on the Miri side.
        this.block_thread(
            BlockReason::Genmc,
            None,
            callback!(
                @capture<'tcx> {}
                |_this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);
                    unreachable!("GenMC should never unblock a thread blocked by an `assume`.");
                }
            ),
        );
        interp_ok(())
    }

    fn intercept_mutex_lock(&mut self, mutex: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        debug!("GenMC: handling Mutex::lock()");
        let this = self.eval_context_mut();
        let genmc_ctx = this.machine.data_race.as_genmc_ref().unwrap();

        let size = mutex.layout.size.bytes();
        assert!(
            size <= MAX_ACCESS_SIZE,
            "Mutex is larger than maximal size of a memory access supported by GenMC ({size} > {MAX_ACCESS_SIZE})"
        );
        let result = genmc_ctx.handle.borrow_mut().pin_mut().handle_mutex_lock(
            genmc_ctx.active_thread_genmc_tid(&this.machine),
            mutex.ptr().addr().bytes(),
            size,
        );
        if let Some(error) = result.error.as_ref() {
            // FIXME(genmc): improve error handling.
            throw_ub_format!("{}", error.to_string_lossy());
        }
        if result.is_reset {
            debug!("GenMC: Mutex::lock: Reset");
            // GenMC informed us to reset and try the lock again later.
            // We block the current thread until GenMC schedules it again.
            this.block_thread(
                crate::BlockReason::Genmc,
                None,
                crate::callback!(
                    @capture<'tcx> {
                        mutex: MPlaceTy<'tcx>,
                    }
                    |this, unblock: crate::UnblockKind| {
                        debug!("GenMC: Mutex::lock: unblocking callback called, attempting to lock the Mutex again.");
                        assert_eq!(unblock, crate::UnblockKind::Ready);
                        this.intercept_mutex_lock(mutex)?;
                        interp_ok(())
                    }
                ),
            );
        } else if result.is_lock_acquired {
            debug!("GenMC: Mutex::lock successfully acquired the Mutex.");
        } else {
            debug!("GenMC: Mutex::lock failed to acquire the Mutex, permanently blocking thread.");
            // NOTE: `handle_mutex_lock` already blocked the current thread on the GenMC side.
            this.block_thread(
                crate::BlockReason::Genmc,
                None,
                crate::callback!(
                    @capture<'tcx> {
                        mutex: MPlaceTy<'tcx>,
                    }
                    |_this, _unblock: crate::UnblockKind| {
                        unreachable!("A thread blocked on `Mutex::lock` should not be unblocked again.");
                    }
                ),
            );
        }
        // NOTE: We don't write anything back to Miri's memory where the Mutex is located, that state is handled only by GenMC.
        interp_ok(())
    }

    fn intercept_mutex_try_lock(
        &mut self,
        mutex: MPlaceTy<'tcx>,
        dest: &crate::PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        debug!("GenMC: handling Mutex::try_lock()");
        let this = self.eval_context_mut();
        let genmc_ctx = this.machine.data_race.as_genmc_ref().unwrap();
        let size = mutex.layout.size.bytes();
        assert!(
            size <= MAX_ACCESS_SIZE,
            "Mutex is larger than maximal size of a memory access supported by GenMC ({size} > {MAX_ACCESS_SIZE})"
        );
        let result = genmc_ctx.handle.borrow_mut().pin_mut().handle_mutex_try_lock(
            genmc_ctx.active_thread_genmc_tid(&this.machine),
            mutex.ptr().addr().bytes(),
            size,
        );
        if let Some(error) = result.error.as_ref() {
            // FIXME(genmc): improve error handling.
            throw_ub_format!("{}", error.to_string_lossy());
        }
        debug!(
            "GenMC: Mutex::try_lock(): is_reset: {}, is_lock_acquired: {}",
            result.is_reset, result.is_lock_acquired
        );
        assert!(!result.is_reset, "GenMC returned 'reset' for a mutex try_lock.");
        // Write the return value of try_lock, i.e., whether we acquired the mutex.
        this.write_scalar(Scalar::from_bool(result.is_lock_acquired), dest)?;
        // NOTE: We don't write anything back to Miri's memory where the Mutex is located, that state is handled only by GenMC.
        interp_ok(())
    }

    fn intercept_mutex_unlock(&self, mutex: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        debug!("GenMC: handling Mutex::unlock()");
        let this = self.eval_context_ref();
        let genmc_ctx = this.machine.data_race.as_genmc_ref().unwrap();
        let result = genmc_ctx.handle.borrow_mut().pin_mut().handle_mutex_unlock(
            genmc_ctx.active_thread_genmc_tid(&this.machine),
            mutex.ptr().addr().bytes(),
            mutex.layout.size.bytes(),
        );
        if let Some(error) = result.error.as_ref() {
            // FIXME(genmc): improve error handling.
            throw_ub_format!("{}", error.to_string_lossy());
        }
        // NOTE: We don't write anything back to Miri's memory where the Mutex is located, that state is handled only by GenMC.}
        interp_ok(())
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Given a `ty::Instance<'tcx>`, do any required special handling.
    /// Returns true if this `instance` should be skipped (i.e., no MIR should be executed for it).
    fn genmc_intercept_function(
        &mut self,
        instance: rustc_middle::ty::Instance<'tcx>,
        args: &[rustc_const_eval::interpret::FnArg<'tcx, crate::Provenance>],
        dest: &crate::PlaceTy<'tcx>,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        assert!(
            this.machine.data_race.as_genmc_ref().is_some(),
            "This function should only be called in GenMC mode."
        );

        // NOTE: When adding new intercepted functions here, they must also be added to `fn get_function_kind` in `concurrency/genmc/scheduling.rs`.
        use rustc_span::sym;
        if this.tcx.is_diagnostic_item(sym::sys_mutex_lock, instance.def_id()) {
            let [mutex] = this.get_fn_args(instance, args)?;
            let mutex = this.deref_pointer(&mutex)?;
            this.intercept_mutex_lock(mutex)?;
        } else if this.tcx.is_diagnostic_item(sym::sys_mutex_try_lock, instance.def_id()) {
            let [mutex] = this.get_fn_args(instance, args)?;
            let mutex = this.deref_pointer(&mutex)?;
            this.intercept_mutex_try_lock(mutex, dest)?;
        } else if this.tcx.is_diagnostic_item(sym::sys_mutex_unlock, instance.def_id()) {
            let [mutex] = this.get_fn_args(instance, args)?;
            let mutex = this.deref_pointer(&mutex)?;
            this.intercept_mutex_unlock(mutex)?;
        } else {
            // Nothing to intercept.
            return interp_ok(false);
        }
        interp_ok(true)
    }

    /// Handle an `assume` statement. This will tell GenMC to block the current thread if the `condition` is false.
    /// Returns `true` if the current thread should be blocked in Miri too.
    fn handle_genmc_verifier_assume(&mut self, condition: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let condition_bool = this.read_scalar(condition)?.to_bool()?;
        debug!("GenMC: handle_genmc_verifier_assume, condition: {condition:?} = {condition_bool}");
        if condition_bool {
            return interp_ok(());
        }
        this.handle_user_assume_block()
    }
}
