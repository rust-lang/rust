use std::time::Duration;

use rustc_target::abi::Size;

use crate::concurrency::init_once::InitOnceStatus;
use crate::*;

#[derive(Copy, Clone)]
struct WindowsInitOnce {
    id: InitOnceId,
}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    // Windows sync primitives are pointer sized.
    // We only use the first 4 bytes for the id.

    fn init_once_get_data(
        &mut self,
        init_once_ptr: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, WindowsInitOnce> {
        let this = self.eval_context_mut();

        let init_once = this.deref_pointer(init_once_ptr)?;
        let init_offset = Size::ZERO;

        this.lazy_sync_get_data(
            &init_once,
            init_offset,
            || throw_ub_format!("`INIT_ONCE` can't be moved after first use"),
            |this| {
                // TODO: check that this is still all-zero.
                let id = this.machine.sync.init_once_create();
                interp_ok(WindowsInitOnce { id })
            },
        )
    }

    /// Returns `true` if we were succssful, `false` if we would block.
    fn init_once_try_begin(
        &mut self,
        id: InitOnceId,
        pending_place: &MPlaceTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();
        interp_ok(match this.init_once_status(id) {
            InitOnceStatus::Uninitialized => {
                this.init_once_begin(id);
                this.write_scalar(this.eval_windows("c", "TRUE"), pending_place)?;
                this.write_scalar(this.eval_windows("c", "TRUE"), dest)?;
                true
            }
            InitOnceStatus::Complete => {
                this.init_once_observe_completed(id);
                this.write_scalar(this.eval_windows("c", "FALSE"), pending_place)?;
                this.write_scalar(this.eval_windows("c", "TRUE"), dest)?;
                true
            }
            InitOnceStatus::Begun => false,
        })
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
#[allow(non_snake_case)]
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn InitOnceBeginInitialize(
        &mut self,
        init_once_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
        pending_op: &OpTy<'tcx>,
        context_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = this.init_once_get_data(init_once_op)?.id;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let pending_place = this.deref_pointer(pending_op)?;
        let context = this.read_pointer(context_op)?;

        if flags != 0 {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        }

        if !this.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        if this.init_once_try_begin(id, &pending_place, dest)? {
            // Done!
            return interp_ok(());
        }

        // We have to block, and then try again when we are woken up.
        let dest = dest.clone();
        this.init_once_enqueue_and_block(
            id,
            callback!(
                @capture<'tcx> {
                    id: InitOnceId,
                    pending_place: MPlaceTy<'tcx>,
                    dest: MPlaceTy<'tcx>,
                }
                @unblock = |this| {
                    let ret = this.init_once_try_begin(id, &pending_place, &dest)?;
                    assert!(ret, "we were woken up but init_once_try_begin still failed");
                    interp_ok(())
                }
            ),
        );
        interp_ok(())
    }

    fn InitOnceComplete(
        &mut self,
        init_once_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
        context_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let id = this.init_once_get_data(init_once_op)?.id;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let context = this.read_pointer(context_op)?;

        let success = if flags == 0 {
            true
        } else if flags == this.eval_windows_u32("c", "INIT_ONCE_INIT_FAILED") {
            false
        } else {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        };

        if !this.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        if this.init_once_status(id) != InitOnceStatus::Begun {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling InitOnceComplete on a one time initialization that has not begun or is already completed"
            );
        }

        if success {
            this.init_once_complete(id)?;
        } else {
            this.init_once_fail(id)?;
        }

        interp_ok(this.eval_windows("c", "TRUE"))
    }

    fn WaitOnAddress(
        &mut self,
        ptr_op: &OpTy<'tcx>,
        compare_op: &OpTy<'tcx>,
        size_op: &OpTy<'tcx>,
        timeout_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr_op)?;
        let compare = this.read_pointer(compare_op)?;
        let size = this.read_target_usize(size_op)?;
        let timeout_ms = this.read_scalar(timeout_op)?.to_u32()?;

        let addr = ptr.addr().bytes();

        if size > 8 || !size.is_power_of_two() {
            let invalid_param = this.eval_windows("c", "ERROR_INVALID_PARAMETER");
            this.set_last_error(invalid_param)?;
            this.write_scalar(Scalar::from_i32(0), dest)?;
            return interp_ok(());
        };
        let size = Size::from_bytes(size);

        let timeout = if timeout_ms == this.eval_windows_u32("c", "INFINITE") {
            None
        } else {
            let duration = Duration::from_millis(timeout_ms.into());
            Some((TimeoutClock::Monotonic, TimeoutAnchor::Relative, duration))
        };

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let layout = this.machine.layouts.uint(size).unwrap();
        let futex_val =
            this.read_scalar_atomic(&this.ptr_to_mplace(ptr, layout), AtomicReadOrd::Relaxed)?;
        let compare_val = this.read_scalar(&this.ptr_to_mplace(compare, layout))?;

        if futex_val == compare_val {
            // If the values are the same, we have to block.
            this.futex_wait(
                addr,
                u32::MAX, // bitset
                timeout,
                Scalar::from_i32(1), // retval_succ
                Scalar::from_i32(0), // retval_timeout
                dest.clone(),
                this.eval_windows("c", "ERROR_TIMEOUT"), // errno_timeout
            );
        }

        this.write_scalar(Scalar::from_i32(1), dest)?;

        interp_ok(())
    }

    fn WakeByAddressSingle(&mut self, ptr_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr_op)?;

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let addr = ptr.addr().bytes();
        this.futex_wake(addr, u32::MAX)?;

        interp_ok(())
    }
    fn WakeByAddressAll(&mut self, ptr_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr_op)?;

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let addr = ptr.addr().bytes();
        while this.futex_wake(addr, u32::MAX)? {}

        interp_ok(())
    }
}
