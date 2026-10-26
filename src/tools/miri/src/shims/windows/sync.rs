use std::time::Duration;

use rustc_abi::{FieldIdx, Size};

use crate::concurrency::init_once::{EvalContextExt as _, InitOnceStatus};
use crate::concurrency::sync::{AccessKind, FutexRef, SyncObj};
use crate::*;

#[derive(Clone)]
struct WindowsInitOnce {
    init_once: InitOnceRef,
}

impl SyncObj for WindowsInitOnce {
    fn on_access<'tcx>(&self, access_kind: AccessKind) -> InterpResult<'tcx> {
        if !self.init_once.queue_is_empty() {
            throw_ub_format!(
                "{access_kind} of `INIT_ONCE` is forbidden while the queue is non-empty"
            );
        }
        interp_ok(())
    }

    fn delete_on_write(&self) -> bool {
        true
    }
}

struct WindowsFutex {
    futex: FutexRef,
}

impl SyncObj for WindowsFutex {}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    // Windows sync primitives are pointer sized.
    // We only use the first byte for the "init" flag.

    fn init_once_get_data<'a>(
        &'a mut self,
        init_once_ptr: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, &'a WindowsInitOnce>
    where
        'tcx: 'a,
    {
        let ecx = self.eval_context_mut();

        let init_once = ecx.deref_pointer_as(init_once_ptr, ecx.windows_ty_layout("INIT_ONCE"))?;
        let init_offset = Size::ZERO;

        ecx.get_immovable_sync_with_static_init(
            &init_once,
            init_offset,
            /* uninit_val */ 0,
            /* init_val */ 1,
            |this| {
                let ptr_field = this.project_field(&init_once, FieldIdx::from_u32(0))?;
                let val = this.read_target_usize(&ptr_field)?;
                if val == 0 {
                    interp_ok(WindowsInitOnce { init_once: InitOnceRef::new() })
                } else {
                    throw_ub_format!("`INIT_ONCE` was not properly initialized at this location, or it got overwritten");
                }
            },
        )
    }

    /// Returns `true` if we were successful, `false` if we would block.
    fn init_once_try_begin(
        &mut self,
        init_once_ref: &InitOnceRef,
        pending_place: &MPlaceTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, bool> {
        let ecx = self.eval_context_mut();
        interp_ok(match init_once_ref.status() {
            InitOnceStatus::Uninitialized => {
                init_once_ref.begin();
                ecx.write_scalar(ecx.eval_windows("c", "TRUE"), pending_place)?;
                ecx.write_scalar(ecx.eval_windows("c", "TRUE"), dest)?;
                true
            }
            InitOnceStatus::Complete => {
                ecx.init_once_observe_completed(init_once_ref)?;
                ecx.write_scalar(ecx.eval_windows("c", "FALSE"), pending_place)?;
                ecx.write_scalar(ecx.eval_windows("c", "TRUE"), dest)?;
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
        let ecx = self.eval_context_mut();

        let init_once = ecx.init_once_get_data(init_once_op)?.init_once.clone();
        let flags = ecx.read_scalar(flags_op)?.to_u32()?;
        // PBOOL is int*
        let pending_place = ecx.deref_pointer_as(pending_op, ecx.machine.layouts.i32)?;
        let context = ecx.read_pointer(context_op)?;

        if flags != 0 {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        }

        if !ecx.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        if ecx.init_once_try_begin(&init_once, &pending_place, dest)? {
            // Done!
            return interp_ok(());
        }

        // We have to block, and then try again when we are woken up.
        let dest = dest.clone();
        ecx.init_once_enqueue_and_block(
            init_once.clone(),
            callback!(
                @capture<'tcx> {
                    init_once: InitOnceRef,
                    pending_place: MPlaceTy<'tcx>,
                    dest: MPlaceTy<'tcx>,
                }
                |this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);
                    let ret = this.init_once_try_begin(&init_once, &pending_place, &dest)?;
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
        let ecx = self.eval_context_mut();

        let init_once = ecx.init_once_get_data(init_once_op)?.init_once.clone();
        let flags = ecx.read_scalar(flags_op)?.to_u32()?;
        let context = ecx.read_pointer(context_op)?;

        let success = if flags == 0 {
            true
        } else if flags == ecx.eval_windows_u32("c", "INIT_ONCE_INIT_FAILED") {
            false
        } else {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        };

        if !ecx.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        if init_once.status() != InitOnceStatus::Begun {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling InitOnceComplete on a one time initialization that has not begun or is already completed"
            );
        }

        if success {
            ecx.init_once_complete(&init_once)?;
        } else {
            ecx.init_once_fail(&init_once)?;
        }

        interp_ok(ecx.eval_windows("c", "TRUE"))
    }

    fn WaitOnAddress(
        &mut self,
        ptr_op: &OpTy<'tcx>,
        compare_op: &OpTy<'tcx>,
        size_op: &OpTy<'tcx>,
        timeout_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let ptr = ecx.read_pointer(ptr_op)?;
        let compare = ecx.read_pointer(compare_op)?;
        let size = ecx.read_target_usize(size_op)?;
        let timeout_ms = ecx.read_scalar(timeout_op)?.to_u32()?;

        if size > 8 || !size.is_power_of_two() {
            let invalid_param = ecx.eval_windows("c", "ERROR_INVALID_PARAMETER");
            ecx.set_last_error(invalid_param)?;
            ecx.write_scalar(Scalar::from_i32(0), dest)?;
            return interp_ok(());
        };
        let size = Size::from_bytes(size);

        let deadline = if timeout_ms == ecx.eval_windows_u32("c", "INFINITE") {
            None
        } else {
            let duration = Duration::from_millis(timeout_ms.into());
            Some(ecx.machine.monotonic_clock.now().add_lossy(duration).into())
        };

        // See the Linux futex implementation for why this fence exists.
        ecx.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let layout = ecx.machine.layouts.uint(size).unwrap();
        let futex_val =
            ecx.read_scalar_atomic(&ecx.ptr_to_mplace(ptr, layout), AtomicReadOrd::Acquire)?;
        let compare_val = ecx.read_scalar(&ecx.ptr_to_mplace(compare, layout))?;

        if futex_val == compare_val {
            // If the values are the same, we have to block.

            // This cannot fail since we already did an atomic acquire read on that pointer.
            let futex_ref = ecx
                .get_sync_or_init(ptr, |_| WindowsFutex { futex: Default::default() })
                .unwrap()
                .futex
                .clone();

            let dest = dest.clone();
            ecx.futex_wait(
                futex_ref,
                u32::MAX, // bitset
                deadline,
                callback!(
                    @capture<'tcx> {
                        dest: MPlaceTy<'tcx>
                    }
                    |this, unblock: UnblockKind| {
                        match unblock {
                            UnblockKind::Ready => {
                                this.write_int(1, &dest)
                            }
                            UnblockKind::TimedOut => {
                                this.set_last_error(IoError::WindowsError("ERROR_TIMEOUT"))?;
                                this.write_int(0, &dest)
                            }
                        }
                    }
                ),
            );
        }

        ecx.write_scalar(Scalar::from_i32(1), dest)?;

        interp_ok(())
    }

    fn WakeByAddressSingle(&mut self, ptr_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let ptr = ecx.read_pointer(ptr_op)?;

        // See the Linux futex implementation for why this fence exists.
        ecx.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let Some(futex_ref) =
            ecx.get_sync_or_init(ptr, |_| WindowsFutex { futex: Default::default() })
        else {
            // Seems like this cannot return an error, so we just wake nobody.
            return interp_ok(());
        };
        let futex_ref = futex_ref.futex.clone();

        ecx.futex_wake(&futex_ref, u32::MAX, 1)?;

        interp_ok(())
    }
    fn WakeByAddressAll(&mut self, ptr_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let ptr = ecx.read_pointer(ptr_op)?;

        // See the Linux futex implementation for why this fence exists.
        ecx.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let Some(futex_ref) =
            ecx.get_sync_or_init(ptr, |_| WindowsFutex { futex: Default::default() })
        else {
            // Seems like this cannot return an error, so we just wake nobody.
            return interp_ok(());
        };
        let futex_ref = futex_ref.futex.clone();

        ecx.futex_wake(&futex_ref, u32::MAX, usize::MAX)?;

        interp_ok(())
    }
}
