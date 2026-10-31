//! Contains FreeBSD-specific synchronization functions

use core::time::Duration;

use rustc_abi::FieldIdx;

use crate::concurrency::sync::FutexRef;
use crate::meta_obj::GetMetaError;
use crate::*;

pub struct FreeBsdFutex {
    futex: FutexRef,
}

impl MetaObj for FreeBsdFutex {
    fn opaque(&self) -> bool {
        false
    }
}

/// Extended variant of the `timespec` struct.
pub struct UmtxTime {
    timeout: Duration,
    abs_time: bool,
    timeout_clock: TimeoutClock,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Implementation of the FreeBSD [`_umtx_op`](https://man.freebsd.org/cgi/man.cgi?query=_umtx_op&sektion=2&manpath=FreeBSD+14.2-RELEASE+and+Ports) syscall.
    /// This is used for futex operations on FreeBSD.
    ///
    /// `obj`: a pointer to the futex object (can be a lot of things, mostly *AtomicU32)
    /// `op`: the futex operation to run
    /// `val`: the current value of the object as a `c_long` (for wait/wake)
    /// `uaddr`: `op`-specific optional parameter, pointer-sized integer or pointer to an `op`-specific struct
    /// `uaddr2`: `op`-specific optional parameter, pointer-sized integer or pointer to an `op`-specific struct
    /// `dest`: the place this syscall returns to, 0 for success, -1 for failure
    ///
    /// # Note
    /// Curently only the WAIT and WAKE operations are implemented.
    fn _umtx_op(
        &mut self,
        obj: &OpTy<'tcx>,
        op: &OpTy<'tcx>,
        val: &OpTy<'tcx>,
        uaddr: &OpTy<'tcx>,
        uaddr2: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let obj_ptr = ecx.read_pointer(obj)?;
        let op = ecx.read_scalar(op)?.to_i32()?;
        let val = ecx.read_target_usize(val)?;
        let uaddr = ecx.read_target_usize(uaddr)?;
        let uaddr2 = ecx.read_pointer(uaddr2)?;

        let wait = ecx.eval_libc_i32("UMTX_OP_WAIT");
        let wait_uint = ecx.eval_libc_i32("UMTX_OP_WAIT_UINT");
        let wait_uint_private = ecx.eval_libc_i32("UMTX_OP_WAIT_UINT_PRIVATE");

        let wake = ecx.eval_libc_i32("UMTX_OP_WAKE");
        let wake_private = ecx.eval_libc_i32("UMTX_OP_WAKE_PRIVATE");

        let timespec_layout = ecx.libc_ty_layout("timespec");
        let umtx_time_layout = ecx.libc_ty_layout("_umtx_time");
        assert!(
            timespec_layout.size != umtx_time_layout.size,
            "`struct timespec` and `struct _umtx_time` should have different sizes."
        );

        match op {
            // UMTX_OP_WAIT_UINT and UMTX_OP_WAIT_UINT_PRIVATE only differ in whether they work across
            // processes or not. For Miri, we can treat them the same.
            op if op == wait || op == wait_uint || op == wait_uint_private => {
                let obj_layout =
                    if op == wait { ecx.machine.layouts.isize } else { ecx.machine.layouts.u32 };
                let obj = ecx.ptr_to_mplace(obj_ptr, obj_layout);

                // Read the Linux futex wait implementation in Miri to understand why this fence is needed.
                ecx.atomic_fence(AtomicFenceOrd::SeqCst)?;
                let obj_val = ecx
                    .read_scalar_atomic(&obj, AtomicReadOrd::Acquire)?
                    .to_bits(obj_layout.size)?; // isize and u32 can have different sizes

                if obj_val == u128::from(val) {
                    // This cannot fail since we already did an atomic acquire read on that pointer.
                    // Acquire reads are only allowed on mutable memory.
                    let futex_ref = ecx
                        .get_or_insert_meta(&obj, |_| {
                            interp_ok(FreeBsdFutex { futex: Default::default() })
                        })?
                        .futex
                        .clone();

                    // From the manual:
                    // The timeout is specified by passing either the address of `struct timespec`, or its
                    // extended variant, `struct _umtx_time`, as the `uaddr2` argument of _umtx_op().
                    // They are distinguished by the `uaddr` value, which must be equal
                    // to the size of the structure pointed to by `uaddr2`, casted to uintptr_t.
                    let deadline = if ecx.ptr_is_null(uaddr2)? {
                        // no timeout parameter
                        None
                    } else {
                        if uaddr == umtx_time_layout.size.bytes() {
                            // `uaddr2` points to a `struct _umtx_time`.
                            let umtx_time_place = ecx.ptr_to_mplace(uaddr2, umtx_time_layout);

                            let Some(umtx_time) = ecx.read_umtx_time(&umtx_time_place)? else {
                                return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
                            };

                            let style = if umtx_time.abs_time {
                                TimeoutStyle::Absolute
                            } else {
                                TimeoutStyle::Relative
                            };

                            Some(ecx.machine.timeout(
                                umtx_time.timeout_clock,
                                style,
                                umtx_time.timeout,
                            ))
                        } else if uaddr == timespec_layout.size.bytes() {
                            // RealTime clock can't be used in isolation mode.
                            ecx.check_no_isolation("`_umtx_op` with `timespec` timeout")?;

                            // `uaddr2` points to a `struct timespec`.
                            let timespec = ecx.ptr_to_mplace(uaddr2, timespec_layout);
                            let Some(duration) = ecx.read_timespec(&timespec)? else {
                                return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
                            };

                            // FreeBSD does not seem to document which clock is used when the timeout
                            // is passed as a `struct timespec*`. Based on discussions online and the source
                            // code (umtx_copyin_umtx_time() in kern_umtx.c), it seems to default to CLOCK_REALTIME,
                            // so that's what we also do.
                            // Discussion in golang: https://github.com/golang/go/issues/17168#issuecomment-250235271
                            Some(ecx.machine.timeout(
                                TimeoutClock::RealTime,
                                TimeoutStyle::Relative,
                                duration,
                            ))
                        } else {
                            return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
                        }
                    };

                    let dest = dest.clone();
                    ecx.futex_wait(
                        futex_ref,
                        u32::MAX, // we set the bitset to include all bits
                        deadline,
                        callback!(
                            @capture<'tcx> {
                                dest: MPlaceTy<'tcx>,
                            }
                            |ecx, unblock: UnblockKind| match unblock {
                                UnblockKind::Ready => {
                                    // From the manual:
                                    // If successful, all requests, except UMTX_SHM_CREAT and UMTX_SHM_LOOKUP
                                    // sub-requests of the UMTX_OP_SHM request, will return zero.
                                    ecx.write_int(0, &dest)
                                }
                                UnblockKind::TimedOut => {
                                    ecx.set_errno_and_return_neg1(LibcError("ETIMEDOUT"), &dest)
                                }
                            }
                        ),
                    );
                    interp_ok(())
                } else {
                    // The manual doesn’t specify what should happen if the futex value doesn’t match the expected one.
                    // On FreeBSD 14.2, testing shows that WAIT operations return 0 even when the value is incorrect.
                    ecx.write_int(0, dest)?;
                    interp_ok(())
                }
            }
            // UMTX_OP_WAKE and UMTX_OP_WAKE_PRIVATE only differ in whether they work across
            // processes or not. For Miri, we can treat them the same.
            op if op == wake || op == wake_private => {
                let futex = match ecx.try_get_meta::<FreeBsdFutex>(obj_ptr) {
                    Ok(f) => f,
                    Err(GetMetaError::NoMem) => {
                        // From Linux implementation:
                        // No AllocId, or no live allocation at that AllocId.
                        // Return an error code. (That seems nicer than silently doing something non-intuitive.)
                        // This means that if an address gets reused by a new allocation,
                        // we'll use an independent futex queue for this... that seems acceptable.
                        return ecx.set_errno_and_return_neg1(LibcError("EFAULT"), dest);
                    }
                    Err(GetMetaError::NoObj) => {
                        // Nobody waiting here, nothing to do.
                        ecx.write_int(0, dest)?;
                        return interp_ok(());
                    }
                };
                let futex_ref = futex.futex.clone();

                // Saturating cast for when usize is smaller than u64.
                let count = usize::try_from(val).unwrap_or(usize::MAX);
                if count == 0 {
                    // Native testing seems to indicate that this still wakes up a thread.
                    // Just consider this unsupported for now.
                    throw_unsup_format!(
                        "UMTX_OP_WAKE with a val (number of threads to wake up) of 0 is not supported"
                    );
                }

                // Read the Linux futex wake implementation in Miri to understand why this fence is needed.
                ecx.atomic_fence(AtomicFenceOrd::SeqCst)?;

                // `_umtx_op` doesn't return the amount of woken threads.
                let _woken = ecx.futex_wake(
                    &futex_ref,
                    u32::MAX, // we set the bitset to include all bits
                    count,
                )?;

                // From the manual:
                // If successful, all requests, except UMTX_SHM_CREAT and UMTX_SHM_LOOKUP
                // sub-requests of the UMTX_OP_SHM request, will return zero.
                ecx.write_int(0, dest)?;
                interp_ok(())
            }
            op => {
                throw_unsup_format!("Miri does not support `_umtx_op` syscall with op={}", op)
            }
        }
    }

    /// Parses a `_umtx_time` struct.
    /// Returns `None` if the underlying `timespec` struct is invalid.
    fn read_umtx_time(&self, ut: &MPlaceTy<'tcx>) -> InterpResult<'tcx, Option<UmtxTime>> {
        let this = self.eval_context_ref();
        // Only flag allowed is UMTX_ABSTIME.
        let abs_time = this.eval_libc_u32("UMTX_ABSTIME");

        let timespec_place = this.project_field(ut, FieldIdx::from_u32(0))?;
        // Inner `timespec` must still be valid.
        let Some(duration) = this.read_timespec(&timespec_place)? else { return interp_ok(None) };

        let flags_place = this.project_field(ut, FieldIdx::from_u32(1))?;
        let mut flags = this.read_scalar(&flags_place)?.to_u32()?;

        let abs_time_flag = if flags & abs_time != 0 {
            flags &= !abs_time;
            true
        } else {
            false
        };
        if flags != 0 {
            throw_unsup_format!("unsupported `_umtx_time` flags: {:#x}", flags);
        }

        let clock_id_place = this.project_field(ut, FieldIdx::from_u32(2))?;
        let clock_id = this.read_scalar(&clock_id_place)?.to_i32()?;
        let Some(timeout_clock) = this.parse_clockid(clock_id) else {
            throw_unsup_format!("unsupported clock")
        };
        if timeout_clock == TimeoutClock::RealTime {
            this.check_no_isolation("`_umtx_op` with `CLOCK_REALTIME`")?;
        }

        interp_ok(Some(UmtxTime { timeout: duration, abs_time: abs_time_flag, timeout_clock }))
    }
}
