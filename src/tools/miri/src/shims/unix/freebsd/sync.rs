//! Contains FreeBSD-specific synchronization functions

use core::time::Duration;

use rustc_abi::FieldIdx;

use crate::concurrency::sync::FutexRef;
use crate::*;

pub struct FreeBsdFutex {
    futex: FutexRef,
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
        let this = self.eval_context_mut();

        let obj = this.read_pointer(obj)?;
        let op = this.read_scalar(op)?.to_i32()?;
        let val = this.read_target_usize(val)?;
        let uaddr = this.read_target_usize(uaddr)?;
        let uaddr2 = this.read_pointer(uaddr2)?;

        let wait = this.eval_libc_i32("UMTX_OP_WAIT");
        let wait_uint = this.eval_libc_i32("UMTX_OP_WAIT_UINT");
        let wait_uint_private = this.eval_libc_i32("UMTX_OP_WAIT_UINT_PRIVATE");

        let wake = this.eval_libc_i32("UMTX_OP_WAKE");
        let wake_private = this.eval_libc_i32("UMTX_OP_WAKE_PRIVATE");

        let timespec_layout = this.libc_ty_layout("timespec");
        let umtx_time_layout = this.libc_ty_layout("_umtx_time");
        assert!(
            timespec_layout.size != umtx_time_layout.size,
            "`struct timespec` and `struct _umtx_time` should have different sizes."
        );

        match op {
            // UMTX_OP_WAIT_UINT and UMTX_OP_WAIT_UINT_PRIVATE only differ in whether they work across
            // processes or not. For Miri, we can treat them the same.
            op if op == wait || op == wait_uint || op == wait_uint_private => {
                let obj_layout =
                    if op == wait { this.machine.layouts.isize } else { this.machine.layouts.u32 };
                let obj = this.ptr_to_mplace(obj, obj_layout);

                // Read the Linux futex wait implementation in Miri to understand why this fence is needed.
                this.atomic_fence(AtomicFenceOrd::SeqCst)?;
                let obj_val = this
                    .read_scalar_atomic(&obj, AtomicReadOrd::Acquire)?
                    .to_bits(obj_layout.size)?; // isize and u32 can have different sizes

                if obj_val == u128::from(val) {
                    // This cannot fail since we already did an atomic acquire read on that pointer.
                    // Acquire reads are only allowed on mutable memory.
                    let futex_ref = this
                        .get_sync_or_init(obj.ptr(), |_| FreeBsdFutex { futex: Default::default() })
                        .unwrap()
                        .futex
                        .clone();

                    // From the manual:
                    // The timeout is specified by passing either the address of `struct timespec`, or its
                    // extended variant, `struct _umtx_time`, as the `uaddr2` argument of _umtx_op().
                    // They are distinguished by the `uaddr` value, which must be equal
                    // to the size of the structure pointed to by `uaddr2`, casted to uintptr_t.
                    let timeout = if this.ptr_is_null(uaddr2)? {
                        // no timeout parameter
                        None
                    } else {
                        if uaddr == umtx_time_layout.size.bytes() {
                            // `uaddr2` points to a `struct _umtx_time`.
                            let umtx_time_place = this.ptr_to_mplace(uaddr2, umtx_time_layout);

                            let umtx_time = match this.read_umtx_time(&umtx_time_place)? {
                                Some(ut) => ut,
                                None => {
                                    return this
                                        .set_last_error_and_return(LibcError("EINVAL"), dest);
                                }
                            };

                            let anchor = if umtx_time.abs_time {
                                TimeoutAnchor::Absolute
                            } else {
                                TimeoutAnchor::Relative
                            };

                            Some((umtx_time.timeout_clock, anchor, umtx_time.timeout))
                        } else if uaddr == timespec_layout.size.bytes() {
                            // RealTime clock can't be used in isolation mode.
                            this.check_no_isolation("`_umtx_op` with `timespec` timeout")?;

                            // `uaddr2` points to a `struct timespec`.
                            let timespec = this.ptr_to_mplace(uaddr2, timespec_layout);
                            let duration = match this.read_timespec(&timespec)? {
                                Some(duration) => duration,
                                None => {
                                    return this
                                        .set_last_error_and_return(LibcError("EINVAL"), dest);
                                }
                            };

                            // FreeBSD does not seem to document which clock is used when the timeout
                            // is passed as a `struct timespec*`. Based on discussions online and the source
                            // code (umtx_copyin_umtx_time() in kern_umtx.c), it seems to default to CLOCK_REALTIME,
                            // so that's what we also do.
                            // Discussion in golang: https://github.com/golang/go/issues/17168#issuecomment-250235271
                            Some((TimeoutClock::RealTime, TimeoutAnchor::Relative, duration))
                        } else {
                            return this.set_last_error_and_return(LibcError("EINVAL"), dest);
                        }
                    };

                    let dest = dest.clone();
                    this.futex_wait(
                        futex_ref,
                        u32::MAX, // we set the bitset to include all bits
                        timeout,
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
                                    ecx.set_last_error_and_return(LibcError("ETIMEDOUT"), &dest)
                                }
                            }
                        ),
                    );
                    interp_ok(())
                } else {
                    // The manual doesn’t specify what should happen if the futex value doesn’t match the expected one.
                    // On FreeBSD 14.2, testing shows that WAIT operations return 0 even when the value is incorrect.
                    this.write_int(0, dest)?;
                    interp_ok(())
                }
            }
            // UMTX_OP_WAKE and UMTX_OP_WAKE_PRIVATE only differ in whether they work across
            // processes or not. For Miri, we can treat them the same.
            op if op == wake || op == wake_private => {
                let Some(futex_ref) =
                    this.get_sync_or_init(obj, |_| FreeBsdFutex { futex: Default::default() })
                else {
                    // From Linux implemenation:
                    // No AllocId, or no live allocation at that AllocId.
                    // Return an error code. (That seems nicer than silently doing something non-intuitive.)
                    // This means that if an address gets reused by a new allocation,
                    // we'll use an independent futex queue for this... that seems acceptable.
                    return this.set_last_error_and_return(LibcError("EFAULT"), dest);
                };
                let futex_ref = futex_ref.futex.clone();

                // Saturating cast for when usize is smaller than u64.
                let count = usize::try_from(val).unwrap_or(usize::MAX);

                // Read the Linux futex wake implementation in Miri to understand why this fence is needed.
                this.atomic_fence(AtomicFenceOrd::SeqCst)?;

                // `_umtx_op` doesn't return the amount of woken threads.
                let _woken = this.futex_wake(
                    &futex_ref,
                    u32::MAX, // we set the bitset to include all bits
                    count,
                )?;

                // From the manual:
                // If successful, all requests, except UMTX_SHM_CREAT and UMTX_SHM_LOOKUP
                // sub-requests of the UMTX_OP_SHM request, will return zero.
                this.write_int(0, dest)?;
                interp_ok(())
            }
            op => {
                throw_unsup_format!("Miri does not support `_umtx_op` syscall with op={}", op)
            }
        }
    }

    /// Parses a `_umtx_time` struct.
    /// Returns `None` if the underlying `timespec` struct is invalid.
    fn read_umtx_time(&mut self, ut: &MPlaceTy<'tcx>) -> InterpResult<'tcx, Option<UmtxTime>> {
        let this = self.eval_context_mut();
        // Only flag allowed is UMTX_ABSTIME.
        let abs_time = this.eval_libc_u32("UMTX_ABSTIME");

        let timespec_place = this.project_field(ut, FieldIdx::from_u32(0))?;
        // Inner `timespec` must still be valid.
        let duration = match this.read_timespec(&timespec_place)? {
            Some(dur) => dur,
            None => return interp_ok(None),
        };

        let flags_place = this.project_field(ut, FieldIdx::from_u32(1))?;
        let flags = this.read_scalar(&flags_place)?.to_u32()?;
        let abs_time_flag = flags == abs_time;

        let clock_id_place = this.project_field(ut, FieldIdx::from_u32(2))?;
        let clock_id = this.read_scalar(&clock_id_place)?;
        let Some(timeout_clock) = this.parse_clockid(clock_id) else {
            throw_unsup_format!("unsupported clock")
        };
        if timeout_clock == TimeoutClock::RealTime {
            this.check_no_isolation("`_umtx_op` with `CLOCK_REALTIME`")?;
        }

        interp_ok(Some(UmtxTime { timeout: duration, abs_time: abs_time_flag, timeout_clock }))
    }
}
