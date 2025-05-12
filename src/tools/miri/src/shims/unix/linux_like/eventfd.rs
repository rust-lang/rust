//! Linux `eventfd` implementation.
use std::cell::{Cell, RefCell};
use std::io;
use std::io::ErrorKind;

use crate::concurrency::VClock;
use crate::shims::files::{FileDescription, FileDescriptionRef, WeakFileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::shims::unix::linux_like::epoll::{EpollReadyEvents, EvalContextExt as _};
use crate::*;

/// Maximum value that the eventfd counter can hold.
const MAX_COUNTER: u64 = u64::MAX - 1;

/// A kind of file descriptor created by `eventfd`.
/// The `Event` type isn't currently written to by `eventfd`.
/// The interface is meant to keep track of objects associated
/// with a file descriptor. For more information see the man
/// page below:
///
/// <https://man.netbsd.org/eventfd.2>
#[derive(Debug)]
struct EventFd {
    /// The object contains an unsigned 64-bit integer (uint64_t) counter that is maintained by the
    /// kernel. This counter is initialized with the value specified in the argument initval.
    counter: Cell<u64>,
    is_nonblock: bool,
    clock: RefCell<VClock>,
    /// A list of thread ids blocked on eventfd::read.
    blocked_read_tid: RefCell<Vec<ThreadId>>,
    /// A list of thread ids blocked on eventfd::write.
    blocked_write_tid: RefCell<Vec<ThreadId>>,
}

impl FileDescription for EventFd {
    fn name(&self) -> &'static str {
        "event"
    }

    fn close<'tcx>(
        self,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        interp_ok(Ok(()))
    }

    /// Read the counter in the buffer and return the counter if succeeded.
    fn read<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        // We're treating the buffer as a `u64`.
        let ty = ecx.machine.layouts.u64;
        // Check the size of slice, and return error only if the size of the slice < 8.
        if len < ty.size.bytes_usize() {
            return finish.call(ecx, Err(ErrorKind::InvalidInput.into()));
        }

        // Turn the pointer into a place at the right type.
        let buf_place = ecx.ptr_to_mplace_unaligned(ptr, ty);

        eventfd_read(buf_place, self, ecx, finish)
    }

    /// A write call adds the 8-byte integer value supplied in
    /// its buffer (in native endianness) to the counter.  The maximum value that may be
    /// stored in the counter is the largest unsigned 64-bit value
    /// minus 1 (i.e., 0xfffffffffffffffe).  If the addition would
    /// cause the counter's value to exceed the maximum, then the
    /// write either blocks until a read is performed on the
    /// file descriptor, or fails with the error EAGAIN if the
    /// file descriptor has been made nonblocking.
    ///
    /// A write fails with the error EINVAL if the size of the
    /// supplied buffer is less than 8 bytes, or if an attempt is
    /// made to write the value 0xffffffffffffffff.
    fn write<'tcx>(
        self: FileDescriptionRef<Self>,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        ecx: &mut MiriInterpCx<'tcx>,
        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
    ) -> InterpResult<'tcx> {
        // We're treating the buffer as a `u64`.
        let ty = ecx.machine.layouts.u64;
        // Check the size of slice, and return error only if the size of the slice < 8.
        if len < ty.layout.size.bytes_usize() {
            return finish.call(ecx, Err(ErrorKind::InvalidInput.into()));
        }

        // Turn the pointer into a place at the right type.
        let buf_place = ecx.ptr_to_mplace_unaligned(ptr, ty);

        eventfd_write(buf_place, self, ecx, finish)
    }

    fn as_unix<'tcx>(&self, _ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        self
    }
}

impl UnixFileDescription for EventFd {
    fn get_epoll_ready_events<'tcx>(&self) -> InterpResult<'tcx, EpollReadyEvents> {
        // We only check the status of EPOLLIN and EPOLLOUT flags for eventfd. If other event flags
        // need to be supported in the future, the check should be added here.

        interp_ok(EpollReadyEvents {
            epollin: self.counter.get() != 0,
            epollout: self.counter.get() != MAX_COUNTER,
            ..EpollReadyEvents::new()
        })
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// This function creates an `Event` that is used as an event wait/notify mechanism by
    /// user-space applications, and by the kernel to notify user-space applications of events.
    /// The `Event` contains an `u64` counter maintained by the kernel. The counter is initialized
    /// with the value specified in the `initval` argument.
    ///
    /// A new file descriptor referring to the `Event` is returned. The `read`, `write`, `poll`,
    /// `select`, and `close` operations can be performed on the file descriptor. For more
    /// information on these operations, see the man page linked below.
    ///
    /// The `flags` are not currently implemented for eventfd.
    /// The `flags` may be bitwise ORed to change the behavior of `eventfd`:
    /// `EFD_CLOEXEC` - Set the close-on-exec (`FD_CLOEXEC`) flag on the new file descriptor.
    /// `EFD_NONBLOCK` - Set the `O_NONBLOCK` file status flag on the new open file description.
    /// `EFD_SEMAPHORE` - miri does not support semaphore-like semantics.
    ///
    /// <https://linux.die.net/man/2/eventfd>
    fn eventfd(&mut self, val: &OpTy<'tcx>, flags: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let val = this.read_scalar(val)?.to_u32()?;
        let mut flags = this.read_scalar(flags)?.to_i32()?;

        let efd_cloexec = this.eval_libc_i32("EFD_CLOEXEC");
        let efd_nonblock = this.eval_libc_i32("EFD_NONBLOCK");
        let efd_semaphore = this.eval_libc_i32("EFD_SEMAPHORE");

        if flags & efd_semaphore == efd_semaphore {
            throw_unsup_format!("eventfd: EFD_SEMAPHORE is unsupported");
        }

        let mut is_nonblock = false;
        // Unset the flag that we support.
        // After unloading, flags != 0 means other flags are used.
        if flags & efd_cloexec == efd_cloexec {
            // cloexec is ignored because Miri does not support exec.
            flags &= !efd_cloexec;
        }
        if flags & efd_nonblock == efd_nonblock {
            flags &= !efd_nonblock;
            is_nonblock = true;
        }
        if flags != 0 {
            throw_unsup_format!("eventfd: encountered unknown unsupported flags {:#x}", flags);
        }

        let fds = &mut this.machine.fds;

        let fd_value = fds.insert_new(EventFd {
            counter: Cell::new(val.into()),
            is_nonblock,
            clock: RefCell::new(VClock::default()),
            blocked_read_tid: RefCell::new(Vec::new()),
            blocked_write_tid: RefCell::new(Vec::new()),
        });

        interp_ok(Scalar::from_i32(fd_value))
    }
}

/// Block thread if the value addition will exceed u64::MAX -1,
/// else just add the user-supplied value to current counter.
fn eventfd_write<'tcx>(
    buf_place: MPlaceTy<'tcx>,
    eventfd: FileDescriptionRef<EventFd>,
    ecx: &mut MiriInterpCx<'tcx>,
    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
) -> InterpResult<'tcx> {
    // Figure out which value we should add.
    let num = ecx.read_scalar(&buf_place)?.to_u64()?;
    // u64::MAX as input is invalid because the maximum value of counter is u64::MAX - 1.
    if num == u64::MAX {
        return finish.call(ecx, Err(ErrorKind::InvalidInput.into()));
    }

    match eventfd.counter.get().checked_add(num) {
        Some(new_count @ 0..=MAX_COUNTER) => {
            // Future `read` calls will synchronize with this write, so update the FD clock.
            ecx.release_clock(|clock| {
                eventfd.clock.borrow_mut().join(clock);
            });

            // Store new counter value.
            eventfd.counter.set(new_count);

            // Unblock *all* threads previously blocked on `read`.
            // We need to take out the blocked thread ids and unblock them together,
            // because `unblock_threads` may block them again and end up re-adding the
            // thread to the blocked list.
            let waiting_threads = std::mem::take(&mut *eventfd.blocked_read_tid.borrow_mut());
            // FIXME: We can randomize the order of unblocking.
            for thread_id in waiting_threads {
                ecx.unblock_thread(thread_id, BlockReason::Eventfd)?;
            }

            // The state changed; we check and update the status of all supported event
            // types for current file description.
            ecx.check_and_update_readiness(eventfd)?;

            // Return how many bytes we consumed from the user-provided buffer.
            return finish.call(ecx, Ok(buf_place.layout.size.bytes_usize()));
        }
        None | Some(u64::MAX) => {
            // We can't update the state, so we have to block.
            if eventfd.is_nonblock {
                return finish.call(ecx, Err(ErrorKind::WouldBlock.into()));
            }

            eventfd.blocked_write_tid.borrow_mut().push(ecx.active_thread());

            let weak_eventfd = FileDescriptionRef::downgrade(&eventfd);
            ecx.block_thread(
                BlockReason::Eventfd,
                None,
                callback!(
                    @capture<'tcx> {
                        num: u64,
                        buf_place: MPlaceTy<'tcx>,
                        finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                        weak_eventfd: WeakFileDescriptionRef<EventFd>,
                    }
                    |this, unblock: UnblockKind| {
                        assert_eq!(unblock, UnblockKind::Ready);
                        // When we get unblocked, try again. We know the ref is still valid,
                        // otherwise there couldn't be a `write` that unblocks us.
                        let eventfd_ref = weak_eventfd.upgrade().unwrap();
                        eventfd_write(buf_place, eventfd_ref, this, finish)
                    }
                ),
            );
        }
    };
    interp_ok(())
}

/// Block thread if the current counter is 0,
/// else just return the current counter value to the caller and set the counter to 0.
fn eventfd_read<'tcx>(
    buf_place: MPlaceTy<'tcx>,
    eventfd: FileDescriptionRef<EventFd>,
    ecx: &mut MiriInterpCx<'tcx>,
    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
) -> InterpResult<'tcx> {
    // Set counter to 0, get old value.
    let counter = eventfd.counter.replace(0);

    // Block when counter == 0.
    if counter == 0 {
        if eventfd.is_nonblock {
            return finish.call(ecx, Err(ErrorKind::WouldBlock.into()));
        }

        eventfd.blocked_read_tid.borrow_mut().push(ecx.active_thread());

        let weak_eventfd = FileDescriptionRef::downgrade(&eventfd);
        ecx.block_thread(
            BlockReason::Eventfd,
            None,
            callback!(
                @capture<'tcx> {
                    buf_place: MPlaceTy<'tcx>,
                    finish: DynMachineCallback<'tcx, Result<usize, IoError>>,
                    weak_eventfd: WeakFileDescriptionRef<EventFd>,
                }
                |this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);
                    // When we get unblocked, try again. We know the ref is still valid,
                    // otherwise there couldn't be a `write` that unblocks us.
                    let eventfd_ref = weak_eventfd.upgrade().unwrap();
                    eventfd_read(buf_place, eventfd_ref, this, finish)
                }
            ),
        );
    } else {
        // Synchronize with all prior `write` calls to this FD.
        ecx.acquire_clock(&eventfd.clock.borrow());

        // Return old counter value into user-space buffer.
        ecx.write_int(counter, &buf_place)?;

        // Unblock *all* threads previously blocked on `write`.
        // We need to take out the blocked thread ids and unblock them together,
        // because `unblock_threads` may block them again and end up re-adding the
        // thread to the blocked list.
        let waiting_threads = std::mem::take(&mut *eventfd.blocked_write_tid.borrow_mut());
        // FIXME: We can randomize the order of unblocking.
        for thread_id in waiting_threads {
            ecx.unblock_thread(thread_id, BlockReason::Eventfd)?;
        }

        // The state changed; we check and update the status of all supported event
        // types for current file description.
        ecx.check_and_update_readiness(eventfd)?;

        // Tell userspace how many bytes we put into the buffer.
        return finish.call(ecx, Ok(buf_place.layout.size.bytes_usize()));
    }
    interp_ok(())
}
