//! Linux `eventfd` implementation.
use std::cell::{Cell, RefCell};
use std::io;
use std::io::ErrorKind;

use crate::concurrency::VClock;
use crate::shims::unix::fd::FileDescriptionRef;
use crate::shims::unix::linux::epoll::{EpollReadyEvents, EvalContextExt as _};
use crate::shims::unix::*;
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
struct Event {
    /// The object contains an unsigned 64-bit integer (uint64_t) counter that is maintained by the
    /// kernel. This counter is initialized with the value specified in the argument initval.
    counter: Cell<u64>,
    is_nonblock: bool,
    clock: RefCell<VClock>,
}

impl FileDescription for Event {
    fn name(&self) -> &'static str {
        "event"
    }

    fn get_epoll_ready_events<'tcx>(&self) -> InterpResult<'tcx, EpollReadyEvents> {
        // We only check the status of EPOLLIN and EPOLLOUT flags for eventfd. If other event flags
        // need to be supported in the future, the check should be added here.

        interp_ok(EpollReadyEvents {
            epollin: self.counter.get() != 0,
            epollout: self.counter.get() != MAX_COUNTER,
            ..EpollReadyEvents::new()
        })
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        interp_ok(Ok(()))
    }

    /// Read the counter in the buffer and return the counter if succeeded.
    fn read<'tcx>(
        &self,
        self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        // We're treating the buffer as a `u64`.
        let ty = ecx.machine.layouts.u64;
        // Check the size of slice, and return error only if the size of the slice < 8.
        if len < ty.size.bytes_usize() {
            return ecx.set_last_error_and_return(ErrorKind::InvalidInput, dest);
        }

        // eventfd read at the size of u64.
        let buf_place = ecx.ptr_to_mplace_unaligned(ptr, ty);

        // Block when counter == 0.
        let counter = self.counter.get();
        if counter == 0 {
            if self.is_nonblock {
                return ecx.set_last_error_and_return(ErrorKind::WouldBlock, dest);
            }

            throw_unsup_format!("eventfd: blocking is unsupported");
        } else {
            // Synchronize with all prior `write` calls to this FD.
            ecx.acquire_clock(&self.clock.borrow());

            // Give old counter value to userspace, and set counter value to 0.
            ecx.write_int(counter, &buf_place)?;
            self.counter.set(0);

            // When any of the event happened, we check and update the status of all supported event
            // types for current file description.
            ecx.check_and_update_readiness(self_ref)?;

            // Tell userspace how many bytes we wrote.
            ecx.write_int(buf_place.layout.size.bytes(), dest)?;
        }

        interp_ok(())
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
        &self,
        self_ref: &FileDescriptionRef,
        _communicate_allowed: bool,
        ptr: Pointer,
        len: usize,
        dest: &MPlaceTy<'tcx>,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx> {
        // We're treating the buffer as a `u64`.
        let ty = ecx.machine.layouts.u64;
        // Check the size of slice, and return error only if the size of the slice < 8.
        if len < ty.layout.size.bytes_usize() {
            return ecx.set_last_error_and_return(ErrorKind::InvalidInput, dest);
        }

        // Read the user supplied value from the pointer.
        let buf_place = ecx.ptr_to_mplace_unaligned(ptr, ty);
        let num = ecx.read_scalar(&buf_place)?.to_u64()?;

        // u64::MAX as input is invalid because the maximum value of counter is u64::MAX - 1.
        if num == u64::MAX {
            return ecx.set_last_error_and_return(ErrorKind::InvalidInput, dest);
        }
        // If the addition does not let the counter to exceed the maximum value, update the counter.
        // Else, block.
        match self.counter.get().checked_add(num) {
            Some(new_count @ 0..=MAX_COUNTER) => {
                // Future `read` calls will synchronize with this write, so update the FD clock.
                if let Some(clock) = &ecx.release_clock() {
                    self.clock.borrow_mut().join(clock);
                }
                self.counter.set(new_count);
            }
            None | Some(u64::MAX) =>
                if self.is_nonblock {
                    return ecx.set_last_error_and_return(ErrorKind::WouldBlock, dest);
                } else {
                    throw_unsup_format!("eventfd: blocking is unsupported");
                },
        };
        // When any of the event happened, we check and update the status of all supported event
        // types for current file description.
        ecx.check_and_update_readiness(self_ref)?;

        // Return how many bytes we read.
        ecx.write_int(buf_place.layout.size.bytes(), dest)
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

        // eventfd is Linux specific.
        this.assert_target_os("linux", "eventfd");

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

        let fd_value = fds.insert_new(Event {
            counter: Cell::new(val.into()),
            is_nonblock,
            clock: RefCell::new(VClock::default()),
        });

        interp_ok(Scalar::from_i32(fd_value))
    }
}
