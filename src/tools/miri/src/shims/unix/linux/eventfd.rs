//! Linux `eventfd` implementation.
//! Currently just a stub.
use std::io;

use rustc_target::abi::Endian;

use crate::shims::unix::*;
use crate::*;

use self::shims::unix::fd::FileDescriptor;

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
    val: u64,
}

impl FileDescription for Event {
    fn name(&self) -> &'static str {
        "event"
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<()>> {
        Ok(Ok(()))
    }

    /// A write call adds the 8-byte integer value supplied in
    /// its buffer (in native endianness) to the counter.  The maximum value that may be
    /// stored in the counter is the largest unsigned 64-bit value
    /// minus 1 (i.e., 0xfffffffffffffffe).  If the addition would
    /// cause the counter's value to exceed the maximum, then the
    /// write either blocks until a read is performed on the
    /// file descriptor, or fails with the error EAGAIN if the
    /// file descriptor has been made nonblocking.

    /// A write fails with the error EINVAL if the size of the
    /// supplied buffer is less than 8 bytes, or if an attempt is
    /// made to write the value 0xffffffffffffffff.
    fn write<'tcx>(
        &mut self,
        _communicate_allowed: bool,
        bytes: &[u8],
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<usize>> {
        let bytes: [u8; 8] = bytes.try_into().unwrap(); // FIXME fail gracefully when this has the wrong size
        // Convert from target endianness to host endianness.
        let num = match ecx.tcx.sess.target.endian {
            Endian::Little => u64::from_le_bytes(bytes),
            Endian::Big => u64::from_be_bytes(bytes),
        };
        // FIXME handle blocking when addition results in exceeding the max u64 value
        // or fail with EAGAIN if the file descriptor is nonblocking.
        self.val = self.val.checked_add(num).unwrap();
        Ok(Ok(8))
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
    fn eventfd(
        &mut self,
        val: &OpTy<'tcx, Provenance>,
        flags: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let val = this.read_scalar(val)?.to_u32()?;
        let flags = this.read_scalar(flags)?.to_i32()?;

        let efd_cloexec = this.eval_libc_i32("EFD_CLOEXEC");
        let efd_nonblock = this.eval_libc_i32("EFD_NONBLOCK");
        let efd_semaphore = this.eval_libc_i32("EFD_SEMAPHORE");

        if flags & (efd_cloexec | efd_nonblock | efd_semaphore) != flags {
            throw_unsup_format!("eventfd: flag {flags:#x} is unsupported");
        }
        if flags & efd_cloexec == efd_cloexec {
            // cloexec does nothing as we don't support `exec`
        }
        if flags & efd_nonblock == efd_nonblock {
            // FIXME remember the nonblock flag
        }
        if flags & efd_semaphore == efd_semaphore {
            throw_unsup_format!("eventfd: EFD_SEMAPHORE is unsupported");
        }

        let fd = this.machine.fds.insert_fd(FileDescriptor::new(Event { val: val.into() }));
        Ok(Scalar::from_i32(fd))
    }
}
