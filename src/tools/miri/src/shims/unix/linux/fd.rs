use rustc_middle::ty::ScalarInt;

use crate::*;
use epoll::{Epoll, EpollEvent};
use event::Event;
use socketpair::SocketPair;

use shims::unix::fs::EvalContextExt as _;

use std::cell::Cell;

pub mod epoll;
pub mod event;
pub mod socketpair;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// This function returns a file descriptor referring to the new `Epoll` instance. This file
    /// descriptor is used for all subsequent calls to the epoll interface. If the `flags` argument
    /// is 0, then this function is the same as `epoll_create()`.
    ///
    /// <https://linux.die.net/man/2/epoll_create1>
    fn epoll_create1(
        &mut self,
        flags: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let flags = this.read_scalar(flags)?.to_i32()?;

        let epoll_cloexec = this.eval_libc_i32("EPOLL_CLOEXEC");
        if flags == epoll_cloexec {
            // Miri does not support exec, so this flag has no effect.
        } else if flags != 0 {
            throw_unsup_format!("epoll_create1 flags {flags} are not implemented");
        }

        let fd = this.machine.file_handler.insert_fd(Box::new(Epoll::default()));
        Ok(Scalar::from_i32(fd))
    }

    /// This function performs control operations on the `Epoll` instance referred to by the file
    /// descriptor `epfd`. It requests that the operation `op` be performed for the target file
    /// descriptor, `fd`.
    ///
    /// Valid values for the op argument are:
    /// `EPOLL_CTL_ADD` - Register the target file descriptor `fd` on the `Epoll` instance referred
    /// to by the file descriptor `epfd` and associate the event `event` with the internal file
    /// linked to `fd`.
    /// `EPOLL_CTL_MOD` - Change the event `event` associated with the target file descriptor `fd`.
    /// `EPOLL_CTL_DEL` - Deregister the target file descriptor `fd` from the `Epoll` instance
    /// referred to by `epfd`. The `event` is ignored and can be null.
    ///
    /// <https://linux.die.net/man/2/epoll_ctl>
    fn epoll_ctl(
        &mut self,
        epfd: &OpTy<'tcx, Provenance>,
        op: &OpTy<'tcx, Provenance>,
        fd: &OpTy<'tcx, Provenance>,
        event: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let epfd = this.read_scalar(epfd)?.to_i32()?;
        let op = this.read_scalar(op)?.to_i32()?;
        let fd = this.read_scalar(fd)?.to_i32()?;
        let _event = this.read_scalar(event)?.to_pointer(this)?;

        let epoll_ctl_add = this.eval_libc_i32("EPOLL_CTL_ADD");
        let epoll_ctl_mod = this.eval_libc_i32("EPOLL_CTL_MOD");
        let epoll_ctl_del = this.eval_libc_i32("EPOLL_CTL_DEL");

        if op == epoll_ctl_add || op == epoll_ctl_mod {
            let event = this.deref_operand_as(event, this.libc_ty_layout("epoll_event"))?;

            let events = this.project_field(&event, 0)?;
            let events = this.read_scalar(&events.into())?.to_u32()?;
            let data = this.project_field(&event, 1)?;
            let data = this.read_scalar(&data.into())?;
            let event = EpollEvent { events, data };

            if let Some(epfd) = this.machine.file_handler.handles.get_mut(&epfd) {
                let epfd = epfd
                    .as_any_mut()
                    .downcast_mut::<Epoll>()
                    .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

                epfd.file_descriptors.insert(fd, event);
                Ok(Scalar::from_i32(0))
            } else {
                Ok(Scalar::from_i32(this.handle_not_found()?))
            }
        } else if op == epoll_ctl_del {
            if let Some(epfd) = this.machine.file_handler.handles.get_mut(&epfd) {
                let epfd = epfd
                    .as_any_mut()
                    .downcast_mut::<Epoll>()
                    .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

                epfd.file_descriptors.remove(&fd);
                Ok(Scalar::from_i32(0))
            } else {
                Ok(Scalar::from_i32(this.handle_not_found()?))
            }
        } else {
            let einval = this.eval_libc("EINVAL");
            this.set_last_error(einval)?;
            Ok(Scalar::from_i32(-1))
        }
    }

    /// The `epoll_wait()` system call waits for events on the `Epoll`
    /// instance referred to by the file descriptor `epfd`. The buffer
    /// pointed to by `events` is used to return information from the ready
    /// list about file descriptors in the interest list that have some
    /// events available. Up to `maxevents` are returned by `epoll_wait()`.
    /// The `maxevents` argument must be greater than zero.

    /// The `timeout` argument specifies the number of milliseconds that
    /// `epoll_wait()` will block. Time is measured against the
    /// CLOCK_MONOTONIC clock.

    /// A call to `epoll_wait()` will block until either:
    /// • a file descriptor delivers an event;
    /// • the call is interrupted by a signal handler; or
    /// • the timeout expires.

    /// Note that the timeout interval will be rounded up to the system
    /// clock granularity, and kernel scheduling delays mean that the
    /// blocking interval may overrun by a small amount. Specifying a
    /// timeout of -1 causes `epoll_wait()` to block indefinitely, while
    /// specifying a timeout equal to zero cause `epoll_wait()` to return
    /// immediately, even if no events are available.
    ///
    /// On success, `epoll_wait()` returns the number of file descriptors
    /// ready for the requested I/O, or zero if no file descriptor became
    /// ready during the requested timeout milliseconds. On failure,
    /// `epoll_wait()` returns -1 and errno is set to indicate the error.
    ///
    /// <https://man7.org/linux/man-pages/man2/epoll_wait.2.html>
    fn epoll_wait(
        &mut self,
        epfd: &OpTy<'tcx, Provenance>,
        events: &OpTy<'tcx, Provenance>,
        maxevents: &OpTy<'tcx, Provenance>,
        timeout: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let epfd = this.read_scalar(epfd)?.to_i32()?;
        let _events = this.read_scalar(events)?.to_pointer(this)?;
        let _maxevents = this.read_scalar(maxevents)?.to_i32()?;
        let _timeout = this.read_scalar(timeout)?.to_i32()?;

        if let Some(epfd) = this.machine.file_handler.handles.get_mut(&epfd) {
            let _epfd = epfd
                .as_any_mut()
                .downcast_mut::<Epoll>()
                .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_wait`"))?;

            // FIXME return number of events ready when scheme for marking events ready exists
            throw_unsup_format!("returning ready events from epoll_wait is not yet implemented");
        } else {
            Ok(Scalar::from_i32(this.handle_not_found()?))
        }
    }

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
    #[expect(clippy::needless_if)]
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

        if flags & (efd_cloexec | efd_nonblock | efd_semaphore) == 0 {
            throw_unsup_format!("{flags} is unsupported");
        }
        // FIXME handle the cloexec and nonblock flags
        if flags & efd_cloexec == efd_cloexec {}
        if flags & efd_nonblock == efd_nonblock {}
        if flags & efd_semaphore == efd_semaphore {
            throw_unsup_format!("EFD_SEMAPHORE is unsupported");
        }

        let fh = &mut this.machine.file_handler;
        let fd = fh.insert_fd(Box::new(Event { val: Cell::new(val.into()) }));
        Ok(Scalar::from_i32(fd))
    }

    /// Currently this function creates new `SocketPair`s without specifying the domain, type, or
    /// protocol of the new socket and these are stored in the socket values `sv` argument.
    ///
    /// This function creates an unnamed pair of connected sockets in the specified domain, of the
    /// specified type, and using the optionally specified protocol.
    ///
    /// The `domain` argument specified a communication domain; this selects the protocol family
    /// used for communication. The socket `type` specifies the communication semantics.
    /// The `protocol` specifies a particular protocol to use with the socket. Normally there's
    /// only a single protocol supported for a particular socket type within a given protocol
    /// family, in which case `protocol` can be specified as 0. It is possible that many protocols
    /// exist and in that case, a particular protocol must be specified.
    ///
    /// For more information on the arguments see the socket manpage:
    /// <https://linux.die.net/man/2/socket>
    ///
    /// <https://linux.die.net/man/2/socketpair>
    fn socketpair(
        &mut self,
        domain: &OpTy<'tcx, Provenance>,
        type_: &OpTy<'tcx, Provenance>,
        protocol: &OpTy<'tcx, Provenance>,
        sv: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let _domain = this.read_scalar(domain)?.to_i32()?;
        let _type_ = this.read_scalar(type_)?.to_i32()?;
        let _protocol = this.read_scalar(protocol)?.to_i32()?;
        let sv = this.deref_operand(sv)?;

        let fh = &mut this.machine.file_handler;
        let sv0 = fh.insert_fd(Box::new(SocketPair));
        let sv0 = ScalarInt::try_from_int(sv0, sv.layout.size).unwrap();
        let sv1 = fh.insert_fd(Box::new(SocketPair));
        let sv1 = ScalarInt::try_from_int(sv1, sv.layout.size).unwrap();

        this.write_scalar(sv0, &sv.into())?;
        this.write_scalar(sv1, &sv.offset(sv.layout.size, sv.layout, this)?.into())?;

        Ok(Scalar::from_i32(0))
    }
}
