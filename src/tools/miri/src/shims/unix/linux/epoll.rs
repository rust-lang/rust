use std::io;

use rustc_data_structures::fx::FxHashMap;

use crate::shims::unix::*;
use crate::*;

use self::shims::unix::fd::FileDescriptor;

/// An `Epoll` file descriptor connects file handles and epoll events
#[derive(Clone, Debug, Default)]
struct Epoll {
    /// The file descriptors we are watching, and what we are watching for.
    file_descriptors: FxHashMap<i32, EpollEvent>,
}

/// Epoll Events associate events with data.
/// These fields are currently unused by miri.
/// This matches the `epoll_event` struct defined
/// by the epoll_ctl man page. For more information
/// see the man page:
///
/// <https://man7.org/linux/man-pages/man2/epoll_ctl.2.html>
#[derive(Clone, Debug)]
struct EpollEvent {
    #[allow(dead_code)]
    events: u32,
    /// `Scalar` is used to represent the
    /// `epoll_data` type union.
    #[allow(dead_code)]
    data: Scalar,
}

impl FileDescription for Epoll {
    fn name(&self) -> &'static str {
        "epoll"
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
    ) -> InterpResult<'tcx, io::Result<()>> {
        Ok(Ok(()))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// This function returns a file descriptor referring to the new `Epoll` instance. This file
    /// descriptor is used for all subsequent calls to the epoll interface. If the `flags` argument
    /// is 0, then this function is the same as `epoll_create()`.
    ///
    /// <https://linux.die.net/man/2/epoll_create1>
    fn epoll_create1(&mut self, flags: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let flags = this.read_scalar(flags)?.to_i32()?;

        let epoll_cloexec = this.eval_libc_i32("EPOLL_CLOEXEC");
        if flags == epoll_cloexec {
            // Miri does not support exec, so this flag has no effect.
        } else if flags != 0 {
            throw_unsup_format!("epoll_create1 flags {flags} are not implemented");
        }

        let fd = this.machine.fds.insert_fd(FileDescriptor::new(Epoll::default()));
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
        epfd: &OpTy<'tcx>,
        op: &OpTy<'tcx>,
        fd: &OpTy<'tcx>,
        event: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let epfd = this.read_scalar(epfd)?.to_i32()?;
        let op = this.read_scalar(op)?.to_i32()?;
        let fd = this.read_scalar(fd)?.to_i32()?;
        let _event = this.read_scalar(event)?.to_pointer(this)?;

        let epoll_ctl_add = this.eval_libc_i32("EPOLL_CTL_ADD");
        let epoll_ctl_mod = this.eval_libc_i32("EPOLL_CTL_MOD");
        let epoll_ctl_del = this.eval_libc_i32("EPOLL_CTL_DEL");

        if op == epoll_ctl_add || op == epoll_ctl_mod {
            let event = this.deref_pointer_as(event, this.libc_ty_layout("epoll_event"))?;

            let events = this.project_field(&event, 0)?;
            let events = this.read_scalar(&events)?.to_u32()?;
            let data = this.project_field(&event, 1)?;
            let data = this.read_scalar(&data)?;
            let event = EpollEvent { events, data };

            let Some(mut epfd) = this.machine.fds.get_mut(epfd) else {
                return Ok(Scalar::from_i32(this.fd_not_found()?));
            };
            let epfd = epfd
                .downcast_mut::<Epoll>()
                .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

            epfd.file_descriptors.insert(fd, event);
            Ok(Scalar::from_i32(0))
        } else if op == epoll_ctl_del {
            let Some(mut epfd) = this.machine.fds.get_mut(epfd) else {
                return Ok(Scalar::from_i32(this.fd_not_found()?));
            };
            let epfd = epfd
                .downcast_mut::<Epoll>()
                .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

            epfd.file_descriptors.remove(&fd);
            Ok(Scalar::from_i32(0))
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
        epfd: &OpTy<'tcx>,
        events: &OpTy<'tcx>,
        maxevents: &OpTy<'tcx>,
        timeout: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let epfd = this.read_scalar(epfd)?.to_i32()?;
        let _events = this.read_scalar(events)?.to_pointer(this)?;
        let _maxevents = this.read_scalar(maxevents)?.to_i32()?;
        let _timeout = this.read_scalar(timeout)?.to_i32()?;

        let Some(mut epfd) = this.machine.fds.get_mut(epfd) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        let _epfd = epfd
            .downcast_mut::<Epoll>()
            .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_wait`"))?;

        // FIXME return number of events ready when scheme for marking events ready exists
        throw_unsup_format!("returning ready events from epoll_wait is not yet implemented");
    }
}
