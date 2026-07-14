use std::io;
use std::rc::Rc;
use std::time::Duration;

use rustc_abi::FieldIdx;

use crate::shims::files::{FileDescription, FileDescriptionRef};
use crate::shims::unix::UnixFileDescription;
use crate::*;

/// An `Epoll` file descriptor connects file handles and epoll events
#[derive(Debug)]
pub struct Epoll {
    /// Watcher used for registering interests in the global readiness
    /// interest table.
    watcher: Rc<ReadinessWatcher>,
}

impl VisitProvenance for FileDescriptionRef<Epoll> {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // No provenance anywhere in this type.
    }
}

impl FileDescription for Epoll {
    fn name(&self) -> &'static str {
        "epoll"
    }

    fn metadata<'tcx>(
        &self,
    ) -> InterpResult<'tcx, Either<io::Result<std::fs::Metadata>, &'static str>> {
        // On Linux, epoll is an "anonymous inode" reported as S_IFREG.
        interp_ok(Either::Right("S_IFREG"))
    }

    fn as_unix<'tcx>(
        self: FileDescriptionRef<Self>,
        _ecx: &MiriInterpCx<'tcx>,
    ) -> FileDescriptionRef<dyn UnixFileDescription> {
        self
    }
}

impl UnixFileDescription for Epoll {}

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

        // Miri does not support exec, so EPOLL_CLOEXEC flag has no effect.
        if flags != epoll_cloexec && flags != 0 {
            throw_unsup_format!(
                "epoll_create1: flag {:#x} is unsupported, only 0 or EPOLL_CLOEXEC are allowed",
                flags
            );
        }

        let fd = this
            .machine
            .fds
            .insert_new(Epoll { watcher: Rc::new(this.machine.readiness_interests.new_watcher()) });
        interp_ok(Scalar::from_i32(fd))
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

        let epfd_value = this.read_scalar(epfd)?.to_i32()?;
        let op = this.read_scalar(op)?.to_i32()?;
        let fd = this.read_scalar(fd)?.to_i32()?;
        let event = this.deref_pointer_as(event, this.libc_ty_layout("epoll_event"))?;

        let epoll_ctl_add = this.eval_libc_i32("EPOLL_CTL_ADD");
        let epoll_ctl_mod = this.eval_libc_i32("EPOLL_CTL_MOD");
        let epoll_ctl_del = this.eval_libc_i32("EPOLL_CTL_DEL");
        let epollin = this.eval_libc_u32("EPOLLIN");
        let epollout = this.eval_libc_u32("EPOLLOUT");
        let epollrdhup = this.eval_libc_u32("EPOLLRDHUP");
        let epollet = this.eval_libc_u32("EPOLLET");
        let epollhup = this.eval_libc_u32("EPOLLHUP");
        let epollerr = this.eval_libc_u32("EPOLLERR");

        // Throw EFAULT if epfd and fd have the same value.
        if epfd_value == fd {
            return this.set_errno_and_return_neg1_i32(LibcError("EFAULT"));
        }

        // Check if epfd is a valid epoll file descriptor.
        let Some(epfd) = this.machine.fds.get(epfd_value) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        let epfd = epfd
            .downcast::<Epoll>()
            .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

        let Some(fd_ref) = this.machine.fds.get(fd) else {
            return this.set_errno_and_return_neg1_i32(LibcError("EBADF"));
        };
        let id = fd_ref.id();
        let interest_key = (id, fd);

        if op == epoll_ctl_add || op == epoll_ctl_mod {
            // Read event bitmask and data from epoll_event passed by caller.
            let mut relevant_bitflag =
                this.read_scalar(&this.project_field(&event, FieldIdx::ZERO)?)?.to_u32()?;
            let data = this.read_scalar(&this.project_field(&event, FieldIdx::ONE)?)?.to_u64()?;

            let is_edge_triggered = if relevant_bitflag & epollet == epollet {
                relevant_bitflag &= !epollet;
                true
            } else {
                false
            };

            // Unset the flag we support to discover if any unsupported flags are used.
            let mut flags = relevant_bitflag;
            // epoll_wait(2) will always wait for epollhup and epollerr; it is not
            // necessary to set it in events when calling epoll_ctl().
            // So we will always set these two event types.
            relevant_bitflag |= epollhup;
            relevant_bitflag |= epollerr;

            if flags & epollin == epollin {
                flags &= !epollin;
            }
            if flags & epollout == epollout {
                flags &= !epollout;
            }
            if flags & epollrdhup == epollrdhup {
                flags &= !epollrdhup;
            }
            if flags & epollhup == epollhup {
                flags &= !epollhup;
            }
            if flags & epollerr == epollerr {
                flags &= !epollerr;
            }
            if flags != 0 {
                throw_unsup_format!(
                    "epoll_ctl: encountered unknown unsupported flags {:#x}",
                    flags
                );
            }

            let relevant = this.epoll_bitflag_to_readiness(relevant_bitflag);

            if op == epoll_ctl_add {
                // Add a new interest to the watcher.
                let result =
                    epfd.watcher.add_interest(fd, relevant, is_edge_triggered, data, this)?;
                if result.is_err() {
                    // We already had an interest in this.
                    return this.set_errno_and_return_neg1_i32(LibcError("EEXIST"));
                }
            } else {
                // Modify the existing interest.
                let result = epfd.watcher.update_interest(interest_key, this, |interest| {
                    interest.is_edge_triggered = is_edge_triggered;
                    interest.relevant = relevant;
                    interest.data = data;
                })?;
                if result.is_none() {
                    // There is no interest registered for the specified key.
                    return this.set_errno_and_return_neg1_i32(LibcError("ENOENT"));
                }
            }
        } else if op == epoll_ctl_del {
            if epfd.watcher.remove_interest(interest_key, this).is_none() {
                // We did not have interest in this.
                return this.set_errno_and_return_neg1_i32(LibcError("ENOENT"));
            };
        } else {
            throw_unsup_format!("unsupported epoll_ctl operation: {op}");
        }

        interp_ok(Scalar::from_i32(0))
    }

    /// The `epoll_wait()` system call waits for events on the `Epoll`
    /// instance referred to by the file descriptor `epfd`. The buffer
    /// pointed to by `events` is used to return information from the ready
    /// list about file descriptors in the interest list that have some
    /// events available. Up to `maxevents` are returned by `epoll_wait()`.
    /// The `maxevents` argument must be greater than zero.
    ///
    /// The `timeout` argument specifies the number of milliseconds that
    /// `epoll_wait()` will block. Time is measured against the
    /// CLOCK_MONOTONIC clock. If the timeout is zero, the function will not block,
    /// while if the timeout is -1, the function will block
    /// until at least one event has been retrieved (or an error
    /// occurred).
    ///
    /// A call to `epoll_wait()` will block until either:
    /// • a file descriptor delivers an event;
    /// • the call is interrupted by a signal handler; or
    /// • the timeout expires.
    ///
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
        events_op: &OpTy<'tcx>,
        maxevents: &OpTy<'tcx>,
        timeout: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let epfd_value = this.read_scalar(epfd)?.to_i32()?;
        let events = this.read_immediate(events_op)?;
        let maxevents = this.read_scalar(maxevents)?.to_i32()?;
        let timeout = this.read_scalar(timeout)?.to_i32()?;

        if epfd_value <= 0 || maxevents <= 0 {
            return this.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
        }

        // This needs to come after the maxevents value check, or else maxevents.try_into().unwrap()
        // will fail.
        let event = this.deref_pointer_as(
            &events,
            this.libc_array_ty_layout("epoll_event", maxevents.try_into().unwrap()),
        )?;

        let Some(epfd) = this.machine.fds.get(epfd_value) else {
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };
        let Some(epfd) = epfd.downcast::<Epoll>() else {
            return this.set_errno_and_return_neg1(LibcError("EBADF"), dest);
        };

        if timeout == 0 || epfd.watcher.ready_count() != 0 {
            // If the timeout is 0 or there is a ready event, we can return immediately.
            this.return_ready_list(&epfd, dest, &event)?;
        } else {
            // Blocking, with a relative timeout.
            let deadline = match timeout {
                0.. => {
                    let duration = Duration::from_millis(timeout.try_into().unwrap());
                    Some(this.machine.monotonic_clock.now().add_lossy(duration).into())
                }
                -1 => None,
                ..-1 => {
                    throw_unsup_format!(
                        "epoll_wait: Only timeout values greater than or equal to -1 are supported."
                    );
                }
            };

            // Record this thread as blocked.
            epfd.watcher.add_blocked_thread(this.active_thread());
            // And block it.
            let dest = dest.clone();
            // We keep a strong ref to the underlying `ReadinessWatcher` to make sure it sticks around.
            // This means there'll be a leak if we never wake up, but that anyway would imply
            // a thread is permanently blocked so this is fine.
            this.block_thread(
                BlockReason::Readiness,
                deadline,
                callback!(
                    @capture<'tcx> {
                        epfd: FileDescriptionRef<Epoll>,
                        dest: MPlaceTy<'tcx>,
                        event: MPlaceTy<'tcx>,
                    }
                    |this, unblock: UnblockKind| {
                        match unblock {
                            UnblockKind::Ready => {
                                let events = this.return_ready_list(&epfd, &dest, &event)?;
                                assert!(events > 0, "we got woken up with no events to deliver");
                                interp_ok(())
                            },
                            UnblockKind::TimedOut => {
                                // Remove the current active thread id from the blocked threads list.
                                epfd.watcher.remove_blocked_thread(this.active_thread());
                                this.write_int(0, &dest)?;
                                interp_ok(())
                            },
                        }
                    }
                ),
            );
        }
        interp_ok(())
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextPrivExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Convert a [`Readiness`] instance into the corresponding epoll
    /// readiness bitflag.
    fn readiness_to_epoll_bitflag(&self, readiness: &Readiness) -> u32 {
        let this = self.eval_context_ref();

        let epollin = this.eval_libc_u32("EPOLLIN");
        let epollout = this.eval_libc_u32("EPOLLOUT");
        let epollrdhup = this.eval_libc_u32("EPOLLRDHUP");
        let epollhup = this.eval_libc_u32("EPOLLHUP");
        let epollerr = this.eval_libc_u32("EPOLLERR");

        let mut bitflag = 0;
        if readiness.readable {
            bitflag |= epollin;
        }
        if readiness.writable {
            bitflag |= epollout;
        }
        if readiness.read_closed {
            bitflag |= epollrdhup;
        }
        if readiness.write_closed {
            bitflag |= epollhup;
        }
        if readiness.error {
            bitflag |= epollerr;
        }
        bitflag
    }

    /// Convert an epoll readiness bitflag into the corresponding
    /// [`Readiness`] instance.
    fn epoll_bitflag_to_readiness(&self, bitflag: u32) -> Readiness {
        let this = self.eval_context_ref();

        let epollin = this.eval_libc_u32("EPOLLIN");
        let epollout = this.eval_libc_u32("EPOLLOUT");
        let epollrdhup = this.eval_libc_u32("EPOLLRDHUP");
        let epollhup = this.eval_libc_u32("EPOLLHUP");
        let epollerr = this.eval_libc_u32("EPOLLERR");

        Readiness {
            readable: bitflag & epollin == epollin,
            writable: bitflag & epollout == epollout,
            read_closed: bitflag & epollrdhup == epollrdhup,
            write_closed: bitflag & epollhup == epollhup,
            error: bitflag & epollerr == epollerr,
        }
    }

    /// Stores the ready list of the `epfd` epoll instance into `events` (which must be an array),
    /// and the number of returned events into `dest`.
    fn return_ready_list(
        &mut self,
        epfd: &FileDescriptionRef<Epoll>,
        dest: &MPlaceTy<'tcx>,
        events: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mut num_of_events = 0i32;
        let mut array_iter = this.project_array_fields(events)?;
        let max_events_num: usize = events.len(this)?.try_into().unwrap();

        // We get up to the first `max_events_num` ready events from the
        // watcher and fill them into the slots of the array.
        for interest in epfd.watcher.get_ready_interests(max_events_num, this)? {
            let (_idx, slot) = array_iter.next(this)?.expect("Array should have slot for interest");
            // Deliver event to caller.
            this.write_int_fields_named(
                &[
                    ("events", this.readiness_to_epoll_bitflag(interest.active()).into()),
                    ("u64", interest.data.into()),
                ],
                &slot,
            )?;
            num_of_events = num_of_events.strict_add(1);
            // Synchronize receiving thread with the event of interest.
            this.acquire_clock(interest.clock())?;
        }
        this.write_int(num_of_events, dest)?;
        interp_ok(num_of_events)
    }
}
