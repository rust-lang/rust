use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::time::Duration;

use rustc_abi::FieldIdx;

use crate::concurrency::VClock;
use crate::shims::files::{
    DynFileDescriptionRef, FdId, FdNum, FileDescription, FileDescriptionRef, WeakFileDescriptionRef,
};
use crate::shims::unix::UnixFileDescription;
use crate::*;

type EpollEventKey = (FdId, FdNum);

/// An `Epoll` file descriptor connects file handles and epoll events
#[derive(Debug, Default)]
struct Epoll {
    /// A map of EpollEventInterests registered under this epoll instance.
    /// Each entry is differentiated using FdId and file descriptor value.
    interest_list: RefCell<BTreeMap<EpollEventKey, EpollEventInterest>>,
    /// A map of EpollEventInstance that will be returned when `epoll_wait` is called.
    /// Similar to interest_list, the entry is also differentiated using FdId
    /// and file descriptor value.
    /// We keep this separate from `interest_list` for two reasons: there might be many
    /// interests but only a few of them ready (so with a separate list it is more efficient
    /// to find a ready event), and having separate `RefCell` lets us mutate the `interest_list`
    /// while unblocking threads which might mutate the `ready_list`.
    ready_list: RefCell<BTreeMap<EpollEventKey, EpollEventInstance>>,
    /// A list of thread ids blocked on this epoll instance.
    blocked_tid: RefCell<Vec<ThreadId>>,
}

impl VisitProvenance for Epoll {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // No provenance anywhere in this type.
    }
}

/// Returns the range of all EpollEventKey for the given FD ID.
fn range_for_id(id: FdId) -> std::ops::RangeInclusive<EpollEventKey> {
    (id, 0)..=(id, i32::MAX)
}

/// EpollEventInstance contains information that will be returned by epoll_wait.
#[derive(Debug)]
pub struct EpollEventInstance {
    /// Bitmask of event types that happened to the file description.
    events: u32,
    /// User-defined data associated with the interest that triggered this instance.
    data: u64,
    /// The release clock associated with this event.
    clock: VClock,
}

impl EpollEventInstance {
    pub fn new(events: u32, data: u64) -> EpollEventInstance {
        EpollEventInstance { events, data, clock: Default::default() }
    }
}

/// EpollEventInterest registers the file description information to an epoll
/// instance during a successful `epoll_ctl` call. It also stores additional
/// information needed to check and update readiness state for `epoll_wait`.
///
/// `events` and `data` field matches the `epoll_event` struct defined
/// by the epoll_ctl man page. For more information
/// see the man page:
///
/// <https://man7.org/linux/man-pages/man2/epoll_ctl.2.html>
#[derive(Debug, Copy, Clone)]
pub struct EpollEventInterest {
    /// The events bitmask retrieved from `epoll_event`.
    events: u32,
    /// The data retrieved from `epoll_event`.
    /// libc's data field in epoll_event can store integer or pointer,
    /// but only u64 is supported for now.
    /// <https://man7.org/linux/man-pages/man3/epoll_event.3type.html>
    data: u64,
}

/// EpollReadyEvents reflects the readiness of a file description.
#[derive(Debug)]
pub struct EpollReadyEvents {
    /// The associated file is available for read(2) operations, in the sense that a read will not block.
    /// (I.e., returning EOF is considered "ready".)
    pub epollin: bool,
    /// The associated file is available for write(2) operations, in the sense that a write will not block.
    pub epollout: bool,
    /// Stream socket peer closed connection, or shut down writing
    /// half of connection.
    pub epollrdhup: bool,
    /// For stream socket, this event merely indicates that the peer
    /// closed its end of the channel.
    /// Unlike epollrdhup, this should only be set when the stream is fully closed.
    /// epollrdhup also gets set when only the write half is closed, which is possible
    /// via `shutdown(_, SHUT_WR)`.
    pub epollhup: bool,
    /// Error condition happened on the associated file descriptor.
    pub epollerr: bool,
}

impl EpollReadyEvents {
    pub fn new() -> Self {
        EpollReadyEvents {
            epollin: false,
            epollout: false,
            epollrdhup: false,
            epollhup: false,
            epollerr: false,
        }
    }

    pub fn get_event_bitmask<'tcx>(&self, ecx: &MiriInterpCx<'tcx>) -> u32 {
        let epollin = ecx.eval_libc_u32("EPOLLIN");
        let epollout = ecx.eval_libc_u32("EPOLLOUT");
        let epollrdhup = ecx.eval_libc_u32("EPOLLRDHUP");
        let epollhup = ecx.eval_libc_u32("EPOLLHUP");
        let epollerr = ecx.eval_libc_u32("EPOLLERR");

        let mut bitmask = 0;
        if self.epollin {
            bitmask |= epollin;
        }
        if self.epollout {
            bitmask |= epollout;
        }
        if self.epollrdhup {
            bitmask |= epollrdhup;
        }
        if self.epollhup {
            bitmask |= epollhup;
        }
        if self.epollerr {
            bitmask |= epollerr;
        }
        bitmask
    }
}

impl FileDescription for Epoll {
    fn name(&self) -> &'static str {
        "epoll"
    }

    fn destroy<'tcx>(
        mut self,
        self_addr: usize,
        _communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        // If we were interested in some FDs, we can remove that now.
        let mut ids = self.interest_list.get_mut().keys().map(|(id, _num)| *id).collect::<Vec<_>>();
        ids.dedup(); // they come out of the map sorted
        for id in ids {
            ecx.machine.epoll_interests.remove(id, self_addr);
        }
        interp_ok(Ok(()))
    }

    fn as_unix<'tcx>(&self, _ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        self
    }
}

impl UnixFileDescription for Epoll {}

/// The table of all EpollEventInterest.
/// This tracks, for each file description, which epoll instances have an interest in events
/// for this file description.
pub struct EpollInterestTable(BTreeMap<FdId, Vec<WeakFileDescriptionRef<Epoll>>>);

impl EpollInterestTable {
    pub(crate) fn new() -> Self {
        EpollInterestTable(BTreeMap::new())
    }

    fn insert(&mut self, id: FdId, epoll: WeakFileDescriptionRef<Epoll>) {
        let epolls = self.0.entry(id).or_default();
        epolls.push(epoll);
    }

    fn remove(&mut self, id: FdId, epoll_addr: usize) {
        let epolls = self.0.entry(id).or_default();
        // FIXME: linear scan. Keep the list sorted so we can do binary search?
        let idx = epolls
            .iter()
            .position(|old_ref| old_ref.addr() == epoll_addr)
            .expect("trying to remove an epoll that's not in the list");
        epolls.remove(idx);
    }

    fn get_epolls(&self, id: FdId) -> Option<&Vec<WeakFileDescriptionRef<Epoll>>> {
        self.0.get(&id)
    }

    pub fn remove_epolls(&mut self, id: FdId) {
        if let Some(epolls) = self.0.remove(&id) {
            for epoll in epolls.iter().filter_map(|e| e.upgrade()) {
                // This is a still-live epoll with interest in this FD. Remove all
                // relevent interests.
                epoll
                    .interest_list
                    .borrow_mut()
                    .extract_if(range_for_id(id), |_, _| true)
                    // Consume the iterator.
                    .for_each(|_| ());
                // Also remove all events from the ready list that refer to this FD.
                epoll
                    .ready_list
                    .borrow_mut()
                    .extract_if(range_for_id(id), |_, _| true)
                    // Consume the iterator.
                    .for_each(|_| ());
            }
        }
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

        // Miri does not support exec, so EPOLL_CLOEXEC flag has no effect.
        if flags != epoll_cloexec && flags != 0 {
            throw_unsup_format!(
                "epoll_create1: flag {:#x} is unsupported, only 0 or EPOLL_CLOEXEC are allowed",
                flags
            );
        }

        let fd = this.machine.fds.insert_new(Epoll::default());
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
            return this.set_last_error_and_return_i32(LibcError("EFAULT"));
        }

        // Check if epfd is a valid epoll file descriptor.
        let Some(epfd) = this.machine.fds.get(epfd_value) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };
        let epfd = epfd
            .downcast::<Epoll>()
            .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

        let mut interest_list = epfd.interest_list.borrow_mut();

        let Some(fd_ref) = this.machine.fds.get(fd) else {
            return this.set_last_error_and_return_i32(LibcError("EBADF"));
        };
        let id = fd_ref.id();

        if op == epoll_ctl_add || op == epoll_ctl_mod {
            // Read event bitmask and data from epoll_event passed by caller.
            let mut events =
                this.read_scalar(&this.project_field(&event, FieldIdx::ZERO)?)?.to_u32()?;
            let data = this.read_scalar(&this.project_field(&event, FieldIdx::ONE)?)?.to_u64()?;

            // Unset the flag we support to discover if any unsupported flags are used.
            let mut flags = events;
            // epoll_wait(2) will always wait for epollhup and epollerr; it is not
            // necessary to set it in events when calling epoll_ctl().
            // So we will always set these two event types.
            events |= epollhup;
            events |= epollerr;

            if events & epollet != epollet {
                // We only support edge-triggered notification for now.
                throw_unsup_format!("epoll_ctl: epollet flag must be included.");
            } else {
                flags &= !epollet;
            }
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

            // Add new interest to list.
            let epoll_key = (id, fd);
            let new_interest = EpollEventInterest { events, data };
            if op == epoll_ctl_add {
                if interest_list.range(range_for_id(id)).next().is_none() {
                    // This is the first time this FD got added to this epoll.
                    // Remember that in the global list so we get notified about FD events.
                    this.machine.epoll_interests.insert(id, FileDescriptionRef::downgrade(&epfd));
                }
                if interest_list.insert(epoll_key, new_interest).is_some() {
                    // We already had interest in this.
                    return this.set_last_error_and_return_i32(LibcError("EEXIST"));
                }
            } else {
                // Modify the existing interest.
                let Some(interest) = interest_list.get_mut(&epoll_key) else {
                    return this.set_last_error_and_return_i32(LibcError("ENOENT"));
                };
                *interest = new_interest;
            };

            // Deliver events for the new interest.
            send_ready_events_to_interests(
                this,
                &epfd,
                fd_ref.as_unix(this).get_epoll_ready_events()?.get_event_bitmask(this),
                std::iter::once((&epoll_key, &new_interest)),
            )?;

            interp_ok(Scalar::from_i32(0))
        } else if op == epoll_ctl_del {
            let epoll_key = (id, fd);

            // Remove epoll_event_interest from interest_list.
            if interest_list.remove(&epoll_key).is_none() {
                // We did not have interest in this.
                return this.set_last_error_and_return_i32(LibcError("ENOENT"));
            };
            // If this was the last interest in this FD, remove us from the global list
            // of who is interested in this FD.
            if interest_list.range(range_for_id(id)).next().is_none() {
                this.machine.epoll_interests.remove(id, epfd.addr());
            }

            // Remove related epoll_interest from ready list.
            epfd.ready_list.borrow_mut().remove(&epoll_key);

            interp_ok(Scalar::from_i32(0))
        } else {
            throw_unsup_format!("unsupported epoll_ctl operation: {op}");
        }
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
            return this.set_last_error_and_return(LibcError("EINVAL"), dest);
        }

        // This needs to come after the maxevents value check, or else maxevents.try_into().unwrap()
        // will fail.
        let event = this.deref_pointer_as(
            &events,
            this.libc_array_ty_layout("epoll_event", maxevents.try_into().unwrap()),
        )?;

        let Some(epfd) = this.machine.fds.get(epfd_value) else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };
        let Some(epfd) = epfd.downcast::<Epoll>() else {
            return this.set_last_error_and_return(LibcError("EBADF"), dest);
        };

        // We just need to know if the ready list is empty and borrow the thread_ids out.
        let ready_list_empty = epfd.ready_list.borrow().is_empty();
        if timeout == 0 || !ready_list_empty {
            // If the ready list is not empty, or the timeout is 0, we can return immediately.
            return_ready_list(&epfd, dest, &event, this)?;
        } else {
            // Blocking
            let timeout = match timeout {
                0.. => {
                    let duration = Duration::from_millis(timeout.try_into().unwrap());
                    Some((TimeoutClock::Monotonic, TimeoutAnchor::Relative, duration))
                }
                -1 => None,
                ..-1 => {
                    throw_unsup_format!(
                        "epoll_wait: Only timeout values greater than or equal to -1 are supported."
                    );
                }
            };
            // Record this thread as blocked.
            epfd.blocked_tid.borrow_mut().push(this.active_thread());
            // And block it.
            let dest = dest.clone();
            // We keep a strong ref to the underlying `Epoll` to make sure it sticks around.
            // This means there'll be a leak if we never wake up, but that anyway would imply
            // a thread is permanently blocked so this is fine.
            this.block_thread(
                BlockReason::Epoll,
                timeout,
                callback!(
                    @capture<'tcx> {
                        epfd: FileDescriptionRef<Epoll>,
                        dest: MPlaceTy<'tcx>,
                        event: MPlaceTy<'tcx>,
                    }
                    |this, unblock: UnblockKind| {
                        match unblock {
                            UnblockKind::Ready => {
                                return_ready_list(&epfd, &dest, &event, this)?;
                                interp_ok(())
                            },
                            UnblockKind::TimedOut => {
                                // Remove the current active thread_id from the blocked thread_id list.
                                epfd
                                    .blocked_tid.borrow_mut()
                                    .retain(|&id| id != this.active_thread());
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

    /// For a specific file description, get its ready events and send it to everyone who registered
    /// interest in this FD. This function should be called whenever an event causes more bytes or
    /// an EOF to become newly readable from an FD, and whenever more bytes can be written to an FD
    /// or no more future writes are possible.
    ///
    /// This *will* report an event if anyone is subscribed to it, without any further filtering, so
    /// do not call this function when an FD didn't have anything happen to it!
    fn epoll_send_fd_ready_events(&mut self, fd_ref: DynFileDescriptionRef) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = fd_ref.id();
        // Figure out who is interested in this. We need to clone this list since we can't prove
        // that `send_ready_events_to_interest` won't mutate it.
        let Some(epolls) = this.machine.epoll_interests.get_epolls(id) else {
            return interp_ok(());
        };
        let epolls = epolls
            .iter()
            .map(|weak| {
                weak.upgrade()
                    .expect("someone forgot to remove the garbage from `machine.epoll_interests`")
            })
            .collect::<Vec<_>>();
        let event_bitmask = fd_ref.as_unix(this).get_epoll_ready_events()?.get_event_bitmask(this);
        for epoll in epolls {
            send_ready_events_to_interests(
                this,
                &epoll,
                event_bitmask,
                epoll.interest_list.borrow().range(range_for_id(id)),
            )?;
        }

        interp_ok(())
    }
}

/// Send the latest ready events for one particular FD (identified by `event_key`) to everyone in
/// the `interests` list, if they are interested in this kind of event.
fn send_ready_events_to_interests<'tcx, 'a>(
    ecx: &mut MiriInterpCx<'tcx>,
    epoll: &Epoll,
    event_bitmask: u32,
    interests: impl Iterator<Item = (&'a EpollEventKey, &'a EpollEventInterest)>,
) -> InterpResult<'tcx> {
    let mut wakeup = false;
    for (&event_key, interest) in interests {
        // This checks if any of the events specified in epoll_event_interest.events
        // match those in ready_events.
        let flags = interest.events & event_bitmask;
        if flags == 0 {
            continue;
        }
        // Geenrate a new event instance, with the flags that this one is interested in.
        let mut new_instance = EpollEventInstance::new(flags, interest.data);
        ecx.release_clock(|clock| {
            new_instance.clock.clone_from(clock);
        })?;
        // Add event to ready list for this epoll instance.
        // Tests confirm that we have to *overwrite* the old instance for the same key.
        let mut ready_list = epoll.ready_list.borrow_mut();
        ready_list.insert(event_key, new_instance);
        wakeup = true;
    }
    if wakeup {
        // Wake up threads that may have been waiting for events on this epoll.
        // Do this only once for all the interests.
        // Edge-triggered notification only notify one thread even if there are
        // multiple threads blocked on the same epoll.
        if let Some(thread_id) = epoll.blocked_tid.borrow_mut().pop() {
            ecx.unblock_thread(thread_id, BlockReason::Epoll)?;
        }
    }

    interp_ok(())
}

/// Stores the ready list of the `epfd` epoll instance into `events` (which must be an array),
/// and the number of returned events into `dest`.
fn return_ready_list<'tcx>(
    epfd: &FileDescriptionRef<Epoll>,
    dest: &MPlaceTy<'tcx>,
    events: &MPlaceTy<'tcx>,
    ecx: &mut MiriInterpCx<'tcx>,
) -> InterpResult<'tcx> {
    let mut ready_list = epfd.ready_list.borrow_mut();
    let mut num_of_events: i32 = 0;
    let mut array_iter = ecx.project_array_fields(events)?;

    while let Some(des) = array_iter.next(ecx)? {
        if let Some((_, epoll_event_instance)) = ready_list.pop_first() {
            ecx.write_int_fields_named(
                &[
                    ("events", epoll_event_instance.events.into()),
                    ("u64", epoll_event_instance.data.into()),
                ],
                &des.1,
            )?;
            // Synchronize waking thread with the event of interest.
            ecx.acquire_clock(&epoll_event_instance.clock)?;

            num_of_events = num_of_events.strict_add(1);
        } else {
            break;
        }
    }
    ecx.write_int(num_of_events, dest)?;
    interp_ok(())
}
