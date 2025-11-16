use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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
    /// A map of EpollEventInterests registered under this epoll instance. Each entry is
    /// differentiated using FdId and file descriptor value.
    interest_list: RefCell<BTreeMap<EpollEventKey, EpollEventInterest>>,
    /// The subset of interests that is currently considered "ready". Stored separately so we
    /// can access it more efficiently.
    ready_set: RefCell<BTreeSet<EpollEventKey>>,
    /// The queue of threads blocked on this epoll instance.
    queue: RefCell<VecDeque<ThreadId>>,
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

/// Tracks the events that this epoll is interested in for a given file descriptor.
#[derive(Debug)]
pub struct EpollEventInterest {
    /// The events bitmask the epoll is interested in.
    relevant_events: u32,
    /// The currently active events for this file descriptor.
    active_events: u32,
    /// The vector clock for wakeups.
    clock: VClock,
    /// User-defined data associated with this interest.
    /// libc's data field in epoll_event can store integer or pointer,
    /// but only u64 is supported for now.
    /// <https://man7.org/linux/man-pages/man3/epoll_event.3type.html>
    data: u64,
}

/// EpollReadyEvents reflects the readiness of a file description.
#[derive(Debug)]
pub struct EpollEvents {
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

impl EpollEvents {
    pub fn new() -> Self {
        EpollEvents {
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
        self_id: FdId,
        _communicate_allowed: bool,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        // If we were interested in some FDs, we can remove that now.
        let mut ids = self.interest_list.get_mut().keys().map(|(id, _num)| *id).collect::<Vec<_>>();
        ids.dedup(); // they come out of the map sorted
        for id in ids {
            ecx.machine.epoll_interests.remove(id, self_id);
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
/// for this file description. The `FdId` is the ID of the epoll instance, so that we can recognize
/// it later when it is slated for removal. The vector is sorted by that ID.
pub struct EpollInterestTable(BTreeMap<FdId, Vec<(FdId, WeakFileDescriptionRef<Epoll>)>>);

impl EpollInterestTable {
    pub(crate) fn new() -> Self {
        EpollInterestTable(BTreeMap::new())
    }

    fn insert(&mut self, id: FdId, epoll: &FileDescriptionRef<Epoll>) {
        let epolls = self.0.entry(id).or_default();
        let idx = epolls
            .binary_search_by_key(&epoll.id(), |&(id, _)| id)
            .expect_err("trying to add an epoll that's already in the list");
        epolls.insert(idx, (epoll.id(), FileDescriptionRef::downgrade(epoll)));
    }

    fn remove(&mut self, id: FdId, epoll_id: FdId) {
        let epolls = self.0.entry(id).or_default();
        let idx = epolls
            .binary_search_by_key(&epoll_id, |&(id, _)| id)
            .expect("trying to remove an epoll that's not in the list");
        epolls.remove(idx);
    }

    fn get_epolls(&self, id: FdId) -> Option<impl Iterator<Item = &WeakFileDescriptionRef<Epoll>>> {
        self.0.get(&id).map(|epolls| epolls.iter().map(|(_id, epoll)| epoll))
    }

    pub fn remove_epolls(&mut self, id: FdId) {
        if let Some(epolls) = self.0.remove(&id) {
            for epoll in epolls.iter().filter_map(|(_id, epoll)| epoll.upgrade()) {
                // This is a still-live epoll with interest in this FD. Remove all
                // relevent interests (including from the ready set).
                epoll
                    .interest_list
                    .borrow_mut()
                    .extract_if(range_for_id(id), |_, _| true)
                    // Consume the iterator.
                    .for_each(|_| ());
                epoll
                    .ready_set
                    .borrow_mut()
                    .extract_if(range_for_id(id), |_| true)
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

            // Add new interest to list. Experiments show that we need to reset all state
            // on `EPOLL_CTL_MOD`, including the edge tracking.
            let epoll_key = (id, fd);
            if op == epoll_ctl_add {
                if interest_list.range(range_for_id(id)).next().is_none() {
                    // This is the first time this FD got added to this epoll.
                    // Remember that in the global list so we get notified about FD events.
                    this.machine.epoll_interests.insert(id, &epfd);
                }
                let new_interest = EpollEventInterest {
                    relevant_events: events,
                    data,
                    active_events: 0,
                    clock: VClock::default(),
                };
                if interest_list.try_insert(epoll_key, new_interest).is_err() {
                    // We already had interest in this.
                    return this.set_last_error_and_return_i32(LibcError("EEXIST"));
                }
            } else {
                // Modify the existing interest.
                let Some(interest) = interest_list.get_mut(&epoll_key) else {
                    return this.set_last_error_and_return_i32(LibcError("ENOENT"));
                };
                interest.relevant_events = events;
                interest.data = data;
            }

            // Deliver events for the new interest.
            update_readiness(
                this,
                &epfd,
                fd_ref.as_unix(this).epoll_active_events()?.get_event_bitmask(this),
                /* force_edge */ true,
                move |callback| {
                    // Need to release the RefCell when this closure returns, so we have to move
                    // it into the closure, so we have to do a re-lookup here.
                    callback(epoll_key, interest_list.get_mut(&epoll_key).unwrap())
                },
            )?;

            interp_ok(Scalar::from_i32(0))
        } else if op == epoll_ctl_del {
            let epoll_key = (id, fd);

            // Remove epoll_event_interest from interest_list and ready_set.
            if interest_list.remove(&epoll_key).is_none() {
                // We did not have interest in this.
                return this.set_last_error_and_return_i32(LibcError("ENOENT"));
            };
            epfd.ready_set.borrow_mut().remove(&epoll_key);
            // If this was the last interest in this FD, remove us from the global list
            // of who is interested in this FD.
            if interest_list.range(range_for_id(id)).next().is_none() {
                this.machine.epoll_interests.remove(id, epfd.id());
            }

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

        if timeout == 0 || !epfd.ready_set.borrow().is_empty() {
            // If the timeout is 0 or there is a ready event, we can return immediately.
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
            epfd.queue.borrow_mut().push_back(this.active_thread());
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
                                let events = return_ready_list(&epfd, &dest, &event, this)?;
                                assert!(events > 0, "we got woken up with no events to deliver");
                                interp_ok(())
                            },
                            UnblockKind::TimedOut => {
                                // Remove the current active thread_id from the blocked thread_id list.
                                epfd
                                    .queue.borrow_mut()
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

    /// For a specific file description, get its currently active events and send it to everyone who
    /// registered interest in this FD. This function must be called whenever the result of
    /// `epoll_active_events` might change.
    ///
    /// If `force_edge` is set, edge-triggered interests will be triggered even if the set of
    /// ready events did not change. This can lead to spurious wakeups. Use with caution!
    fn update_epoll_active_events(
        &mut self,
        fd_ref: DynFileDescriptionRef,
        force_edge: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = fd_ref.id();
        // Figure out who is interested in this. We need to clone this list since we can't prove
        // that `send_active_events_to_interest` won't mutate it.
        let Some(epolls) = this.machine.epoll_interests.get_epolls(id) else {
            return interp_ok(());
        };
        let epolls = epolls
            .map(|weak| {
                weak.upgrade()
                    .expect("someone forgot to remove the garbage from `machine.epoll_interests`")
            })
            .collect::<Vec<_>>();
        let active_events = fd_ref.as_unix(this).epoll_active_events()?.get_event_bitmask(this);
        for epoll in epolls {
            update_readiness(this, &epoll, active_events, force_edge, |callback| {
                for (&key, interest) in epoll.interest_list.borrow_mut().range_mut(range_for_id(id))
                {
                    callback(key, interest)?;
                }
                interp_ok(())
            })?;
        }

        interp_ok(())
    }
}

/// Call this when the interests denoted by `for_each_interest` have their active event set changed
/// to `active_events`. The list is provided indirectly via the `for_each_interest` closure, which
/// will call its argument closure for each relevant interest.
///
/// Any `RefCell` should be released by the time `for_each_interest` returns since we will then
/// be waking up threads which might require access to those `RefCell`.
fn update_readiness<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    epoll: &Epoll,
    active_events: u32,
    force_edge: bool,
    for_each_interest: impl FnOnce(
        &mut dyn FnMut(EpollEventKey, &mut EpollEventInterest) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx>,
) -> InterpResult<'tcx> {
    let mut ready_set = epoll.ready_set.borrow_mut();
    for_each_interest(&mut |key, interest| {
        // Update the ready events tracked in this interest.
        let new_readiness = interest.relevant_events & active_events;
        let prev_readiness = std::mem::replace(&mut interest.active_events, new_readiness);
        if new_readiness == 0 {
            // Un-trigger this, there's nothing left to report here.
            ready_set.remove(&key);
        } else if force_edge || new_readiness != prev_readiness & new_readiness {
            // Either we force an "edge" to be detected, or there's a bit set in `new`
            // that was not set in `prev`. In both cases, this is ready now.
            ready_set.insert(key);
            ecx.release_clock(|clock| {
                interest.clock.join(clock);
            })?;
        }
        interp_ok(())
    })?;
    // While there are events ready to be delivered, wake up a thread to receive them.
    while !ready_set.is_empty()
        && let Some(thread_id) = epoll.queue.borrow_mut().pop_front()
    {
        drop(ready_set); // release the "lock" so the unblocked thread can have it
        ecx.unblock_thread(thread_id, BlockReason::Epoll)?;
        ready_set = epoll.ready_set.borrow_mut();
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
) -> InterpResult<'tcx, i32> {
    let mut interest_list = epfd.interest_list.borrow_mut();
    let mut ready_set = epfd.ready_set.borrow_mut();
    let mut num_of_events: i32 = 0;
    let mut array_iter = ecx.project_array_fields(events)?;

    // Sanity-check to ensure that all event info is up-to-date.
    if cfg!(debug_assertions) {
        for (key, interest) in interest_list.iter() {
            // Ensure this matches the latest readiness of this FD.
            // We have to do an FD lookup by ID for this. The FdNum might be already closed.
            let fd = &ecx.machine.fds.fds.values().find(|fd| fd.id() == key.0).unwrap();
            let current_active = fd.as_unix(ecx).epoll_active_events()?.get_event_bitmask(ecx);
            assert_eq!(interest.active_events, current_active & interest.relevant_events);
        }
    }

    // While there is a slot to store another event, and an event to store, deliver that event.
    while let Some(slot) = array_iter.next(ecx)?
        && let Some(&key) = ready_set.first()
    {
        let interest = interest_list.get_mut(&key).expect("non-existent event in ready set");
        // Deliver event to caller.
        ecx.write_int_fields_named(
            &[("events", interest.active_events.into()), ("u64", interest.data.into())],
            &slot.1,
        )?;
        num_of_events = num_of_events.strict_add(1);
        // Synchronize receiving thread with the event of interest.
        ecx.acquire_clock(&interest.clock)?;
        // Since currently, all events are edge-triggered, we remove them from the ready set when
        // they get delivered.
        ready_set.remove(&key);
    }
    ecx.write_int(num_of_events, dest)?;
    interp_ok(num_of_events)
}
