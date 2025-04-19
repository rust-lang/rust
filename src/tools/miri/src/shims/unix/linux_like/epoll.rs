use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::rc::{Rc, Weak};
use std::time::Duration;

use crate::concurrency::VClock;
use crate::shims::files::{
    DynFileDescriptionRef, FdId, FileDescription, FileDescriptionRef, WeakFileDescriptionRef,
};
use crate::shims::unix::UnixFileDescription;
use crate::*;

/// An `Epoll` file descriptor connects file handles and epoll events
#[derive(Debug, Default)]
struct Epoll {
    /// A map of EpollEventInterests registered under this epoll instance.
    /// Each entry is differentiated using FdId and file descriptor value.
    interest_list: RefCell<BTreeMap<(FdId, i32), Rc<RefCell<EpollEventInterest>>>>,
    /// A map of EpollEventInstance that will be returned when `epoll_wait` is called.
    /// Similar to interest_list, the entry is also differentiated using FdId
    /// and file descriptor value.
    ready_list: ReadyList,
    /// A list of thread ids blocked on this epoll instance.
    blocked_tid: RefCell<Vec<ThreadId>>,
}

impl VisitProvenance for Epoll {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {
        // No provenance anywhere in this type.
    }
}

/// EpollEventInstance contains information that will be returned by epoll_wait.
#[derive(Debug)]
pub struct EpollEventInstance {
    /// Xor-ed event types that happened to the file description.
    events: u32,
    /// Original data retrieved from `epoll_event` during `epoll_ctl`.
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
#[derive(Debug)]
pub struct EpollEventInterest {
    /// The file descriptor value of the file description registered.
    /// This is only used for ready_list, to inform userspace which FD triggered an event.
    /// For that, it is crucial to preserve the original FD number.
    /// This FD number must never be "dereferenced" to a file description inside Miri.
    fd_num: i32,
    /// The events bitmask retrieved from `epoll_event`.
    events: u32,
    /// The data retrieved from `epoll_event`.
    /// libc's data field in epoll_event can store integer or pointer,
    /// but only u64 is supported for now.
    /// <https://man7.org/linux/man-pages/man3/epoll_event.3type.html>
    data: u64,
    /// The epoll file description that this EpollEventInterest is registered under.
    /// This is weak to avoid cycles, but an upgrade is always guaranteed to succeed
    /// because only the `Epoll` holds a strong ref to a `EpollEventInterest`.
    weak_epfd: WeakFileDescriptionRef<Epoll>,
}

/// EpollReadyEvents reflects the readiness of a file description.
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

#[derive(Debug, Default)]
struct ReadyList {
    mapping: RefCell<BTreeMap<(FdId, i32), EpollEventInstance>>,
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

    fn close<'tcx>(
        self,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        interp_ok(Ok(()))
    }

    fn as_unix<'tcx>(&self, _ecx: &MiriInterpCx<'tcx>) -> &dyn UnixFileDescription {
        self
    }
}

impl UnixFileDescription for Epoll {}

/// The table of all EpollEventInterest.
/// The BTreeMap key is the FdId of an active file description registered with
/// any epoll instance. The value is a list of EpollEventInterest associated
/// with that file description.
pub struct EpollInterestTable(BTreeMap<FdId, Vec<Weak<RefCell<EpollEventInterest>>>>);

impl EpollInterestTable {
    pub(crate) fn new() -> Self {
        EpollInterestTable(BTreeMap::new())
    }

    pub fn insert_epoll_interest(&mut self, id: FdId, fd: Weak<RefCell<EpollEventInterest>>) {
        match self.0.get_mut(&id) {
            Some(fds) => {
                fds.push(fd);
            }
            None => {
                let vec = vec![fd];
                self.0.insert(id, vec);
            }
        }
    }

    pub fn get_epoll_interest(&self, id: FdId) -> Option<&Vec<Weak<RefCell<EpollEventInterest>>>> {
        self.0.get(&id)
    }

    pub fn get_epoll_interest_mut(
        &mut self,
        id: FdId,
    ) -> Option<&mut Vec<Weak<RefCell<EpollEventInterest>>>> {
        self.0.get_mut(&id)
    }

    pub fn remove(&mut self, id: FdId) {
        self.0.remove(&id);
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

        // Throw EINVAL if epfd and fd have the same value.
        if epfd_value == fd {
            return this.set_last_error_and_return_i32(LibcError("EINVAL"));
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
            let mut events = this.read_scalar(&this.project_field(&event, 0)?)?.to_u32()?;
            let data = this.read_scalar(&this.project_field(&event, 1)?)?.to_u64()?;

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

            let epoll_key = (id, fd);

            // Check the existence of fd in the interest list.
            if op == epoll_ctl_add {
                if interest_list.contains_key(&epoll_key) {
                    return this.set_last_error_and_return_i32(LibcError("EEXIST"));
                }
            } else {
                if !interest_list.contains_key(&epoll_key) {
                    return this.set_last_error_and_return_i32(LibcError("ENOENT"));
                }
            }

            if op == epoll_ctl_add {
                // Create an epoll_interest.
                let interest = Rc::new(RefCell::new(EpollEventInterest {
                    fd_num: fd,
                    events,
                    data,
                    weak_epfd: FileDescriptionRef::downgrade(&epfd),
                }));
                // Notification will be returned for current epfd if there is event in the file
                // descriptor we registered.
                check_and_update_one_event_interest(&fd_ref, &interest, id, this)?;

                // Insert an epoll_interest to global epoll_interest list.
                this.machine.epoll_interests.insert_epoll_interest(id, Rc::downgrade(&interest));
                interest_list.insert(epoll_key, interest);
            } else {
                // Modify the existing interest.
                let epoll_interest = interest_list.get_mut(&epoll_key).unwrap();
                {
                    let mut epoll_interest = epoll_interest.borrow_mut();
                    epoll_interest.events = events;
                    epoll_interest.data = data;
                }
                // Updating an FD interest triggers events.
                check_and_update_one_event_interest(&fd_ref, epoll_interest, id, this)?;
            }

            interp_ok(Scalar::from_i32(0))
        } else if op == epoll_ctl_del {
            let epoll_key = (id, fd);

            // Remove epoll_event_interest from interest_list.
            let Some(epoll_interest) = interest_list.remove(&epoll_key) else {
                return this.set_last_error_and_return_i32(LibcError("ENOENT"));
            };
            // All related Weak<EpollEventInterest> will fail to upgrade after the drop.
            drop(epoll_interest);

            // Remove related epoll_interest from ready list.
            epfd.ready_list.mapping.borrow_mut().remove(&epoll_key);

            // Remove dangling EpollEventInterest from its global table.
            // .unwrap() below should succeed because the file description id must have registered
            // at least one epoll_interest, if not, it will fail when removing epoll_interest from
            // interest list.
            this.machine
                .epoll_interests
                .get_epoll_interest_mut(id)
                .unwrap()
                .retain(|event| event.upgrade().is_some());

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
        let ready_list_empty = epfd.ready_list.mapping.borrow().is_empty();
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

    /// For a specific file description, get its ready events and update the corresponding ready
    /// list. This function should be called whenever an event causes more bytes or an EOF to become
    /// newly readable from an FD, and whenever more bytes can be written to an FD or no more future
    /// writes are possible.
    ///
    /// This *will* report an event if anyone is subscribed to it, without any further filtering, so
    /// do not call this function when an FD didn't have anything happen to it!
    fn check_and_update_readiness(
        &mut self,
        fd_ref: DynFileDescriptionRef,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let id = fd_ref.id();
        let mut waiter = Vec::new();
        // Get a list of EpollEventInterest that is associated to a specific file description.
        if let Some(epoll_interests) = this.machine.epoll_interests.get_epoll_interest(id) {
            for weak_epoll_interest in epoll_interests {
                if let Some(epoll_interest) = weak_epoll_interest.upgrade() {
                    let is_updated =
                        check_and_update_one_event_interest(&fd_ref, &epoll_interest, id, this)?;
                    if is_updated {
                        // Edge-triggered notification only notify one thread even if there are
                        // multiple threads blocked on the same epfd.

                        // This unwrap can never fail because if the current epoll instance were
                        // closed, the upgrade of weak_epoll_interest
                        // above would fail. This guarantee holds because only the epoll instance
                        // holds a strong ref to epoll_interest.
                        let epfd = epoll_interest.borrow().weak_epfd.upgrade().unwrap();
                        // FIXME: We can randomly pick a thread to unblock.
                        if let Some(thread_id) = epfd.blocked_tid.borrow_mut().pop() {
                            waiter.push(thread_id);
                        };
                    }
                }
            }
        }
        waiter.sort();
        waiter.dedup();
        for thread_id in waiter {
            this.unblock_thread(thread_id, BlockReason::Epoll)?;
        }
        interp_ok(())
    }
}

/// This function takes in ready list and returns EpollEventInstance with file description
/// that is not closed.
fn ready_list_next(
    ecx: &MiriInterpCx<'_>,
    ready_list: &mut BTreeMap<(FdId, i32), EpollEventInstance>,
) -> Option<EpollEventInstance> {
    while let Some((epoll_key, epoll_event_instance)) = ready_list.pop_first() {
        // This ensures that we only return events that we are interested. The FD might have been closed since
        // the event was generated, in which case we are not interested anymore.
        // When a file description is fully closed, it gets removed from `machine.epoll_interests`,
        // so we skip events whose FD is not in that map anymore.
        if ecx.machine.epoll_interests.get_epoll_interest(epoll_key.0).is_some() {
            return Some(epoll_event_instance);
        }
    }
    None
}

/// This helper function checks whether an epoll notification should be triggered for a specific
/// epoll_interest and, if necessary, triggers the notification, and returns whether the
/// notification was added/updated. Unlike check_and_update_readiness, this function sends a
/// notification to only one epoll instance.
fn check_and_update_one_event_interest<'tcx>(
    fd_ref: &DynFileDescriptionRef,
    interest: &RefCell<EpollEventInterest>,
    id: FdId,
    ecx: &MiriInterpCx<'tcx>,
) -> InterpResult<'tcx, bool> {
    // Get the bitmask of ready events for a file description.
    let ready_events_bitmask = fd_ref.as_unix(ecx).get_epoll_ready_events()?.get_event_bitmask(ecx);
    let epoll_event_interest = interest.borrow();
    let epfd = epoll_event_interest.weak_epfd.upgrade().unwrap();
    // This checks if any of the events specified in epoll_event_interest.events
    // match those in ready_events.
    let flags = epoll_event_interest.events & ready_events_bitmask;
    // If there is any event that we are interested in being specified as ready,
    // insert an epoll_return to the ready list.
    if flags != 0 {
        let epoll_key = (id, epoll_event_interest.fd_num);
        let mut ready_list = epfd.ready_list.mapping.borrow_mut();
        let mut event_instance = EpollEventInstance::new(flags, epoll_event_interest.data);
        // If we are tracking data races, remember the current clock so we can sync with it later.
        ecx.release_clock(|clock| {
            event_instance.clock.clone_from(clock);
        });
        // Triggers the notification by inserting it to the ready list.
        ready_list.insert(epoll_key, event_instance);
        interp_ok(true)
    } else {
        interp_ok(false)
    }
}

/// Stores the ready list of the `epfd` epoll instance into `events` (which must be an array),
/// and the number of returned events into `dest`.
fn return_ready_list<'tcx>(
    epfd: &FileDescriptionRef<Epoll>,
    dest: &MPlaceTy<'tcx>,
    events: &MPlaceTy<'tcx>,
    ecx: &mut MiriInterpCx<'tcx>,
) -> InterpResult<'tcx> {
    let mut ready_list = epfd.ready_list.mapping.borrow_mut();
    let mut num_of_events: i32 = 0;
    let mut array_iter = ecx.project_array_fields(events)?;

    while let Some(des) = array_iter.next(ecx)? {
        if let Some(epoll_event_instance) = ready_list_next(ecx, &mut ready_list) {
            ecx.write_int_fields_named(
                &[
                    ("events", epoll_event_instance.events.into()),
                    ("u64", epoll_event_instance.data.into()),
                ],
                &des.1,
            )?;
            // Synchronize waking thread with the event of interest.
            ecx.acquire_clock(&epoll_event_instance.clock);

            num_of_events = num_of_events.strict_add(1);
        } else {
            break;
        }
    }
    ecx.write_int(num_of_events, dest)?;
    interp_ok(())
}
