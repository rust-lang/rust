use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::rc::{Rc, Weak};
use std::time::Duration;

use crate::shims::unix::fd::{FdId, FileDescriptionRef, WeakFileDescriptionRef};
use crate::shims::unix::*;
use crate::*;

/// An `Epoll` file descriptor connects file handles and epoll events
#[derive(Clone, Debug, Default)]
struct Epoll {
    /// A map of EpollEventInterests registered under this epoll instance.
    /// Each entry is differentiated using FdId and file descriptor value.
    interest_list: RefCell<BTreeMap<(FdId, i32), Rc<RefCell<EpollEventInterest>>>>,
    /// A map of EpollEventInstance that will be returned when `epoll_wait` is called.
    /// Similar to interest_list, the entry is also differentiated using FdId
    /// and file descriptor value.
    // This is an Rc because EpollInterest need to hold a reference to update
    // it.
    ready_list: Rc<RefCell<BTreeMap<(FdId, i32), EpollEventInstance>>>,
    /// A list of thread ids blocked on this epoll instance.
    thread_id: RefCell<Vec<ThreadId>>,
}

/// EpollEventInstance contains information that will be returned by epoll_wait.
#[derive(Debug)]
pub struct EpollEventInstance {
    /// Xor-ed event types that happened to the file description.
    events: u32,
    /// Original data retrieved from `epoll_event` during `epoll_ctl`.
    data: u64,
}

impl EpollEventInstance {
    pub fn new(events: u32, data: u64) -> EpollEventInstance {
        EpollEventInstance { events, data }
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
#[derive(Clone, Debug)]
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
    /// Ready list of the epoll instance under which this EpollEventInterest is registered.
    ready_list: Rc<RefCell<BTreeMap<(FdId, i32), EpollEventInstance>>>,
    /// The epoll file description that this EpollEventInterest is registered under.
    weak_epfd: WeakFileDescriptionRef,
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

impl Epoll {
    fn get_ready_list(&self) -> Rc<RefCell<BTreeMap<(FdId, i32), EpollEventInstance>>> {
        Rc::clone(&self.ready_list)
    }
}

impl FileDescription for Epoll {
    fn name(&self) -> &'static str {
        "epoll"
    }

    fn close<'tcx>(
        self: Box<Self>,
        _communicate_allowed: bool,
        _ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, io::Result<()>> {
        Ok(Ok(()))
    }
}

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

        let mut epoll_instance = Epoll::default();
        epoll_instance.ready_list = Rc::new(RefCell::new(BTreeMap::new()));

        let fd = this.machine.fds.insert_new(Epoll::default());
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

        // Fail on unsupported operations.
        if op & epoll_ctl_add != epoll_ctl_add
            && op & epoll_ctl_mod != epoll_ctl_mod
            && op & epoll_ctl_del != epoll_ctl_del
        {
            throw_unsup_format!("epoll_ctl: encountered unknown unsupported operation {:#x}", op);
        }

        // Throw EINVAL if epfd and fd have the same value.
        if epfd_value == fd {
            let einval = this.eval_libc("EINVAL");
            this.set_last_error(einval)?;
            return Ok(Scalar::from_i32(-1));
        }

        // Check if epfd is a valid epoll file descriptor.
        let Some(epfd) = this.machine.fds.get(epfd_value) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        let epoll_file_description = epfd
            .downcast::<Epoll>()
            .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_ctl`"))?;

        let mut interest_list = epoll_file_description.interest_list.borrow_mut();
        let ready_list = &epoll_file_description.ready_list;

        let Some(fd_ref) = this.machine.fds.get(fd) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        let id = fd_ref.get_id();

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
                    let eexist = this.eval_libc("EEXIST");
                    this.set_last_error(eexist)?;
                    return Ok(Scalar::from_i32(-1));
                }
            } else {
                if !interest_list.contains_key(&epoll_key) {
                    let enoent = this.eval_libc("ENOENT");
                    this.set_last_error(enoent)?;
                    return Ok(Scalar::from_i32(-1));
                }
            }

            // Create an epoll_interest.
            let interest = Rc::new(RefCell::new(EpollEventInterest {
                fd_num: fd,
                events,
                data,
                ready_list: Rc::clone(ready_list),
                weak_epfd: epfd.downgrade(),
            }));

            if op == epoll_ctl_add {
                // Insert an epoll_interest to global epoll_interest list.
                this.machine.epoll_interests.insert_epoll_interest(id, Rc::downgrade(&interest));
                interest_list.insert(epoll_key, Rc::clone(&interest));
            } else {
                // Directly modify the epoll_interest so the global epoll_event_interest table
                // will be updated too.
                let mut epoll_interest = interest_list.get_mut(&epoll_key).unwrap().borrow_mut();
                epoll_interest.events = events;
                epoll_interest.data = data;
            }

            // Notification will be returned for current epfd if there is event in the file
            // descriptor we registered.
            check_and_update_one_event_interest(&fd_ref, interest, id, this)?;
            return Ok(Scalar::from_i32(0));
        } else if op == epoll_ctl_del {
            let epoll_key = (id, fd);

            // Remove epoll_event_interest from interest_list.
            let Some(epoll_interest) = interest_list.remove(&epoll_key) else {
                let enoent = this.eval_libc("ENOENT");
                this.set_last_error(enoent)?;
                return Ok(Scalar::from_i32(-1));
            };
            // All related Weak<EpollEventInterest> will fail to upgrade after the drop.
            drop(epoll_interest);

            // Remove related epoll_interest from ready list.
            ready_list.borrow_mut().remove(&epoll_key);

            // Remove dangling EpollEventInterest from its global table.
            // .unwrap() below should succeed because the file description id must have registered
            // at least one epoll_interest, if not, it will fail when removing epoll_interest from
            // interest list.
            this.machine
                .epoll_interests
                .get_epoll_interest_mut(id)
                .unwrap()
                .retain(|event| event.upgrade().is_some());

            return Ok(Scalar::from_i32(0));
        }
        Ok(Scalar::from_i32(-1))
    }

    /// The `epoll_wait()` system call waits for events on the `Epoll`
    /// instance referred to by the file descriptor `epfd`. The buffer
    /// pointed to by `events` is used to return information from the ready
    /// list about file descriptors in the interest list that have some
    /// events available. Up to `maxevents` are returned by `epoll_wait()`.
    /// The `maxevents` argument must be greater than zero.

    /// The `timeout` argument specifies the number of milliseconds that
    /// `epoll_wait()` will block. Time is measured against the
    /// CLOCK_MONOTONIC clock. If the timeout is zero, the function will not block,
    /// while if the timeout is -1, the function will block
    /// until at least one event has been retrieved (or an error
    /// occurred).

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
            let einval = this.eval_libc("EINVAL");
            this.set_last_error(einval)?;
            this.write_int(-1, dest)?;
            return Ok(());
        }

        // This needs to come after the maxevents value check, or else maxevents.try_into().unwrap()
        // will fail.
        let event = this.deref_pointer_as(
            &events,
            this.libc_array_ty_layout("epoll_event", maxevents.try_into().unwrap()),
        )?;

        let Some(epfd) = this.machine.fds.get(epfd_value) else {
            let result_value: i32 = this.fd_not_found()?;
            this.write_int(result_value, dest)?;
            return Ok(());
        };
        // Create a weak ref of epfd and pass it to callback so we will make sure that epfd
        // is not close after the thread unblocks.
        let weak_epfd = epfd.downgrade();

        // We just need to know if the ready list is empty and borrow the thread_ids out.
        // The whole logic is wrapped inside a block so we don't need to manually drop epfd later.
        let ready_list_empty;
        let mut thread_ids;
        {
            let epoll_file_description = epfd
                .downcast::<Epoll>()
                .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_wait`"))?;
            let binding = epoll_file_description.get_ready_list();
            ready_list_empty = binding.borrow_mut().is_empty();
            thread_ids = epoll_file_description.thread_id.borrow_mut();
        }
        if timeout == 0 || !ready_list_empty {
            // If the ready list is not empty, or the timeout is 0, we can return immediately.
            blocking_epoll_callback(epfd_value, weak_epfd, dest, &event, this)?;
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
            thread_ids.push(this.active_thread());
            let dest = dest.clone();
            this.block_thread(
                BlockReason::Epoll,
                timeout,
                callback!(
                    @capture<'tcx> {
                        epfd_value: i32,
                        weak_epfd: WeakFileDescriptionRef,
                        dest: MPlaceTy<'tcx>,
                        event: MPlaceTy<'tcx>,
                    }
                    @unblock = |this| {
                        blocking_epoll_callback(epfd_value, weak_epfd, &dest, &event, this)?;
                        Ok(())
                    }
                    @timeout = |this| {
                        // No notification after blocking timeout.
                        let Some(epfd) = weak_epfd.upgrade() else {
                            throw_unsup_format!("epoll FD {epfd_value} got closed while blocking.")
                        };
                        // Remove the current active thread_id from the blocked thread_id list.
                        epfd.downcast::<Epoll>()
                            .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_wait`"))?
                            .thread_id.borrow_mut()
                            .retain(|&id| id != this.active_thread());
                        this.write_int(0, &dest)?;
                        Ok(())
                    }
                ),
            );
        }
        Ok(())
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
        fd_ref: &FileDescriptionRef,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let id = fd_ref.get_id();
        let mut waiter = Vec::new();
        // Get a list of EpollEventInterest that is associated to a specific file description.
        if let Some(epoll_interests) = this.machine.epoll_interests.get_epoll_interest(id) {
            for weak_epoll_interest in epoll_interests {
                if let Some(epoll_interest) = weak_epoll_interest.upgrade() {
                    let is_updated = check_and_update_one_event_interest(
                        fd_ref,
                        epoll_interest.clone(),
                        id,
                        this,
                    )?;
                    if is_updated {
                        // Edge-triggered notification only notify one thread even if there are
                        // multiple threads block on the same epfd.

                        // This unwrap can never fail because if the current epoll instance were
                        // closed, the upgrade of weak_epoll_interest
                        // above would fail. This guarantee holds because only the epoll instance
                        // holds a strong ref to epoll_interest.
                        let epfd = epoll_interest.borrow().weak_epfd.upgrade().unwrap();
                        // FIXME: We can randomly pick a thread to unblock.
                        if let Some(thread_id) =
                            epfd.downcast::<Epoll>().unwrap().thread_id.borrow_mut().pop()
                        {
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
        Ok(())
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
    return None;
}

/// This helper function checks whether an epoll notification should be triggered for a specific
/// epoll_interest and, if necessary, triggers the notification, and returns whether the
/// notification was added/updated. Unlike check_and_update_readiness, this function sends a
/// notification to only one epoll instance.
fn check_and_update_one_event_interest<'tcx>(
    fd_ref: &FileDescriptionRef,
    interest: Rc<RefCell<EpollEventInterest>>,
    id: FdId,
    ecx: &MiriInterpCx<'tcx>,
) -> InterpResult<'tcx, bool> {
    // Get the bitmask of ready events for a file description.
    let ready_events_bitmask = fd_ref.get_epoll_ready_events()?.get_event_bitmask(ecx);
    let epoll_event_interest = interest.borrow();
    // This checks if any of the events specified in epoll_event_interest.events
    // match those in ready_events.
    let flags = epoll_event_interest.events & ready_events_bitmask;
    // If there is any event that we are interested in being specified as ready,
    // insert an epoll_return to the ready list.
    if flags != 0 {
        let epoll_key = (id, epoll_event_interest.fd_num);
        let ready_list = &mut epoll_event_interest.ready_list.borrow_mut();
        let event_instance = EpollEventInstance::new(flags, epoll_event_interest.data);
        // Triggers the notification by inserting it to the ready list.
        ready_list.insert(epoll_key, event_instance);
        return Ok(true);
    }
    return Ok(false);
}

/// Callback function after epoll_wait unblocks
fn blocking_epoll_callback<'tcx>(
    epfd_value: i32,
    weak_epfd: WeakFileDescriptionRef,
    dest: &MPlaceTy<'tcx>,
    events: &MPlaceTy<'tcx>,
    ecx: &mut MiriInterpCx<'tcx>,
) -> InterpResult<'tcx> {
    let Some(epfd) = weak_epfd.upgrade() else {
        throw_unsup_format!("epoll FD {epfd_value} got closed while blocking.")
    };

    let epoll_file_description = epfd
        .downcast::<Epoll>()
        .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_wait`"))?;

    let ready_list = epoll_file_description.get_ready_list();
    let mut ready_list = ready_list.borrow_mut();
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
            num_of_events = num_of_events.strict_add(1);
        } else {
            break;
        }
    }
    ecx.write_int(num_of_events, dest)?;
    Ok(())
}
