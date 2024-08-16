use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io;
use std::rc::{Rc, Weak};

use crate::shims::unix::fd::{FdId, FileDescriptionRef};
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
    file_descriptor: i32,
    /// The events bitmask retrieved from `epoll_event`.
    events: u32,
    /// The data retrieved from `epoll_event`.
    /// libc's data field in epoll_event can store integer or pointer,
    /// but only u64 is supported for now.
    /// <https://man7.org/linux/man-pages/man3/epoll_event.3type.html>
    data: u64,
    /// Ready list of the epoll instance under which this EpollEventInterest is registered.
    ready_list: Rc<RefCell<BTreeMap<(FdId, i32), EpollEventInstance>>>,
}

/// EpollReadyEvents reflects the readiness of a file description.
pub struct EpollReadyEvents {
    /// The associated file is available for read(2) operations.
    pub epollin: bool,
    /// The associated file is available for write(2) operations.
    pub epollout: bool,
    /// Stream socket peer closed connection, or shut down writing
    /// half of connection.
    pub epollrdhup: bool,
}

impl EpollReadyEvents {
    pub fn new() -> Self {
        EpollReadyEvents { epollin: false, epollout: false, epollrdhup: false }
    }

    pub fn get_event_bitmask<'tcx>(&self, ecx: &MiriInterpCx<'tcx>) -> u32 {
        let epollin = ecx.eval_libc_u32("EPOLLIN");
        let epollout = ecx.eval_libc_u32("EPOLLOUT");
        let epollrdhup = ecx.eval_libc_u32("EPOLLRDHUP");

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

        // Fail on unsupported operations.
        if op & epoll_ctl_add != epoll_ctl_add
            && op & epoll_ctl_mod != epoll_ctl_mod
            && op & epoll_ctl_del != epoll_ctl_del
        {
            throw_unsup_format!("epoll_ctl: encountered unknown unsupported operation {:#x}", op);
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

        let Some(file_descriptor) = this.machine.fds.get(fd) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        let id = file_descriptor.get_id();

        if op == epoll_ctl_add || op == epoll_ctl_mod {
            // Read event bitmask and data from epoll_event passed by caller.
            let events = this.read_scalar(&this.project_field(&event, 0)?)?.to_u32()?;
            let data = this.read_scalar(&this.project_field(&event, 1)?)?.to_u64()?;

            // Unset the flag we support to discover if any unsupported flags are used.
            let mut flags = events;
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

            let id = file_descriptor.get_id();
            // Create an epoll_interest.
            let interest = Rc::new(RefCell::new(EpollEventInterest {
                file_descriptor: fd,
                events,
                data,
                ready_list: Rc::clone(ready_list),
            }));

            if op == epoll_ctl_add {
                // Insert an epoll_interest to global epoll_interest list.
                this.machine.epoll_interests.insert_epoll_interest(id, Rc::downgrade(&interest));
                interest_list.insert(epoll_key, interest);
            } else {
                // Directly modify the epoll_interest so the global epoll_event_interest table
                // will be updated too.
                let mut epoll_interest = interest_list.get_mut(&epoll_key).unwrap().borrow_mut();
                epoll_interest.events = events;
                epoll_interest.data = data;
            }

            // Readiness will be updated immediately when the epoll_event_interest is added or modified.
            this.check_and_update_readiness(&file_descriptor)?;

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
        events_op: &OpTy<'tcx>,
        maxevents: &OpTy<'tcx>,
        timeout: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let epfd = this.read_scalar(epfd)?.to_i32()?;
        let maxevents = this.read_scalar(maxevents)?.to_i32()?;
        let event = this.deref_pointer_as(
            events_op,
            this.libc_array_ty_layout("epoll_event", maxevents.try_into().unwrap()),
        )?;
        let timeout = this.read_scalar(timeout)?.to_i32()?;

        if epfd <= 0 {
            let einval = this.eval_libc("EINVAL");
            this.set_last_error(einval)?;
            return Ok(Scalar::from_i32(-1));
        }
        // FIXME: Implement blocking support
        if timeout != 0 {
            throw_unsup_format!("epoll_wait: timeout value can only be 0");
        }

        let Some(epfd) = this.machine.fds.get(epfd) else {
            return Ok(Scalar::from_i32(this.fd_not_found()?));
        };
        let epoll_file_description = epfd
            .downcast::<Epoll>()
            .ok_or_else(|| err_unsup_format!("non-epoll FD passed to `epoll_wait`"))?;

        let ready_list = epoll_file_description.get_ready_list();
        let mut ready_list = ready_list.borrow_mut();
        let mut num_of_events: i32 = 0;
        let mut array_iter = this.project_array_fields(&event)?;

        while let Some((epoll_key, epoll_return)) = ready_list.pop_first() {
            // If the file description is fully close, the entry for corresponding FdID in the
            // global epoll event interest table would be empty.
            if this.machine.epoll_interests.get_epoll_interest(epoll_key.0).is_some() {
                // Return notification to the caller if the file description is not fully closed.
                if let Some(des) = array_iter.next(this)? {
                    this.write_int_fields_named(
                        &[
                            ("events", epoll_return.events.into()),
                            ("u64", epoll_return.data.into()),
                        ],
                        &des.1,
                    )?;
                    num_of_events = num_of_events.checked_add(1).unwrap();
                } else {
                    break;
                }
            }
        }
        Ok(Scalar::from_i32(num_of_events))
    }

    /// For a specific file description, get its ready events and update
    /// the corresponding ready list. This function is called whenever a file description
    /// is registered with epoll, or when its readiness *might* have changed.
    fn check_and_update_readiness(&self, fd_ref: &FileDescriptionRef) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_ref();
        let id = fd_ref.get_id();
        // Get a list of EpollEventInterest that is associated to a specific file description.
        if let Some(epoll_interests) = this.machine.epoll_interests.get_epoll_interest(id) {
            let epoll_ready_events = fd_ref.get_epoll_ready_events()?;
            // Get the bitmask of ready events.
            let ready_events = epoll_ready_events.get_event_bitmask(this);

            for weak_epoll_interest in epoll_interests {
                if let Some(epoll_interest) = weak_epoll_interest.upgrade() {
                    // This checks if any of the events specified in epoll_event_interest.events
                    // match those in ready_events.
                    let epoll_event_interest = epoll_interest.borrow();
                    let flags = epoll_event_interest.events & ready_events;
                    // If there is any event that we are interested in being specified as ready,
                    // insert an epoll_return to the ready list.
                    if flags != 0 {
                        let epoll_key = (id, epoll_event_interest.file_descriptor);
                        let ready_list = &mut epoll_event_interest.ready_list.borrow_mut();
                        let event_instance =
                            EpollEventInstance::new(flags, epoll_event_interest.data);
                        ready_list.insert(epoll_key, event_instance);
                    }
                }
            }
        }
        Ok(())
    }
}
