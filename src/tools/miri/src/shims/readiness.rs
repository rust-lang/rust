use std::cell::{Ref, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::rc::{Rc, Weak};

use crate::concurrency::VClock;
use crate::shims::files::{DynFileDescriptionRef, FdNum};
use crate::shims::*;
use crate::*;

/// Struct reflecting the readiness of a file description.
#[derive(Debug, Clone, PartialEq)]
pub struct Readiness {
    /// Boolean whether the file description is readable.
    pub readable: bool,
    /// Boolean whether the file description is writable.
    pub writable: bool,
    /// Boolean whether the read end of the file description
    /// is closed.
    pub read_closed: bool,
    /// Boolean whether the write end of the file description
    /// is closed.
    pub write_closed: bool,
    /// Boolean whether the file description has an error.
    pub error: bool,
}

impl std::ops::BitAnd for Readiness {
    type Output = Readiness;

    fn bitand(self, rhs: Readiness) -> Self::Output {
        Readiness {
            readable: self.readable && rhs.readable,
            writable: self.writable && rhs.writable,
            read_closed: self.read_closed && rhs.read_closed,
            write_closed: self.write_closed && rhs.write_closed,
            error: self.error && rhs.error,
        }
    }
}

impl std::ops::BitOrAssign for Readiness {
    fn bitor_assign(&mut self, rhs: Self) {
        self.readable |= rhs.readable;
        self.writable |= rhs.writable;
        self.read_closed |= rhs.read_closed;
        self.write_closed |= rhs.write_closed;
        self.error |= rhs.error;
    }
}

impl Readiness {
    pub const EMPTY: Readiness = Readiness {
        readable: false,
        writable: false,
        read_closed: false,
        write_closed: false,
        error: false,
    };
}

pub type ReadinessInterestKey = (FdId, FdNum);

/// Returns the range of all [`ReadinessInterestKey`] for the given FD ID.
fn range_for_id(id: FdId) -> std::ops::RangeInclusive<ReadinessInterestKey> {
    (id, 0)..=(id, FdNum::MAX)
}

#[derive(Debug, Clone)]
pub struct ReadinessInterest {
    /// The mask of events the interest is interested in
    /// for this file descriptor.
    pub relevant: Readiness,
    /// Boolean whether this is an edge-triggered interest.
    /// When [`false`] it's a level-triggered interest instead.
    pub is_edge_triggered: bool,
    /// Data attached to the interest.
    // FIXME: In the future we might want to support more data types,
    // then this should no longer be a `u64` but a `dyn Any` instead.
    pub data: u64,
    /// The currently active readiness for this file descriptor.
    active: Readiness,
    /// The vector clock for wakeups.
    clock: VClock,
}

impl ReadinessInterest {
    pub fn active(&self) -> &Readiness {
        &self.active
    }

    pub fn clock(&self) -> &VClock {
        &self.clock
    }
}

type ReadinessWatcherId = usize;

/// A struct which stores [`ReadinessInterest`]s for a set of file descriptions
/// together with which interests are currently satisfied, and a list of
/// threads which should be unblocked once a [`ReadinessInterest`] of the
/// watcher is fulfilled.
#[derive(Debug)]
pub struct ReadinessWatcher {
    /// Globally unique identifier of the watcher.
    id: ReadinessWatcherId,
    /// A map of [`ReadinessInterest`]s registered for this watcher. Each entry is
    /// identified using a [`FdId`] [`FdNum`] tuple.
    interests: RefCell<BTreeMap<ReadinessInterestKey, ReadinessInterest>>,
    /// The subset of interests that is currently considered "ready". Stored separately so we
    /// can access it more efficiently.
    /// This is implemented as a queue so that with level-triggered interests, all events eventually
    /// get returned from [`ReadinessWatcher::get_ready_interests`]. The queue does not contain any
    /// duplicates.
    ready: RefCell<VecDeque<ReadinessInterestKey>>,
    /// The queue of threads blocked on this watcher.
    queue: RefCell<VecDeque<ThreadId>>,
}

impl ReadinessWatcher {
    /// Get a reference to the map of registered interests of the watcher
    /// together with their keys.
    pub fn interests(&self) -> Ref<'_, BTreeMap<ReadinessInterestKey, ReadinessInterest>> {
        self.interests.borrow()
    }

    /// Add an interest for the file description to which the file descriptor
    /// `fd_num` belongs.
    /// `relevant` contains the readiness mask of relevant events.
    /// `is_edge_triggered` specifies whether the interest is edge-triggered
    /// ([`true`]) or level-triggered ([`false`]).
    /// `data` is the user-data which is associated with the interest.
    ///
    /// The function returns `Ok(())` when the interest was successfully
    /// added, and `Err(())` when an interest with this key was already registered.
    pub fn add_interest<'tcx>(
        self: &Rc<Self>,
        fd_num: FdNum,
        relevant: Readiness,
        is_edge_triggered: bool,
        data: u64,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<(), ()>> {
        let fd_ref = ecx.machine.fds.get(fd_num).expect("File description should exist");
        let fd_id = fd_ref.id();
        let key = (fd_id, fd_num);

        let interest = ReadinessInterest {
            active: Readiness::EMPTY,
            clock: VClock::default(),
            relevant,
            is_edge_triggered,
            data,
        };
        let mut interests = self.interests.borrow_mut();
        if interests.range(range_for_id(fd_id)).next().is_none() {
            // This is the first time this FD got added to the watcher.
            // We need to remember that in the global list such that we
            // get notified about FD events.
            ecx.machine.readiness_interests.insert(fd_id, self);
        }
        if interests.try_insert(key, interest).is_err() {
            return interp_ok(Err(()));
        }

        // After adding a new interest for a fd, we need to forcefully update
        // the readiness of this fd.

        ecx.update_readiness(
            self,
            fd_ref.readiness()?,
            /* force_edge */ true,
            move |callback| {
                // Need to release the RefCell when this closure returns, so we have to move
                // it into the closure, so we have to do a re-lookup here.
                callback(key, interests.get_mut(&key).unwrap())
            },
        )?;

        interp_ok(Ok(()))
    }

    /// Update the interest which is registered for `key`.
    /// `cb` gets invoked with a mutable reference to the registered
    /// [`ReadinessInterest`].
    ///
    /// This function returns [`None`] when no interest is registered
    /// for the specified `key`.
    pub fn update_interest<'tcx>(
        self: &Rc<Self>,
        key: ReadinessInterestKey,
        ecx: &mut MiriInterpCx<'tcx>,
        cb: impl FnOnce(&mut ReadinessInterest),
    ) -> InterpResult<'tcx, Option<()>> {
        let mut interests = self.interests.borrow_mut();
        let Some(interest) = interests.get_mut(&key) else { return interp_ok(None) };
        cb(interest);

        // After updating an interest for a fd, we need to forcefully update
        // the readiness of this fd.

        let fd_ref = ecx.machine.fds.get(key.1).expect("File description should exist");
        ecx.update_readiness(
            self,
            fd_ref.readiness()?,
            /* force_edge */ true,
            move |callback| {
                // Need to release the RefCell when this closure returns, so we have to move
                // it into the closure, so we have to do a re-lookup here.
                callback(key, interests.get_mut(&key).unwrap())
            },
        )?;

        interp_ok(Some(()))
    }

    /// Remove the interest registered for `key`.
    ///
    /// This function returns [`None`] when no interest is registered
    /// for the specified `key`.
    pub fn remove_interest<'tcx>(
        self: &Rc<ReadinessWatcher>,
        key: ReadinessInterestKey,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> Option<()> {
        let mut interests = self.interests.borrow_mut();

        if interests.remove(&key).is_none() {
            // We did not have interest in this.
            return None;
        }

        // Remove the ready event for this key, should one exist.
        let mut ready_events = self.ready.borrow_mut();
        if let Some(idx) = ready_events.iter().position(|k| k == &key) {
            ready_events.remove(idx);
        }
        // If this was the last interest in this FD, remove us from the global list
        // of who is interested in this FD.
        if interests.range(range_for_id(key.0)).next().is_none() {
            ecx.machine.readiness_interests.remove(key.0, self);
        }

        Some(())
    }

    /// Add the thread with id `thread_id` to the queue of
    /// blocked threads which will be unblocked when the
    /// watcher becomes ready.
    pub fn add_blocked_thread(&self, thread_id: ThreadId) {
        self.queue.borrow_mut().push_back(thread_id);
    }

    /// Remove all threads with id `thread_id` from the queue
    /// of blocked threads which will be unblocked when the
    /// watcher becomes ready.
    pub fn remove_blocked_thread(&self, thread_id: ThreadId) {
        self.queue.borrow_mut().retain(|id| id != &thread_id);
    }

    /// Get the amount of interests which are registered to this
    /// watcher and which are currently ready.
    pub fn ready_count(&self) -> usize {
        self.ready.borrow().len()
    }

    /// Get at most the first `count` ready interests from the ready queue.
    ///
    /// If the interest is a level-triggered interest, it's automatically
    /// added to the end of the queue again such that it will only be reported
    /// after all other ready interest have been returned.
    ///
    /// This method returns at most every event from the ready queue once.
    /// This ensures that every returned interest is unique, even when there
    /// are level-triggered interests.
    pub fn get_ready_interests<'tcx>(
        &self,
        count: usize,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Vec<ReadinessInterest>> {
        let interests = self.interests.borrow();
        let mut ready = self.ready.borrow_mut();

        // Sanity-check to ensure that all event info is up-to-date.
        if cfg!(debug_assertions) {
            for (key, interest) in interests.iter() {
                // Ensure this matches the latest readiness of this FD.
                // We have to do an FD lookup by ID for this. The FdNum might be already closed.
                let fd = ecx.machine.fds.fds.values().find(|fd| fd.id() == key.0).unwrap();
                let current_active = fd.readiness()?;
                assert_eq!(interest.active(), &(current_active & interest.relevant.clone()));
            }
        }

        // Compute how many events we can actually return.
        let count = count.min(ready.len());
        let mut ready_interests = Vec::with_capacity(count);

        // We have to bound this iterator by `count`. Iterating until `ready` is empty would not
        // work since we are re-adding level-triggered events to `ready` during the loop.
        while ready_interests.len() < count {
            let key = ready.pop_front().unwrap();
            let interest = interests.get(&key).expect("non-existing interest in ready set");

            if !interest.is_edge_triggered {
                // This is a level-triggered interest, so we need to re-add the event
                // at the end of the ready queue like Linux does with epoll:
                // <https://github.com/torvalds/linux/blob/HEAD/fs/eventpoll.c#L1835-L1847>
                ready.push_back(key);
            }

            ready_interests.push(interest.clone());
        }

        interp_ok(ready_interests)
    }

    /// Destroy the watcher instance.
    ///
    /// This also deregisters all interests of the watcher
    /// from the global readiness interest table.
    pub fn destroy<'tcx>(self, ecx: &mut MiriInterpCx<'tcx>) {
        // If we were interested in some FDs, we can remove that now.
        let mut ids = self.interests.borrow().keys().map(|(id, _num)| *id).collect::<Vec<_>>();
        // Because the ids come out of the map sorted,
        // deduping only keeps all unique entries.
        ids.dedup();
        for id in ids {
            ecx.machine.readiness_interests.remove(id, &self);
        }
    }
}

impl VisitProvenance for ReadinessWatcher {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
}

/// The table with all [`ReadinessWatcher`]s.
/// This tracks, for each file description, which watchers have an interest in events
/// for this file description.
pub struct ReadinessInterestTable {
    /// The id of the next [`ReadinessWatcher`] created through
    /// [`ReadinessInterestTable::new_watcher`].
    next_watcher_id: ReadinessWatcherId,
    /// Maps each file description (identified by its [`FdId`]) to the list of watchers that are
    /// interested in that FD. We also store the [`ReadinessWatcher`]s ID
    /// separately so we can access it without calling `upgrade`. The list
    /// is sorted by that id. We use an ID so that we can identify the watcher even after it has
    /// been moved, e.g. in [`ReadinessWatcher::destroy`].
    interests: BTreeMap<FdId, Vec<(ReadinessWatcherId, Weak<ReadinessWatcher>)>>,
}

impl ReadinessInterestTable {
    pub(crate) fn new() -> Self {
        ReadinessInterestTable { interests: BTreeMap::new(), next_watcher_id: 0 }
    }

    /// Create a new [`ReadinessWatcher`] with a globally unique id.
    /// Every watcher gets a sequentially increasing id such that no two
    /// watchers ever get the same id.
    pub fn new_watcher(&mut self) -> ReadinessWatcher {
        let id = self.next_watcher_id;
        self.next_watcher_id = id.strict_add(1);
        ReadinessWatcher {
            id,
            interests: RefCell::new(BTreeMap::new()),
            ready: RefCell::new(VecDeque::new()),
            queue: RefCell::new(VecDeque::new()),
        }
    }

    /// Add an interest for `watcher` for the file description with id `fd_id`.
    fn insert(&mut self, fd_id: FdId, watcher: &Rc<ReadinessWatcher>) {
        let watchers = self.interests.entry(fd_id).or_default();
        let idx = watchers
            .binary_search_by_key(&watcher.id, |&(id, _)| id)
            .expect_err("watcher already has a registered interest in the provided fd");
        watchers.insert(idx, (watcher.id, Rc::downgrade(watcher)));
    }

    /// Remove the interest of `watcher` for the file description with id `fd_id`.
    fn remove(&mut self, fd_id: FdId, watcher: &ReadinessWatcher) {
        let watchers = self.interests.entry(fd_id).or_default();
        let idx = watchers
            .binary_search_by_key(&watcher.id, |&(id, _)| id)
            .expect("watcher has no registered interest in the provided fd");
        watchers.remove(idx);
    }

    /// Get all watchers which have a registered interest in the file description
    /// with id `fd_id`.
    fn get_watchers_for_fd(
        &self,
        fd_id: FdId,
    ) -> Option<impl Iterator<Item = Rc<ReadinessWatcher>>> {
        let watchers = self.interests.get(&fd_id)?;
        Some(watchers.iter().map(|(_id, watcher)| {
            watcher
                .upgrade()
                .expect("someone forgot to remove the garbage from `machine.readiness_interests`")
        }))
    }

    /// Remove all watchers for the file description with id `fd_id`.
    pub fn remove_watchers_for_fd(&mut self, fd_id: FdId) {
        let Some(watchers) = self.interests.remove(&fd_id) else {
            return;
        };

        for watcher in watchers.iter().filter_map(|(_id, watcher)| Weak::upgrade(watcher)) {
            // This is a still-live watcher with interest in this FD. Remove all
            // relevant interests (including from the ready set).
            watcher
                .interests
                .borrow_mut()
                .extract_if(range_for_id(fd_id), |_, _| true)
                // Consume the iterator.
                .for_each(drop);
            // Remove the ready interests for this file description.
            watcher.ready.borrow_mut().retain(|(id, _)| id != &fd_id);
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Returns whether the given FD has any readiness watcher with a blocked thread watching it.
    fn has_watcher_with_blocked_thread(&self, fd_id: FdId) -> bool {
        let this = self.eval_context_ref();
        let Some(mut watchers) = this.machine.readiness_interests.get_watchers_for_fd(fd_id) else {
            return false;
        };
        // See if any of those watchers has a blocked thread.
        watchers.any(|w| w.queue.borrow().len() > 0)
    }

    /// For a specific file description, get its current readiness and send it to everyone who
    /// registered interest in this FD. This function must be called whenever the result of
    /// `FileDescription::readiness` might change.
    ///
    /// If `force_edge` is set, edge-triggered interests will be triggered even if the set of
    /// ready events did not change. This can lead to spurious wakeups. Use with caution!
    fn update_fd_readiness(
        &mut self,
        fd: DynFileDescriptionRef,
        force_edge: bool,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let fd_id = fd.id();

        let Some(watchers) = this.machine.readiness_interests.get_watchers_for_fd(fd_id) else {
            return interp_ok(());
        };
        let watchers = watchers.collect::<Vec<_>>(); // need to make a copy so below we can unblock threads
        let active_readiness = fd.readiness()?;
        for watcher in watchers {
            this.update_readiness(&watcher, active_readiness.clone(), force_edge, |callback| {
                for (&key, interest) in
                    watcher.interests.borrow_mut().range_mut(range_for_id(fd_id))
                {
                    callback(key, interest)?;
                }
                interp_ok(())
            })?;
        }

        interp_ok(())
    }
}

impl<'tcx> EvalContextPrivExt<'tcx> for MiriInterpCx<'tcx> {}
pub trait EvalContextPrivExt<'tcx>: MiriInterpCxExt<'tcx> {
    /// Call this when the interests denoted by `for_each_interest` have their active readiness changed
    /// to `active`. The list is provided indirectly via the `for_each_interest` closure, which
    /// will call its argument closure for each relevant interest.
    ///
    /// Any [`RefCell`]s should be released by the time `for_each_interest` returns since we will then
    /// be waking up threads which might require access to those [`RefCell`]s.
    fn update_readiness(
        &mut self,
        watcher: &Rc<ReadinessWatcher>,
        active: Readiness,
        force_edge: bool,
        for_each_interest: impl FnOnce(
            &mut dyn FnMut(ReadinessInterestKey, &mut ReadinessInterest) -> InterpResult<'tcx>,
        ) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let mut ready = watcher.ready.borrow_mut();
        for_each_interest(&mut |key, interest| {
            let new_readiness = interest.relevant.clone() & active.clone();
            let prev_readiness = std::mem::replace(&mut interest.active, new_readiness.clone());
            if new_readiness == Readiness::EMPTY {
                // Un-trigger this, there's nothing left to report here.
                if let Some(idx) = ready.iter().position(|k| k == &key) {
                    ready.remove(idx);
                }
            } else if force_edge || new_readiness != prev_readiness & new_readiness.clone() {
                // Either we force an "edge" to be detected or there's a bit set in `new_readiness`
                // that was not set in `prev_readiness`. In both cases, this is ready now.

                // We need to ensure that this event is not already part of the `ready` queue
                // before enqueueing, as Linux does it with epoll:
                // <https://github.com/torvalds/linux/blob/HEAD/fs/eventpoll.c#L1292-L1296>
                if !ready.contains(&key) {
                    ready.push_back(key);
                }

                // No matter whether this is newly ready or just re-triggered,
                // the waiter fetching this event should sync with the current thread.
                this.release_clock(|clock| {
                    interest.clock.join(clock);
                })?;
            }
            interp_ok(())
        })?;

        // While there are events ready to be delivered, wake up a thread to receive them.
        while !ready.is_empty()
            && let Some(thread_id) = watcher.queue.borrow_mut().pop_front()
        {
            drop(ready);
            this.unblock_thread(thread_id, BlockReason::Readiness)?;
            ready = watcher.ready.borrow_mut();
        }
        interp_ok(())
    }
}
