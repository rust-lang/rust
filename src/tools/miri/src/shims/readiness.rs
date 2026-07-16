use std::assert_matches;
use std::cell::{Ref, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::rc::{Rc, Weak};

use crate::concurrency::VClock;
use crate::shims::files::{DynFileDescriptionRef, FdNum, WeakDynFileDescriptionRef};
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

impl std::ops::BitOr for Readiness {
    type Output = Readiness;

    fn bitor(self, rhs: Readiness) -> Self::Output {
        Readiness {
            readable: self.readable | rhs.readable,
            writable: self.writable | rhs.writable,
            read_closed: self.read_closed | rhs.read_closed,
            write_closed: self.write_closed | rhs.write_closed,
            error: self.error | rhs.error,
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
    /// The FD we are interested in.
    watched_fd: WeakDynFileDescriptionRef,
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

/// A struct which stores [`ReadinessInterest`]s for a set of file descriptions
/// together with which interests are currently satisfied, and a list of
/// threads which should be unblocked once a [`ReadinessInterest`] of the
/// watcher is fulfilled.
#[derive(Debug, Default)]
pub struct ReadinessWatcher {
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
            watched_fd: FileDescriptionRef::downgrade(&fd_ref),
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
            ecx.machine.readiness_interests.insert(&fd_ref, self);
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

        #[allow(clippy::question_mark)] // avoid hidden control flow
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

    /// Get at most the first `max` ready interests from the ready queue.
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
        max: usize,
        ecx: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Vec<ReadinessInterest>> {
        // The goal below is to do work proportyional to how much we're actually reporting to the
        // client, i.e., how many interests of this watcher are ready. In particular, we do *not*
        // iterate over all interests of this watcher, or all watchers, or all FDs being watched, or
        // anything like that.

        // Process delayed readiness updates, to ensure we have fully up-to-date information.
        // This is almost always a NOP.
        DelayedReadinessUpdates::process(ecx)?;

        let interests = self.interests.borrow();
        let mut ready = self.ready.borrow_mut();

        // Sanity-check to ensure that all event info is up-to-date.
        if cfg!(debug_assertions) {
            for interest in interests.values() {
                // Ensure this matches the latest readiness of this FD, if the FD is still open.
                let Some(fd) = interest.watched_fd.upgrade() else { continue };
                let current_active = fd.readiness()?;
                assert_eq!(interest.active(), &(current_active & interest.relevant.clone()));
            }
        }

        let mut ready_interests = Vec::with_capacity(max.min(ready.len()));
        let mut re_add_interests = Vec::new();

        // Continue while we haven't yet gotten `max` events, and while there are more to add.
        while ready_interests.len() < max
            && let Some(key) = ready.pop_front()
        {
            let interest = interests.get(&key).expect("non-existing interest in ready set");
            if interest.watched_fd.is_closed() {
                // "A file descriptor is removed from an interest list only after all the file
                // descriptors referring to the underlying open file description have been closed."
                // So, we should have removed this FD from the interest list, we just didn't get
                // around to that yet. Pretend it does not exist. It will not be re-added to the
                // ready list, and it will eventually be cleaned up by the GC.
                continue;
            }

            if !interest.is_edge_triggered {
                // This is a level-triggered interest, so we need to re-add the event:
                // <https://github.com/torvalds/linux/blob/HEAD/fs/eventpoll.c#L1835-L1847>.
                // We delay adding them back so they do not get picked up by later loop iterations.
                re_add_interests.push(key);
            }

            ready_interests.push(interest.clone());
        }
        // Add back the level-triggered ones.
        ready.extend(re_add_interests);

        interp_ok(ready_interests)
    }
}

impl VisitProvenance for ReadinessWatcher {
    fn visit_provenance(&self, _visit: &mut VisitWith<'_>) {}
}

/// The table with all [`ReadinessWatcher`]s.
/// This tracks, for each file description, which watchers have an interest in events
/// for this file description.
pub struct ReadinessInterestTable {
    /// Maps each file description (identified by its [`FdId`]) to the list of watchers that are
    /// interested in that FD.
    /// We also store a weak reference to the watched file description, so we can clean them up
    /// when they get freed.
    /// FIXME: a better place to store this would probably be the relevant FD itself. Experiment
    /// with delegation to easily do that for all FD types, once that is ready.
    interests: BTreeMap<FdId, (WeakDynFileDescriptionRef, Vec<Weak<ReadinessWatcher>>)>,
}

impl ReadinessInterestTable {
    pub(crate) fn new() -> Self {
        ReadinessInterestTable { interests: BTreeMap::new() }
    }

    /// Add an interest for `watcher` for the `watched_fd` file description.
    fn insert(&mut self, watched_fd: &DynFileDescriptionRef, watcher: &Rc<ReadinessWatcher>) {
        let (_watched, watchers) = self
            .interests
            .entry(watched_fd.id())
            .or_insert_with(|| (FileDescriptionRef::downgrade(watched_fd), Vec::new()));
        if cfg!(debug_assertions) {
            // Ensure uniqueness.
            assert_matches!(
                watchers.iter().find(|elem| elem.as_ptr() == Rc::as_ptr(watcher)),
                None,
                "watcher has already been added to this watched fd",
            );
        }
        watchers.push(Rc::downgrade(watcher));
    }

    /// Remove the interest of `watcher` for the file description with id `fd_id`.
    fn remove(&mut self, watched_fd: FdId, watcher: &Rc<ReadinessWatcher>) {
        let (_watched, watchers) = self
            .interests
            .get_mut(&watched_fd)
            .expect("watcher has no registered interest in the provided watched fd");
        // We need to do a linear scan to find the watcher to remove. That's not ideal, but removing
        // a watched FD from an epoll is rare so it's not worth the non-trivial effort it would take
        // to make this more efficient.
        let idx = watchers
            .iter()
            .position(|elem| elem.as_ptr() == Rc::as_ptr(watcher))
            .expect("watcher has no registered interest in the provided watched fd");
        watchers.remove(idx);
    }

    /// Get all watchers which have a registered interest in the file description with id `fd_id`.
    fn get_watchers_for_fd(
        &self,
        fd_id: FdId,
    ) -> Option<impl Iterator<Item = Rc<ReadinessWatcher>>> {
        let (_watched, watchers) = self.interests.get(&fd_id)?;
        // Ignore weak refs that cannot be upgraded -- those correspond to closed
        // file descriptions and will be cleaned up by the GC eventually.
        Some(watchers.iter().filter_map(|watcher| watcher.upgrade()))
    }

    /// Run garbage collector to remove all dropped watchers and dropped watched FDs.
    pub fn run_gc(&mut self) {
        self.interests.retain(|&fd_id, (watched, watchers)| {
            if watched.is_closed() {
                // An FD we were watching got closed. Remove it from the watcher as well.
                for watcher in watchers.iter().filter_map(Weak::upgrade) {
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
                return false; // delete this one
            }
            // Keep this one, but clean up the watchers.
            watchers.retain(|watcher| watcher.strong_count() > 0);
            true
        });
    }
}

/// If a file description's readiness is known to change but we don't have an `ecx` around to update
/// it immediately, we arange for a referene to the delayed readiness updates queue to be available
/// and perform the update on the next scheduler call.
#[derive(Default, Debug)]
pub struct DelayedReadinessUpdates {
    to_update: RefCell<Vec<DynFileDescriptionRef>>,
}

impl DelayedReadinessUpdates {
    pub fn add(&self, fd: DynFileDescriptionRef) {
        self.to_update.borrow_mut().push(fd);
    }

    pub fn process<'tcx>(ecx: &mut MiriInterpCx<'tcx>) -> InterpResult<'tcx> {
        loop {
            // Avoid keeping the RefCell open over the `update_fd_readiness` as that can invoke
            // arbitrary code via the unblock callback.
            let Some(fd) = ecx.machine.delayed_readiness_updates.to_update.borrow_mut().pop()
            else {
                return interp_ok(());
            };
            ecx.update_fd_readiness(fd, /* force_edge */ false)?;
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
                // <https://github.com/torvalds/linux/blob/13dce771bbad42da6ecf086446d8ddfd1fce3a1b/fs/eventpoll.c#L1290-L1293>
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
