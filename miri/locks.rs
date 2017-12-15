use super::*;
use rustc::middle::region;

////////////////////////////////////////////////////////////////////////////////
// Locks
////////////////////////////////////////////////////////////////////////////////

/// Information about a lock that is currently held.
#[derive(Clone, Debug)]
pub struct LockInfo<'tcx> {
    /// Stores for which lifetimes (of the original write lock) we got
    /// which suspensions.
    suspended: HashMap<WriteLockId<'tcx>, Vec<region::Scope>>,
    /// The current state of the lock that's actually effective.
    pub active: Lock,
}

/// Write locks are identified by a stack frame and an "abstract" (untyped) place.
/// It may be tempting to use the lifetime as identifier, but that does not work
/// for two reasons:
/// * First of all, due to subtyping, the same lock may be referred to with different
///   lifetimes.
/// * Secondly, different write locks may actually have the same lifetime.  See `test2`
///   in `run-pass/many_shr_bor.rs`.
/// The Id is "captured" when the lock is first suspended; at that point, the borrow checker
/// considers the path frozen and hence the Id remains stable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WriteLockId<'tcx> {
    frame: usize,
    path: AbsPlace<'tcx>,
}


use rustc::mir::interpret::Lock::*;
use rustc::mir::interpret::Lock;

impl<'tcx> Default for LockInfo<'tcx> {
    fn default() -> Self {
        LockInfo::new(NoLock)
    }
}

impl<'tcx> LockInfo<'tcx> {
    fn new(lock: Lock) -> LockInfo<'tcx> {
        LockInfo {
            suspended: HashMap::new(),
            active: lock,
        }
    }

    fn access_permitted(&self, frame: Option<usize>, access: AccessKind) -> bool {
        use super::AccessKind::*;
        match (&self.active, access) {
            (&NoLock, _) => true,
            (&ReadLock(ref lfts), Read) => {
                assert!(!lfts.is_empty(), "Someone left an empty read lock behind.");
                // Read access to read-locked region is okay, no matter who's holding the read lock.
                true
            }
            (&WriteLock(ref lft), _) => {
                // All access is okay if we are the ones holding it
                Some(lft.frame) == frame
            }
            _ => false, // Nothing else is okay.
        }
    }
}

pub trait MemoryExt<'tcx> {
    fn check_locks(
        &self,
        ptr: MemoryPointer,
        len: u64,
        access: AccessKind,
    ) -> EvalResult<'tcx>;
    fn acquire_lock(
        &mut self,
        ptr: MemoryPointer,
        len: u64,
        region: Option<region::Scope>,
        kind: AccessKind,
    ) -> EvalResult<'tcx>;
    fn suspend_write_lock(
        &mut self,
        ptr: MemoryPointer,
        len: u64,
        lock_path: &AbsPlace<'tcx>,
        suspend: Option<region::Scope>,
    ) -> EvalResult<'tcx>;
    fn recover_write_lock(
        &mut self,
        ptr: MemoryPointer,
        len: u64,
        lock_path: &AbsPlace<'tcx>,
        lock_region: Option<region::Scope>,
        suspended_region: region::Scope,
    ) -> EvalResult<'tcx>;
    fn locks_lifetime_ended(&mut self, ending_region: Option<region::Scope>);
}


impl<'a, 'tcx: 'a> MemoryExt<'tcx> for Memory<'a, 'tcx, Evaluator<'tcx>> {
    fn check_locks(
        &self,
        ptr: MemoryPointer,
        len: u64,
        access: AccessKind,
    ) -> EvalResult<'tcx> {
        if len == 0 {
            return Ok(());
        }
        let locks = match self.data.locks.get(&ptr.alloc_id.0) {
            Some(locks) => locks,
            // immutable static or other constant memory
            None => return Ok(()),
        };
        let frame = self.cur_frame;
        locks
            .check(Some(frame), ptr.offset, len, access)
            .map_err(|lock| {
                EvalErrorKind::MemoryLockViolation {
                    ptr,
                    len,
                    frame,
                    access,
                    lock: lock.active,
                }.into()
            })
    }

    /// Acquire the lock for the given lifetime
    fn acquire_lock(
        &mut self,
        ptr: MemoryPointer,
        len: u64,
        region: Option<region::Scope>,
        kind: AccessKind,
    ) -> EvalResult<'tcx> {
        let frame = self.cur_frame;
        assert!(len > 0);
        trace!(
            "Frame {} acquiring {:?} lock at {:?}, size {} for region {:?}",
            frame,
            kind,
            ptr,
            len,
            region
        );
        self.check_bounds(ptr.offset(len, &*self)?, true)?; // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)

        let locks = match self.data.locks.get_mut(&ptr.alloc_id.0) {
            Some(locks) => locks,
            // immutable static or other constant memory
            None => return Ok(()),
        };

        // Iterate over our range and acquire the lock.  If the range is already split into pieces,
        // we have to manipulate all of them.
        let lifetime = DynamicLifetime { frame, region };
        for lock in locks.iter_mut(ptr.offset, len) {
            if !lock.access_permitted(None, kind) {
                return err!(MemoryAcquireConflict {
                    ptr,
                    len,
                    kind,
                    lock: lock.active.clone(),
                });
            }
            // See what we have to do
            match (&mut lock.active, kind) {
                (active @ &mut NoLock, AccessKind::Write) => {
                    *active = WriteLock(lifetime);
                }
                (active @ &mut NoLock, AccessKind::Read) => {
                    *active = ReadLock(vec![lifetime]);
                }
                (&mut ReadLock(ref mut lifetimes), AccessKind::Read) => {
                    lifetimes.push(lifetime);
                }
                _ => bug!("We already checked that there is no conflicting lock"),
            }
        }
        Ok(())
    }

    /// Release or suspend a write lock of the given lifetime prematurely.
    /// When releasing, if there is a read lock or someone else's write lock, that's an error.
    /// If no lock is held, that's fine.  This can happen when e.g. a local is initialized
    /// from a constant, and then suspended.
    /// When suspending, the same cases are fine; we just register an additional suspension.
    fn suspend_write_lock(
        &mut self,
        ptr: MemoryPointer,
        len: u64,
        lock_path: &AbsPlace<'tcx>,
        suspend: Option<region::Scope>,
    ) -> EvalResult<'tcx> {
        assert!(len > 0);
        let cur_frame = self.cur_frame;
        let locks = match self.data.locks.get_mut(&ptr.alloc_id.0) {
            Some(locks) => locks,
            // immutable static or other constant memory
            None => return Ok(()),
        };

        'locks: for lock in locks.iter_mut(ptr.offset, len) {
            let is_our_lock = match lock.active {
                WriteLock(lft) =>
                    // Double-check that we are holding the lock.
                    // (Due to subtyping, checking the region would not make any sense.)
                    lft.frame == cur_frame,
                ReadLock(_) | NoLock => false,
            };
            if is_our_lock {
                trace!("Releasing {:?}", lock.active);
                // Disable the lock
                lock.active = NoLock;
            } else {
                trace!(
                    "Not touching {:?} as it is not our lock",
                    lock.active,
                );
            }
            // Check if we want to register a suspension
            if let Some(suspend_region) = suspend {
                let lock_id = WriteLockId {
                    frame: cur_frame,
                    path: lock_path.clone(),
                };
                trace!("Adding suspension to {:?}", lock_id);
                let mut new_suspension = false;
                lock.suspended
                    .entry(lock_id)
                    // Remember whether we added a new suspension or not
                    .or_insert_with(|| { new_suspension = true; Vec::new() })
                    .push(suspend_region);
                // If the suspension is new, we should have owned this.
                // If there already was a suspension, we should NOT have owned this.
                if new_suspension == is_our_lock {
                    // All is well
                    continue 'locks;
                }
            } else {
                if !is_our_lock {
                    // All is well.
                    continue 'locks;
                }
            }
            // If we get here, releasing this is an error except for NoLock.
            if lock.active != NoLock {
                return err!(InvalidMemoryLockRelease {
                    ptr,
                    len,
                    frame: cur_frame,
                    lock: lock.active.clone(),
                });
            }
        }

        Ok(())
    }

    /// Release a suspension from the write lock.  If this is the last suspension or if there is no suspension, acquire the lock.
    fn recover_write_lock(
        &mut self,
        ptr: MemoryPointer,
        len: u64,
        lock_path: &AbsPlace<'tcx>,
        lock_region: Option<region::Scope>,
        suspended_region: region::Scope,
    ) -> EvalResult<'tcx> {
        assert!(len > 0);
        let cur_frame = self.cur_frame;
        let lock_id = WriteLockId {
            frame: cur_frame,
            path: lock_path.clone(),
        };
        let locks = match self.data.locks.get_mut(&ptr.alloc_id.0) {
            Some(locks) => locks,
            // immutable static or other constant memory
            None => return Ok(()),
        };

        for lock in locks.iter_mut(ptr.offset, len) {
            // Check if we have a suspension here
            let (got_the_lock, remove_suspension) = match lock.suspended.get_mut(&lock_id) {
                None => {
                    trace!("No suspension around, we can just acquire");
                    (true, false)
                }
                Some(suspensions) => {
                    trace!("Found suspension of {:?}, removing it", lock_id);
                    // That's us!  Remove suspension (it should be in there).  The same suspension can
                    // occur multiple times (when there are multiple shared borrows of this that have the same
                    // lifetime); only remove one of them.
                    let idx = match suspensions.iter().enumerate().find(|&(_, re)| re == &suspended_region) {
                        None => // TODO: Can the user trigger this?
                            bug!("We have this lock suspended, but not for the given region."),
                        Some((idx, _)) => idx
                    };
                    suspensions.remove(idx);
                    let got_lock = suspensions.is_empty();
                    if got_lock {
                        trace!("All suspensions are gone, we can have the lock again");
                    }
                    (got_lock, got_lock)
                }
            };
            if remove_suspension {
                // with NLL, we could do that up in the match above...
                assert!(got_the_lock);
                lock.suspended.remove(&lock_id);
            }
            if got_the_lock {
                match lock.active {
                    ref mut active @ NoLock => {
                        *active = WriteLock(
                            DynamicLifetime {
                                frame: cur_frame,
                                region: lock_region,
                            }
                        );
                    }
                    _ => {
                        return err!(MemoryAcquireConflict {
                            ptr,
                            len,
                            kind: AccessKind::Write,
                            lock: lock.active.clone(),
                        })
                    }
                }
            }
        }

        Ok(())
    }

    fn locks_lifetime_ended(&mut self, ending_region: Option<region::Scope>) {
        let cur_frame = self.cur_frame;
        trace!(
            "Releasing frame {} locks that expire at {:?}",
            cur_frame,
            ending_region
        );
        let has_ended = |lifetime: &DynamicLifetime| -> bool {
            if lifetime.frame != cur_frame {
                return false;
            }
            match ending_region {
                None => true, // When a function ends, we end *all* its locks. It's okay for a function to still have lifetime-related locks
                // when it returns, that can happen e.g. with NLL when a lifetime can, but does not have to, extend beyond the
                // end of a function.  Same for a function still having recoveries.
                Some(ending_region) => lifetime.region == Some(ending_region),
            }
        };

        for alloc_locks in self.data.locks.values_mut() {
            for lock in alloc_locks.iter_mut_all() {
                // Delete everything that ends now -- i.e., keep only all the other lifetimes.
                let lock_ended = match lock.active {
                    WriteLock(ref lft) => has_ended(lft),
                    ReadLock(ref mut lfts) => {
                        lfts.retain(|lft| !has_ended(lft));
                        lfts.is_empty()
                    }
                    NoLock => false,
                };
                if lock_ended {
                    lock.active = NoLock;
                }
                // Also clean up suspended write locks when the function returns
                if ending_region.is_none() {
                    lock.suspended.retain(|id, _suspensions| id.frame != cur_frame);
                }
            }
            // Clean up the map
            alloc_locks.retain(|lock| match lock.active {
                NoLock => lock.suspended.len() > 0,
                _ => true,
            });
        }
    }
}

impl<'tcx> RangeMap<LockInfo<'tcx>> {
    pub fn check(
        &self,
        frame: Option<usize>,
        offset: u64,
        len: u64,
        access: AccessKind,
    ) -> Result<(), LockInfo<'tcx>> {
        if len == 0 {
            return Ok(());
        }
        for lock in self.iter(offset, len) {
            // Check if the lock is in conflict with the access.
            if !lock.access_permitted(frame, access) {
                return Err(lock.clone());
            }
        }
        Ok(())
    }
}
