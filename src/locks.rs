#![allow(unused)]

use super::*;
use rustc::middle::region;
use rustc::ty::layout::Size;

////////////////////////////////////////////////////////////////////////////////
// Locks
////////////////////////////////////////////////////////////////////////////////

// Just some dummy to keep this compiling; I think some of this will be useful later
type AbsPlace<'tcx> = ::rustc::ty::Ty<'tcx>;

/// Information about a lock that is currently held.
#[derive(Clone, Debug, PartialEq, Eq)]
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
