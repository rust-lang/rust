//! Non-poisoning synchronous locks.
//!
//! The difference from the locks in the [`poison`] module is that the locks in this module will not
//! become poisoned when a thread panics while holding a guard.
//!
//! [`poison`]: super::poison

use crate::fmt;

/// A type alias for the result of a nonblocking locking method.
#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub type TryLockResult<Guard> = Result<Guard, WouldBlock>;

/// A lock could not be acquired at this time because the operation would otherwise block.
#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub struct WouldBlock;

#[unstable(feature = "sync_nonpoison", issue = "134645")]
impl fmt::Debug for WouldBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "WouldBlock".fmt(f)
    }
}

#[unstable(feature = "sync_nonpoison", issue = "134645")]
impl fmt::Display for WouldBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "try_lock failed because the operation would block".fmt(f)
    }
}

#[unstable(feature = "nonpoison_condvar", issue = "134645")]
pub use self::condvar::Condvar;
#[unstable(feature = "mapped_lock_guards", issue = "117108")]
pub use self::mutex::MappedMutexGuard;
#[unstable(feature = "nonpoison_mutex", issue = "134645")]
pub use self::mutex::{Mutex, MutexGuard};
#[unstable(feature = "mapped_lock_guards", issue = "117108")]
pub use self::rwlock::{MappedRwLockReadGuard, MappedRwLockWriteGuard};
#[unstable(feature = "nonpoison_rwlock", issue = "134645")]
pub use self::rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

mod condvar;
mod mutex;
mod rwlock;
