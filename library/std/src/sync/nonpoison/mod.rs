//! Non-poisoning synchronous locks.
//!
//! The difference from the locks in the [`poison`] module is that the locks in this module will not
//! become poisoned when a thread panics while holding a guard.

/// A type alias for the result of a nonblocking locking method.
pub type TryLockResult<Guard> = Result<Guard, WouldBlock>;

/// A lock could not be acquired at this time because the operation would otherwise block.
pub struct WouldBlock;

#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub use self::mutex::MappedMutexGuard;
#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub use self::mutex::{Mutex, MutexGuard};

mod mutex;
