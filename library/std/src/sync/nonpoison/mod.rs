//! Non-poisoning synchronous locks.
//!
//! The difference from the locks in the [`poison`] module is that the locks in this module will not
//! become poisoned when a thread panics while holding a guard.

#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub use self::mutex::MappedMutexGuard;
#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub use self::mutex::{Mutex, MutexGuard};

mod mutex;
