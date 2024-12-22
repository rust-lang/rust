//! Non-poisoning syncronous locks.
//!
//! The locks found on this module will not become poisoned when a thread panics whilst holding a guard.

#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub use self::mutex::MappedMutexGuard;
#[unstable(feature = "sync_nonpoison", issue = "134645")]
pub use self::mutex::{Mutex, MutexGuard};

mod mutex;
