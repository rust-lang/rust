//! This module defines various operations and types that are implemented in
//! one way for the serial compiler, and another way the parallel compiler.
//!
//! Operations
//! ----------
//! The parallel versions of operations use Rayon to execute code in parallel,
//! while the serial versions degenerate straightforwardly to serial execution.
//! The operations include `join`, `parallel`, `par_iter`, and `par_for_each`.
//!
//! Types
//! -----
//! The parallel versions of types provide various kinds of synchronization,
//! while the serial compiler versions do not.
//!
//! The following table shows how the types are implemented internally. Except
//! where noted otherwise, the type in column one is defined as a
//! newtype around the type from column two or three.
//!
//! | Type                    | Serial version      | Parallel version                |
//! | ----------------------- | ------------------- | ------------------------------- |
//! | `Lock<T>`               | `RefCell<T>`        | `RefCell<T>` or                 |
//! |                         |                     | `parking_lot::Mutex<T>`         |
//! | `RwLock<T>`             | `RefCell<T>`        | `parking_lot::RwLock<T>`        |

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

pub use horde::collect;
pub use parking_lot::{
    MappedRwLockReadGuard as MappedReadGuard, MappedRwLockWriteGuard as MappedWriteGuard,
    RwLockReadGuard as ReadGuard, RwLockWriteGuard as WriteGuard,
};

pub use self::atomic::AtomicU64;
pub use self::freeze::{FreezeLock, FreezeReadGuard, FreezeWriteGuard};
#[doc(no_inline)]
pub use self::lock::{Lock, LockGuard, Mode};
pub use self::mode::{
    FromDyn, check_dyn_thread_safe, is_dyn_thread_safe, set_dyn_thread_safe_mode,
};
pub use self::parallel::{
    broadcast, par_fns, par_for_each_in, par_join, par_map, parallel_guard, spawn,
    try_par_for_each_in,
};
pub use self::sync_table::{IntoPointer, LockedWrite, Read, SyncTable};
pub use self::vec::{AppendOnlyIndexVec, AppendOnlyVec};
pub use self::worker_local::{Registry, WorkerLocal};
pub use crate::marker::*;

mod freeze;
mod lock;
mod parallel;
mod sync_table;
mod vec;
mod worker_local;

/// Keep the conditional imports together in a submodule, so that import-sorting
/// doesn't split them up.
mod atomic {
    // Most hosts can just use a regular AtomicU64.
    #[cfg(target_has_atomic = "64")]
    pub use std::sync::atomic::AtomicU64;

    // Some 32-bit hosts don't have AtomicU64, so use a fallback.
    #[cfg(not(target_has_atomic = "64"))]
    pub use portable_atomic::AtomicU64;
}

mod mode {
    use std::sync::atomic::{AtomicU8, Ordering};

    use crate::sync::{DynSend, DynSync};

    const UNINITIALIZED: u8 = 0;
    const DYN_NOT_THREAD_SAFE: u8 = 1;
    const DYN_THREAD_SAFE: u8 = 2;

    static DYN_THREAD_SAFE_MODE: AtomicU8 = AtomicU8::new(UNINITIALIZED);

    // Whether thread safety is enabled (due to running under multiple threads).
    #[inline]
    pub fn check_dyn_thread_safe() -> Option<FromDyn<()>> {
        is_dyn_thread_safe().then_some(FromDyn(()))
    }

    // Whether thread safety is enabled (due to running under multiple threads).
    #[inline]
    pub fn is_dyn_thread_safe() -> bool {
        match DYN_THREAD_SAFE_MODE.load(Ordering::Relaxed) {
            DYN_NOT_THREAD_SAFE => false,
            DYN_THREAD_SAFE => true,
            _ => panic!("uninitialized dyn_thread_safe mode!"),
        }
    }

    // Whether thread safety might be enabled.
    #[inline]
    pub(super) fn might_be_dyn_thread_safe() -> bool {
        DYN_THREAD_SAFE_MODE.load(Ordering::Relaxed) != DYN_NOT_THREAD_SAFE
    }

    // Only set by the `-Z threads` compile option
    pub fn set_dyn_thread_safe_mode(mode: bool) {
        let set: u8 = if mode { DYN_THREAD_SAFE } else { DYN_NOT_THREAD_SAFE };
        let previous = DYN_THREAD_SAFE_MODE.compare_exchange(
            UNINITIALIZED,
            set,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );

        // Check that the mode was either uninitialized or was already set to the requested mode.
        assert!(previous.is_ok() || previous == Err(set));
    }

    #[derive(Copy, Clone)]
    pub struct FromDyn<T>(T);

    impl<T> FromDyn<T> {
        #[inline(always)]
        pub fn derive<O>(&self, val: O) -> FromDyn<O> {
            // We already did the check for `sync::is_dyn_thread_safe()` when creating `Self`
            FromDyn(val)
        }

        #[inline(always)]
        pub fn into_inner(self) -> T {
            self.0
        }
    }

    // `FromDyn` is `Send` if `T` is `DynSend`, since it ensures that sync::is_dyn_thread_safe() is true.
    unsafe impl<T: DynSend> Send for FromDyn<T> {}

    // `FromDyn` is `Sync` if `T` is `DynSync`, since it ensures that sync::is_dyn_thread_safe() is true.
    unsafe impl<T: DynSync> Sync for FromDyn<T> {}

    impl<T> std::ops::Deref for FromDyn<T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T> std::ops::DerefMut for FromDyn<T> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
}

/// This makes locks panic if they are already held.
/// It is only useful when you are running in a single thread
const ERROR_CHECKING: bool = false;

#[derive(Default)]
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

pub trait HashMapExt<K, V> {
    /// Same as HashMap::insert, but it may panic if there's already an
    /// entry for `key` with a value not equal to `value`
    fn insert_same(&mut self, key: K, value: V);
}

impl<K: Eq + Hash, V: Eq, S: BuildHasher> HashMapExt<K, V> for HashMap<K, V, S> {
    fn insert_same(&mut self, key: K, value: V) {
        self.entry(key).and_modify(|old| assert!(*old == value)).or_insert(value);
    }
}

#[derive(Debug, Default)]
pub struct RwLock<T>(parking_lot::RwLock<T>);

impl<T> RwLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        RwLock(parking_lot::RwLock::new(inner))
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    #[inline(always)]
    pub fn read(&self) -> ReadGuard<'_, T> {
        if ERROR_CHECKING {
            self.0.try_read().expect("lock was already held")
        } else {
            self.0.read()
        }
    }

    #[inline(always)]
    pub fn try_write(&self) -> Result<WriteGuard<'_, T>, ()> {
        self.0.try_write().ok_or(())
    }

    #[inline(always)]
    pub fn write(&self) -> WriteGuard<'_, T> {
        if ERROR_CHECKING {
            self.0.try_write().expect("lock was already held")
        } else {
            self.0.write()
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow(&self) -> ReadGuard<'_, T> {
        self.read()
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow_mut(&self) -> WriteGuard<'_, T> {
        self.write()
    }
}
