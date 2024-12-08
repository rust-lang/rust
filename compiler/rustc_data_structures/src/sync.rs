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
//! | `Lrc<T>`                | `rc::Rc<T>`         | `sync::Arc<T>`                  |
//! |` Weak<T>`               | `rc::Weak<T>`       | `sync::Weak<T>`                 |
//! | `LRef<'a, T>` [^2]      | `&'a mut T`         | `&'a T`                         |
//! |                         |                     |                                 |
//! | `AtomicBool`            | `Cell<bool>`        | `atomic::AtomicBool`            |
//! | `AtomicU32`             | `Cell<u32>`         | `atomic::AtomicU32`             |
//! | `AtomicU64`             | `Cell<u64>`         | `atomic::AtomicU64`             |
//! | `AtomicUsize`           | `Cell<usize>`       | `atomic::AtomicUsize`           |
//! |                         |                     |                                 |
//! | `Lock<T>`               | `RefCell<T>`        | `RefCell<T>` or                 |
//! |                         |                     | `parking_lot::Mutex<T>`         |
//! | `RwLock<T>`             | `RefCell<T>`        | `parking_lot::RwLock<T>`        |
//! | `MTLock<T>`        [^1] | `T`                 | `Lock<T>`                       |
//! | `MTLockRef<'a, T>` [^2] | `&'a mut MTLock<T>` | `&'a MTLock<T>`                 |
//! |                         |                     |                                 |
//! | `ParallelIterator`      | `Iterator`          | `rayon::iter::ParallelIterator` |
//!
//! [^1]: `MTLock` is similar to `Lock`, but the serial version avoids the cost
//! of a `RefCell`. This is appropriate when interior mutability is not
//! required.
//!
//! [^2]: `MTRef`, `MTLockRef` are type aliases.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

pub use crate::marker::*;

mod lock;
#[doc(no_inline)]
pub use lock::{Lock, LockGuard, Mode};

mod worker_local;
pub use worker_local::{Registry, WorkerLocal};

mod parallel;
pub use parallel::{join, par_for_each_in, par_map, parallel_guard, scope, try_par_for_each_in};
pub use vec::{AppendOnlyIndexVec, AppendOnlyVec};

mod vec;

mod freeze;
pub use freeze::{FreezeLock, FreezeReadGuard, FreezeWriteGuard};

mod mode {
    use std::sync::atomic::{AtomicU8, Ordering};

    const UNINITIALIZED: u8 = 0;
    const DYN_NOT_THREAD_SAFE: u8 = 1;
    const DYN_THREAD_SAFE: u8 = 2;

    static DYN_THREAD_SAFE_MODE: AtomicU8 = AtomicU8::new(UNINITIALIZED);

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
    pub fn might_be_dyn_thread_safe() -> bool {
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
}

// FIXME(parallel_compiler): Get rid of these aliases across the compiler.

pub use std::marker::{Send, Sync};
// Use portable AtomicU64 for targets without native 64-bit atomics
#[cfg(target_has_atomic = "64")]
pub use std::sync::atomic::AtomicU64;
pub use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};
pub use std::sync::{Arc as Lrc, OnceLock, Weak};

pub use mode::{is_dyn_thread_safe, set_dyn_thread_safe_mode};
pub use parking_lot::{
    MappedMutexGuard as MappedLockGuard, MappedRwLockReadGuard as MappedReadGuard,
    MappedRwLockWriteGuard as MappedWriteGuard, RwLockReadGuard as ReadGuard,
    RwLockWriteGuard as WriteGuard,
};
#[cfg(not(target_has_atomic = "64"))]
pub use portable_atomic::AtomicU64;

pub type LRef<'a, T> = &'a T;

#[derive(Debug, Default)]
pub struct MTLock<T>(Lock<T>);

impl<T> MTLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        MTLock(Lock::new(inner))
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
    pub fn lock(&self) -> LockGuard<'_, T> {
        self.0.lock()
    }

    #[inline(always)]
    pub fn lock_mut(&self) -> LockGuard<'_, T> {
        self.lock()
    }
}

use parking_lot::RwLock as InnerRwLock;

/// This makes locks panic if they are already held.
/// It is only useful when you are running in a single thread
const ERROR_CHECKING: bool = false;

pub type MTLockRef<'a, T> = LRef<'a, MTLock<T>>;

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
pub struct RwLock<T>(InnerRwLock<T>);

impl<T> RwLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        RwLock(InnerRwLock::new(inner))
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
    #[track_caller]
    pub fn with_read_lock<F: FnOnce(&T) -> R, R>(&self, f: F) -> R {
        f(&*self.read())
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
    pub fn with_write_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.write())
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

    #[inline(always)]
    pub fn leak(&self) -> &T {
        let guard = self.read();
        let ret = unsafe { &*(&raw const *guard) };
        std::mem::forget(guard);
        ret
    }
}

// FIXME: Probably a bad idea
impl<T: Clone> Clone for RwLock<T> {
    #[inline]
    fn clone(&self) -> Self {
        RwLock::new(self.borrow().clone())
    }
}
