//! This module implements a lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
//! It implements `DynSend` and `DynSync` instead of the typical `Send` and `Sync` traits.

use std::fmt;

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    NoSync,
    Sync,
}

use std::cell::{Cell, UnsafeCell};
use std::intrinsics::unlikely;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

use parking_lot::RawMutex;
use parking_lot::lock_api::RawMutex as _;

use crate::sync::{DynSend, DynSync, mode};

/// A guard holding mutable access to a `Lock` which is in a locked state.
#[must_use = "if unused the Lock will immediately unlock"]
pub struct LockGuard<'a, T> {
    lock: &'a Lock<T>,
    marker: PhantomData<&'a mut T>,

    /// The synchronization mode of the lock. This is explicitly passed to let LLVM relate it
    /// to the original lock operation.
    mode: Mode,
}

impl<'a, T: 'a> Deref for LockGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We have shared access to the mutable access owned by this type,
        // so we can give out a shared reference.
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T: 'a> DerefMut for LockGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We have mutable access to the data so we can give out a mutable reference.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<'a, T: 'a> Drop for LockGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY (union access): We get `self.mode` from the lock operation so it is consistent
        // with the `lock.mode` state. This means we access the right union fields.
        match self.mode {
            Mode::NoSync => {
                let cell = unsafe { &self.lock.mode_union.no_sync };
                debug_assert!(cell.get());
                cell.set(false);
            }
            // SAFETY (unlock): We know that the lock is locked as this type is a proof of that.
            Mode::Sync => unsafe { self.lock.mode_union.sync.unlock() },
        }
    }
}

union ModeUnion {
    /// Indicates if the cell is locked. Only used if `Lock.mode` is `NoSync`.
    no_sync: ManuallyDrop<Cell<bool>>,

    /// A lock implementation that's only used if `Lock.mode` is `Sync`.
    sync: ManuallyDrop<RawMutex>,
}

/// The value representing a locked state for the `Cell`.
const LOCKED: bool = true;

/// A lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
/// It implements `DynSend` and `DynSync` instead of the typical `Send` and `Sync`.
pub struct Lock<T> {
    /// Indicates if synchronization is used via `mode_union.sync` if it's `Sync`, or if a
    /// not thread safe cell is used via `mode_union.no_sync` if it's `NoSync`.
    /// This is set on initialization and never changed.
    mode: Mode,

    mode_union: ModeUnion,
    data: UnsafeCell<T>,
}

impl<T> Lock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        let (mode, mode_union) = if unlikely(mode::might_be_dyn_thread_safe()) {
            // Create the lock with synchronization enabled using the `RawMutex` type.
            (Mode::Sync, ModeUnion { sync: ManuallyDrop::new(RawMutex::INIT) })
        } else {
            // Create the lock with synchronization disabled.
            (Mode::NoSync, ModeUnion { no_sync: ManuallyDrop::new(Cell::new(!LOCKED)) })
        };
        Lock { mode, mode_union, data: UnsafeCell::new(inner) }
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    #[inline(always)]
    pub fn try_lock(&self) -> Option<LockGuard<'_, T>> {
        let mode = self.mode;
        // SAFETY: This is safe since the union fields are used in accordance with `self.mode`.
        match mode {
            Mode::NoSync => {
                let cell = unsafe { &self.mode_union.no_sync };
                let was_unlocked = cell.get() != LOCKED;
                if was_unlocked {
                    cell.set(LOCKED);
                }
                was_unlocked
            }
            Mode::Sync => unsafe { self.mode_union.sync.try_lock() },
        }
        .then(|| LockGuard { lock: self, marker: PhantomData, mode })
    }

    /// This acquires the lock assuming synchronization is in a specific mode.
    ///
    /// Safety
    /// This method must only be called with `Mode::Sync` if `might_be_dyn_thread_safe` was
    /// true on lock creation.
    #[inline(always)]
    #[track_caller]
    pub unsafe fn lock_assume(&self, mode: Mode) -> LockGuard<'_, T> {
        #[inline(never)]
        #[track_caller]
        #[cold]
        fn lock_held() -> ! {
            panic!("lock was already held")
        }

        // SAFETY: This is safe since the union fields are used in accordance with `mode`
        // which also must match `self.mode` due to the safety precondition.
        unsafe {
            match mode {
                Mode::NoSync => {
                    if unlikely(self.mode_union.no_sync.replace(LOCKED) == LOCKED) {
                        lock_held()
                    }
                }
                Mode::Sync => self.mode_union.sync.lock(),
            }
        }
        LockGuard { lock: self, marker: PhantomData, mode }
    }

    #[inline(always)]
    #[track_caller]
    pub fn lock(&self) -> LockGuard<'_, T> {
        unsafe { self.lock_assume(self.mode) }
    }
}

unsafe impl<T: DynSend> DynSend for Lock<T> {}
unsafe impl<T: DynSend> DynSync for Lock<T> {}

impl<T> Lock<T> {
    #[inline(always)]
    #[track_caller]
    pub fn with_lock<F: FnOnce(&mut T) -> R, R>(&self, f: F) -> R {
        f(&mut *self.lock())
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow(&self) -> LockGuard<'_, T> {
        self.lock()
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow_mut(&self) -> LockGuard<'_, T> {
        self.lock()
    }
}

impl<T: Default> Default for Lock<T> {
    #[inline]
    fn default() -> Self {
        Lock::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for Lock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.try_lock() {
            Some(guard) => f.debug_struct("Lock").field("data", &&*guard).finish(),
            None => {
                struct LockedPlaceholder;
                impl fmt::Debug for LockedPlaceholder {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("<locked>")
                    }
                }

                f.debug_struct("Lock").field("data", &LockedPlaceholder).finish()
            }
        }
    }
}
