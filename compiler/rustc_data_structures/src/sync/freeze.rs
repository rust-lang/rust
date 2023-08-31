use crate::sync::{AtomicBool, Lock, LockGuard};
#[cfg(parallel_compiler)]
use crate::sync::{DynSend, DynSync};
use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    sync::atomic::Ordering,
};

/// A type which allows mutation using a lock until
/// the value is frozen and can be accessed lock-free.
#[derive(Default)]
pub struct Freeze<T> {
    data: UnsafeCell<T>,
    frozen: AtomicBool,
    lock: Lock<()>,
}

#[cfg(parallel_compiler)]
unsafe impl<T: DynSync + DynSend> DynSync for Freeze<T> {}

impl<T> Freeze<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Self { data: UnsafeCell::new(value), frozen: AtomicBool::new(false), lock: Lock::new(()) }
    }

    #[inline]
    pub fn read(&self) -> FreezeReadGuard<'_, T> {
        FreezeReadGuard {
            _lock_guard: if self.frozen.load(Ordering::Acquire) {
                None
            } else {
                Some(self.lock.lock())
            },
            // SAFETY: If this is not frozen, `_lock_guard` holds the lock to the `UnsafeCell` so
            // this has shared access until the `FreezeReadGuard` is dropped. If this is frozen,
            // the data cannot be modified and shared access is sound.
            data: unsafe { &*self.data.get() },
        }
    }

    #[inline]
    #[track_caller]
    pub fn write(&self) -> FreezeWriteGuard<'_, T> {
        let _lock_guard = self.lock.lock();
        assert!(!self.frozen.load(Ordering::Relaxed), "still mutable");
        FreezeWriteGuard {
            _lock_guard,
            // SAFETY: `_lock_guard` holds the lock to the `UnsafeCell` so this has mutable access
            // until the `FreezeWriteGuard` is dropped.
            data: unsafe { &mut *self.data.get() },
        }
    }

    #[inline]
    pub fn freeze(&self) -> &T {
        if !self.frozen.load(Ordering::Acquire) {
            // Get the lock to ensure no concurrent writes.
            let _lock = self.lock.lock();
            self.frozen.store(true, Ordering::Release);
        }

        // SAFETY: This is frozen so the data cannot be modified and shared access is sound.
        unsafe { &*self.data.get() }
    }
}

/// A guard holding shared access to a `Freeze` which is in a locked state or frozen.
#[must_use = "if unused the Freeze may immediately unlock"]
pub struct FreezeReadGuard<'a, T> {
    _lock_guard: Option<LockGuard<'a, ()>>,
    data: &'a T,
}

impl<'a, T: 'a> Deref for FreezeReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.data
    }
}

/// A guard holding mutable access to a `Freeze` which is in a locked state or frozen.
#[must_use = "if unused the Freeze may immediately unlock"]
pub struct FreezeWriteGuard<'a, T> {
    _lock_guard: LockGuard<'a, ()>,
    data: &'a mut T,
}

impl<'a, T: 'a> Deref for FreezeWriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.data
    }
}

impl<'a, T: 'a> DerefMut for FreezeWriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.data
    }
}
