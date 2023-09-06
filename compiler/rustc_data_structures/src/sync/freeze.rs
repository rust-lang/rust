use crate::sync::{AtomicBool, ReadGuard, RwLock, WriteGuard};
#[cfg(parallel_compiler)]
use crate::sync::{DynSend, DynSync};
use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::atomic::Ordering,
};

/// A type which allows mutation using a lock until
/// the value is frozen and can be accessed lock-free.
///
/// Unlike `RwLock`, it can be used to prevent mutation past a point.
#[derive(Default)]
pub struct FreezeLock<T> {
    data: UnsafeCell<T>,
    frozen: AtomicBool,

    /// This lock protects writes to the `data` and `frozen` fields.
    lock: RwLock<()>,
}

#[cfg(parallel_compiler)]
unsafe impl<T: DynSync + DynSend> DynSync for FreezeLock<T> {}

impl<T> FreezeLock<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Self { data: UnsafeCell::new(value), frozen: AtomicBool::new(false), lock: RwLock::new(()) }
    }

    #[inline]
    pub fn read(&self) -> FreezeReadGuard<'_, T> {
        FreezeReadGuard {
            _lock_guard: if self.frozen.load(Ordering::Acquire) {
                None
            } else {
                Some(self.lock.read())
            },
            lock: self,
        }
    }

    #[inline]
    #[track_caller]
    pub fn write(&self) -> FreezeWriteGuard<'_, T> {
        let _lock_guard = self.lock.write();
        // Use relaxed ordering since we're in the write lock.
        assert!(!self.frozen.load(Ordering::Relaxed), "still mutable");
        FreezeWriteGuard { _lock_guard, lock: self, marker: PhantomData }
    }

    #[inline]
    pub fn freeze(&self) -> &T {
        if !self.frozen.load(Ordering::Acquire) {
            // Get the lock to ensure no concurrent writes and that we release the latest write.
            let _lock = self.lock.write();
            self.frozen.store(true, Ordering::Release);
        }

        // SAFETY: This is frozen so the data cannot be modified and shared access is sound.
        unsafe { &*self.data.get() }
    }
}

/// A guard holding shared access to a `FreezeLock` which is in a locked state or frozen.
#[must_use = "if unused the FreezeLock may immediately unlock"]
pub struct FreezeReadGuard<'a, T> {
    _lock_guard: Option<ReadGuard<'a, ()>>,
    lock: &'a FreezeLock<T>,
}

impl<'a, T: 'a> Deref for FreezeReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: If `lock` is not frozen, `_lock_guard` holds the lock to the `UnsafeCell` so
        // this has shared access until the `FreezeReadGuard` is dropped. If `lock` is frozen,
        // the data cannot be modified and shared access is sound.
        unsafe { &*self.lock.data.get() }
    }
}

/// A guard holding mutable access to a `FreezeLock` which is in a locked state or frozen.
#[must_use = "if unused the FreezeLock may immediately unlock"]
pub struct FreezeWriteGuard<'a, T> {
    _lock_guard: WriteGuard<'a, ()>,
    lock: &'a FreezeLock<T>,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'a> Deref for FreezeWriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `self._lock_guard` holds the lock to the `UnsafeCell` so this has shared access.
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T: 'a> DerefMut for FreezeWriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `self._lock_guard` holds the lock to the `UnsafeCell` so this has mutable access.
        unsafe { &mut *self.lock.data.get() }
    }
}
