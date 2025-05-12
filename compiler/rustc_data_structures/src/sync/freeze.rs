use std::cell::UnsafeCell;
use std::intrinsics::likely;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::sync::{DynSend, DynSync, ReadGuard, RwLock, WriteGuard};

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

unsafe impl<T: DynSync + DynSend> DynSync for FreezeLock<T> {}

impl<T> FreezeLock<T> {
    #[inline]
    pub fn new(value: T) -> Self {
        Self::with(value, false)
    }

    #[inline]
    pub fn frozen(value: T) -> Self {
        Self::with(value, true)
    }

    #[inline]
    pub fn with(value: T, frozen: bool) -> Self {
        Self {
            data: UnsafeCell::new(value),
            frozen: AtomicBool::new(frozen),
            lock: RwLock::new(()),
        }
    }

    /// Clones the inner value along with the frozen state.
    #[inline]
    pub fn clone(&self) -> Self
    where
        T: Clone,
    {
        let lock = self.read();
        Self::with(lock.clone(), self.is_frozen())
    }

    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::Acquire)
    }

    /// Get the inner value if frozen.
    #[inline]
    pub fn get(&self) -> Option<&T> {
        if likely(self.frozen.load(Ordering::Acquire)) {
            // SAFETY: This is frozen so the data cannot be modified.
            unsafe { Some(&*self.data.get()) }
        } else {
            None
        }
    }

    #[inline]
    pub fn read(&self) -> FreezeReadGuard<'_, T> {
        FreezeReadGuard {
            _lock_guard: if self.frozen.load(Ordering::Acquire) {
                None
            } else {
                Some(self.lock.read())
            },
            data: unsafe { NonNull::new_unchecked(self.data.get()) },
        }
    }

    #[inline]
    pub fn borrow(&self) -> FreezeReadGuard<'_, T> {
        self.read()
    }

    #[inline]
    #[track_caller]
    pub fn write(&self) -> FreezeWriteGuard<'_, T> {
        self.try_write().expect("data should not be frozen if we're still attempting to mutate it")
    }

    #[inline]
    pub fn try_write(&self) -> Option<FreezeWriteGuard<'_, T>> {
        let _lock_guard = self.lock.write();
        // Use relaxed ordering since we're in the write lock.
        if self.frozen.load(Ordering::Relaxed) {
            None
        } else {
            Some(FreezeWriteGuard {
                _lock_guard,
                data: unsafe { NonNull::new_unchecked(self.data.get()) },
                frozen: &self.frozen,
                marker: PhantomData,
            })
        }
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
pub struct FreezeReadGuard<'a, T: ?Sized> {
    _lock_guard: Option<ReadGuard<'a, ()>>,
    data: NonNull<T>,
}

impl<'a, T: ?Sized + 'a> Deref for FreezeReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: If the lock is not frozen, `_lock_guard` holds the lock to the `UnsafeCell` so
        // this has shared access until the `FreezeReadGuard` is dropped. If the lock is frozen,
        // the data cannot be modified and shared access is sound.
        unsafe { &*self.data.as_ptr() }
    }
}

impl<'a, T: ?Sized> FreezeReadGuard<'a, T> {
    #[inline]
    pub fn map<U: ?Sized>(this: Self, f: impl FnOnce(&T) -> &U) -> FreezeReadGuard<'a, U> {
        FreezeReadGuard { data: NonNull::from(f(&*this)), _lock_guard: this._lock_guard }
    }
}

/// A guard holding mutable access to a `FreezeLock` which is in a locked state or frozen.
#[must_use = "if unused the FreezeLock may immediately unlock"]
pub struct FreezeWriteGuard<'a, T: ?Sized> {
    _lock_guard: WriteGuard<'a, ()>,
    frozen: &'a AtomicBool,
    data: NonNull<T>,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> FreezeWriteGuard<'a, T> {
    pub fn freeze(self) -> &'a T {
        self.frozen.store(true, Ordering::Release);

        // SAFETY: This is frozen so the data cannot be modified and shared access is sound.
        unsafe { &*self.data.as_ptr() }
    }
}

impl<'a, T: ?Sized> FreezeWriteGuard<'a, T> {
    #[inline]
    pub fn map<U: ?Sized>(
        mut this: Self,
        f: impl FnOnce(&mut T) -> &mut U,
    ) -> FreezeWriteGuard<'a, U> {
        FreezeWriteGuard {
            data: NonNull::from(f(&mut *this)),
            _lock_guard: this._lock_guard,
            frozen: this.frozen,
            marker: PhantomData,
        }
    }
}

impl<'a, T: ?Sized + 'a> Deref for FreezeWriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `self._lock_guard` holds the lock to the `UnsafeCell` so this has shared access.
        unsafe { &*self.data.as_ptr() }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for FreezeWriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `self._lock_guard` holds the lock to the `UnsafeCell` so this has mutable access.
        unsafe { &mut *self.data.as_ptr() }
    }
}
