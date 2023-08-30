//! This module implements a lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
//! It implements `DynSend` and `DynSync` instead of the typical `Send` and `Sync` traits.
//!
//! When `cfg(parallel_compiler)` is not set, the lock is instead a wrapper around `RefCell`.

#[cfg(not(parallel_compiler))]
use std::cell::RefCell;
#[cfg(parallel_compiler)]
use {
    crate::cold_path,
    crate::sync::DynSend,
    crate::sync::DynSync,
    parking_lot::lock_api::RawMutex,
    std::cell::Cell,
    std::cell::UnsafeCell,
    std::fmt,
    std::intrinsics::{likely, unlikely},
    std::marker::PhantomData,
    std::mem::ManuallyDrop,
    std::ops::{Deref, DerefMut},
};

#[cfg(not(parallel_compiler))]
pub use std::cell::RefMut as LockGuard;

#[cfg(not(parallel_compiler))]
#[derive(Debug)]
pub struct Lock<T>(RefCell<T>);

#[cfg(not(parallel_compiler))]
impl<T> Lock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        Lock(RefCell::new(inner))
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
    pub fn try_lock(&self) -> Option<LockGuard<'_, T>> {
        self.0.try_borrow_mut().ok()
    }

    #[inline(always)]
    #[track_caller]
    pub fn lock(&self) -> LockGuard<'_, T> {
        self.0.borrow_mut()
    }
}

/// A guard holding mutable access to a `Lock` which is in a locked state.
#[cfg(parallel_compiler)]
#[must_use = "if unused the Lock will immediately unlock"]
pub struct LockGuard<'a, T> {
    lock: &'a Lock<T>,
    marker: PhantomData<&'a mut T>,
}

#[cfg(parallel_compiler)]
impl<'a, T: 'a> Deref for LockGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We have shared access to the mutable access owned by this type,
        // so we can give out a shared reference.
        unsafe { &*self.lock.data.get() }
    }
}

#[cfg(parallel_compiler)]
impl<'a, T: 'a> DerefMut for LockGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We have mutable access to the data so we can give out a mutable reference.
        unsafe { &mut *self.lock.data.get() }
    }
}

#[cfg(parallel_compiler)]
impl<'a, T: 'a> Drop for LockGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: We know that the lock is in a locked
        // state because it is a invariant of this type.
        unsafe { self.lock.raw.unlock() };
    }
}

#[cfg(parallel_compiler)]
union LockRawUnion {
    /// Indicates if the cell is locked. Only used if `LockRaw.sync` is false.
    cell: ManuallyDrop<Cell<bool>>,

    /// A lock implementation that's only used if `LockRaw.sync` is true.
    lock: ManuallyDrop<parking_lot::RawMutex>,
}

/// A raw lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
/// It contains no associated data and is used in the implementation of `Lock` which does have such data.
///
/// A manual implementation of a tagged union is used with the `sync` field and the `LockRawUnion` instead
/// of using enums as it results in better code generation.
#[cfg(parallel_compiler)]
struct LockRaw {
    /// Indicates if synchronization is used via `opt.lock` if true,
    /// or if a non-thread safe cell is used via `opt.cell`. This is set on initialization and never changed.
    sync: bool,
    opt: LockRawUnion,
}

#[cfg(parallel_compiler)]
impl LockRaw {
    fn new() -> Self {
        if unlikely(super::mode::might_be_dyn_thread_safe()) {
            // Create the lock with synchronization enabled using the `RawMutex` type.
            LockRaw {
                sync: true,
                opt: LockRawUnion { lock: ManuallyDrop::new(parking_lot::RawMutex::INIT) },
            }
        } else {
            // Create the lock with synchronization disabled.
            LockRaw { sync: false, opt: LockRawUnion { cell: ManuallyDrop::new(Cell::new(false)) } }
        }
    }

    #[inline(always)]
    fn try_lock(&self) -> bool {
        // SAFETY: This is safe since the union fields are used in accordance with `self.sync`.
        unsafe {
            if likely(!self.sync) {
                if self.opt.cell.get() {
                    false
                } else {
                    self.opt.cell.set(true);
                    true
                }
            } else {
                self.opt.lock.try_lock()
            }
        }
    }

    #[inline(always)]
    fn lock(&self) {
        if super::ERROR_CHECKING {
            // We're in the debugging mode, so assert that the lock is not held so we
            // get a panic instead of waiting for the lock.
            assert_eq!(self.try_lock(), true, "lock must not be hold");
        } else {
            // SAFETY: This is safe since the union fields are used in accordance with `self.sync`.
            unsafe {
                if likely(!self.sync) {
                    if unlikely(self.opt.cell.replace(true)) {
                        cold_path(|| panic!("lock was already held"))
                    }
                } else {
                    self.opt.lock.lock();
                }
            }
        }
    }

    /// This unlocks the lock.
    ///
    /// Safety
    /// This method may only be called if the lock is currently held.
    #[inline(always)]
    unsafe fn unlock(&self) {
        // SAFETY: The union use is safe since the union fields are used in accordance with
        // `self.sync` and the `unlock` method precondition is upheld by the caller.
        unsafe {
            if likely(!self.sync) {
                debug_assert_eq!(self.opt.cell.get(), true);
                self.opt.cell.set(false);
            } else {
                self.opt.lock.unlock();
            }
        }
    }
}

/// A lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
/// It implements `DynSend` and `DynSync` instead of the typical `Send` and `Sync`.
#[cfg(parallel_compiler)]
pub struct Lock<T> {
    raw: LockRaw,
    data: UnsafeCell<T>,
}

#[cfg(parallel_compiler)]
impl<T> Lock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        Lock { raw: LockRaw::new(), data: UnsafeCell::new(inner) }
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
        if self.raw.try_lock() { Some(LockGuard { lock: self, marker: PhantomData }) } else { None }
    }

    #[inline(always)]
    pub fn lock(&self) -> LockGuard<'_, T> {
        self.raw.lock();
        LockGuard { lock: self, marker: PhantomData }
    }
}

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

#[cfg(parallel_compiler)]
unsafe impl<T: DynSend> DynSend for Lock<T> {}
#[cfg(parallel_compiler)]
unsafe impl<T: DynSend> DynSync for Lock<T> {}

#[cfg(parallel_compiler)]
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

impl<T: Default> Default for Lock<T> {
    #[inline]
    fn default() -> Self {
        Lock::new(T::default())
    }
}
