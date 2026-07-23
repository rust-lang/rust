use core::fmt;
use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::{hint, mem};

use parking_lot::RawRwLock;
use parking_lot::lock_api::RawRwLock as _;

use crate::sync::mode::{self, Mode};

// Mimic `RefCell` borrow counting
type BorrowCounter = isize;
const UNUSED: BorrowCounter = 0;

#[inline(always)]
const fn is_writing(x: BorrowCounter) -> bool {
    x < UNUSED
}

#[inline(always)]
const fn is_reading(x: BorrowCounter) -> bool {
    x > UNUSED
}

/// A guard holding shared access to a `RwLock` which is in a read-locked state.
#[must_use = "if unused the Lock will immediately unlock"]
pub struct ReadGuard<'a, T> {
    rw_lock: &'a RwLock<T>,
    marker: PhantomData<&'a T>,

    /// The synchronization mode of the lock. This is explicitly passed to let LLVM relate it
    /// to the original lock operation.
    mode: Mode,
}

/// # SAFETY
///
/// - union_access: `mode` and `mode_union` must be in the same state
/// - unlock: `mode_union` must be in a read-locked state
#[inline(always)]
unsafe fn drop_read_lock(mode: Mode, mode_union: &ModeUnion) {
    match mode {
        Mode::NoSync => {
            let borrow = unsafe { &mode_union.no_sync };
            debug_assert!(is_reading(borrow.get()));
            borrow.update(|b| b - 1);
        }
        Mode::Sync => unsafe { mode_union.sync.unlock_shared() },
    }
}

impl<'a, T: 'a> Drop for ReadGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY (union access): We get `self.mode` from the lock operation so it is consistent
        // with the `lock.mode` state. This means we access the right union fields.
        // SAFETY (unlock): We know that the rwlock is read-locked as this type is a proof of that.
        unsafe {
            drop_read_lock(self.mode, &self.rw_lock.mode_union);
        }
    }
}

impl<'a, T: 'a> Deref for ReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We have shared access to the shared access of this type,
        // so we can give out a shared reference.
        unsafe { &*self.rw_lock.data.get() }
    }
}

#[must_use = "if unused the RwLock will immediately unlock"]
pub struct MappedReadGuard<'a, T> {
    mode_union: &'a ModeUnion,
    data: *const T,
    marker: PhantomData<&'a T>,

    /// The synchronization mode of the lock. This is explicitly passed to let LLVM relate it
    /// to the original lock operation.
    mode: Mode,
}

impl<'a, T: 'a> Drop for MappedReadGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY (union access): We get `self.mode` from the lock operation so it is consistent
        // with the `lock.mode` state. This means we access the right union fields.
        // SAFETY (unlock): We know that the rwlock is read-locked as this type is a proof of that.
        unsafe {
            drop_read_lock(self.mode, self.mode_union);
        }
    }
}

impl<'a, T: 'a> Deref for MappedReadGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We have shared access to the shared access of this type,
        // so we can give out a shared reference.
        unsafe { &*self.data }
    }
}

impl<'a, T: 'a> ReadGuard<'a, T> {
    pub fn map<U, F>(s: Self, f: F) -> MappedReadGuard<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        let mode = s.mode;
        let mode_union = &s.rw_lock.mode_union;
        let data = f(unsafe { &*s.rw_lock.data.get() });
        mem::forget(s);
        MappedReadGuard { mode_union, data, marker: PhantomData, mode }
    }
}

/// A guard holding exclusive access to a `RwLock` which is in a write-locked state.
#[must_use = "if unused the Lock will immediately unlock"]
pub struct WriteGuard<'a, T> {
    rw_lock: &'a RwLock<T>,
    marker: PhantomData<&'a mut T>,

    /// The synchronization mode of the lock. This is explicitly passed to let LLVM relate it
    /// to the original lock operation.
    mode: Mode,
}

/// # SAFETY
///
/// - union_access: `mode` and `mode_union` must be in the same state
/// - unlock: `mode_union` must be in a write-locked state
#[inline(always)]
unsafe fn drop_write_lock(mode: Mode, mode_union: &ModeUnion) {
    // SAFETY: Caller guarantees union_access.
    match mode {
        Mode::NoSync => {
            let borrow = unsafe { &mode_union.no_sync };
            debug_assert!(is_writing(borrow.get()));
            borrow.update(|b| b + 1);
        }
        // SAFETY: Caller guarantees write-locked state.
        Mode::Sync => unsafe { mode_union.sync.unlock_exclusive() },
    }
}

impl<'a, T: 'a> Drop for WriteGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY (union access): We get `mode` from the lock operation so it is consistent
        // with the `lock.mode` state. This means we access the right union fields.
        // SAFETY (unlock): We know that the rwlock is write-locked as this type is a proof of that.
        unsafe { drop_write_lock(self.mode, &self.rw_lock.mode_union) };
    }
}

impl<'a, T: 'a> Deref for WriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We have shared access to the shared access of this type,
        // so we can give out a shared reference.
        unsafe { &*self.rw_lock.data.get() }
    }
}

impl<'a, T: 'a> DerefMut for WriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We have exclusive access to the exclusive access of this type,
        // so we can give out a exclusive reference.
        unsafe { &mut *self.rw_lock.data.get() }
    }
}

#[must_use = "if unused the RwLock will immediately unlock"]
pub struct MappedWriteGuard<'a, T> {
    mode_union: &'a ModeUnion,
    data: *mut T,
    marker: PhantomData<&'a mut T>,

    /// The synchronization mode of the lock. This is explicitly passed to let LLVM relate it
    /// to the original lock operation.
    mode: Mode,
}

impl<'a, T: 'a> Drop for MappedWriteGuard<'a, T> {
    fn drop(&mut self) {
        // SAFETY (union access): We get `self.mode` from the lock operation so it is consistent
        // with the `lock.mode` state. This means we access the right union fields.
        // SAFETY (unlock): We know that the rwlock is write-locked as this type is a proof of that.
        unsafe {
            drop_write_lock(self.mode, &self.mode_union);
        }
    }
}

impl<'a, T: 'a> Deref for MappedWriteGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We have shared access to the exclusive access of this type,
        // so we can give out a shared reference.
        unsafe { &*self.data }
    }
}

impl<'a, T: 'a> DerefMut for MappedWriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We have exclusive access to the exclusive access of this type,
        // so we can give out a exclusive reference.
        unsafe { &mut *self.data }
    }
}

impl<'a, T: 'a> WriteGuard<'a, T> {
    pub fn map<U, F>(s: Self, f: F) -> MappedWriteGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let mode = s.mode;
        let mode_union = &s.rw_lock.mode_union;
        // SAFETY: We have owned access to the exclusive access of this type,
        // so we can give out a exclusive reference.
        let data = f(unsafe { &mut *s.rw_lock.data.get() });
        mem::forget(s);
        MappedWriteGuard { mode_union, data, marker: PhantomData, mode }
    }
}

union ModeUnion {
    /// Mimics the borrow counting of `RefCell` that is only used if `RwLock.mode` is `NoSync`.
    no_sync: ManuallyDrop<Cell<BorrowCounter>>,

    /// A RwLock of parking_lot that is only used if `RwLock.mode` is `Sync`
    sync: ManuallyDrop<RawRwLock>,
}

pub struct RwLock<T> {
    mode: Mode,

    mode_union: ModeUnion,
    data: UnsafeCell<T>,
}

/// This makes locks panic if they are already held.
/// It is only useful when you are running in a single thread
const ERROR_CHECKING: bool = false;

impl<T> RwLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        let (mode, mode_union) = if mode::might_be_dyn_thread_safe() {
            hint::cold_path();
            // Create the lock with synchronization enabled using the `RawRwLock` type.
            (Mode::Sync, ModeUnion { sync: ManuallyDrop::new(RawRwLock::INIT) })
        } else {
            // Create the lock with synchronization disabled.
            (Mode::NoSync, ModeUnion { no_sync: ManuallyDrop::new(Cell::new(UNUSED)) })
        };
        RwLock { mode, mode_union, data: UnsafeCell::new(inner) }
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    pub fn read(&self) -> ReadGuard<'_, T> {
        if ERROR_CHECKING {
            self.try_read().expect("lock was already held")
        } else {
            let mode = self.mode;

            // SAFETY: This is safe since the union fields are used in accordance with `self.mode`.
            match mode {
                Mode::NoSync => {
                    let borrow = unsafe { &self.mode_union.no_sync };
                    if is_writing(borrow.get()) {
                        panic!("RwLock already mutably borrowed");
                    }
                    borrow.update(|b| b + 1);
                }
                Mode::Sync => unsafe { self.mode_union.sync.lock_shared() },
            }
            ReadGuard { rw_lock: self, marker: PhantomData, mode }
        }
    }

    pub fn try_read(&self) -> Result<ReadGuard<'_, T>, ()> {
        let mode = self.mode;

        // SAFETY: This is safe since the union fields are used in accordance with `self.mode`.
        match mode {
            Mode::NoSync => {
                let borrow = unsafe { &self.mode_union.no_sync };
                let is_reading = !is_writing(borrow.get());
                if is_reading {
                    borrow.update(|b| b + 1);
                }
                is_reading
            }
            Mode::Sync => unsafe { self.mode_union.sync.try_lock_shared() },
        }
        .then(|| ReadGuard { rw_lock: self, marker: PhantomData, mode })
        .ok_or(())
    }

    pub fn write(&self) -> WriteGuard<'_, T> {
        if ERROR_CHECKING {
            self.try_write().expect("lock was already held")
        } else {
            let mode = self.mode;

            // SAFETY: This is safe since the union fields are used in accordance with `self.mode`.
            match mode {
                Mode::NoSync => {
                    let borrow = unsafe { &self.mode_union.no_sync };
                    if borrow.get() != UNUSED {
                        panic!("lock was already held")
                    }
                    borrow.replace(UNUSED - 1);
                }
                Mode::Sync => unsafe { self.mode_union.sync.lock_exclusive() },
            }
            WriteGuard { rw_lock: self, marker: PhantomData, mode }
        }
    }

    pub fn try_write(&self) -> Result<WriteGuard<'_, T>, ()> {
        let mode = self.mode;

        // SAFETY: This is safe since the union fields are used in accordance with `self.mode`.
        match mode {
            Mode::NoSync => {
                let borrow = unsafe { &self.mode_union.no_sync };
                let unused = borrow.get() == UNUSED;
                if unused {
                    borrow.replace(UNUSED - 1);
                }
                unused
            }
            Mode::Sync => unsafe { self.mode_union.sync.try_lock_exclusive() },
        }
        .then(|| WriteGuard { rw_lock: self, marker: PhantomData, mode })
        .ok_or(())
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

impl<T: Default> Default for RwLock<T> {
    fn default() -> Self {
        RwLock::new(T::default())
    }
}

impl<T: fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("RwLock");
        match self.try_read() {
            Ok(guard) => debug.field("data", &&*guard).finish(),
            Err(()) => {
                // Additional format_args! here is to remove quotes around <locked> in debug output.
                debug.field("data", &format_args!("<locked>")).finish()
            }
        }
    }
}

impl<'a, T: fmt::Debug + 'a> fmt::Debug for MappedWriteGuard<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<'a, T: fmt::Debug + 'a> fmt::Debug for MappedReadGuard<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}
