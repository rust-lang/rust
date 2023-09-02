//! This module implements a lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
//! It implements `DynSend` and `DynSync` instead of the typical `Send` and `Sync` traits.
//!
//! When `cfg(parallel_compiler)` is not set, the lock is instead a wrapper around `RefCell`.

#![allow(dead_code)]

use std::fmt;

#[cfg(not(parallel_compiler))]
pub use disabled::*;
#[cfg(parallel_compiler)]
pub use enabled::*;

#[derive(Clone, Copy, PartialEq)]
pub enum Assume {
    NoSync,
    Sync,
}

mod enabled {
    use super::Assume;
    use crate::sync::mode;
    #[cfg(parallel_compiler)]
    use crate::sync::{DynSend, DynSync};
    use parking_lot::lock_api::RawMutex as _;
    use parking_lot::RawMutex;
    use std::cell::Cell;
    use std::cell::UnsafeCell;
    use std::hint::unreachable_unchecked;
    use std::intrinsics::unlikely;
    use std::marker::PhantomData;
    use std::ops::{Deref, DerefMut};

    /// A guard holding mutable access to a `Lock` which is in a locked state.
    #[must_use = "if unused the Lock will immediately unlock"]
    pub struct LockGuard<'a, T> {
        lock: &'a Lock<T>,
        marker: PhantomData<&'a mut T>,

        /// The syncronization mode of the lock. This is explicitly passed to let LLVM relate it
        /// to the original lock operation.
        assume: Assume,
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
            // SAFETY (dispatch): We get `self.assume` from the lock operation so it is consistent
            // with the lock state.
            // SAFETY (unlock): We know that the lock is locked as this type is a proof of that.
            unsafe {
                self.lock.dispatch(
                    self.assume,
                    |cell| {
                        debug_assert_eq!(cell.get(), true);
                        cell.set(false);
                        Some(())
                    },
                    |lock| lock.unlock(),
                );
            };
        }
    }

    enum LockMode {
        NoSync(Cell<bool>),
        Sync(RawMutex),
    }

    impl LockMode {
        #[inline(always)]
        fn to_assume(&self) -> Assume {
            match self {
                LockMode::NoSync(..) => Assume::NoSync,
                LockMode::Sync(..) => Assume::Sync,
            }
        }
    }

    /// The value representing a locked state for the `Cell`.
    const LOCKED: bool = true;

    /// A lock which only uses synchronization if `might_be_dyn_thread_safe` is true.
    /// It implements `DynSend` and `DynSync` instead of the typical `Send` and `Sync`.
    pub struct Lock<T> {
        mode: LockMode,
        data: UnsafeCell<T>,
    }

    impl<T> Lock<T> {
        #[inline(always)]
        pub fn new(inner: T) -> Self {
            Lock {
                mode: if unlikely(mode::might_be_dyn_thread_safe()) {
                    // Create the lock with synchronization enabled using the `RawMutex` type.
                    LockMode::Sync(RawMutex::INIT)
                } else {
                    // Create the lock with synchronization disabled.
                    LockMode::NoSync(Cell::new(!LOCKED))
                },
                data: UnsafeCell::new(inner),
            }
        }

        #[inline(always)]
        pub fn into_inner(self) -> T {
            self.data.into_inner()
        }

        #[inline(always)]
        pub fn get_mut(&mut self) -> &mut T {
            self.data.get_mut()
        }

        /// This dispatches on the `LockMode` and gives access to its variants depending on
        /// `assume`. If `no_sync` returns `None` this will panic.
        ///
        /// Safety
        /// This method must only be called if `might_be_dyn_thread_safe` on lock creation matches
        /// matches the `assume` argument.
        #[inline(always)]
        #[track_caller]
        unsafe fn dispatch<R>(
            &self,
            assume: Assume,
            no_sync: impl FnOnce(&Cell<bool>) -> Option<R>,
            sync: impl FnOnce(&RawMutex) -> R,
        ) -> R {
            #[inline(never)]
            #[track_caller]
            #[cold]
            fn lock_held() -> ! {
                panic!("lock was already held")
            }

            match assume {
                Assume::NoSync => {
                    let LockMode::NoSync(cell) = &self.mode else {
                        unsafe { unreachable_unchecked() }
                    };
                    if let Some(v) = no_sync(cell) {
                        v
                    } else {
                        // Call this here instead of in `no_sync` so `track_caller` gets properly
                        // passed along.
                        lock_held()
                    }
                }
                Assume::Sync => {
                    let LockMode::Sync(lock) = &self.mode else {
                        unsafe { unreachable_unchecked() }
                    };
                    sync(lock)
                }
            }
        }

        #[inline(always)]
        pub fn try_lock(&self) -> Option<LockGuard<'_, T>> {
            let assume = self.mode.to_assume();
            unsafe {
                self.dispatch(
                    assume,
                    |cell| Some((cell.get() != LOCKED).then(|| cell.set(LOCKED)).is_some()),
                    RawMutex::try_lock,
                )
                .then(|| LockGuard { lock: self, marker: PhantomData, assume })
            }
        }

        #[inline(always)]
        #[track_caller]
        pub unsafe fn lock_assume(&self, assume: Assume) -> LockGuard<'_, T> {
            unsafe {
                self.dispatch(
                    assume,
                    |cell| (cell.replace(LOCKED) != LOCKED).then(|| ()),
                    RawMutex::lock,
                );
                LockGuard { lock: self, marker: PhantomData, assume }
            }
        }

        #[inline(always)]
        #[track_caller]
        pub fn lock(&self) -> LockGuard<'_, T> {
            unsafe { self.lock_assume(self.mode.to_assume()) }
        }
    }

    #[cfg(parallel_compiler)]
    unsafe impl<T: DynSend> DynSend for Lock<T> {}
    #[cfg(parallel_compiler)]
    unsafe impl<T: DynSend> DynSync for Lock<T> {}
}

mod disabled {
    use super::Assume;
    use std::cell::RefCell;

    pub use std::cell::RefMut as LockGuard;

    pub struct Lock<T>(RefCell<T>);

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
        // This is unsafe to match the API for the `parallel_compiler` case.
        pub unsafe fn lock_assume(&self, _assume: Assume) -> LockGuard<'_, T> {
            self.0.borrow_mut()
        }

        #[inline(always)]
        #[track_caller]
        pub fn lock(&self) -> LockGuard<'_, T> {
            self.0.borrow_mut()
        }
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
