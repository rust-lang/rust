use super::abi;
use crate::{
    cell::UnsafeCell,
    convert::TryFrom,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// A mutex implemented by `dis_dsp` (for intra-core synchronization) and a
/// spinlock (for inter-core synchronization).
pub struct SpinMutex<T = ()> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

impl<T> SpinMutex<T> {
    #[inline]
    pub const fn new(x: T) -> Self {
        Self { locked: AtomicBool::new(false), data: UnsafeCell::new(x) }
    }

    /// Acquire a lock.
    #[inline]
    pub fn with_locked<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        struct SpinMutexGuard<'a>(&'a AtomicBool);

        impl Drop for SpinMutexGuard<'_> {
            #[inline]
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
                unsafe { abi::ena_dsp() };
            }
        }

        let _guard;
        if unsafe { abi::sns_dsp() } == 0 {
            let er = unsafe { abi::dis_dsp() };
            debug_assert!(er >= 0);

            // Wait until the current processor acquires a lock.
            while self.locked.swap(true, Ordering::Acquire) {}

            _guard = SpinMutexGuard(&self.locked);
        }

        f(unsafe { &mut *self.data.get() })
    }
}

/// `OnceCell<(abi::ID, T)>` implemented by `dis_dsp` (for intra-core
/// synchronization) and a spinlock (for inter-core synchronization).
///
/// It's assumed that `0` is not a valid ID, and all kernel
/// object IDs fall into range `1..=usize::MAX`.
pub struct SpinIdOnceCell<T = ()> {
    id: AtomicUsize,
    spin: SpinMutex<()>,
    extra: UnsafeCell<MaybeUninit<T>>,
}

const ID_UNINIT: usize = 0;

impl<T> SpinIdOnceCell<T> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            id: AtomicUsize::new(ID_UNINIT),
            extra: UnsafeCell::new(MaybeUninit::uninit()),
            spin: SpinMutex::new(()),
        }
    }

    #[inline]
    pub fn get(&self) -> Option<(abi::ID, &T)> {
        match self.id.load(Ordering::Acquire) {
            ID_UNINIT => None,
            id => Some((id as abi::ID, unsafe { (&*self.extra.get()).assume_init_ref() })),
        }
    }

    #[inline]
    pub fn get_mut(&mut self) -> Option<(abi::ID, &mut T)> {
        match *self.id.get_mut() {
            ID_UNINIT => None,
            id => Some((id as abi::ID, unsafe { (&mut *self.extra.get()).assume_init_mut() })),
        }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self) -> (abi::ID, &T) {
        (self.id.load(Ordering::Acquire) as abi::ID, unsafe {
            (&*self.extra.get()).assume_init_ref()
        })
    }

    /// Assign the content without checking if it's already initialized or
    /// being initialized.
    pub unsafe fn set_unchecked(&self, (id, extra): (abi::ID, T)) {
        debug_assert!(self.get().is_none());

        // Assumption: A positive `abi::ID` fits in `usize`.
        debug_assert!(id >= 0);
        debug_assert!(usize::try_from(id).is_ok());
        let id = id as usize;

        unsafe { *self.extra.get() = MaybeUninit::new(extra) };
        self.id.store(id, Ordering::Release);
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty. If the cell was empty and `f` failed, an
    /// error is returned.
    ///
    /// Warning: `f` must not perform a blocking operation, which
    /// includes panicking.
    #[inline]
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<(abi::ID, &T), E>
    where
        F: FnOnce() -> Result<(abi::ID, T), E>,
    {
        // Fast path
        if let Some(x) = self.get() {
            return Ok(x);
        }

        self.initialize(f)?;

        debug_assert!(self.get().is_some());

        // Safety: The inner value has been initialized
        Ok(unsafe { self.get_unchecked() })
    }

    fn initialize<F, E>(&self, f: F) -> Result<(), E>
    where
        F: FnOnce() -> Result<(abi::ID, T), E>,
    {
        self.spin.with_locked(|_| {
            if self.id.load(Ordering::Relaxed) == ID_UNINIT {
                let (initialized_id, initialized_extra) = f()?;

                // Assumption: A positive `abi::ID` fits in `usize`.
                debug_assert!(initialized_id >= 0);
                debug_assert!(usize::try_from(initialized_id).is_ok());
                let initialized_id = initialized_id as usize;

                // Store the initialized contents. Use the release ordering to
                // make sure the write is visible to the callers of `get`.
                unsafe { *self.extra.get() = MaybeUninit::new(initialized_extra) };
                self.id.store(initialized_id, Ordering::Release);
            }
            Ok(())
        })
    }
}

impl<T> Drop for SpinIdOnceCell<T> {
    #[inline]
    fn drop(&mut self) {
        if self.get_mut().is_some() {
            unsafe { (&mut *self.extra.get()).assume_init_drop() };
        }
    }
}
