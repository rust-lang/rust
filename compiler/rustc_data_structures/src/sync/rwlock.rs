use std::cell::{Ref, RefCell, RefMut};
use std::intrinsics::cold_path;
use std::ops::{Deref, DerefMut};

use crate::sync::mode::might_be_dyn_thread_safe;

#[derive(Debug)]
pub enum RwLock<T> {
    Sync(parking_lot::RwLock<T>),
    NoSync(RefCell<T>),
}

#[clippy::has_significant_drop]
#[must_use = "if unused the RwLock will immediately unlock"]
#[derive(Debug)]
pub enum ReadGuard<'a, T> {
    Sync(parking_lot::RwLockReadGuard<'a, T>),
    NoSync(Ref<'a, T>),
}

#[clippy::has_significant_drop]
#[must_use = "if unused the RwLock will immediately unlock"]
#[derive(Debug)]
pub enum WriteGuard<'a, T> {
    Sync(parking_lot::RwLockWriteGuard<'a, T>),
    NoSync(RefMut<'a, T>),
}

#[clippy::has_significant_drop]
#[must_use = "if unused the RwLock will immediately unlock"]
#[derive(Debug)]
pub enum MappedReadGuard<'a, T> {
    Sync(parking_lot::MappedRwLockReadGuard<'a, T>),
    NoSync(Ref<'a, T>),
}

#[clippy::has_significant_drop]
#[must_use = "if unused the RwLock will immediately unlock"]
#[derive(Debug)]
pub enum MappedWriteGuard<'a, T> {
    Sync(parking_lot::MappedRwLockWriteGuard<'a, T>),
    NoSync(RefMut<'a, T>),
}

#[derive(Debug)]
pub struct ReadError;

#[derive(Debug)]
pub struct WriteError;

impl<T> RwLock<T> {
    #[inline(always)]
    pub fn new(inner: T) -> Self {
        if might_be_dyn_thread_safe() {
            cold_path();
            RwLock::Sync(parking_lot::RwLock::new(inner))
        } else {
            RwLock::NoSync(RefCell::new(inner))
        }
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        match self {
            RwLock::Sync(inner) => {
                cold_path();
                parking_lot::RwLock::into_inner(inner)
            }
            RwLock::NoSync(inner) => RefCell::into_inner(inner),
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        match self {
            RwLock::Sync(inner) => {
                cold_path();
                parking_lot::RwLock::get_mut(inner)
            }
            RwLock::NoSync(inner) => RefCell::get_mut(inner),
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn read(&self) -> ReadGuard<'_, T> {
        match self {
            RwLock::Sync(inner) => {
                cold_path();
                ReadGuard::Sync(inner.read())
            }
            RwLock::NoSync(inner) => ReadGuard::NoSync(inner.borrow()),
        }
    }

    #[inline(always)]
    pub fn try_read(&self) -> Result<ReadGuard<'_, T>, ReadError> {
        match self {
            RwLock::Sync(inner) => {
                cold_path();
                Ok(ReadGuard::Sync(inner.try_read().ok_or(ReadError)?))
            }
            RwLock::NoSync(inner) => {
                Ok(ReadGuard::NoSync(inner.try_borrow().map_err(|_| ReadError)?))
            }
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn write(&self) -> WriteGuard<'_, T> {
        match self {
            RwLock::Sync(inner) => {
                cold_path();
                WriteGuard::Sync(inner.write())
            }
            RwLock::NoSync(inner) => WriteGuard::NoSync(inner.borrow_mut()),
        }
    }

    #[inline(always)]
    pub fn try_write(&self) -> Result<WriteGuard<'_, T>, WriteError> {
        match self {
            RwLock::Sync(inner) => {
                cold_path();
                Ok(WriteGuard::Sync(inner.try_write().ok_or(WriteError)?))
            }
            RwLock::NoSync(inner) => {
                Ok(WriteGuard::NoSync(inner.try_borrow_mut().map_err(|_| WriteError)?))
            }
        }
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow(&self) -> ReadGuard<'_, T> {
        self.read()
    }

    #[inline(always)]
    pub fn try_borrow(&self) -> Result<ReadGuard<'_, T>, ReadError> {
        self.try_read()
    }

    #[inline(always)]
    #[track_caller]
    pub fn borrow_mut(&self) -> WriteGuard<'_, T> {
        self.write()
    }

    #[inline(always)]
    pub fn try_borrow_mut(&self) -> Result<WriteGuard<'_, T>, WriteError> {
        self.try_write()
    }
}

impl<T: Default> Default for RwLock<T> {
    #[inline(always)]
    fn default() -> Self {
        RwLock::<T>::new(Default::default())
    }
}

impl<'a, T> ReadGuard<'a, T> {
    #[inline(always)]
    pub fn map<U, F>(s: Self, f: F) -> MappedReadGuard<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        match s {
            ReadGuard::Sync(guard) => {
                cold_path();
                MappedReadGuard::Sync(parking_lot::RwLockReadGuard::map(guard, f))
            }
            ReadGuard::NoSync(guard) => MappedReadGuard::NoSync(Ref::map(guard, f)),
        }
    }
}

impl<'a, T> WriteGuard<'a, T> {
    #[inline(always)]
    pub fn map<U, F>(s: Self, f: F) -> MappedWriteGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        match s {
            WriteGuard::Sync(guard) => {
                cold_path();
                MappedWriteGuard::Sync(parking_lot::RwLockWriteGuard::map(guard, f))
            }
            WriteGuard::NoSync(guard) => MappedWriteGuard::NoSync(RefMut::map(guard, f)),
        }
    }
}

impl<'a, T> Deref for ReadGuard<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        match self {
            ReadGuard::Sync(guard) => {
                cold_path();
                Deref::deref(guard)
            }
            ReadGuard::NoSync(guard) => Deref::deref(guard),
        }
    }
}

impl<'a, T> Deref for WriteGuard<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        match self {
            WriteGuard::Sync(guard) => {
                cold_path();
                Deref::deref(guard)
            }
            WriteGuard::NoSync(guard) => Deref::deref(guard),
        }
    }
}

impl<'a, T> DerefMut for WriteGuard<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        match self {
            WriteGuard::Sync(guard) => {
                cold_path();
                DerefMut::deref_mut(guard)
            }
            WriteGuard::NoSync(guard) => DerefMut::deref_mut(guard),
        }
    }
}

impl<'a, T> MappedReadGuard<'a, T> {
    #[inline(always)]
    pub fn map<U, F>(s: Self, f: F) -> MappedReadGuard<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        match s {
            MappedReadGuard::Sync(guard) => {
                cold_path();
                MappedReadGuard::Sync(parking_lot::MappedRwLockReadGuard::map(guard, f))
            }
            MappedReadGuard::NoSync(guard) => MappedReadGuard::NoSync(Ref::map(guard, f)),
        }
    }
}

impl<'a, T> MappedWriteGuard<'a, T> {
    #[inline(always)]
    pub fn map<U, F>(s: Self, f: F) -> MappedWriteGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        match s {
            MappedWriteGuard::Sync(guard) => {
                cold_path();
                MappedWriteGuard::Sync(parking_lot::MappedRwLockWriteGuard::map(guard, f))
            }
            MappedWriteGuard::NoSync(guard) => MappedWriteGuard::NoSync(RefMut::map(guard, f)),
        }
    }
}

impl<'a, T> Deref for MappedReadGuard<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        match self {
            MappedReadGuard::Sync(guard) => {
                cold_path();
                Deref::deref(guard)
            }
            MappedReadGuard::NoSync(guard) => Deref::deref(guard),
        }
    }
}

impl<'a, T> Deref for MappedWriteGuard<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        match self {
            MappedWriteGuard::Sync(guard) => {
                cold_path();
                Deref::deref(guard)
            }
            MappedWriteGuard::NoSync(guard) => Deref::deref(guard),
        }
    }
}

impl<'a, T> DerefMut for MappedWriteGuard<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        match self {
            MappedWriteGuard::Sync(guard) => {
                cold_path();
                DerefMut::deref_mut(guard)
            }
            MappedWriteGuard::NoSync(guard) => DerefMut::deref_mut(guard),
        }
    }
}
