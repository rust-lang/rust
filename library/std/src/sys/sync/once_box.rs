//! A racily-initialized alternative to `OnceLock<Box<T>>`.
//!
//! This is used to implement synchronization primitives that need allocation,
//! like the pthread versions.

#![allow(dead_code)] // Only used on some platforms.

use crate::mem::replace;
use crate::ptr::null_mut;
use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};

pub(crate) struct OnceBox<T> {
    ptr: AtomicPtr<T>,
}

impl<T> OnceBox<T> {
    #[inline]
    pub const fn new() -> Self {
        Self { ptr: AtomicPtr::new(null_mut()) }
    }

    /// Gets access to the value, assuming it is already initialized and this
    /// initialization has been observed by the current thread.
    ///
    /// Since all modifications to the pointer have already been observed, the
    /// pointer load in this function can be performed with relaxed ordering,
    /// potentially allowing the optimizer to turn code like this:
    /// ```rust, ignore
    /// once_box.get_or_init(|| Box::new(42));
    /// unsafe { once_box.get_unchecked() }
    /// ```
    /// into
    /// ```rust, ignore
    /// once_box.get_or_init(|| Box::new(42))
    /// ```
    ///
    /// # Safety
    /// This causes undefined behavior if the assumption above is violated.
    #[inline]
    pub unsafe fn get_unchecked(&self) -> &T {
        unsafe { &*self.ptr.load(Relaxed) }
    }

    #[inline]
    pub fn get_or_init(&self, f: impl FnOnce() -> Box<T>) -> &T {
        let ptr = self.ptr.load(Acquire);
        match unsafe { ptr.as_ref() } {
            Some(val) => val,
            None => self.initialize(f),
        }
    }

    #[inline]
    pub fn take(&mut self) -> Option<Box<T>> {
        let ptr = replace(self.ptr.get_mut(), null_mut());
        if !ptr.is_null() { Some(unsafe { Box::from_raw(ptr) }) } else { None }
    }

    #[cold]
    fn initialize(&self, f: impl FnOnce() -> Box<T>) -> &T {
        let new_ptr = Box::into_raw(f());
        match self.ptr.compare_exchange(null_mut(), new_ptr, Release, Acquire) {
            Ok(_) => unsafe { &*new_ptr },
            Err(ptr) => {
                // Lost the race to another thread.
                // Drop the value we created, and use the one from the other thread instead.
                drop(unsafe { Box::from_raw(new_ptr) });
                unsafe { &*ptr }
            }
        }
    }
}

unsafe impl<T: Send> Send for OnceBox<T> {}
unsafe impl<T: Send + Sync> Sync for OnceBox<T> {}

impl<T> Drop for OnceBox<T> {
    fn drop(&mut self) {
        self.take();
    }
}
