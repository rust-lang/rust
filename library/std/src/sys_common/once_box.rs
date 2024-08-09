#![allow(dead_code)] // Only used on some platforms.

// This is used to wrap pthread {Mutex, Condvar, RwLock} in.

use crate::mem::replace;
use crate::pin::Pin;
use crate::ptr::null_mut;
use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::Ordering::{AcqRel, Acquire};

pub(crate) struct OnceBox<T> {
    ptr: AtomicPtr<T>,
}

impl<T> OnceBox<T> {
    #[inline]
    pub const fn new() -> Self {
        Self { ptr: AtomicPtr::new(null_mut()) }
    }

    #[inline]
    pub fn get(&self) -> Option<Pin<&T>> {
        let val = unsafe { self.ptr.load(Acquire).as_ref()? };
        Some(unsafe { Pin::new_unchecked(val) })
    }

    #[inline]
    pub fn get_or_init(&self, f: impl FnOnce() -> Pin<Box<T>>) -> Pin<&T> {
        match self.get() {
            Some(val) => val,
            None => self.init(f),
        }
    }

    #[cold]
    fn init(&self, f: impl FnOnce() -> Pin<Box<T>>) -> Pin<&T> {
        let val = f();
        let ptr = Box::into_raw(unsafe { Pin::into_inner_unchecked(val) });
        let val = match self.ptr.compare_exchange(null_mut(), ptr, AcqRel, Acquire) {
            Ok(_) => ptr,
            Err(new) => {
                drop(unsafe { Box::from_raw(ptr) });
                new
            }
        };

        unsafe { Pin::new_unchecked(&*val) }
    }

    #[inline]
    pub fn take(&mut self) -> Option<Pin<Box<T>>> {
        let ptr = replace(self.ptr.get_mut(), null_mut());
        if !ptr.is_null() { Some(unsafe { Box::from_raw(ptr).into() }) } else { None }
    }
}

impl<T> Drop for OnceBox<T> {
    fn drop(&mut self) {
        drop(self.take())
    }
}
