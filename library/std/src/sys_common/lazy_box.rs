#![allow(dead_code)] // Only used on some platforms.

// This is used to wrap pthread {Mutex, Condvar, RwLock} in.

use crate::marker::PhantomData;
use crate::ops::{Deref, DerefMut};
use crate::ptr::null_mut;
use crate::sync::atomic::{
    AtomicPtr,
    Ordering::{AcqRel, Acquire},
};

pub(crate) struct LazyBox<T: LazyInit> {
    ptr: AtomicPtr<T>,
    _phantom: PhantomData<T>,
}

pub(crate) trait LazyInit {
    /// This is called before the box is allocated, to provide the value to
    /// move into the new box.
    ///
    /// It might be called more than once per LazyBox, as multiple threads
    /// might race to initialize it concurrently, each constructing and initializing
    /// their own box. (All but one of them will be destroyed right after.)
    fn init() -> Box<Self>;
}

impl<T: LazyInit> LazyBox<T> {
    #[inline]
    pub const fn new() -> Self {
        Self { ptr: AtomicPtr::new(null_mut()), _phantom: PhantomData }
    }

    #[inline]
    fn get_pointer(&self) -> *mut T {
        let ptr = self.ptr.load(Acquire);
        if ptr.is_null() { self.initialize() } else { ptr }
    }

    #[cold]
    fn initialize(&self) -> *mut T {
        let new_ptr = Box::into_raw(T::init());
        match self.ptr.compare_exchange(null_mut(), new_ptr, AcqRel, Acquire) {
            Ok(_) => new_ptr,
            Err(ptr) => {
                // Lost the race to another thread.
                // Drop the box we created, and use the one from the other thread instead.
                drop(unsafe { Box::from_raw(new_ptr) });
                ptr
            }
        }
    }
}

impl<T: LazyInit> Deref for LazyBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.get_pointer() }
    }
}

impl<T: LazyInit> DerefMut for LazyBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.get_pointer() }
    }
}

impl<T: LazyInit> Drop for LazyBox<T> {
    fn drop(&mut self) {
        let ptr = *self.ptr.get_mut();
        if !ptr.is_null() {
            drop(unsafe { Box::from_raw(ptr) });
        }
    }
}
