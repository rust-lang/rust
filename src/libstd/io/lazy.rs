use crate::cell::Cell;
use crate::ptr;
use crate::sync::{Arc, RawMutex};
use crate::sys_common;

pub struct Lazy<T> {
    lock: RawMutex,
    ptr: Cell<*mut Arc<T>>,
}

#[inline]
const fn done<T>() -> *mut Arc<T> { 1_usize as *mut _ }

unsafe impl<T> Sync for Lazy<T> {}

impl<T> Lazy<T> {
    pub const fn new() -> Lazy<T> {
        Lazy {
            lock: RawMutex::new(),
            ptr: Cell::new(ptr::null_mut()),
        }
    }
}

impl<T: Send + Sync + 'static> Lazy<T> {
    /// Warning: `init` must not call `get` on the variable that is being
    /// initialized. Doing so will cause a deadlock.
    pub fn get(&'static self, init: fn() -> Arc<T>) -> Option<Arc<T>> {
        let _guard = self.lock.lock();
        let ptr = self.ptr.get();
        if ptr.is_null() {
            Some(unsafe { self.init(init) })
        } else if ptr == done() {
            None
        } else {
            Some(unsafe { (*ptr).clone() })
        }
    }

    // Must only be called with `lock` held
    unsafe fn init(&'static self, init: fn() -> Arc<T>) -> Arc<T> {
        // If we successfully register an at exit handler, then we cache the
        // `Arc` allocation in our own internal box (it will get deallocated by
        // the at exit handler). Otherwise we just return the freshly allocated
        // `Arc`.
        let registered = sys_common::at_exit(move || {
            let ptr = {
                let _guard = self.lock.lock();
                self.ptr.replace(done())
            };
            drop(Box::from_raw(ptr))
        });
        // This could reentrantly call `init` again, which is a problem
        // because our `lock` allows reentrancy!
        // That's why `get` is unsafe and requires the caller to ensure no reentrancy happens.
        let ret = init();
        if registered.is_ok() {
            self.ptr.set(Box::into_raw(Box::new(ret.clone())));
        }
        ret
    }
}
