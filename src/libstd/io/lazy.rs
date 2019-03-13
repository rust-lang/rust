use crate::cell::Cell;
use crate::ptr;
use crate::sync::Arc;
use crate::sys_common;
use crate::sys_common::mutex::Mutex;

pub struct Lazy<T> {
    // We never call `lock.init()`, so it is UB to attempt to acquire this mutex reentrantly!
    lock: Mutex,
    ptr: Cell<*mut Arc<T>>,
}

#[inline]
const fn done<T>() -> *mut Arc<T> { 1_usize as *mut _ }

unsafe impl<T> Sync for Lazy<T> {}

impl<T> Lazy<T> {
    pub const fn new() -> Lazy<T> {
        Lazy {
            lock: Mutex::new(),
            ptr: Cell::new(ptr::null_mut()),
        }
    }
}

impl<T: Send + Sync + 'static> Lazy<T> {
    /// Safety: `init` must not call `get` on the variable that is being
    /// initialized.
    pub unsafe fn get(&'static self, init: fn() -> Arc<T>) -> Option<Arc<T>> {
        let _guard = self.lock.lock();
        let ptr = self.ptr.get();
        if ptr.is_null() {
            Some(self.init(init))
        } else if ptr == done() {
            None
        } else {
            Some((*ptr).clone())
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
