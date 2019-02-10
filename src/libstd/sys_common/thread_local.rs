//! OS-based thread local storage
//!
//! This module provides an implementation of OS-based thread local storage,
//! using the native OS-provided facilities (think `TlsAlloc` or
//! `pthread_setspecific`). The interface of this differs from the other types
//! of thread-local-storage provided in this crate in that OS-based TLS can only
//! get/set pointers,
//!
//! This module also provides two flavors of TLS. One is intended for static
//! initialization, and does not contain a `Drop` implementation to deallocate
//! the OS-TLS key. The other is a type which does implement `Drop` and hence
//! has a safe interface.
//!
//! # Usage
//!
//! This module should likely not be used directly unless other primitives are
//! being built on. types such as `thread_local::spawn::Key` are likely much
//! more useful in practice than this OS-based version which likely requires
//! unsafe code to interoperate with.
//!
//! # Examples
//!
//! Using a dynamically allocated TLS key. Note that this key can be shared
//! among many threads via an `Arc`.
//!
//! ```ignore (cannot-doctest-private-modules)
//! let key = Key::new(None);
//! assert!(key.get().is_null());
//! key.set(1 as *mut u8);
//! assert!(!key.get().is_null());
//!
//! drop(key); // deallocate this TLS slot.
//! ```
//!
//! Sometimes a statically allocated key is either required or easier to work
//! with, however.
//!
//! ```ignore (cannot-doctest-private-modules)
//! static KEY: StaticKey = INIT;
//!
//! unsafe {
//!     assert!(KEY.get().is_null());
//!     KEY.set(1 as *mut u8);
//! }
//! ```

#![allow(non_camel_case_types)]
#![unstable(feature = "thread_local_internals", issue = "0")]
#![allow(dead_code)] // sys isn't exported yet

use crate::ptr;
use crate::sync::atomic::{self, AtomicUsize, Ordering};
use crate::sys::thread_local as imp;
use crate::sys_common::mutex::Mutex;

/// A type for TLS keys that are statically allocated.
///
/// This type is entirely `unsafe` to use as it does not protect against
/// use-after-deallocation or use-during-deallocation.
///
/// The actual OS-TLS key is lazily allocated when this is used for the first
/// time. The key is also deallocated when the Rust runtime exits or `destroy`
/// is called, whichever comes first.
///
/// # Examples
///
/// ```ignore (cannot-doctest-private-modules)
/// use tls::os::{StaticKey, INIT};
///
/// static KEY: StaticKey = INIT;
///
/// unsafe {
///     assert!(KEY.get().is_null());
///     KEY.set(1 as *mut u8);
/// }
/// ```
pub struct StaticKey {
    /// Inner static TLS key (internals).
    key: AtomicUsize,
    /// Destructor for the TLS value.
    ///
    /// See `Key::new` for information about when the destructor runs and how
    /// it runs.
    dtor: Option<unsafe extern fn(*mut u8)>,
}

/// A type for a safely managed OS-based TLS slot.
///
/// This type allocates an OS TLS key when it is initialized and will deallocate
/// the key when it falls out of scope. When compared with `StaticKey`, this
/// type is entirely safe to use.
///
/// Implementations will likely, however, contain unsafe code as this type only
/// operates on `*mut u8`, a raw pointer.
///
/// # Examples
///
/// ```ignore (cannot-doctest-private-modules)
/// use tls::os::Key;
///
/// let key = Key::new(None);
/// assert!(key.get().is_null());
/// key.set(1 as *mut u8);
/// assert!(!key.get().is_null());
///
/// drop(key); // deallocate this TLS slot.
/// ```
pub struct Key {
    key: imp::Key,
}

/// Constant initialization value for static TLS keys.
///
/// This value specifies no destructor by default.
pub const INIT: StaticKey = StaticKey::new(None);

impl StaticKey {
    pub const fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> StaticKey {
        StaticKey {
            key: atomic::AtomicUsize::new(0),
            dtor,
        }
    }

    /// Gets the value associated with this TLS key
    ///
    /// This will lazily allocate a TLS key from the OS if one has not already
    /// been allocated.
    #[inline]
    pub unsafe fn get(&self) -> *mut u8 { imp::get(self.key()) }

    /// Sets this TLS key to a new value.
    ///
    /// This will lazily allocate a TLS key from the OS if one has not already
    /// been allocated.
    #[inline]
    pub unsafe fn set(&self, val: *mut u8) { imp::set(self.key(), val) }

    #[inline]
    unsafe fn key(&self) -> imp::Key {
        match self.key.load(Ordering::Relaxed) {
            0 => self.lazy_init() as imp::Key,
            n => n as imp::Key
        }
    }

    unsafe fn lazy_init(&self) -> usize {
        // Currently the Windows implementation of TLS is pretty hairy, and
        // it greatly simplifies creation if we just synchronize everything.
        //
        // Additionally a 0-index of a tls key hasn't been seen on windows, so
        // we just simplify the whole branch.
        if imp::requires_synchronized_create() {
            // We never call `INIT_LOCK.init()`, so it is UB to attempt to
            // acquire this mutex reentrantly!
            static INIT_LOCK: Mutex = Mutex::new();
            let _guard = INIT_LOCK.lock();
            let mut key = self.key.load(Ordering::SeqCst);
            if key == 0 {
                key = imp::create(self.dtor) as usize;
                self.key.store(key, Ordering::SeqCst);
            }
            rtassert!(key != 0);
            return key
        }

        // POSIX allows the key created here to be 0, but the compare_and_swap
        // below relies on using 0 as a sentinel value to check who won the
        // race to set the shared TLS key. As far as I know, there is no
        // guaranteed value that cannot be returned as a posix_key_create key,
        // so there is no value we can initialize the inner key with to
        // prove that it has not yet been set. As such, we'll continue using a
        // value of 0, but with some gyrations to make sure we have a non-0
        // value returned from the creation routine.
        // FIXME: this is clearly a hack, and should be cleaned up.
        let key1 = imp::create(self.dtor);
        let key = if key1 != 0 {
            key1
        } else {
            let key2 = imp::create(self.dtor);
            imp::destroy(key1);
            key2
        };
        rtassert!(key != 0);
        match self.key.compare_and_swap(0, key as usize, Ordering::SeqCst) {
            // The CAS succeeded, so we've created the actual key
            0 => key as usize,
            // If someone beat us to the punch, use their key instead
            n => { imp::destroy(key); n }
        }
    }
}

impl Key {
    /// Creates a new managed OS TLS key.
    ///
    /// This key will be deallocated when the key falls out of scope.
    ///
    /// The argument provided is an optionally-specified destructor for the
    /// value of this TLS key. When a thread exits and the value for this key
    /// is non-null the destructor will be invoked. The TLS value will be reset
    /// to null before the destructor is invoked.
    ///
    /// Note that the destructor will not be run when the `Key` goes out of
    /// scope.
    #[inline]
    pub fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> Key {
        Key { key: unsafe { imp::create(dtor) } }
    }

    /// See StaticKey::get
    #[inline]
    pub fn get(&self) -> *mut u8 {
        unsafe { imp::get(self.key) }
    }

    /// See StaticKey::set
    #[inline]
    pub fn set(&self, val: *mut u8) {
        unsafe { imp::set(self.key, val) }
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        // Right now Windows doesn't support TLS key destruction, but this also
        // isn't used anywhere other than tests, so just leak the TLS key.
        // unsafe { imp::destroy(self.key) }
    }
}

pub unsafe fn register_dtor_fallback(t: *mut u8,
                                     dtor: unsafe extern fn(*mut u8)) {
    // The fallback implementation uses a vanilla OS-based TLS key to track
    // the list of destructors that need to be run for this thread. The key
    // then has its own destructor which runs all the other destructors.
    //
    // The destructor for DTORS is a little special in that it has a `while`
    // loop to continuously drain the list of registered destructors. It
    // *should* be the case that this loop always terminates because we
    // provide the guarantee that a TLS key cannot be set after it is
    // flagged for destruction.

    static DTORS: StaticKey = StaticKey::new(Some(run_dtors));
    type List = Vec<(*mut u8, unsafe extern fn(*mut u8))>;
    if DTORS.get().is_null() {
        let v: Box<List> = box Vec::new();
        DTORS.set(Box::into_raw(v) as *mut u8);
    }
    let list: &mut List = &mut *(DTORS.get() as *mut List);
    list.push((t, dtor));

    unsafe extern fn run_dtors(mut ptr: *mut u8) {
        while !ptr.is_null() {
            let list: Box<List> = Box::from_raw(ptr as *mut List);
            for (ptr, dtor) in list.into_iter() {
                dtor(ptr);
            }
            ptr = DTORS.get();
            DTORS.set(ptr::null_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Key, StaticKey};

    fn assert_sync<T: Sync>() {}
    fn assert_send<T: Send>() {}

    #[test]
    fn smoke() {
        assert_sync::<Key>();
        assert_send::<Key>();

        let k1 = Key::new(None);
        let k2 = Key::new(None);
        assert!(k1.get().is_null());
        assert!(k2.get().is_null());
        k1.set(1 as *mut _);
        k2.set(2 as *mut _);
        assert_eq!(k1.get() as usize, 1);
        assert_eq!(k2.get() as usize, 2);
    }

    #[test]
    fn statik() {
        static K1: StaticKey = StaticKey::new(None);
        static K2: StaticKey = StaticKey::new(None);

        unsafe {
            assert!(K1.get().is_null());
            assert!(K2.get().is_null());
            K1.set(1 as *mut _);
            K2.set(2 as *mut _);
            assert_eq!(K1.get() as usize, 1);
            assert_eq!(K2.get() as usize, 2);
        }
    }
}
