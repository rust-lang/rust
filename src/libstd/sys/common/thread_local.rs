// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

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
//! ```rust,ignore
//! let key = OsKey::new(None);
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
//! ```rust,ignore
//! static KEY: StaticOsKey = INIT;
//!
//! unsafe {
//!     assert!(KEY.get().is_null());
//!     KEY.set(1 as *mut u8);
//! }
//! ```

use sync::atomic::{self, AtomicUsize, Ordering};
use core::nonzero::NonZero;
use cell::{Cell, UnsafeCell};
use boxed::Box;
use marker;
use ptr;

pub trait OsKeyImp: Sized {
    unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Self;
    unsafe fn get(&self) -> *mut u8;
    unsafe fn set(&self, value: *mut u8);
    unsafe fn destroy(&self);

    unsafe fn from_usize(value: usize) -> Self;
    unsafe fn into_usize(&self) -> NonZero<usize>;
}

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
/// ```ignore
/// use sys::os::{StaticOsKey, INIT};
///
/// static KEY: StaticOsKey = INIT;
///
/// unsafe {
///     assert!(KEY.get().is_null());
///     KEY.set(1 as *mut u8);
/// }
/// ```
pub struct StaticOsKey<K> {
    /// Inner static TLS key (internals).
    key: AtomicUsize,
    /// Destructor for the TLS value.
    ///
    /// See `OsKey::new` for information about when the destructor runs and how
    /// it runs.
    dtor: Option<unsafe extern fn(*mut u8)>,
    data: marker::PhantomData<K>,
}

/// A type for a safely managed OS-based TLS slot.
///
/// This type allocates an OS TLS key when it is initialized and will deallocate
/// the key when it falls out of scope. When compared with `StaticOsKey`, this
/// type is entirely safe to use.
///
/// Implementations will likely, however, contain unsafe code as this type only
/// operates on `*mut u8`, a raw pointer.
///
/// # Examples
///
/// ```rust,ignore
/// use sys::os::OsKey;
///
/// let key = OsKey::new(None);
/// assert!(key.get().is_null());
/// key.set(1 as *mut u8);
/// assert!(!key.get().is_null());
///
/// drop(key); // deallocate this TLS slot.
/// ```
pub struct OsKey<K: OsKeyImp> {
    key: K,
}

impl<K: OsKeyImp> StaticOsKey<K> {
    /// Gets the value associated with this TLS key
    ///
    /// This will lazily allocate a TLS key from the OS if one has not already
    /// been allocated.
    #[inline]
    pub unsafe fn get(&self) -> *mut u8 { self.key().get() }

    /// Sets this TLS key to a new value.
    ///
    /// This will lazily allocate a TLS key from the OS if one has not already
    /// been allocated.
    #[inline]
    pub unsafe fn set(&self, val: *mut u8) { self.key().set(val) }

    /// Deallocates this OS TLS key.
    ///
    /// This function is unsafe as there is no guarantee that the key is not
    /// currently in use by other threads or will not ever be used again.
    ///
    /// Note that this does *not* run the user-provided destructor if one was
    /// specified at definition time. Doing so must be done manually.
    pub unsafe fn destroy(&self) {
        match self.key.swap(0, Ordering::SeqCst) {
            0 => {}
            n => { K::from_usize(n).destroy() }
        }
    }

    pub const fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> Self {
        StaticOsKey {
            key: atomic::AtomicUsize::new(0),
            data: marker::PhantomData,
            dtor: dtor
        }
    }

    #[inline]
    unsafe fn key(&self) -> K {
        match self.key.load(Ordering::Relaxed) {
            0 => self.lazy_init(),
            n => K::from_usize(n)
        }
    }

    unsafe fn lazy_init(&self) -> K {
        let key = K::create(self.dtor);
        match self.key.compare_and_swap(0, *key.into_usize(), Ordering::SeqCst) {
            // The CAS succeeded, so we've created the actual key
            0 => { key },
            // If someone beat us to the punch, use their key instead
            n => { key.destroy(); K::from_usize(n) }
        }
    }
}

impl<K: OsKeyImp> OsKey<K> {
    /// Creates a new managed OS TLS key.
    ///
    /// This key will be deallocated when the key falls out of scope.
    ///
    /// The argument provided is an optionally-specified destructor for the
    /// value of this TLS key. When a thread exits and the value for this key
    /// is non-null the destructor will be invoked. The TLS value will be reset
    /// to null before the destructor is invoked.
    ///
    /// Note that the destructor will not be run when the `OsKey` goes out of
    /// scope.
    #[inline]
    pub fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> Self {
        OsKey { key: unsafe { K::create(dtor) } }
    }

    /// See StaticOsKey::get
    #[inline]
    pub fn get(&self) -> *mut u8 {
        unsafe { self.key.get() }
    }

    /// See StaticOsKey::set
    #[inline]
    pub fn set(&self, val: *mut u8) {
        unsafe { self.key.set(val) }
    }
}

impl<K: OsKeyImp> Drop for OsKey<K> {
    fn drop(&mut self) {
        unsafe { self.key.destroy() }
    }
}

pub struct Key<T, K> {
    // OS-TLS key that we'll use to key off.
    os: StaticOsKey<K>,
    marker: marker::PhantomData<Cell<T>>,
}

unsafe impl<T, K: OsKeyImp> marker::Sync for Key<T, K> { }

struct Value<T: 'static, K: 'static> {
    key: &'static Key<T, K>,
    value: UnsafeCell<Option<T>>,
}

impl<T: 'static, K: OsKeyImp + 'static> Key<T, K> {
    pub const fn new() -> Self {
        Key {
            os: StaticOsKey::new(Some(destroy_value::<T, K>)),
            marker: marker::PhantomData
        }
    }

    pub unsafe fn get(&'static self) -> Option<&'static UnsafeCell<Option<T>>> {
        let ptr = self.os.get() as *mut Value<T, K>;
        if !ptr.is_null() {
            if ptr as usize == 1 {
                return None
            }
            return Some(&(*ptr).value);
        }

        // If the lookup returned null, we haven't initialized our own local
        // copy, so do that now.
        let ptr: Box<Value<T, K>> = Box::new(Value {
            key: self,
            value: UnsafeCell::new(None),
        });
        let ptr = Box::into_raw(ptr);
        self.os.set(ptr as *mut u8);
        Some(&(*ptr).value)
    }
}

unsafe extern fn destroy_value<T: 'static, K: OsKeyImp + 'static>(ptr: *mut u8) {
    // The OS TLS ensures that this key contains a NULL value when this
    // destructor starts to run. We set it back to a sentinel value of 1 to
    // ensure that any future calls to `get` for this thread will return
    // `None`.
    //
    // Note that to prevent an infinite loop we reset it back to null right
    // before we return from the destructor ourselves.
    let ptr = Box::from_raw(ptr as *mut Value<T, K>);
    let key = ptr.key;
    key.os.set(1 as *mut u8);
    drop(ptr);
    key.os.set(ptr::null_mut());
}

#[cfg(test)]
mod tests {
    use super::{Key, StaticOsKey};

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
        static K1: StaticOsKey = StaticOsKey::new(None);
        static K2: StaticOsKey = StaticOsKey::new(None);

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
