// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

use core::sync::atomic::{self, AtomicUsize, Ordering};

use unix::thread_local as imp;
use thread_local as sys;

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
pub struct StaticOsKey {
    /// Inner static TLS key (internals).
    key: AtomicUsize,
    /// Destructor for the TLS value.
    ///
    /// See `OsKey::new` for information about when the destructor runs and how
    /// it runs.
    dtor: Option<unsafe extern fn(*mut u8)>,
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
pub struct OsKey {
    key: imp::Key,
}

impl sys::StaticOsKey for StaticOsKey {
    /// Gets the value associated with this TLS key
    ///
    /// This will lazily allocate a TLS key from the OS if one has not already
    /// been allocated.
    #[inline]
    unsafe fn get(&self) -> *mut u8 { imp::get(self.key()) }

    /// Sets this TLS key to a new value.
    ///
    /// This will lazily allocate a TLS key from the OS if one has not already
    /// been allocated.
    #[inline]
    unsafe fn set(&self, val: *mut u8) { imp::set(self.key(), val) }

    /// Deallocates this OS TLS key.
    ///
    /// This function is unsafe as there is no guarantee that the key is not
    /// currently in use by other threads or will not ever be used again.
    ///
    /// Note that this does *not* run the user-provided destructor if one was
    /// specified at definition time. Doing so must be done manually.
    unsafe fn destroy(&self) {
        match self.key.swap(0, Ordering::SeqCst) {
            0 => {}
            n => { imp::destroy(n as imp::Key) }
        }
    }
}

impl StaticOsKey {
    pub const fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> StaticOsKey {
        StaticOsKey {
            key: atomic::AtomicUsize::new(0),
            dtor: dtor
        }
    }

    #[inline]
    unsafe fn key(&self) -> imp::Key {
        match self.key.load(Ordering::Relaxed) {
            0 => self.lazy_init() as imp::Key,
            n => n as imp::Key
        }
    }

    unsafe fn lazy_init(&self) -> usize {
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
        assert!(key != 0);
        match self.key.compare_and_swap(0, key as usize, Ordering::SeqCst) {
            // The CAS succeeded, so we've created the actual key
            0 => key as usize,
            // If someone beat us to the punch, use their key instead
            n => { imp::destroy(key); n }
        }
    }
}

impl sys::OsKey for OsKey {
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
    fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> OsKey {
        OsKey { key: unsafe { imp::create(dtor) } }
    }

    /// See StaticOsKey::get
    #[inline]
    fn get(&self) -> *mut u8 {
        unsafe { imp::get(self.key) }
    }

    /// See StaticOsKey::set
    #[inline]
    fn set(&self, val: *mut u8) {
        unsafe { imp::set(self.key, val) }
    }
}

impl Drop for OsKey {
    fn drop(&mut self) {
        unsafe { imp::destroy(self.key) }
    }
}

#[cfg(all(any(target_os = "macos", target_os = "linux"),
          not(target_arch = "aarch64"),
          not(no_elf_tls)))]
#[doc(hidden)]
pub mod key_imp {
    use thread_local::{self as sys, StaticOsKey};
    use alloc::boxed::Box;
    use collections::Vec;
    use core::cell::{Cell, UnsafeCell};
    use core::marker;
    use core::intrinsics;
    use core::ptr;

    pub struct Key<T> {
        inner: UnsafeCell<Option<T>>,

        // Metadata to keep track of the state of the destructor. Remember that
        // these variables are thread-local, not global.
        dtor_registered: Cell<bool>,
        dtor_running: Cell<bool>,
    }

    unsafe impl<T> marker::Sync for Key<T> { }

    impl<T> Key<T> {
        pub const fn new() -> Key<T> {
            Key {
                inner: UnsafeCell::new(None),
                dtor_registered: Cell::new(false),
                dtor_running: Cell::new(false)
            }
        }

        unsafe fn register_dtor(&self) {
            if !intrinsics::needs_drop::<T>() || self.dtor_registered.get() {
                return
            }

            register_dtor(self as *const _ as *mut u8,
                          destroy_value::<T>);
            self.dtor_registered.set(true);
        }
    }

    impl<T> sys::Key<T> for Key<T> {
        unsafe fn get(&'static self) -> Option<&'static UnsafeCell<Option<T>>> {
            if intrinsics::needs_drop::<T>() && self.dtor_running.get() {
                return None
            }
            self.register_dtor();
            Some(&self.inner)
        }
    }

    // Since what appears to be glibc 2.18 this symbol has been shipped which
    // GCC and clang both use to invoke destructors in thread_local globals, so
    // let's do the same!
    //
    // Note, however, that we run on lots older linuxes, as well as cross
    // compiling from a newer linux to an older linux, so we also have a
    // fallback implementation to use as well.
    //
    // Due to rust-lang/rust#18804, make sure this is not generic!
    #[cfg(target_os = "linux")]
    unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
        use core::mem;
        use libc;

        extern {
            #[linkage = "extern_weak"]
            static __dso_handle: *mut u8;
            #[linkage = "extern_weak"]
            static __cxa_thread_atexit_impl: *const libc::c_void;
        }
        if !__cxa_thread_atexit_impl.is_null() {
            type F = unsafe extern fn(dtor: unsafe extern fn(*mut u8),
                                      arg: *mut u8,
                                      dso_handle: *mut u8) -> libc::c_int;
            mem::transmute::<*const libc::c_void, F>(__cxa_thread_atexit_impl)
            (dtor, t, &__dso_handle as *const _ as *mut _);
            return
        }

        // The fallback implementation uses a vanilla OS-based TLS key to track
        // the list of destructors that need to be run for this thread. The key
        // then has its own destructor which runs all the other destructors.
        //
        // The destructor for DTORS is a little special in that it has a `while`
        // loop to continuously drain the list of registered destructors. It
        // *should* be the case that this loop always terminates because we
        // provide the guarantee that a TLS key cannot be set after it is
        // flagged for destruction.
        static DTORS: super::StaticOsKey = super::StaticOsKey::new(Some(run_dtors));
        type List = Vec<(*mut u8, unsafe extern fn(*mut u8))>;
        if DTORS.get().is_null() {
            let v: Box<List> = Box::new(Vec::new());
            DTORS.set(Box::into_raw(v) as *mut u8);
        }
        let list: &mut List = &mut *(DTORS.get() as *mut List);
        list.push((t, dtor));

        unsafe extern fn run_dtors(mut ptr: *mut u8) {
            while !ptr.is_null() {
                let list: Box<List> = Box::from_raw(ptr as *mut List);
                for &(ptr, dtor) in list.iter() {
                    dtor(ptr);
                }
                ptr = DTORS.get();
                DTORS.set(ptr::null_mut());
            }
        }
    }

    // OSX's analog of the above linux function is this _tlv_atexit function.
    // The disassembly of thread_local globals in C++ (at least produced by
    // clang) will have this show up in the output.
    #[cfg(target_os = "macos")]
    unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
        extern {
            fn _tlv_atexit(dtor: unsafe extern fn(*mut u8),
                           arg: *mut u8);
        }
        _tlv_atexit(dtor, t);
    }

    pub unsafe extern fn destroy_value<T>(ptr: *mut u8) {
        let ptr = ptr as *mut Key<T>;
        // Right before we run the user destructor be sure to flag the
        // destructor as running for this thread so calls to `get` will return
        // `None`.
        (*ptr).dtor_running.set(true);

        // The OSX implementation of TLS apparently had an odd aspect to it
        // where the pointer we have may be overwritten while this destructor
        // is running. Specifically if a TLS destructor re-accesses TLS it may
        // trigger a re-initialization of all TLS variables, paving over at
        // least some destroyed ones with initial values.
        //
        // This means that if we drop a TLS value in place on OSX that we could
        // revert the value to its original state halfway through the
        // destructor, which would be bad!
        //
        // Hence, we use `ptr::read` on OSX (to move to a "safe" location)
        // instead of drop_in_place.
        if cfg!(target_os = "macos") {
            ptr::read((*ptr).inner.get());
        } else {
            intrinsics::drop_in_place((*ptr).inner.get());
        }
    }
}

#[cfg(any(not(any(target_os = "macos", target_os = "linux")),
          target_arch = "aarch64",
          no_elf_tls))]
#[doc(hidden)]
pub mod key_imp {
    pub use thread_local::os::Key;
}

pub use self::key_imp::Key;

#[cfg(test)]
mod tests {
    use prelude::v1::*;
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
