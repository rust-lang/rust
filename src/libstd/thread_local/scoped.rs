// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Scoped thread-local storage
//!
//! This module provides the ability to generate *scoped* thread-local
//! variables. In this sense, scoped indicates that thread local storage
//! actually stores a reference to a value, and this reference is only placed
//! in storage for a scoped amount of time.
//!
//! There are no restrictions on what types can be placed into a scoped
//! variable, but all scoped variables are initialized to the equivalent of
//! null. Scoped thread local stor is useful when a value is present for a known
//! period of time and it is not required to relinquish ownership of the
//! contents.
//!
//! # Example
//!
//! ```
//! scoped_thread_local!(static FOO: uint)
//!
//! // Initially each scoped slot is empty.
//! assert!(!FOO.is_set());
//!
//! // When inserting a value, the value is only in place for the duration
//! // of the closure specified.
//! FOO.set(&1, || {
//!     FOO.with(|slot| {
//!         assert_eq!(*slot, 1);
//!     });
//! });
//! ```

#![macro_escape]

use prelude::*;

// macro hygiene sure would be nice, wouldn't it?
#[doc(hidden)] pub use self::imp::KeyInner;
#[doc(hidden)] pub use sys_common::thread_local::INIT as OS_INIT;

/// Type representing a thread local storage key corresponding to a reference
/// to the type parameter `T`.
///
/// Keys are statically allocated and can contain a reference to an instance of
/// type `T` scoped to a particular lifetime. Keys provides two methods, `set`
/// and `with`, both of which currently use closures to control the scope of
/// their contents.
pub struct Key<T> { #[doc(hidden)] pub inner: KeyInner<T> }

/// Declare a new scoped thread local storage key.
///
/// This macro declares a `static` item on which methods are used to get and
/// set the value stored within.
#[macro_export]
macro_rules! scoped_thread_local(
    (static $name:ident: $t:ty) => (
        __scoped_thread_local_inner!(static $name: $t)
    );
    (pub static $name:ident: $t:ty) => (
        __scoped_thread_local_inner!(pub static $name: $t)
    );
)

#[macro_export]
#[doc(hidden)]
macro_rules! __scoped_thread_local_inner(
    (static $name:ident: $t:ty) => (
        #[cfg_attr(not(any(windows, target_os = "android", target_os = "ios")),
                   thread_local)]
        static $name: ::std::thread_local::scoped::Key<$t> =
            __scoped_thread_local_inner!($t);
    );
    (pub static $name:ident: $t:ty) => (
        #[cfg_attr(not(any(windows, target_os = "android", target_os = "ios")),
                   thread_local)]
        pub static $name: ::std::thread_local::scoped::Key<$t> =
            __scoped_thread_local_inner!($t);
    );
    ($t:ty) => ({
        use std::thread_local::scoped::Key as __Key;

        #[cfg(not(any(windows, target_os = "android", target_os = "ios")))]
        const INIT: __Key<$t> = __Key {
            inner: ::std::thread_local::scoped::KeyInner {
                inner: ::std::cell::UnsafeCell { value: 0 as *mut _ },
            }
        };

        #[cfg(any(windows, target_os = "android", target_os = "ios"))]
        const INIT: __Key<$t> = __Key {
            inner: ::std::thread_local::scoped::KeyInner {
                inner: ::std::thread_local::scoped::OS_INIT,
                marker: ::std::kinds::marker::InvariantType,
            }
        };

        INIT
    })
)

impl<T> Key<T> {
    /// Insert a value into this scoped thread local storage slot for a
    /// duration of a closure.
    ///
    /// While `cb` is running, the value `t` will be returned by `get` unless
    /// this function is called recursively inside of `cb`.
    ///
    /// Upon return, this function will restore the previous value, if any
    /// was available.
    ///
    /// # Example
    ///
    /// ```
    /// scoped_thread_local!(static FOO: uint)
    ///
    /// FOO.set(&100, || {
    ///     let val = FOO.with(|v| *v);
    ///     assert_eq!(val, 100);
    ///
    ///     // set can be called recursively
    ///     FOO.set(&101, || {
    ///         // ...
    ///     });
    ///
    ///     // Recursive calls restore the previous value.
    ///     let val = FOO.with(|v| *v);
    ///     assert_eq!(val, 100);
    /// });
    /// ```
    pub fn set<R>(&'static self, t: &T, cb: || -> R) -> R {
        struct Reset<'a, T: 'a> {
            key: &'a KeyInner<T>,
            val: *mut T,
        }
        #[unsafe_destructor]
        impl<'a, T> Drop for Reset<'a, T> {
            fn drop(&mut self) {
                unsafe { self.key.set(self.val) }
            }
        }

        let prev = unsafe {
            let prev = self.inner.get();
            self.inner.set(t as *const T as *mut T);
            prev
        };

        let _reset = Reset { key: &self.inner, val: prev };
        cb()
    }

    /// Get a value out of this scoped variable.
    ///
    /// This function takes a closure which receives the value of this
    /// variable.
    ///
    /// # Panics
    ///
    /// This function will panic if `set` has not previously been called.
    ///
    /// # Example
    ///
    /// ```no_run
    /// scoped_thread_local!(static FOO: uint)
    ///
    /// FOO.with(|slot| {
    ///     // work with `slot`
    /// });
    /// ```
    pub fn with<R>(&'static self, cb: |&T| -> R) -> R {
        unsafe {
            let ptr = self.inner.get();
            assert!(!ptr.is_null(), "cannot access a scoped thread local \
                                     variable without calling `set` first");
            cb(&*ptr)
        }
    }

    /// Test whether this TLS key has been `set` for the current thread.
    pub fn is_set(&'static self) -> bool {
        unsafe { !self.inner.get().is_null() }
    }
}

#[cfg(not(any(windows, target_os = "android", target_os = "ios")))]
mod imp {
    use std::cell::UnsafeCell;

    // FIXME: Should be a `Cell`, but that's not `Sync`
    #[doc(hidden)]
    pub struct KeyInner<T> { pub inner: UnsafeCell<*mut T> }

    #[doc(hidden)]
    impl<T> KeyInner<T> {
        #[doc(hidden)]
        pub unsafe fn set(&self, ptr: *mut T) { *self.inner.get() = ptr; }
        #[doc(hidden)]
        pub unsafe fn get(&self) -> *mut T { *self.inner.get() }
    }
}

#[cfg(any(windows, target_os = "android", target_os = "ios"))]
mod imp {
    use kinds::marker;
    use sys_common::thread_local::StaticKey as OsStaticKey;

    #[doc(hidden)]
    pub struct KeyInner<T> {
        pub inner: OsStaticKey,
        pub marker: marker::InvariantType<T>,
    }

    #[doc(hidden)]
    impl<T> KeyInner<T> {
        #[doc(hidden)]
        pub unsafe fn set(&self, ptr: *mut T) { self.inner.set(ptr as *mut _) }
        #[doc(hidden)]
        pub unsafe fn get(&self) -> *mut T { self.inner.get() as *mut _ }
    }
}


#[cfg(test)]
mod tests {
    use cell::Cell;
    use prelude::*;

    #[test]
    fn smoke() {
        scoped_thread_local!(static BAR: uint)

        assert!(!BAR.is_set());
        BAR.set(&1, || {
            assert!(BAR.is_set());
            BAR.with(|slot| {
                assert_eq!(*slot, 1);
            });
        });
        assert!(!BAR.is_set());
    }

    #[test]
    fn cell_allowed() {
        scoped_thread_local!(static BAR: Cell<uint>)

        BAR.set(&Cell::new(1), || {
            BAR.with(|slot| {
                assert_eq!(slot.get(), 1);
            });
        });
    }
}

