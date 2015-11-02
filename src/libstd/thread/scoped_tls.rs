// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
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
//! null. Scoped thread local storage is useful when a value is present for a known
//! period of time and it is not required to relinquish ownership of the
//! contents.
//!
//! # Examples
//!
//! ```
//! #![feature(scoped_tls)]
//!
//! scoped_thread_local!(static FOO: u32);
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

#![unstable(feature = "thread_local_internals", issue = "0")]

#[doc(hidden)]
pub use self::imp::KeyInner as __KeyInner;

/// Type representing a thread local storage key corresponding to a reference
/// to the type parameter `T`.
///
/// Keys are statically allocated and can contain a reference to an instance of
/// type `T` scoped to a particular lifetime. Keys provides two methods, `set`
/// and `with`, both of which currently use closures to control the scope of
/// their contents.
#[unstable(feature = "scoped_tls",
           reason = "scoped TLS has yet to have wide enough use to fully consider \
                     stabilizing its interface",
           issue = "27715")]
pub struct ScopedKey<T:'static> { inner: fn() -> &'static imp::KeyInner<T> }

/// Declare a new scoped thread local storage key.
///
/// This macro declares a `static` item on which methods are used to get and
/// set the value stored within.
///
/// See [ScopedKey documentation](thread/struct.ScopedKey.html) for more
/// information.
#[macro_export]
#[allow_internal_unstable]
macro_rules! scoped_thread_local {
    (static $name:ident: $t:ty) => (
        static $name: $crate::thread::ScopedKey<$t> =
            __scoped_thread_local_inner!($t);
    );
    (pub static $name:ident: $t:ty) => (
        pub static $name: $crate::thread::ScopedKey<$t> =
            __scoped_thread_local_inner!($t);
    );
}

#[doc(hidden)]
#[unstable(feature = "thread_local_internals",
           reason = "should not be necessary")]
#[macro_export]
#[allow_internal_unstable]
macro_rules! __scoped_thread_local_inner {
    ($t:ty) => {{
        #[cfg_attr(not(any(windows, not(no_elf_tls),
                           target_os = "android",
                           target_os = "ios",
                           target_os = "netbsd",
                           target_os = "openbsd",
                           target_arch = "aarch64")),
                   thread_local)]
        static _KEY: $crate::thread::__ScopedKeyInner<$t> =
            $crate::thread::__ScopedKeyInner::new();
        fn _getit() -> &'static $crate::thread::__ScopedKeyInner<$t> { &_KEY }
        $crate::thread::ScopedKey::new(_getit)
    }}
}

#[unstable(feature = "scoped_tls",
           reason = "scoped TLS has yet to have wide enough use to fully consider \
                     stabilizing its interface",
           issue = "27715")]
impl<T> ScopedKey<T> {
    #[doc(hidden)]
    pub const fn new(inner: fn() -> &'static imp::KeyInner<T>) -> ScopedKey<T> {
        ScopedKey { inner: inner }
    }

    /// Inserts a value into this scoped thread local storage slot for a
    /// duration of a closure.
    ///
    /// While `cb` is running, the value `t` will be returned by `get` unless
    /// this function is called recursively inside of `cb`.
    ///
    /// Upon return, this function will restore the previous value, if any
    /// was available.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(scoped_tls)]
    ///
    /// scoped_thread_local!(static FOO: u32);
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
    pub fn set<R, F>(&'static self, t: &T, cb: F) -> R where
        F: FnOnce() -> R,
    {
        struct Reset<'a, T: 'a> {
            key: &'a imp::KeyInner<T>,
            val: *mut T,
        }
                impl<'a, T> Drop for Reset<'a, T> {
            fn drop(&mut self) {
                unsafe { self.key.set(self.val) }
            }
        }

        let inner = (self.inner)();
        let prev = unsafe {
            let prev = inner.get();
            inner.set(t as *const T as *mut T);
            prev
        };

        let _reset = Reset { key: inner, val: prev };
        cb()
    }

    /// Gets a value out of this scoped variable.
    ///
    /// This function takes a closure which receives the value of this
    /// variable.
    ///
    /// # Panics
    ///
    /// This function will panic if `set` has not previously been called.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(scoped_tls)]
    ///
    /// scoped_thread_local!(static FOO: u32);
    ///
    /// FOO.with(|slot| {
    ///     // work with `slot`
    /// });
    /// ```
    pub fn with<R, F>(&'static self, cb: F) -> R where
        F: FnOnce(&T) -> R
    {
        unsafe {
            let ptr = (self.inner)().get();
            assert!(!ptr.is_null(), "cannot access a scoped thread local \
                                     variable without calling `set` first");
            cb(&*ptr)
        }
    }

    /// Test whether this TLS key has been `set` for the current thread.
    pub fn is_set(&'static self) -> bool {
        unsafe { !(self.inner)().get().is_null() }
    }
}

#[cfg(not(any(windows,
              target_os = "android",
              target_os = "ios",
              target_os = "netbsd",
              target_os = "openbsd",
              target_arch = "aarch64",
              no_elf_tls)))]
#[doc(hidden)]
mod imp {
    use cell::Cell;
    use ptr;

    pub struct KeyInner<T> { inner: Cell<*mut T> }

    unsafe impl<T> ::marker::Sync for KeyInner<T> { }

    impl<T> KeyInner<T> {
        pub const fn new() -> KeyInner<T> {
            KeyInner { inner: Cell::new(ptr::null_mut()) }
        }
        pub unsafe fn set(&self, ptr: *mut T) { self.inner.set(ptr); }
        pub unsafe fn get(&self) -> *mut T { self.inner.get() }
    }
}

#[cfg(any(windows,
          target_os = "android",
          target_os = "ios",
          target_os = "netbsd",
          target_os = "openbsd",
          target_arch = "aarch64",
          no_elf_tls))]
#[doc(hidden)]
mod imp {
    use cell::Cell;
    use marker;
    use sys::thread_local::StaticOsKey;

    pub struct KeyInner<T> {
        pub inner: StaticOsKey,
        pub marker: marker::PhantomData<Cell<T>>,
    }

    unsafe impl<T> marker::Sync for KeyInner<T> { }

    impl<T> KeyInner<T> {
        pub const fn new() -> KeyInner<T> {
            KeyInner {
                inner: StaticOsKey::new(None),
                marker: marker::PhantomData
            }
        }
        pub unsafe fn set(&self, ptr: *mut T) { self.inner.set(ptr as *mut _) }
        pub unsafe fn get(&self) -> *mut T { self.inner.get() as *mut _ }
    }
}


#[cfg(test)]
mod tests {
    use cell::Cell;

    scoped_thread_local!(static FOO: u32);

    #[test]
    fn smoke() {
        scoped_thread_local!(static BAR: u32);

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
        scoped_thread_local!(static BAR: Cell<u32>);

        BAR.set(&Cell::new(1), || {
            BAR.with(|slot| {
                assert_eq!(slot.get(), 1);
            });
        });
    }

    #[test]
    fn scope_item_allowed() {
        assert!(!FOO.is_set());
        FOO.set(&1, || {
            assert!(FOO.is_set());
            FOO.with(|slot| {
                assert_eq!(*slot, 1);
            });
        });
        assert!(!FOO.is_set());
    }
}
