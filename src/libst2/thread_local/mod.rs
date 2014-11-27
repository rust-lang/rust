// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Thread local storage
//!
//! This module provides an implementation of thread local storage for Rust
//! programs. Thread local storage is a method of storing data into a global
//! variable which each thread in the program will have its own copy of.
//! Threads do not share this data, so accesses do not need to be synchronized.
//!
//! At a high level, this module provides two variants of storage:
//!
//! * Owning thread local storage. This is a type of thread local key which
//!   owns the value that it contains, and will destroy the value when the
//!   thread exits. This variant is created with the `thread_local!` macro and
//!   can contain any value which is `'static` (no borrowed pointers.
//!
//! * Scoped thread local storage. This type of key is used to store a reference
//!   to a value into local storage temporarily for the scope of a function
//!   call. There are no restrictions on what types of values can be placed
//!   into this key.
//!
//! Both forms of thread local storage provide an accessor function, `with`,
//! which will yield a shared reference to the value to the specified
//! closure. Thread local keys only allow shared access to values as there is no
//! way to guarantee uniqueness if a mutable borrow was allowed. Most values
//! will want to make use of some form of **interior mutability** through the
//! `Cell` or `RefCell` types.

#![macro_escape]
#![experimental]

use prelude::*;

use cell::UnsafeCell;

// Sure wish we had macro hygiene, no?
#[doc(hidden)] pub use self::imp::Key as KeyInner;
#[doc(hidden)] pub use self::imp::destroy_value;
#[doc(hidden)] pub use sys_common::thread_local::INIT_INNER as OS_INIT_INNER;
#[doc(hidden)] pub use sys_common::thread_local::StaticKey as OsStaticKey;

pub mod scoped;

/// A thread local storage key which owns its contents.
///
/// This key uses the fastest possible implementation available to it for the
/// target platform. It is instantiated with the `thread_local!` macro and the
/// primary method is the `with` method.
///
/// The `with` method yields a reference to the contained value which cannot be
/// sent across tasks or escape the given closure.
///
/// # Initialization and Destruction
///
/// Initialization is dynamically performed on the first call to `with()`
/// within a thread, and values support destructors which will be run when a
/// thread exits.
///
/// # Example
///
/// ```
/// use std::cell::RefCell;
///
/// thread_local!(static FOO: RefCell<uint> = RefCell::new(1));
///
/// FOO.with(|f| {
///     assert_eq!(*f.borrow(), 1);
///     *f.borrow_mut() = 2;
/// });
///
/// // each thread starts out with the initial value of 1
/// spawn(proc() {
///     FOO.with(|f| {
///         assert_eq!(*f.borrow(), 1);
///         *f.borrow_mut() = 3;
///     });
/// });
///
/// // we retain our original value of 2 despite the child thread
/// FOO.with(|f| {
///     assert_eq!(*f.borrow(), 2);
/// });
/// ```
pub struct Key<T> {
    // The key itself may be tagged with #[thread_local], and this `Key` is
    // stored as a `static`, and it's not valid for a static to reference the
    // address of another thread_local static. For this reason we kinda wonkily
    // work around this by generating a shim function which will give us the
    // address of the inner TLS key at runtime.
    //
    // This is trivially devirtualizable by LLVM because we never store anything
    // to this field and rustc can declare the `static` as constant as well.
    #[doc(hidden)]
    pub inner: fn() -> &'static KeyInner<UnsafeCell<Option<T>>>,

    // initialization routine to invoke to create a value
    #[doc(hidden)]
    pub init: fn() -> T,
}

/// Declare a new thread local storage key of type `std::thread_local::Key`.
#[macro_export]
#[doc(hidden)]
macro_rules! thread_local(
    (static $name:ident: $t:ty = $init:expr) => (
        static $name: ::std::thread_local::Key<$t> = {
            use std::cell::UnsafeCell as __UnsafeCell;
            use std::thread_local::KeyInner as __KeyInner;
            use std::option::Option as __Option;
            use std::option::None as __None;

            __thread_local_inner!(static __KEY: __UnsafeCell<__Option<$t>> = {
                __UnsafeCell { value: __None }
            })
            fn __init() -> $t { unimplemented!() }
            fn __getit() -> &'static __KeyInner<__UnsafeCell<__Option<$t>>> { unimplemented!() }
            ::std::thread_local::Key { inner: __getit, init: __init }
        };
    );
    (pub static $name:ident: $t:ty = $init:expr) => (
        pub static $name: ::std::thread_local::Key<$t> = {
            use std::cell::UnsafeCell as __UnsafeCell;
            use std::thread_local::KeyInner as __KeyInner;
            use std::option::Option as __Option;
            use std::option::None as __None;

            __thread_local_inner!(static __KEY: __UnsafeCell<__Option<$t>> = {
                __UnsafeCell { value: __None }
            })
            fn __init() -> $t { unimplemented!() }
            fn __getit() -> &'static __KeyInner<__UnsafeCell<__Option<$t>>> { unimplemented!() }
            ::std::thread_local::Key { inner: __getit, init: __init }
        };
    );
)

// Macro pain #4586:
//
// When cross compiling, rustc will load plugins and macros from the *host*
// platform before search for macros from the target platform. This is primarily
// done to detect, for example, plugins. Ideally the macro below would be
// defined once per module below, but unfortunately this means we have the
// following situation:
//
// 1. We compile libstd for x86_64-unknown-linux-gnu, this thread_local!() macro
//    will inject #[thread_local] statics.
// 2. We then try to compile a program for arm-linux-androideabi
// 3. The compiler has a host of linux and a target of android, so it loads
//    macros from the *linux* libstd.
// 4. The macro generates a #[thread_local] field, but the android libstd does
//    not use #[thread_local]
// 5. Compile error about structs with wrong fields.
//
// To get around this, we're forced to inject the #[cfg] logic into the macro
// itself. Woohoo.

#[macro_export]
macro_rules! __thread_local_inner(
    (static $name:ident: $t:ty = $init:expr) => (
        #[cfg_attr(any(target_os = "macos", target_os = "linux"), thread_local)]
        static $name: ::std::thread_local::KeyInner<$t> =
            __thread_local_inner!($init, $t);
    );
    (pub static $name:ident: $t:ty = $init:expr) => (
        #[cfg_attr(any(target_os = "macos", target_os = "linux"), thread_local)]
        pub static $name: ::std::thread_local::KeyInner<$t> =
            __thread_local_inner!($init, $t);
    );
    ($init:expr, $t:ty) => ({
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        const INIT: ::std::thread_local::KeyInner<$t> = {
            ::std::thread_local::KeyInner {
                inner: ::std::cell::UnsafeCell { value: $init },
                dtor_registered: ::std::cell::UnsafeCell { value: false },
                dtor_running: ::std::cell::UnsafeCell { value: false },
                marker: ::std::kinds::marker::NoCopy,
            }
        };

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        const INIT: ::std::thread_local::KeyInner<$t> = {
            unsafe extern fn __destroy(ptr: *mut u8) { unimplemented!() }
            ::std::thread_local::KeyInner {
                inner: ::std::cell::UnsafeCell { value: $init },
                os: ::std::thread_local::OsStaticKey {
                    inner: ::std::thread_local::OS_INIT_INNER,
                    dtor: ::std::option::Some(__destroy),
                },
            }
        };

        INIT
    });
)

impl<T: 'static> Key<T> {
    /// Acquire a reference to the value in this TLS key.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// This function will `panic!()` if the key currently has its
    /// destructor running, and it **may** panic if the destructor has
    /// previously been run for this thread.
    pub fn with<R>(&'static self, f: |&T| -> R) -> R { unimplemented!() }

    /// Test this TLS key to determine whether its value has been destroyed for
    /// the current thread or not.
    ///
    /// This will not initialize the key if it is not already initialized.
    pub fn destroyed(&'static self) -> bool { unimplemented!() }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod imp {
    use prelude::*;

    use cell::UnsafeCell;
    use intrinsics;
    use kinds::marker;
    use ptr;

    #[doc(hidden)]
    pub struct Key<T> {
        // Place the inner bits in an `UnsafeCell` to currently get around the
        // "only Sync statics" restriction. This allows any type to be placed in
        // the cell.
        //
        // Note that all access requires `T: 'static` so it can't be a type with
        // any borrowed pointers still.
        pub inner: UnsafeCell<T>,

        // Metadata to keep track of the state of the destructor. Remember that
        // these variables are thread-local, not global.
        pub dtor_registered: UnsafeCell<bool>, // should be Cell
        pub dtor_running: UnsafeCell<bool>, // should be Cell

        // These shouldn't be copied around.
        pub marker: marker::NoCopy,
    }

    #[doc(hidden)]
    impl<T> Key<T> {
        pub unsafe fn get(&'static self) -> Option<&'static T> { unimplemented!() }

        unsafe fn register_dtor(&self) { unimplemented!() }
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
    unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) { unimplemented!() }

    // OSX's analog of the above linux function is this _tlv_atexit function.
    // The disassembly of thread_local globals in C++ (at least produced by
    // clang) will have this show up in the output.
    #[cfg(target_os = "macos")]
    unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) { unimplemented!() }

    #[doc(hidden)]
    pub unsafe extern fn destroy_value<T>(ptr: *mut u8) { unimplemented!() }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod imp {
    use prelude::*;

    use cell::UnsafeCell;
    use mem;
    use sys_common::thread_local::StaticKey as OsStaticKey;

    #[doc(hidden)]
    pub struct Key<T> {
        // Statically allocated initialization expression, using an `UnsafeCell`
        // for the same reasons as above.
        pub inner: UnsafeCell<T>,

        // OS-TLS key that we'll use to key off.
        pub os: OsStaticKey,
    }

    struct Value<T: 'static> {
        key: &'static Key<T>,
        value: T,
    }

    #[doc(hidden)]
    impl<T> Key<T> {
        pub unsafe fn get(&'static self) -> Option<&'static T> { unimplemented!() }

        unsafe fn ptr(&'static self) -> Option<*mut T> { unimplemented!() }
    }

    #[doc(hidden)]
    pub unsafe extern fn destroy_value<T: 'static>(ptr: *mut u8) { unimplemented!() }
}
