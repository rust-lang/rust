// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
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

#![stable]

use prelude::v1::*;

use cell::UnsafeCell;

#[macro_use]
pub mod scoped;

// Sure wish we had macro hygiene, no?
#[doc(hidden)]
pub mod __impl {
    pub use super::imp::Key as KeyInner;
    pub use super::imp::destroy_value;
    pub use sys_common::thread_local::INIT_INNER as OS_INIT_INNER;
    pub use sys_common::thread_local::StaticKey as OsStaticKey;
}

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
/// use std::thread::Thread;
///
/// thread_local!(static FOO: RefCell<uint> = RefCell::new(1));
///
/// FOO.with(|f| {
///     assert_eq!(*f.borrow(), 1);
///     *f.borrow_mut() = 2;
/// });
///
/// // each thread starts out with the initial value of 1
/// Thread::spawn(move|| {
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
#[stable]
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
    pub inner: fn() -> &'static __impl::KeyInner<UnsafeCell<Option<T>>>,

    // initialization routine to invoke to create a value
    #[doc(hidden)]
    pub init: fn() -> T,
}

/// Declare a new thread local storage key of type `std::thread_local::Key`.
#[macro_export]
#[stable]
macro_rules! thread_local {
    (static $name:ident: $t:ty = $init:expr) => (
        static $name: ::std::thread_local::Key<$t> = {
            use std::cell::UnsafeCell as __UnsafeCell;
            use std::thread_local::__impl::KeyInner as __KeyInner;
            use std::option::Option as __Option;
            use std::option::Option::None as __None;

            __thread_local_inner!(static __KEY: __UnsafeCell<__Option<$t>> = {
                __UnsafeCell { value: __None }
            });
            fn __init() -> $t { $init }
            fn __getit() -> &'static __KeyInner<__UnsafeCell<__Option<$t>>> {
                &__KEY
            }
            ::std::thread_local::Key { inner: __getit, init: __init }
        };
    );
    (pub static $name:ident: $t:ty = $init:expr) => (
        pub static $name: ::std::thread_local::Key<$t> = {
            use std::cell::UnsafeCell as __UnsafeCell;
            use std::thread_local::__impl::KeyInner as __KeyInner;
            use std::option::Option as __Option;
            use std::option::Option::None as __None;

            __thread_local_inner!(static __KEY: __UnsafeCell<__Option<$t>> = {
                __UnsafeCell { value: __None }
            });
            fn __init() -> $t { $init }
            fn __getit() -> &'static __KeyInner<__UnsafeCell<__Option<$t>>> {
                &__KEY
            }
            ::std::thread_local::Key { inner: __getit, init: __init }
        };
    );
}

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
#[doc(hidden)]
macro_rules! __thread_local_inner {
    (static $name:ident: $t:ty = $init:expr) => (
        #[cfg_attr(all(any(target_os = "macos", target_os = "linux"),
                       not(target_arch = "aarch64")),
                   thread_local)]
        static $name: ::std::thread_local::__impl::KeyInner<$t> =
            __thread_local_inner!($init, $t);
    );
    (pub static $name:ident: $t:ty = $init:expr) => (
        #[cfg_attr(all(any(target_os = "macos", target_os = "linux"),
                       not(target_arch = "aarch64")),
                   thread_local)]
        pub static $name: ::std::thread_local::__impl::KeyInner<$t> =
            __thread_local_inner!($init, $t);
    );
    ($init:expr, $t:ty) => ({
        #[cfg(all(any(target_os = "macos", target_os = "linux"), not(target_arch = "aarch64")))]
        const _INIT: ::std::thread_local::__impl::KeyInner<$t> = {
            ::std::thread_local::__impl::KeyInner {
                inner: ::std::cell::UnsafeCell { value: $init },
                dtor_registered: ::std::cell::UnsafeCell { value: false },
                dtor_running: ::std::cell::UnsafeCell { value: false },
            }
        };

        #[cfg(any(not(any(target_os = "macos", target_os = "linux")), target_arch = "aarch64"))]
        const _INIT: ::std::thread_local::__impl::KeyInner<$t> = {
            unsafe extern fn __destroy(ptr: *mut u8) {
                ::std::thread_local::__impl::destroy_value::<$t>(ptr);
            }

            ::std::thread_local::__impl::KeyInner {
                inner: ::std::cell::UnsafeCell { value: $init },
                os: ::std::thread_local::__impl::OsStaticKey {
                    inner: ::std::thread_local::__impl::OS_INIT_INNER,
                    dtor: ::std::option::Option::Some(__destroy as unsafe extern fn(*mut u8)),
                },
            }
        };

        _INIT
    });
}

/// Indicator of the state of a thread local storage key.
#[unstable = "state querying was recently added"]
#[derive(Eq, PartialEq, Copy)]
pub enum State {
    /// All keys are in this state whenever a thread starts. Keys will
    /// transition to the `Valid` state once the first call to `with` happens
    /// and the initialization expression succeeds.
    ///
    /// Keys in the `Uninitialized` state will yield a reference to the closure
    /// passed to `with` so long as the initialization routine does not panic.
    Uninitialized,

    /// Once a key has been accessed successfully, it will enter the `Valid`
    /// state. Keys in the `Valid` state will remain so until the thread exits,
    /// at which point the destructor will be run and the key will enter the
    /// `Destroyed` state.
    ///
    /// Keys in the `Valid` state will be guaranteed to yield a reference to the
    /// closure passed to `with`.
    Valid,

    /// When a thread exits, the destructors for keys will be run (if
    /// necessary). While a destructor is running, and possibly after a
    /// destructor has run, a key is in the `Destroyed` state.
    ///
    /// Keys in the `Destroyed` states will trigger a panic when accessed via
    /// `with`.
    Destroyed,
}

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
    #[stable]
    pub fn with<F, R>(&'static self, f: F) -> R
                      where F: FnOnce(&T) -> R {
        let slot = (self.inner)();
        unsafe {
            let slot = slot.get().expect("cannot access a TLS value during or \
                                          after it is destroyed");
            f(match *slot.get() {
                Some(ref inner) => inner,
                None => self.init(slot),
            })
        }
    }

    unsafe fn init(&self, slot: &UnsafeCell<Option<T>>) -> &T {
        // Execute the initialization up front, *then* move it into our slot,
        // just in case initialization fails.
        let value = (self.init)();
        let ptr = slot.get();
        *ptr = Some(value);
        (*ptr).as_ref().unwrap()
    }

    /// Query the current state of this key.
    ///
    /// A key is initially in the `Uninitialized` state whenever a thread
    /// starts. It will remain in this state up until the first call to `with`
    /// within a thread has run the initialization expression successfully.
    ///
    /// Once the initialization expression succeeds, the key transitions to the
    /// `Valid` state which will guarantee that future calls to `with` will
    /// succeed within the thread.
    ///
    /// When a thread exits, each key will be destroyed in turn, and as keys are
    /// destroyed they will enter the `Destroyed` state just before the
    /// destructor starts to run. Keys may remain in the `Destroyed` state after
    /// destruction has completed. Keys without destructors (e.g. with types
    /// that are `Copy`), may never enter the `Destroyed` state.
    ///
    /// Keys in the `Uninitialized` can be accessed so long as the
    /// initialization does not panic. Keys in the `Valid` state are guaranteed
    /// to be able to be accessed. Keys in the `Destroyed` state will panic on
    /// any call to `with`.
    #[unstable = "state querying was recently added"]
    pub fn state(&'static self) -> State {
        unsafe {
            match (self.inner)().get() {
                Some(cell) => {
                    match *cell.get() {
                        Some(..) => State::Valid,
                        None => State::Uninitialized,
                    }
                }
                None => State::Destroyed,
            }
        }
    }

    /// Deprecated
    #[deprecated = "function renamed to state() and returns more info"]
    pub fn destroyed(&'static self) -> bool { self.state() == State::Destroyed }
}

#[cfg(all(any(target_os = "macos", target_os = "linux"), not(target_arch = "aarch64")))]
mod imp {
    use prelude::v1::*;

    use cell::UnsafeCell;
    use intrinsics;
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
    }

    unsafe impl<T> ::marker::Sync for Key<T> { }

    #[doc(hidden)]
    impl<T> Key<T> {
        pub unsafe fn get(&'static self) -> Option<&'static T> {
            if intrinsics::needs_drop::<T>() && *self.dtor_running.get() {
                return None
            }
            self.register_dtor();
            Some(&*self.inner.get())
        }

        unsafe fn register_dtor(&self) {
            if !intrinsics::needs_drop::<T>() || *self.dtor_registered.get() {
                return
            }

            register_dtor(self as *const _ as *mut u8,
                          destroy_value::<T>);
            *self.dtor_registered.get() = true;
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
        use mem;
        use libc;
        use sys_common::thread_local as os;

        extern {
            static __dso_handle: *mut u8;
            #[linkage = "extern_weak"]
            static __cxa_thread_atexit_impl: *const ();
        }
        if !__cxa_thread_atexit_impl.is_null() {
            type F = unsafe extern fn(dtor: unsafe extern fn(*mut u8),
                                      arg: *mut u8,
                                      dso_handle: *mut u8) -> libc::c_int;
            mem::transmute::<*const (), F>(__cxa_thread_atexit_impl)
            (dtor, t, __dso_handle);
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
        static DTORS: os::StaticKey = os::StaticKey {
            inner: os::INIT_INNER,
            dtor: Some(run_dtors as unsafe extern "C" fn(*mut u8)),
        };
        type List = Vec<(*mut u8, unsafe extern fn(*mut u8))>;
        if DTORS.get().is_null() {
            let v: Box<List> = box Vec::new();
            DTORS.set(mem::transmute(v));
        }
        let list: &mut List = &mut *(DTORS.get() as *mut List);
        list.push((t, dtor));

        unsafe extern fn run_dtors(mut ptr: *mut u8) {
            while !ptr.is_null() {
                let list: Box<List> = mem::transmute(ptr);
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

    #[doc(hidden)]
    pub unsafe extern fn destroy_value<T>(ptr: *mut u8) {
        let ptr = ptr as *mut Key<T>;
        // Right before we run the user destructor be sure to flag the
        // destructor as running for this thread so calls to `get` will return
        // `None`.
        *(*ptr).dtor_running.get() = true;
        ptr::read((*ptr).inner.get());
    }
}

#[cfg(any(not(any(target_os = "macos", target_os = "linux")), target_arch = "aarch64"))]
mod imp {
    use prelude::v1::*;

    use cell::UnsafeCell;
    use mem;
    use ptr;
    use sys_common::thread_local::StaticKey as OsStaticKey;

    #[doc(hidden)]
    pub struct Key<T> {
        // Statically allocated initialization expression, using an `UnsafeCell`
        // for the same reasons as above.
        pub inner: UnsafeCell<T>,

        // OS-TLS key that we'll use to key off.
        pub os: OsStaticKey,
    }

    unsafe impl<T> ::marker::Sync for Key<T> { }

    struct Value<T: 'static> {
        key: &'static Key<T>,
        value: T,
    }

    #[doc(hidden)]
    impl<T> Key<T> {
        pub unsafe fn get(&'static self) -> Option<&'static T> {
            self.ptr().map(|p| &*p)
        }

        unsafe fn ptr(&'static self) -> Option<*mut T> {
            let ptr = self.os.get() as *mut Value<T>;
            if !ptr.is_null() {
                if ptr as uint == 1 {
                    return None
                }
                return Some(&mut (*ptr).value as *mut T);
            }

            // If the lookup returned null, we haven't initialized our own local
            // copy, so do that now.
            //
            // Also note that this transmute_copy should be ok because the value
            // `inner` is already validated to be a valid `static` value, so we
            // should be able to freely copy the bits.
            let ptr: Box<Value<T>> = box Value {
                key: self,
                value: mem::transmute_copy(&self.inner),
            };
            let ptr: *mut Value<T> = mem::transmute(ptr);
            self.os.set(ptr as *mut u8);
            Some(&mut (*ptr).value as *mut T)
        }
    }

    #[doc(hidden)]
    pub unsafe extern fn destroy_value<T: 'static>(ptr: *mut u8) {
        // The OS TLS ensures that this key contains a NULL value when this
        // destructor starts to run. We set it back to a sentinel value of 1 to
        // ensure that any future calls to `get` for this thread will return
        // `None`.
        //
        // Note that to prevent an infinite loop we reset it back to null right
        // before we return from the destructor ourselves.
        let ptr: Box<Value<T>> = mem::transmute(ptr);
        let key = ptr.key;
        key.os.set(1 as *mut u8);
        drop(ptr);
        key.os.set(ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use sync::mpsc::{channel, Sender};
    use cell::UnsafeCell;
    use super::State;
    use thread::Thread;

    struct Foo(Sender<()>);

    impl Drop for Foo {
        fn drop(&mut self) {
            let Foo(ref s) = *self;
            s.send(()).unwrap();
        }
    }

    #[test]
    fn smoke_no_dtor() {
        thread_local!(static FOO: UnsafeCell<int> = UnsafeCell { value: 1 });

        FOO.with(|f| unsafe {
            assert_eq!(*f.get(), 1);
            *f.get() = 2;
        });
        let (tx, rx) = channel();
        let _t = Thread::spawn(move|| {
            FOO.with(|f| unsafe {
                assert_eq!(*f.get(), 1);
            });
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();

        FOO.with(|f| unsafe {
            assert_eq!(*f.get(), 2);
        });
    }

    #[test]
    fn states() {
        struct Foo;
        impl Drop for Foo {
            fn drop(&mut self) {
                assert!(FOO.state() == State::Destroyed);
            }
        }
        fn foo() -> Foo {
            assert!(FOO.state() == State::Uninitialized);
            Foo
        }
        thread_local!(static FOO: Foo = foo());

        Thread::scoped(|| {
            assert!(FOO.state() == State::Uninitialized);
            FOO.with(|_| {
                assert!(FOO.state() == State::Valid);
            });
            assert!(FOO.state() == State::Valid);
        }).join().ok().unwrap();
    }

    #[test]
    fn smoke_dtor() {
        thread_local!(static FOO: UnsafeCell<Option<Foo>> = UnsafeCell {
            value: None
        });

        let (tx, rx) = channel();
        let _t = Thread::spawn(move|| unsafe {
            let mut tx = Some(tx);
            FOO.with(|f| {
                *f.get() = Some(Foo(tx.take().unwrap()));
            });
        });
        rx.recv().unwrap();
    }

    #[test]
    fn circular() {
        struct S1;
        struct S2;
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell {
            value: None
        });
        thread_local!(static K2: UnsafeCell<Option<S2>> = UnsafeCell {
            value: None
        });
        static mut HITS: uint = 0;

        impl Drop for S1 {
            fn drop(&mut self) {
                unsafe {
                    HITS += 1;
                    if K2.state() == State::Destroyed {
                        assert_eq!(HITS, 3);
                    } else {
                        if HITS == 1 {
                            K2.with(|s| *s.get() = Some(S2));
                        } else {
                            assert_eq!(HITS, 3);
                        }
                    }
                }
            }
        }
        impl Drop for S2 {
            fn drop(&mut self) {
                unsafe {
                    HITS += 1;
                    assert!(K1.state() != State::Destroyed);
                    assert_eq!(HITS, 2);
                    K1.with(|s| *s.get() = Some(S1));
                }
            }
        }

        Thread::scoped(move|| {
            drop(S1);
        }).join().ok().unwrap();
    }

    #[test]
    fn self_referential() {
        struct S1;
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell {
            value: None
        });

        impl Drop for S1 {
            fn drop(&mut self) {
                assert!(K1.state() == State::Destroyed);
            }
        }

        Thread::scoped(move|| unsafe {
            K1.with(|s| *s.get() = Some(S1));
        }).join().ok().unwrap();
    }

    #[test]
    fn dtors_in_dtors_in_dtors() {
        struct S1(Sender<()>);
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell {
            value: None
        });
        thread_local!(static K2: UnsafeCell<Option<Foo>> = UnsafeCell {
            value: None
        });

        impl Drop for S1 {
            fn drop(&mut self) {
                let S1(ref tx) = *self;
                unsafe {
                    if K2.state() != State::Destroyed {
                        K2.with(|s| *s.get() = Some(Foo(tx.clone())));
                    }
                }
            }
        }

        let (tx, rx) = channel();
        let _t = Thread::spawn(move|| unsafe {
            let mut tx = Some(tx);
            K1.with(|s| *s.get() = Some(S1(tx.take().unwrap())));
        });
        rx.recv().unwrap();
    }
}

#[cfg(test)]
mod dynamic_tests {
    use prelude::v1::*;

    use cell::RefCell;
    use collections::HashMap;

    #[test]
    fn smoke() {
        fn square(i: int) -> int { i * i }
        thread_local!(static FOO: int = square(3));

        FOO.with(|f| {
            assert_eq!(*f, 9);
        });
    }

    #[test]
    fn hashmap() {
        fn map() -> RefCell<HashMap<int, int>> {
            let mut m = HashMap::new();
            m.insert(1, 2);
            RefCell::new(m)
        }
        thread_local!(static FOO: RefCell<HashMap<int, int>> = map());

        FOO.with(|map| {
            assert_eq!(map.borrow()[1], 2);
        });
    }

    #[test]
    fn refcell_vec() {
        thread_local!(static FOO: RefCell<Vec<uint>> = RefCell::new(vec![1, 2, 3]));

        FOO.with(|vec| {
            assert_eq!(vec.borrow().len(), 3);
            vec.borrow_mut().push(4);
            assert_eq!(vec.borrow()[3], 4);
        });
    }
}
