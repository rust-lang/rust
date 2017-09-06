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

#![unstable(feature = "thread_local_internals", issue = "0")]

use fmt;
use cell::UnsafeCell;

/// A thread local storage key which owns its contents.
///
/// This key uses the fastest possible implementation available to it for the
/// target platform. It is instantiated with the [`thread_local!`] macro and the
/// primary method is the [`with`] method.
///
/// The [`with`] method yields a reference to the contained value which cannot be
/// sent across threads or escape the given closure.
///
/// # Initialization and Destruction
///
/// Initialization is dynamically performed on the first call to [`with`]
/// within a thread, and values that implement [`Drop`] get destructed when a
/// thread exits. Some caveats apply, which are explained below.
///
/// # Examples
///
/// ```
/// use std::cell::RefCell;
/// use std::thread;
///
/// thread_local!(static FOO: RefCell<u32> = RefCell::new(1));
///
/// FOO.with(|f| {
///     assert_eq!(*f.borrow(), 1);
///     *f.borrow_mut() = 2;
/// });
///
/// // each thread starts out with the initial value of 1
/// thread::spawn(move|| {
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
///
/// # Platform-specific behavior
///
/// Note that a "best effort" is made to ensure that destructors for types
/// stored in thread local storage are run, but not all platforms can guarantee
/// that destructors will be run for all types in thread local storage. For
/// example, there are a number of known caveats where destructors are not run:
///
/// 1. On Unix systems when pthread-based TLS is being used, destructors will
///    not be run for TLS values on the main thread when it exits. Note that the
///    application will exit immediately after the main thread exits as well.
/// 2. On all platforms it's possible for TLS to re-initialize other TLS slots
///    during destruction. Some platforms ensure that this cannot happen
///    infinitely by preventing re-initialization of any slot that has been
///    destroyed, but not all platforms have this guard. Those platforms that do
///    not guard typically have a synthetic limit after which point no more
///    destructors are run.
/// 3. On macOS, initializing TLS during destruction of other TLS slots can
///    sometimes cancel *all* destructors for the current thread, whether or not
///    the slots have already had their destructors run or not.
///
/// [`with`]: ../../std/thread/struct.LocalKey.html#method.with
/// [`thread_local!`]: ../../std/macro.thread_local.html
/// [`Drop`]: ../../std/ops/trait.Drop.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LocalKey<T: 'static> {
    // This outer `LocalKey<T>` type is what's going to be stored in statics,
    // but actual data inside will sometimes be tagged with #[thread_local].
    // It's not valid for a true static to reference a #[thread_local] static,
    // so we get around that by exposing an accessor through a layer of function
    // indirection (the 'get' thunk).
    //
    // Note that the thunk is itself unsafe because the returned lifetime of the
    // slot where data lives, `'static`, is not actually valid. The lifetime
    // here is actually slightly shorter than the currently running thread!
    //
    // Although this is an extra layer of indirection, it should in theory be
    // trivially devirtualizable by LLVM because the value of `inner` never
    // changes and the constant should be readonly within a crate. This mainly
    // only runs into problems when TLS statics are exported across crates.

    // The user-supplied initialization function that constructs a new T.
    init: fn() -> T,

    // Platform-specific callbacks.

    // Get a reference to the value. get can be called at any time during the
    // value's life cycle (Uninitialized, Initializing, Valid, and Destroyed).
    get: unsafe fn() -> &'static UnsafeCell<LocalKeyValue<T>>,

    // Register a destructor for the value. register_dtor should be called
    // after the key has been transitioned into the Valid state.
    register_dtor: unsafe fn(),
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T: 'static> fmt::Debug for LocalKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("LocalKey { .. }")
    }
}

/// Declare a new thread local storage key of type [`std::thread::LocalKey`].
///
/// # Syntax
///
/// The macro wraps any number of static declarations and makes them thread local.
/// Publicity and attributes for each static are allowed. Example:
///
/// ```
/// use std::cell::RefCell;
/// thread_local! {
///     pub static FOO: RefCell<u32> = RefCell::new(1);
///
///     #[allow(unused)]
///     static BAR: RefCell<f32> = RefCell::new(1.0);
/// }
/// # fn main() {}
/// ```
///
/// See [LocalKey documentation][`std::thread::LocalKey`] for more
/// information.
///
/// [`std::thread::LocalKey`]: ../std/thread/struct.LocalKey.html
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow_internal_unstable]
macro_rules! thread_local {
    // empty (base case for the recursion)
    () => {};

    // process multiple declarations
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr; $($rest:tt)*) => (
        __thread_local_inner!($(#[$attr])* $vis $name, $t, $init);
        thread_local!($($rest)*);
    );

    // handle a single declaration
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr) => (
        __thread_local_inner!($(#[$attr])* $vis $name, $t, $init);
    );
}

#[doc(hidden)]
#[unstable(feature = "thread_local_internals",
           reason = "should not be necessary",
           issue = "0")]
#[macro_export]
#[allow_internal_unstable]
#[allow_internal_unsafe]
macro_rules! __thread_local_inner {
    (@key $(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $init:expr) => {
        {
            #[thread_local]
            #[cfg(target_thread_local)]
            static __KEY: $crate::thread::__FastLocalKeyInner<$t> =
                $crate::thread::__FastLocalKeyInner::new();

            #[cfg(not(target_thread_local))]
            static __KEY: $crate::thread::__OsLocalKeyInner<$t> =
                $crate::thread::__OsLocalKeyInner::new();

            unsafe fn __get() -> &'static $crate::cell::UnsafeCell<$crate::thread::
                __LocalKeyValue<$t>> { __KEY.get() }

            // Only the fast implementation needs a destructor explicitly
            // registered - the OS implementation's destructor is registered
            // automatically by std::sys_common::thread_local::StaticKey.
            #[cfg(target_thread_local)]
            unsafe fn __register_dtor() { __KEY.register_dtor() }
            #[cfg(not(target_thread_local))]
            unsafe fn __register_dtor() {}

            #[inline]
            fn __init() -> $t { $init }

            unsafe { $crate::thread::LocalKey::new(__get, __register_dtor, __init) }
        }
    };
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $init:expr) => {
        #[cfg(stage0)]
        $(#[$attr])* $vis static $name: $crate::thread::LocalKey<$t> =
            __thread_local_inner!(@key $(#[$attr])* $vis $name, $t, $init);

        #[cfg(not(stage0))]
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            __thread_local_inner!(@key $(#[$attr])* $vis $name, $t, $init);
    }
}

#[unstable(feature = "thread_local_state",
           reason = "state querying was recently added",
           issue = "27716")]
#[doc(hidden)]
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum LocalKeyValue<T> {
    Uninitialized,
    Initializing,
    Valid(T),
    Destroyed,
}

/// Indicator of the state of a thread local storage key.
#[unstable(feature = "thread_local_state",
           reason = "state querying was recently added",
           issue = "27716")]
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum LocalKeyState {
    /// All keys are in this state whenever a thread starts. On the first call
    /// to [`with`], keys are initialized - they are moved into the `Initializing`
    /// state, then the initializer is called, and finally, if the initializer
    /// succeeds (does not panic), they are moved into the `Valid` state.
    ///
    /// Keys in the `Uninitialized` state will yield a reference to the closure
    /// passed to [`with`] so long as the initialization routine does not panic.
    ///
    /// [`with`]: ../../std/thread/struct.LocalKey.html#method.with
    Uninitialized,

    /// On the first call to [`with`], the key is moved into this state. After
    /// The initialization routine successfully completes (does not panic),
    /// the key is moved into the `Valid` state (if it does panic, the key is
    /// moved back into the `Uninitialized` state). Any code that may be called
    /// from the initialization routine for a particular key - and that may
    /// also access that key itself - should make sure that the key is not
    /// currently being initialized by either querying the key's state or by
    /// calling [`try_with`] instead of [`with`].
    ///
    /// Keys in the `Initializing` state will trigger a panic when accessed via
    /// [`with`].
    ///
    /// Note to allocator implementors: On some platforms, initializing a TLS
    /// key causes allocation to happen _before_ the key is moved into the
    /// `Initializing` state. Thus, it is not necessarily safe for a global
    /// allocator to use this TLS mechanism.
    ///
    /// [`try_with`]: ../../std/thread/struct.LocalKey.html#method.try_with
    /// [`with`]: ../../std/thread/struct.LocalKey.html#method.with
    Initializing,

    /// Once a key has been initialized successfully, it will enter the `Valid`
    /// state. Keys in the `Valid` state will remain so until the thread exits,
    /// at which point the destructor will be run and the key will enter the
    /// `Destroyed` state.
    ///
    /// Keys in the `Valid` state will be guaranteed to yield a reference to the
    /// closure passed to [`with`].
    ///
    /// [`with`]: ../../std/thread/struct.LocalKey.html#method.with
    Valid,

    /// When a thread exits, the destructors for keys will be run (if
    /// necessary). While a destructor is running, and possibly after a
    /// destructor has run, a key is in the `Destroyed` state.
    ///
    /// Keys in the `Destroyed` state will trigger a panic when accessed via
    /// [`with`].
    ///
    /// [`with`]: ../../std/thread/struct.LocalKey.html#method.with
    Destroyed,
}

/// An error returned by [`LocalKey::try_with`](struct.LocalKey.html#method.try_with).
#[unstable(feature = "thread_local_state",
           reason = "state querying was recently added",
           issue = "27716")]
pub struct AccessError {
    // whether the error was due to the key being in the Initializing
    // state - false means the key was in the Destroyed state
    init: bool,
}

impl AccessError {
    /// Determines whether the `AccessError` was due to the key being initialized.
    ///
    /// If `is_initializing` returns true, this `AccessError` was returned because
    /// the key was in the `Initializing` state when it was accessed.
    #[unstable(feature = "thread_local_state",
               reason = "state querying was recently added",
               issue = "27716")]
    pub fn is_initializing(&self) -> bool {
        self.init
    }

    /// Determines whether the `AccessError` was due to the key being destroyed.
    ///
    /// If `is_destroyed` returns true, this `AccessError` was returned because
    /// the key was in the `Destroyed` state when it was accessed.
    #[unstable(feature = "thread_local_state",
               reason = "state querying was recently added",
               issue = "27716")]
    pub fn is_destroyed(&self) -> bool {
        !self.init
    }
}

#[unstable(feature = "thread_local_state",
           reason = "state querying was recently added",
           issue = "27716")]
impl fmt::Debug for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AccessError").finish()
    }
}

#[unstable(feature = "thread_local_state",
           reason = "state querying was recently added",
           issue = "27716")]
impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.init {
            fmt::Display::fmt("currently being initialized", f)
        } else {
            fmt::Display::fmt("already destroyed", f)
        }
    }
}

impl<T: 'static> LocalKey<T> {
    #[doc(hidden)]
    #[unstable(feature = "thread_local_internals",
               reason = "recently added to create a key",
               issue = "0")]
   pub const unsafe fn new(get: unsafe fn() -> &'static UnsafeCell<LocalKeyValue<T>>,
                           register_dtor: unsafe fn(), init: fn() -> T) -> LocalKey<T> {
       LocalKey {
           init,
           get,
           register_dtor,
       }
   }

    /// Acquires a reference to the value in this TLS key.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// This function will `panic!()` if the key is currently being initialized
    /// or destructed, and it **may** panic if the destructor has previously
    /// been run for this thread.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with<F, R>(&'static self, f: F) -> R
                      where F: FnOnce(&T) -> R {
        self.try_with(f).expect("cannot access a TLS value while it is being initialized \
                                 or during or after it is destroyed")
    }

    unsafe fn init(&self, slot: &UnsafeCell<LocalKeyValue<T>>) {
        struct RollbackOnPanic<'a, T: 'a> {
            panicking: bool,
            slot: &'a UnsafeCell<LocalKeyValue<T>>,
        }

        impl<'a, T: 'a> Drop for RollbackOnPanic<'a, T> {
            fn drop(&mut self) {
                if self.panicking {
                    unsafe { *self.slot.get() = LocalKeyValue::Uninitialized }
                }
            }
        }

        let mut reset = RollbackOnPanic {
            panicking: true,
            slot,
        };

        // Transition into Initializing in preparation for calling self.init.
        *slot.get() = LocalKeyValue::Initializing;
        // Call the initializer; if this panics, the Drop method of RollbackOnPanic
        // will roll back to LocalKeyValue::Uninitialized.
        let val = (self.init)();
        reset.panicking = false;
        // If we get here, self.init didn't panic, so move into state Valid and
        // register the destructor.
        *slot.get() = LocalKeyValue::Valid(val);
        // NOTE(joshlf): Calling self.register_dtor here (after transitioning
        // into Initializing and then into Valid) guarantees that, for the fast
        // implementation, no allocation happens until after a key has transitioned
        // into Initializing. This allows a global allocator to make use of this
        // TLS implementation and still be able to detect a recusive call to alloc.
        //
        // Unfortunately, no such guarantee exists on platforms that cannot use
        // the fast implementation - std::sys_common::thread_local::StaticKey's
        // get and set methods both allocate under the hood, and the only way to
        // get or modify thread-local data is to use those methods. Thus, we cannot
        // guarantee that TLS is always safe to use in the implementation of global
        // allocators. If somebody in the future figures out a way to postpone
        // allocation until after the transition to Initializing, that would be
        // great.
        (self.register_dtor)();
    }

    /// Query the current state of this key.
    ///
    /// A key is initially in the `Uninitialized` state whenever a thread
    /// starts. It will remain in this state up until the first call to [`with`]
    /// or [`try_with`]. At this point, it will transition to the `Initializing`
    /// state for the duration of the initialization.
    ///
    /// Once the initialization expression succeeds, the key transitions to the
    /// `Valid` state which will guarantee that future calls to [`with`] will
    /// succeed within the thread.
    ///
    /// When a thread exits, each key will be destroyed in turn, and as keys are
    /// destroyed they will enter the `Destroyed` state just before the
    /// destructor starts to run. Keys may remain in the `Destroyed` state after
    /// destruction has completed. Keys without destructors (e.g. with types
    /// that are [`Copy`]), may never enter the `Destroyed` state.
    ///
    /// Keys in the `Uninitialized` state can be accessed so long as the
    /// initialization does not panic. Keys in the `Valid` state are guaranteed
    /// to be able to be accessed. Keys in the `Initializing` or  `Destroyed`
    /// states will panic on any call to [`with`].
    ///
    /// [`with`]: ../../std/thread/struct.LocalKey.html#method.with
    /// [`try_with`]: ../../std/thread/struct.LocalKey.html#method.try_with
    /// [`Copy`]: ../../std/marker/trait.Copy.html
    #[unstable(feature = "thread_local_state",
               reason = "state querying was recently added",
               issue = "27716")]
    pub fn state(&'static self) -> LocalKeyState {
        unsafe {
            match &*(self.get)().get() {
                &LocalKeyValue::Uninitialized => LocalKeyState::Uninitialized,
                &LocalKeyValue::Initializing => LocalKeyState::Initializing,
                &LocalKeyValue::Valid(_) => LocalKeyState::Valid,
                &LocalKeyValue::Destroyed => LocalKeyState::Destroyed,
            }
        }
    }

    /// Acquires a reference to the value in this TLS key.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet. If the key has been destroyed (which may happen if this is called
    /// in a destructor) or is being initialized (which may happen if this is called in
    /// the key's initialization expression), this function will return a ThreadLocalError.
    ///
    /// # Panics
    ///
    /// This function will still `panic!()` if the key is uninitialized and the
    /// key's initializer panics.
    #[unstable(feature = "thread_local_state",
               reason = "state querying was recently added",
               issue = "27716")]
    pub fn try_with<F, R>(&'static self, f: F) -> Result<R, AccessError>
                      where F: FnOnce(&T) -> R {
        unsafe {
            let slot = (self.get)();
            if let &LocalKeyValue::Valid(ref inner) = &*slot.get() {
                // Do this in a separate if branch (rather than just part of the
                // match statement in the else block) to increase the performance
                // of the fast path.
                Ok(f(inner))
            } else {
                match *slot.get() {
                    LocalKeyValue::Uninitialized => {
                        self.init(slot);
                        // Now that the value is initialized, we're guaranteed
                        // not to enter this else block in the recursive call.
                        self.try_with(f)
                    }
                    LocalKeyValue::Initializing => Err(AccessError { init: true }),
                    LocalKeyValue::Destroyed => Err(AccessError { init: false }),
                    LocalKeyValue::Valid(_) => unreachable!(),
                }
            }
        }
    }
}

#[doc(hidden)]
#[cfg(target_thread_local)]
pub mod fast {
    use cell::{Cell, UnsafeCell};
    use fmt;
    use mem;
    use sys::fast_thread_local::register_dtor;
    use thread::__LocalKeyValue;

    pub struct Key<T> {
        inner: UnsafeCell<__LocalKeyValue<T>>,

        // Keep track of whether the destructor has been registered (it's
        // registered by LocalKey::init after the initializer successfully
        // returns). Remember that this variable is thread-local, not global.
        dtor_registered: Cell<bool>,
    }

    impl<T> fmt::Debug for Key<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.pad("Key { .. }")
        }
    }

    impl<T> Key<T> {
        pub const fn new() -> Key<T> {
            Key {
                inner: UnsafeCell::new(__LocalKeyValue::Uninitialized),
                dtor_registered: Cell::new(false),
            }
        }

        pub unsafe fn get(&self) -> &'static UnsafeCell<__LocalKeyValue<T>> {
            &*(&self.inner as *const _)
        }

        pub unsafe fn register_dtor(&self) {
            if !mem::needs_drop::<T>() || self.dtor_registered.get() {
                return
            }

            register_dtor(self as *const _ as *mut u8,
                          destroy_value::<T>);
            self.dtor_registered.set(true);
        }
    }

    unsafe extern fn destroy_value<T>(ptr: *mut u8) {
        let ptr = ptr as *mut Key<T>;
        // Set inner to Destroyed before we drop tmp so that a
        // recursive call to get (called from the destructor) will
        // be able to detect that the value is already being dropped.
        let tmp = mem::replace(&mut *(*ptr).inner.get(), __LocalKeyValue::Destroyed);
        drop(tmp);
    }
}

#[doc(hidden)]
pub mod os {
    use cell::UnsafeCell;
    use fmt;
    use ptr;
    use sys_common::thread_local::StaticKey as OsStaticKey;
    use thread::__LocalKeyValue;

    pub struct Key<T> {
        // OS-TLS key that we'll use to key off.
        os: OsStaticKey,

        // Dummy value that can be used once the Value<T> has already been
        // deallocated. Its value is always Destroyed.
        dummy_destroyed: UnsafeCell<__LocalKeyValue<T>>,
    }

    impl<T> fmt::Debug for Key<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.pad("Key { .. }")
        }
    }

    unsafe impl<T> ::marker::Sync for Key<T> { }

    struct Value<T: 'static> {
        key: &'static Key<T>,
        value: UnsafeCell<__LocalKeyValue<T>>,
    }

    impl<T: 'static> Key<T> {
        pub const fn new() -> Key<T> {
            Key {
                os: OsStaticKey::new(Some(destroy_value::<T>)),
                dummy_destroyed: UnsafeCell::new(__LocalKeyValue::Destroyed),
            }
        }

        pub unsafe fn get(&'static self) -> &'static UnsafeCell<__LocalKeyValue<T>> {
            let ptr = self.os.get() as *mut Value<T>;
            if !ptr.is_null() {
                if ptr as usize == 1 {
                    // The destructor was already called (and set self.os to 1).
                    return &self.dummy_destroyed;
                }
                return &(*ptr).value;
            }
            // If the lookup returned null, we haven't initialized our own
            // local copy, so do that now.
            let ptr: Box<Value<T>> = box Value {
                key: &*(self as *const _),
                value: UnsafeCell::new(__LocalKeyValue::Uninitialized),
            };
            let ptr = Box::into_raw(ptr);
            self.os.set(ptr as *mut u8);
            &(*ptr).value
        }
    }

    unsafe extern fn destroy_value<T: 'static>(ptr: *mut u8) {
        // The OS TLS ensures that this key contains a NULL value when this
        // destructor starts to run. We set it back to a sentinel value of 1 to
        // ensure that any future calls to `get` for this thread will return
        // `None`.
        //
        // Note that to prevent an infinite loop we reset it back to null right
        // before we return from the destructor ourselves.
        //
        // FIXME: Setting this back to null means that, after this destructor
        // returns, future accesses of this key will think that the state is
        // Uninitialized (rather than Destroyed). We should figure out a way
        // to fix this.
        let ptr = Box::from_raw(ptr as *mut Value<T>);
        let key = ptr.key;
        key.os.set(1 as *mut u8);
        drop(ptr);
        key.os.set(ptr::null_mut());
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use sync::mpsc::{channel, Sender};
    use cell::{Cell, UnsafeCell};
    use super::LocalKeyState;
    use thread;

    struct Foo(Sender<()>);

    impl Drop for Foo {
        fn drop(&mut self) {
            let Foo(ref s) = *self;
            s.send(()).unwrap();
        }
    }

    #[test]
    fn smoke_no_dtor() {
        thread_local!(static FOO: Cell<i32> = Cell::new(1));

        FOO.with(|f| {
            assert_eq!(f.get(), 1);
            f.set(2);
        });
        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            FOO.with(|f| {
                assert_eq!(f.get(), 1);
            });
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();

        FOO.with(|f| {
            assert_eq!(f.get(), 2);
        });
    }

    #[test]
    fn states() {
        struct Foo;
        impl Drop for Foo {
            fn drop(&mut self) {
                assert!(FOO.state() == LocalKeyState::Destroyed);
            }
        }
        fn foo() -> Foo {
            assert!(FOO.state() == LocalKeyState::Initializing);
            Foo
        }
        thread_local!(static FOO: Foo = foo());

        thread::spawn(|| {
            assert!(FOO.state() == LocalKeyState::Uninitialized);
            FOO.with(|_| {
                assert!(FOO.state() == LocalKeyState::Valid);
            });
            assert!(FOO.state() == LocalKeyState::Valid);
        }).join().ok().unwrap();
    }

    #[test]
    fn smoke_dtor() {
        thread_local!(static FOO: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));

        let (tx, rx) = channel();
        let _t = thread::spawn(move|| unsafe {
            let mut tx = Some(tx);
            FOO.with(|f| {
                *f.get() = Some(Foo(tx.take().unwrap()));
            });
        });
        rx.recv().unwrap();
    }

    #[test]
    fn circular_init() {
        struct S1;
        struct S2;
        thread_local!(static K1: S1 = S1::new());
        thread_local!(static K2: S2 = S2::new());
        static mut HITS: u32 = 0;

        impl S1 {
            fn new() -> S1 {
                unsafe {
                    HITS += 1;
                    if K2.state() == LocalKeyState::Initializing {
                        assert_eq!(HITS, 3);
                    } else {
                        if HITS == 1 {
                            K2.with(|_| {});
                        } else {
                            assert_eq!(HITS, 3);
                        }
                    }
                }
                S1
            }
        }
        impl S2 {
            fn new() -> S2 {
                unsafe {
                    HITS += 1;
                    assert!(K1.state() != LocalKeyState::Initializing);
                    assert_eq!(HITS, 2);
                    K1.with(|_| {});
                }
                S2
            }
        }

        thread::spawn(move|| {
            drop(S1::new());
        }).join().ok().unwrap();
    }

    #[test]
    fn circular_drop() {
        struct S1;
        struct S2;
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
        thread_local!(static K2: UnsafeCell<Option<S2>> = UnsafeCell::new(None));
        static mut HITS: u32 = 0;

        impl Drop for S1 {
            fn drop(&mut self) {
                unsafe {
                    HITS += 1;
                    if K2.state() == LocalKeyState::Destroyed {
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
                    assert!(K1.state() != LocalKeyState::Destroyed);
                    assert_eq!(HITS, 2);
                    K1.with(|s| *s.get() = Some(S1));
                }
            }
        }

        thread::spawn(move|| {
            drop(S1);
        }).join().ok().unwrap();
    }

    #[test]
    fn self_referential_init() {
        struct S1;
        thread_local!(static K1: S1 = S1::new());

        impl S1 {
            fn new() -> S1 {
                assert!(K1.state() == LocalKeyState::Initializing);
                S1
            }
        }

        thread::spawn(move|| K1.with(|_| {})).join().ok().unwrap();
    }

    #[test]
    fn self_referential_drop() {
        struct S1;
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));

        impl Drop for S1 {
            fn drop(&mut self) {
                assert!(K1.state() == LocalKeyState::Destroyed);
            }
        }

        thread::spawn(move|| unsafe {
            K1.with(|s| *s.get() = Some(S1));
        }).join().ok().unwrap();
    }

    // Note that this test will deadlock if TLS destructors aren't run (this
    // requires the destructor to be run to pass the test). macOS has a known bug
    // where dtors-in-dtors may cancel other destructors, so we just ignore this
    // test on macOS.
    #[test]
    #[cfg_attr(target_os = "macos", ignore)]
    fn dtors_in_dtors_in_dtors() {
        struct S1(Sender<()>);
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
        thread_local!(static K2: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));

        impl Drop for S1 {
            fn drop(&mut self) {
                let S1(ref tx) = *self;
                unsafe {
                    if K2.state() != LocalKeyState::Destroyed {
                        K2.with(|s| *s.get() = Some(Foo(tx.clone())));
                    }
                }
            }
        }

        let (tx, rx) = channel();
        let _t = thread::spawn(move|| unsafe {
            let mut tx = Some(tx);
            K1.with(|s| *s.get() = Some(S1(tx.take().unwrap())));
        });
        rx.recv().unwrap();
    }
}

#[cfg(test)]
mod dynamic_tests {
    use cell::RefCell;
    use collections::HashMap;

    #[test]
    fn smoke() {
        fn square(i: i32) -> i32 { i * i }
        thread_local!(static FOO: i32 = square(3));

        FOO.with(|f| {
            assert_eq!(*f, 9);
        });
    }

    #[test]
    fn hashmap() {
        fn map() -> RefCell<HashMap<i32, i32>> {
            let mut m = HashMap::new();
            m.insert(1, 2);
            RefCell::new(m)
        }
        thread_local!(static FOO: RefCell<HashMap<i32, i32>> = map());

        FOO.with(|map| {
            assert_eq!(map.borrow()[&1], 2);
        });
    }

    #[test]
    fn refcell_vec() {
        thread_local!(static FOO: RefCell<Vec<u32>> = RefCell::new(vec![1, 2, 3]));

        FOO.with(|vec| {
            assert_eq!(vec.borrow().len(), 3);
            vec.borrow_mut().push(4);
            assert_eq!(vec.borrow()[3], 4);
        });
    }
}
