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
    // If it returns Some, then the key value is in the Valid state. Otherwise,
    // if is in one of the other three states (use the get_state callback to
    // check which one).
    get: unsafe fn() -> &'static Option<T>,
    // Query the value's current state.
    get_state: unsafe fn() -> LocalKeyState,
    // Begin initialization, moving the value into the Initializing state, and
    // performing any platform-specific initialization.
    //
    // After pre_init has been called, it must be safe to call rollback_init
    // and then pre_init an arbitrary number of times - if init panics, we
    // call rollback_init to move back to the Uninitialized state, and we
    // may try to initialize again in a future call to with or try_with.
    pre_init: unsafe fn(),
    // Finalize initialization, using the provided value as the initial value
    // for this key. Move the value into the Initialized state.
    post_init: unsafe fn(T),
    // Roll back a failed initialization (caused by init panicking). Move the
    // value back into the Uninitialized state.
    rollback_init: unsafe fn(),
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
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $init:expr) => {
        $(#[$attr])* $vis static $name: $crate::thread::LocalKey<$t> = {
            #[thread_local]
            #[cfg(target_thread_local)]
            static __KEY: $crate::thread::__FastLocalKeyInner<$t> =
                $crate::thread::__FastLocalKeyInner::new();

            #[cfg(not(target_thread_local))]
            static __KEY: $crate::thread::__OsLocalKeyInner<$t> =
                $crate::thread::__OsLocalKeyInner::new();

            unsafe fn __getit() -> &'static Option<$t> { __KEY.get() }

            unsafe fn __get_state() -> $crate::thread::LocalKeyState { __KEY.get_state() }

            unsafe fn __pre_init() { __KEY.pre_init() }

            fn __init() -> $t { $init }

            unsafe fn __post_init(val: $t) { __KEY.post_init(val) }

            unsafe fn __rollback_init() { __KEY.rollback_init() }

            unsafe {
                $crate::thread::LocalKey::new(__getit,
                                              __get_state,
                                              __pre_init,
                                              __init,
                                              __post_init,
                                              __rollback_init)
            }
        };
    }
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
   pub const unsafe fn new(get: unsafe fn() -> &'static Option<T>,
                    get_state: unsafe fn() -> LocalKeyState,
                    pre_init: unsafe fn(),
                    init: fn() -> T,
                    post_init: unsafe fn(T),
                    rollback_init: unsafe fn())
                    -> LocalKey<T> {
       LocalKey {
           get,
           get_state,
           pre_init,
           init,
           post_init,
           rollback_init,
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

    unsafe fn init(&self) {
        struct RollbackOnPanic {
            panicking: bool,
            rollback: unsafe fn(),
        }

        impl Drop for RollbackOnPanic {
            fn drop(&mut self) {
                if self.panicking {
                    unsafe { (self.rollback)() }
                }
            }
        }

        let mut reset = RollbackOnPanic {
            panicking: true,
            rollback: self.rollback_init,
        };

        // Transition into Valid, perform any pre-init work (e.g., registering
        // destructors). If self.init panics, the Drop method of RollbackOnPanic
        // will call self.rollback, rolling back the state to Uninitialized.
        //
        // Note that once self.pre_init has been called once, it must be safe to call
        // self.rollback_init and then self.pre_init an arbitrary number of times - if
        // self.init panics, a future call to with or try_with will again try to
        // initialize the value, causing this routine to be run.
        (self.pre_init)();
        let val = (self.init)();
        reset.panicking = false;
        // Transition into Valid, set the value to val.
        (self.post_init)(val);
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
        unsafe { (self.get_state)() }
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
            if let &Some(ref inner) = (self.get)() {
                Ok(f(inner))
            } else {
                match self.state() {
                    LocalKeyState::Uninitialized => {
                        self.init();
                        // Now that the value is initialized, we're guaranteed
                        // not to enter this else block in the recursive call.
                        self.try_with(f)
                    }
                    LocalKeyState::Initializing => Err(AccessError { init: true }),
                    LocalKeyState::Destroyed => Err(AccessError { init: false }),
                    LocalKeyState::Valid => unreachable!(),
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
    use ptr;
    use sys::fast_thread_local::register_dtor;
    use thread::LocalKeyState;

    pub struct Key<T> {
        inner: UnsafeCell<Option<T>>,
        state: UnsafeCell<LocalKeyState>,

        // Keep track of whether the destructor has been registered (this is
        // not the same thing as not being in the Uninitialized state - we
        // can transition back into that state in rollback_init). Remember
        // that this variable is thread-local, not global.
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
                inner: UnsafeCell::new(None),
                state: UnsafeCell::new(LocalKeyState::Uninitialized),
                dtor_registered: Cell::new(false),
            }
        }

        pub fn get(&self) -> &'static Option<T> {
            unsafe { &*self.inner.get() }
        }

        pub fn get_state(&self) -> LocalKeyState {
            unsafe { *self.state.get() }
        }

        pub fn pre_init(&self) {
            unsafe {
                // It's critical that we set the state to Initializing before
                // registering destructors - if registering destructors causes
                // allocation, and the global allocator uses TLS, then the
                // allocator needs to be able to detect that the TLS is in
                // the Initializing state and perform appropriate fallback
                // logic rather than recursing infinitely.
                *self.state.get() = LocalKeyState::Initializing;
                self.register_dtor();
            }
        }

        pub fn post_init(&self, val: T) {
            unsafe {
                *self.inner.get() = Some(val);
                *self.state.get() = LocalKeyState::Valid;
            }
        }

        pub fn rollback_init(&self) {
            unsafe {
                *self.inner.get() = None;
                *self.state.get() = LocalKeyState::Uninitialized;
            }
        }

        unsafe fn register_dtor(&self) {
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
        let tmp = ptr::read((*ptr).inner.get());
        // Set inner to None and set the state to Destroyed before we drop tmp
        // so that a recursive call to get (called from the destructor) will
        // be able to detect that the value is already being dropped.
        ptr::write((*ptr).inner.get(), None);
        *(*ptr).state.get() = LocalKeyState::Destroyed;
        drop(tmp);
    }
}

#[doc(hidden)]
pub mod os {
    use cell::UnsafeCell;
    use fmt;
    use sys_common::thread_local::StaticKey as OsStaticKey;
    use thread::LocalKeyState;

    pub struct Key<T> {
        // OS-TLS key that we'll use to key off.
        os: OsStaticKey,
        // The state of this key. os is only guaranteed to point to an
        // allocated Value<T> if this state is Valid.
        state: UnsafeCell<LocalKeyState>,
        // Store a value here in addition to in the OsStaticKey itself so
        // that if the OsStaticKey hasn't yet been allocated, we still have
        // an Option that we can return a reference to. Unlike the real
        // value, this value will always be None (because when the real
        // value is initialized, we just use it directly).
        dummy_value: UnsafeCell<Option<T>>,
    }

    impl<T> fmt::Debug for Key<T> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.pad("Key { .. }")
        }
    }

    unsafe impl<T> ::marker::Sync for Key<T> { }

    struct Value<T: 'static> {
        key: &'static Key<T>,
        value: UnsafeCell<Option<T>>,
    }

    impl<T: 'static> Key<T> {
        pub const fn new() -> Key<T> {
            Key {
                os: OsStaticKey::new(Some(destroy_value::<T>)),
                state: UnsafeCell::new(LocalKeyState::Uninitialized),
                dummy_value: UnsafeCell::new(None),
            }
        }

        pub fn get(&self) -> &'static Option<T> {
            unsafe {
                match *self.state.get() {
                    LocalKeyState::Valid => {
                        // Since the state is Valid, we know that os points to
                        // an allocated Value<T>.
                        let ptr = self.os.get() as *mut Value<T>;
                        debug_assert!(!ptr.is_null());
                        &*(*ptr).value.get()
                    }
                    _ => {
                        // The dummy_value is guaranteed to always be None. This
                        // allows us to avoid allocating if the state isn't Valid.
                        // This is critical because it means that an allocator can
                        // call try_with and detect that the key is in state
                        // Initializing without recursing infinitely.
                        &*self.dummy_value.get()
                    }
                }
            }
        }

        pub fn get_state(&self) -> LocalKeyState {
            unsafe { *self.state.get() }
        }

        pub fn pre_init(&self) {
            unsafe {
                *self.state.get() = LocalKeyState::Initializing;
            }
        }

        pub fn rollback_init(&self) {
            unsafe {
                *self.state.get() = LocalKeyState::Uninitialized;
            }
        }

        pub fn post_init(&self, val: T) {
            unsafe {
                let ptr: Box<Value<T>> = box Value {
                                                 key: &*(self as *const _),
                                                 value: UnsafeCell::new(Some(val)),
                                             };
                let ptr = Box::into_raw(ptr);
                self.os.set(ptr as *mut u8);
                *self.state.get() = LocalKeyState::Valid;
            }
        }
    }

    unsafe extern fn destroy_value<T: 'static>(ptr: *mut u8) {
        let ptr = Box::from_raw(ptr as *mut Value<T>);
        // Set the state to Destroyed before we drop ptr so that any recursive
        // calls to get can detect that the destructor is already being called.
        *ptr.key.state.get() = LocalKeyState::Destroyed;
        drop(ptr);
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
