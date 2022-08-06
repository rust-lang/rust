//! Thread local storage

#![unstable(feature = "thread_local_internals", issue = "none")]

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

#[cfg(test)]
mod dynamic_tests;

use crate::cell::{Cell, RefCell};
use crate::error::Error;
use crate::fmt;

/// A thread local storage key which owns its contents.
///
/// This key uses the fastest possible implementation available to it for the
/// target platform. It is instantiated with the [`thread_local!`] macro and the
/// primary method is the [`with`] method.
///
/// The [`with`] method yields a reference to the contained value which cannot be
/// sent across threads or escape the given closure.
///
/// [`thread_local!`]: crate::thread_local
///
/// # Initialization and Destruction
///
/// Initialization is dynamically performed on the first call to [`with`]
/// within a thread, and values that implement [`Drop`] get destructed when a
/// thread exits. Some caveats apply, which are explained below.
///
/// A `LocalKey`'s initializer cannot recursively depend on itself, and using
/// a `LocalKey` in this way will cause the initializer to infinitely recurse
/// on the first call to `with`.
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
/// let t = thread::spawn(move|| {
///     FOO.with(|f| {
///         assert_eq!(*f.borrow(), 1);
///         *f.borrow_mut() = 3;
///     });
/// });
///
/// // wait for the thread to complete and bail out on panic
/// t.join().unwrap();
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
/// 3. When the process exits on Windows systems, TLS destructors may only be
///    run on the thread that causes the process to exit. This is because the
///    other threads may be forcibly terminated.
///
/// ## Synchronization in thread-local destructors
///
/// On Windows, synchronization operations (such as [`JoinHandle::join`]) in
/// thread local destructors are prone to deadlocks and so should be avoided.
/// This is because the [loader lock] is held while a destructor is run. The
/// lock is acquired whenever a thread starts or exits or when a DLL is loaded
/// or unloaded. Therefore these events are blocked for as long as a thread
/// local destructor is running.
///
/// [loader lock]: https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
/// [`JoinHandle::join`]: crate::thread::JoinHandle::join
/// [`with`]: LocalKey::with
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LocalKey<T: 'static> {
    // This outer `LocalKey<T>` type is what's going to be stored in statics,
    // but actual data inside will sometimes be tagged with #[thread_local].
    // It's not valid for a true static to reference a #[thread_local] static,
    // so we get around that by exposing an accessor through a layer of function
    // indirection (this thunk).
    //
    // Note that the thunk is itself unsafe because the returned lifetime of the
    // slot where data lives, `'static`, is not actually valid. The lifetime
    // here is actually slightly shorter than the currently running thread!
    //
    // Although this is an extra layer of indirection, it should in theory be
    // trivially devirtualizable by LLVM because the value of `inner` never
    // changes and the constant should be readonly within a crate. This mainly
    // only runs into problems when TLS statics are exported across crates.
    inner: unsafe fn(Option<&mut Option<T>>) -> Option<&'static T>,
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T: 'static> fmt::Debug for LocalKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalKey").finish_non_exhaustive()
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
/// See [`LocalKey` documentation][`std::thread::LocalKey`] for more
/// information.
///
/// [`std::thread::LocalKey`]: crate::thread::LocalKey
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "thread_local_macro")]
#[allow_internal_unstable(thread_local_internals)]
macro_rules! thread_local {
    // empty (base case for the recursion)
    () => {};

    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = const { $init:expr }; $($rest:tt)*) => (
        $crate::__thread_local_inner!($(#[$attr])* $vis $name, $t, const $init);
        $crate::thread_local!($($rest)*);
    );

    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = const { $init:expr }) => (
        $crate::__thread_local_inner!($(#[$attr])* $vis $name, $t, const $init);
    );

    // process multiple declarations
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr; $($rest:tt)*) => (
        $crate::__thread_local_inner!($(#[$attr])* $vis $name, $t, $init);
        $crate::thread_local!($($rest)*);
    );

    // handle a single declaration
    ($(#[$attr:meta])* $vis:vis static $name:ident: $t:ty = $init:expr) => (
        $crate::__thread_local_inner!($(#[$attr])* $vis $name, $t, $init);
    );
}

#[doc(hidden)]
#[unstable(feature = "thread_local_internals", reason = "should not be necessary", issue = "none")]
#[macro_export]
#[allow_internal_unstable(thread_local_internals, cfg_target_thread_local, thread_local)]
#[allow_internal_unsafe]
macro_rules! __thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, const $init:expr) => {{
        #[cfg_attr(not(windows), inline)] // see comments below
        #[deny(unsafe_op_in_unsafe_fn)]
        unsafe fn __getit(
            _init: $crate::option::Option<&mut $crate::option::Option<$t>>,
        ) -> $crate::option::Option<&'static $t> {
            const INIT_EXPR: $t = $init;

            // wasm without atomics maps directly to `static mut`, and dtors
            // aren't implemented because thread dtors aren't really a thing
            // on wasm right now
            //
            // FIXME(#84224) this should come after the `target_thread_local`
            // block.
            #[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
            {
                static mut VAL: $t = INIT_EXPR;
                unsafe { $crate::option::Option::Some(&VAL) }
            }

            // If the platform has support for `#[thread_local]`, use it.
            #[cfg(all(
                target_thread_local,
                not(all(target_family = "wasm", not(target_feature = "atomics"))),
            ))]
            {
                #[thread_local]
                static mut VAL: $t = INIT_EXPR;

                // If a dtor isn't needed we can do something "very raw" and
                // just get going.
                if !$crate::mem::needs_drop::<$t>() {
                    unsafe {
                        return $crate::option::Option::Some(&VAL)
                    }
                }

                // 0 == dtor not registered
                // 1 == dtor registered, dtor not run
                // 2 == dtor registered and is running or has run
                #[thread_local]
                static mut STATE: $crate::primitive::u8 = 0;

                unsafe extern "C" fn destroy(ptr: *mut $crate::primitive::u8) {
                    let ptr = ptr as *mut $t;

                    unsafe {
                        $crate::debug_assert_eq!(STATE, 1);
                        STATE = 2;
                        $crate::ptr::drop_in_place(ptr);
                    }
                }

                unsafe {
                    match STATE {
                        // 0 == we haven't registered a destructor, so do
                        //   so now.
                        0 => {
                            $crate::thread::__FastLocalKeyInner::<$t>::register_dtor(
                                $crate::ptr::addr_of_mut!(VAL) as *mut $crate::primitive::u8,
                                destroy,
                            );
                            STATE = 1;
                            $crate::option::Option::Some(&VAL)
                        }
                        // 1 == the destructor is registered and the value
                        //   is valid, so return the pointer.
                        1 => $crate::option::Option::Some(&VAL),
                        // otherwise the destructor has already run, so we
                        // can't give access.
                        _ => $crate::option::Option::None,
                    }
                }
            }

            // On platforms without `#[thread_local]` we fall back to the
            // same implementation as below for os thread locals.
            #[cfg(all(
                not(target_thread_local),
                not(all(target_family = "wasm", not(target_feature = "atomics"))),
            ))]
            {
                #[inline]
                const fn __init() -> $t { INIT_EXPR }
                static __KEY: $crate::thread::__OsLocalKeyInner<$t> =
                    $crate::thread::__OsLocalKeyInner::new();
                #[allow(unused_unsafe)]
                unsafe {
                    __KEY.get(move || {
                        if let $crate::option::Option::Some(init) = _init {
                            if let $crate::option::Option::Some(value) = init.take() {
                                return value;
                            } else if $crate::cfg!(debug_assertions) {
                                $crate::unreachable!("missing initial value");
                            }
                        }
                        __init()
                    })
                }
            }
        }

        unsafe {
            $crate::thread::LocalKey::new(__getit)
        }
    }};

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {
        {
            #[inline]
            fn __init() -> $t { $init }

            // When reading this function you might ask "why is this inlined
            // everywhere other than Windows?", and that's a very reasonable
            // question to ask. The short story is that it segfaults rustc if
            // this function is inlined. The longer story is that Windows looks
            // to not support `extern` references to thread locals across DLL
            // boundaries. This appears to at least not be supported in the ABI
            // that LLVM implements.
            //
            // Because of this we never inline on Windows, but we do inline on
            // other platforms (where external references to thread locals
            // across DLLs are supported). A better fix for this would be to
            // inline this function on Windows, but only for "statically linked"
            // components. For example if two separately compiled rlibs end up
            // getting linked into a DLL then it's fine to inline this function
            // across that boundary. It's only not fine to inline this function
            // across a DLL boundary. Unfortunately rustc doesn't currently
            // have this sort of logic available in an attribute, and it's not
            // clear that rustc is even equipped to answer this (it's more of a
            // Cargo question kinda). This means that, unfortunately, Windows
            // gets the pessimistic path for now where it's never inlined.
            //
            // The issue of "should enable on Windows sometimes" is #84933
            #[cfg_attr(not(windows), inline)]
            unsafe fn __getit(
                init: $crate::option::Option<&mut $crate::option::Option<$t>>,
            ) -> $crate::option::Option<&'static $t> {
                #[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
                static __KEY: $crate::thread::__StaticLocalKeyInner<$t> =
                    $crate::thread::__StaticLocalKeyInner::new();

                #[thread_local]
                #[cfg(all(
                    target_thread_local,
                    not(all(target_family = "wasm", not(target_feature = "atomics"))),
                ))]
                static __KEY: $crate::thread::__FastLocalKeyInner<$t> =
                    $crate::thread::__FastLocalKeyInner::new();

                #[cfg(all(
                    not(target_thread_local),
                    not(all(target_family = "wasm", not(target_feature = "atomics"))),
                ))]
                static __KEY: $crate::thread::__OsLocalKeyInner<$t> =
                    $crate::thread::__OsLocalKeyInner::new();

                // FIXME: remove the #[allow(...)] marker when macros don't
                // raise warning for missing/extraneous unsafe blocks anymore.
                // See https://github.com/rust-lang/rust/issues/74838.
                #[allow(unused_unsafe)]
                unsafe {
                    __KEY.get(move || {
                        if let $crate::option::Option::Some(init) = init {
                            if let $crate::option::Option::Some(value) = init.take() {
                                return value;
                            } else if $crate::cfg!(debug_assertions) {
                                $crate::unreachable!("missing default value");
                            }
                        }
                        __init()
                    })
                }
            }

            unsafe {
                $crate::thread::LocalKey::new(__getit)
            }
        }
    };
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $($init:tt)*) => {
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::__thread_local_inner!(@key $t, $($init)*);
    }
}

/// An error returned by [`LocalKey::try_with`](struct.LocalKey.html#method.try_with).
#[stable(feature = "thread_local_try_with", since = "1.26.0")]
#[non_exhaustive]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct AccessError;

#[stable(feature = "thread_local_try_with", since = "1.26.0")]
impl fmt::Debug for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccessError").finish()
    }
}

#[stable(feature = "thread_local_try_with", since = "1.26.0")]
impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt("already destroyed", f)
    }
}

#[stable(feature = "thread_local_try_with", since = "1.26.0")]
impl Error for AccessError {}

impl<T: 'static> LocalKey<T> {
    #[doc(hidden)]
    #[unstable(
        feature = "thread_local_internals",
        reason = "recently added to create a key",
        issue = "none"
    )]
    #[rustc_const_unstable(feature = "thread_local_internals", issue = "none")]
    pub const unsafe fn new(
        inner: unsafe fn(Option<&mut Option<T>>) -> Option<&'static T>,
    ) -> LocalKey<T> {
        LocalKey { inner }
    }

    /// Acquires a reference to the value in this TLS key.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// This function will `panic!()` if the key currently has its
    /// destructor running, and it **may** panic if the destructor has
    /// previously been run for this thread.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        self.try_with(f).expect(
            "cannot access a Thread Local Storage value \
             during or after destruction",
        )
    }

    /// Acquires a reference to the value in this TLS key.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet. If the key has been destroyed (which may happen if this is called
    /// in a destructor), this function will return an [`AccessError`].
    ///
    /// # Panics
    ///
    /// This function will still `panic!()` if the key is uninitialized and the
    /// key's initializer panics.
    #[stable(feature = "thread_local_try_with", since = "1.26.0")]
    #[inline]
    pub fn try_with<F, R>(&'static self, f: F) -> Result<R, AccessError>
    where
        F: FnOnce(&T) -> R,
    {
        unsafe {
            let thread_local = (self.inner)(None).ok_or(AccessError)?;
            Ok(f(thread_local))
        }
    }

    /// Acquires a reference to the value in this TLS key, initializing it with
    /// `init` if it wasn't already initialized on this thread.
    ///
    /// If `init` was used to initialize the thread local variable, `None` is
    /// passed as the first argument to `f`. If it was already initialized,
    /// `Some(init)` is passed to `f`.
    ///
    /// # Panics
    ///
    /// This function will panic if the key currently has its destructor
    /// running, and it **may** panic if the destructor has previously been run
    /// for this thread.
    fn initialize_with<F, R>(&'static self, init: T, f: F) -> R
    where
        F: FnOnce(Option<T>, &T) -> R,
    {
        unsafe {
            let mut init = Some(init);
            let reference = (self.inner)(Some(&mut init)).expect(
                "cannot access a Thread Local Storage value \
                 during or after destruction",
            );
            f(init, reference)
        }
    }
}

impl<T: 'static> LocalKey<Cell<T>> {
    /// Sets or initializes the contained value.
    ///
    /// Unlike the other methods, this will *not* run the lazy initializer of
    /// the thread local. Instead, it will be directly initialized with the
    /// given value if it wasn't initialized yet.
    ///
    /// # Panics
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<i32> = panic!("!");
    /// }
    ///
    /// // Calling X.get() here would result in a panic.
    ///
    /// X.set(123); // But X.set() is fine, as it skips the initializer above.
    ///
    /// assert_eq!(X.get(), 123);
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn set(&'static self, value: T) {
        self.initialize_with(Cell::new(value), |value, cell| {
            if let Some(value) = value {
                // The cell was already initialized, so `value` wasn't used to
                // initialize it. So we overwrite the current value with the
                // new one instead.
                cell.set(value.into_inner());
            }
        });
    }

    /// Returns a copy of the contained value.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<i32> = Cell::new(1);
    /// }
    ///
    /// assert_eq!(X.get(), 1);
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn get(&'static self) -> T
    where
        T: Copy,
    {
        self.with(|cell| cell.get())
    }

    /// Takes the contained value, leaving `Default::default()` in its place.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<Option<i32>> = Cell::new(Some(1));
    /// }
    ///
    /// assert_eq!(X.take(), Some(1));
    /// assert_eq!(X.take(), None);
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn take(&'static self) -> T
    where
        T: Default,
    {
        self.with(|cell| cell.take())
    }

    /// Replaces the contained value, returning the old value.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<i32> = Cell::new(1);
    /// }
    ///
    /// assert_eq!(X.replace(2), 1);
    /// assert_eq!(X.replace(3), 2);
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn replace(&'static self, value: T) -> T {
        self.with(|cell| cell.replace(value))
    }
}

impl<T: 'static> LocalKey<RefCell<T>> {
    /// Acquires a reference to the contained value.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed.
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::RefCell;
    ///
    /// thread_local! {
    ///     static X: RefCell<Vec<i32>> = RefCell::new(Vec::new());
    /// }
    ///
    /// X.with_borrow(|v| assert!(v.is_empty()));
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn with_borrow<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        self.with(|cell| f(&cell.borrow()))
    }

    /// Acquires a mutable reference to the contained value.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Example
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::RefCell;
    ///
    /// thread_local! {
    ///     static X: RefCell<Vec<i32>> = RefCell::new(Vec::new());
    /// }
    ///
    /// X.with_borrow_mut(|v| v.push(1));
    ///
    /// X.with_borrow(|v| assert_eq!(*v, vec![1]));
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn with_borrow_mut<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        self.with(|cell| f(&mut cell.borrow_mut()))
    }

    /// Sets or initializes the contained value.
    ///
    /// Unlike the other methods, this will *not* run the lazy initializer of
    /// the thread local. Instead, it will be directly initialized with the
    /// given value if it wasn't initialized yet.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::RefCell;
    ///
    /// thread_local! {
    ///     static X: RefCell<Vec<i32>> = panic!("!");
    /// }
    ///
    /// // Calling X.with() here would result in a panic.
    ///
    /// X.set(vec![1, 2, 3]); // But X.set() is fine, as it skips the initializer above.
    ///
    /// X.with_borrow(|v| assert_eq!(*v, vec![1, 2, 3]));
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn set(&'static self, value: T) {
        self.initialize_with(RefCell::new(value), |value, cell| {
            if let Some(value) = value {
                // The cell was already initialized, so `value` wasn't used to
                // initialize it. So we overwrite the current value with the
                // new one instead.
                *cell.borrow_mut() = value.into_inner();
            }
        });
    }

    /// Takes the contained value, leaving `Default::default()` in its place.
    ///
    /// This will lazily initialize the value if this thread has not referenced
    /// this key yet.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::RefCell;
    ///
    /// thread_local! {
    ///     static X: RefCell<Vec<i32>> = RefCell::new(Vec::new());
    /// }
    ///
    /// X.with_borrow_mut(|v| v.push(1));
    ///
    /// let a = X.take();
    ///
    /// assert_eq!(a, vec![1]);
    ///
    /// X.with_borrow(|v| assert!(v.is_empty()));
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn take(&'static self) -> T
    where
        T: Default,
    {
        self.with(|cell| cell.take())
    }

    /// Replaces the contained value, returning the old value.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    ///
    /// Panics if the key currently has its destructor running,
    /// and it **may** panic if the destructor has previously been run for this thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_methods)]
    /// use std::cell::RefCell;
    ///
    /// thread_local! {
    ///     static X: RefCell<Vec<i32>> = RefCell::new(Vec::new());
    /// }
    ///
    /// let prev = X.replace(vec![1, 2, 3]);
    /// assert!(prev.is_empty());
    ///
    /// X.with_borrow(|v| assert_eq!(*v, vec![1, 2, 3]));
    /// ```
    #[unstable(feature = "local_key_cell_methods", issue = "92122")]
    pub fn replace(&'static self, value: T) -> T {
        self.with(|cell| cell.replace(value))
    }
}

mod lazy {
    use crate::cell::UnsafeCell;
    use crate::hint;
    use crate::mem;

    pub struct LazyKeyInner<T> {
        inner: UnsafeCell<Option<T>>,
    }

    impl<T> LazyKeyInner<T> {
        pub const fn new() -> LazyKeyInner<T> {
            LazyKeyInner { inner: UnsafeCell::new(None) }
        }

        pub unsafe fn get(&self) -> Option<&'static T> {
            // SAFETY: The caller must ensure no reference is ever handed out to
            // the inner cell nor mutable reference to the Option<T> inside said
            // cell. This make it safe to hand a reference, though the lifetime
            // of 'static is itself unsafe, making the get method unsafe.
            unsafe { (*self.inner.get()).as_ref() }
        }

        /// The caller must ensure that no reference is active: this method
        /// needs unique access.
        pub unsafe fn initialize<F: FnOnce() -> T>(&self, init: F) -> &'static T {
            // Execute the initialization up front, *then* move it into our slot,
            // just in case initialization fails.
            let value = init();
            let ptr = self.inner.get();

            // SAFETY:
            //
            // note that this can in theory just be `*ptr = Some(value)`, but due to
            // the compiler will currently codegen that pattern with something like:
            //
            //      ptr::drop_in_place(ptr)
            //      ptr::write(ptr, Some(value))
            //
            // Due to this pattern it's possible for the destructor of the value in
            // `ptr` (e.g., if this is being recursively initialized) to re-access
            // TLS, in which case there will be a `&` and `&mut` pointer to the same
            // value (an aliasing violation). To avoid setting the "I'm running a
            // destructor" flag we just use `mem::replace` which should sequence the
            // operations a little differently and make this safe to call.
            //
            // The precondition also ensures that we are the only one accessing
            // `self` at the moment so replacing is fine.
            unsafe {
                let _ = mem::replace(&mut *ptr, Some(value));
            }

            // SAFETY: With the call to `mem::replace` it is guaranteed there is
            // a `Some` behind `ptr`, not a `None` so `unreachable_unchecked`
            // will never be reached.
            unsafe {
                // After storing `Some` we want to get a reference to the contents of
                // what we just stored. While we could use `unwrap` here and it should
                // always work it empirically doesn't seem to always get optimized away,
                // which means that using something like `try_with` can pull in
                // panicking code and cause a large size bloat.
                match *ptr {
                    Some(ref x) => x,
                    None => hint::unreachable_unchecked(),
                }
            }
        }

        /// The other methods hand out references while taking &self.
        /// As such, callers of this method must ensure no `&` and `&mut` are
        /// available and used at the same time.
        #[allow(unused)]
        pub unsafe fn take(&mut self) -> Option<T> {
            // SAFETY: See doc comment for this method.
            unsafe { (*self.inner.get()).take() }
        }
    }
}

/// On some targets like wasm there's no threads, so no need to generate
/// thread locals and we can instead just use plain statics!
#[doc(hidden)]
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
pub mod statik {
    use super::lazy::LazyKeyInner;
    use crate::fmt;

    pub struct Key<T> {
        inner: LazyKeyInner<T>,
    }

    unsafe impl<T> Sync for Key<T> {}

    impl<T> fmt::Debug for Key<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Key").finish_non_exhaustive()
        }
    }

    impl<T> Key<T> {
        pub const fn new() -> Key<T> {
            Key { inner: LazyKeyInner::new() }
        }

        pub unsafe fn get(&self, init: impl FnOnce() -> T) -> Option<&'static T> {
            // SAFETY: The caller must ensure no reference is ever handed out to
            // the inner cell nor mutable reference to the Option<T> inside said
            // cell. This make it safe to hand a reference, though the lifetime
            // of 'static is itself unsafe, making the get method unsafe.
            let value = unsafe {
                match self.inner.get() {
                    Some(ref value) => value,
                    None => self.inner.initialize(init),
                }
            };

            Some(value)
        }
    }
}

#[doc(hidden)]
#[cfg(target_thread_local)]
pub mod fast {
    use super::lazy::LazyKeyInner;
    use crate::cell::Cell;
    use crate::fmt;
    use crate::mem;
    use crate::sys::thread_local_dtor::register_dtor;

    #[derive(Copy, Clone)]
    enum DtorState {
        Unregistered,
        Registered,
        RunningOrHasRun,
    }

    // This data structure has been carefully constructed so that the fast path
    // only contains one branch on x86. That optimization is necessary to avoid
    // duplicated tls lookups on OSX.
    //
    // LLVM issue: https://bugs.llvm.org/show_bug.cgi?id=41722
    pub struct Key<T> {
        // If `LazyKeyInner::get` returns `None`, that indicates either:
        //   * The value has never been initialized
        //   * The value is being recursively initialized
        //   * The value has already been destroyed or is being destroyed
        // To determine which kind of `None`, check `dtor_state`.
        //
        // This is very optimizer friendly for the fast path - initialized but
        // not yet dropped.
        inner: LazyKeyInner<T>,

        // Metadata to keep track of the state of the destructor. Remember that
        // this variable is thread-local, not global.
        dtor_state: Cell<DtorState>,
    }

    impl<T> fmt::Debug for Key<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Key").finish_non_exhaustive()
        }
    }

    impl<T> Key<T> {
        pub const fn new() -> Key<T> {
            Key { inner: LazyKeyInner::new(), dtor_state: Cell::new(DtorState::Unregistered) }
        }

        // note that this is just a publicly-callable function only for the
        // const-initialized form of thread locals, basically a way to call the
        // free `register_dtor` function defined elsewhere in libstd.
        pub unsafe fn register_dtor(a: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
            unsafe {
                register_dtor(a, dtor);
            }
        }

        pub unsafe fn get<F: FnOnce() -> T>(&self, init: F) -> Option<&'static T> {
            // SAFETY: See the definitions of `LazyKeyInner::get` and
            // `try_initialize` for more information.
            //
            // The caller must ensure no mutable references are ever active to
            // the inner cell or the inner T when this is called.
            // The `try_initialize` is dependant on the passed `init` function
            // for this.
            unsafe {
                match self.inner.get() {
                    Some(val) => Some(val),
                    None => self.try_initialize(init),
                }
            }
        }

        // `try_initialize` is only called once per fast thread local variable,
        // except in corner cases where thread_local dtors reference other
        // thread_local's, or it is being recursively initialized.
        //
        // Macos: Inlining this function can cause two `tlv_get_addr` calls to
        // be performed for every call to `Key::get`.
        // LLVM issue: https://bugs.llvm.org/show_bug.cgi?id=41722
        #[inline(never)]
        unsafe fn try_initialize<F: FnOnce() -> T>(&self, init: F) -> Option<&'static T> {
            // SAFETY: See comment above (this function doc).
            if !mem::needs_drop::<T>() || unsafe { self.try_register_dtor() } {
                // SAFETY: See comment above (this function doc).
                Some(unsafe { self.inner.initialize(init) })
            } else {
                None
            }
        }

        // `try_register_dtor` is only called once per fast thread local
        // variable, except in corner cases where thread_local dtors reference
        // other thread_local's, or it is being recursively initialized.
        unsafe fn try_register_dtor(&self) -> bool {
            match self.dtor_state.get() {
                DtorState::Unregistered => {
                    // SAFETY: dtor registration happens before initialization.
                    // Passing `self` as a pointer while using `destroy_value<T>`
                    // is safe because the function will build a pointer to a
                    // Key<T>, which is the type of self and so find the correct
                    // size.
                    unsafe { register_dtor(self as *const _ as *mut u8, destroy_value::<T>) };
                    self.dtor_state.set(DtorState::Registered);
                    true
                }
                DtorState::Registered => {
                    // recursively initialized
                    true
                }
                DtorState::RunningOrHasRun => false,
            }
        }
    }

    unsafe extern "C" fn destroy_value<T>(ptr: *mut u8) {
        let ptr = ptr as *mut Key<T>;

        // SAFETY:
        //
        // The pointer `ptr` has been built just above and comes from
        // `try_register_dtor` where it is originally a Key<T> coming from `self`,
        // making it non-NUL and of the correct type.
        //
        // Right before we run the user destructor be sure to set the
        // `Option<T>` to `None`, and `dtor_state` to `RunningOrHasRun`. This
        // causes future calls to `get` to run `try_initialize_drop` again,
        // which will now fail, and return `None`.
        unsafe {
            let value = (*ptr).inner.take();
            (*ptr).dtor_state.set(DtorState::RunningOrHasRun);
            drop(value);
        }
    }
}

#[doc(hidden)]
#[cfg(not(target_thread_local))]
pub mod os {
    use super::lazy::LazyKeyInner;
    use crate::cell::Cell;
    use crate::fmt;
    use crate::marker;
    use crate::ptr;
    use crate::sys_common::thread_local_key::StaticKey as OsStaticKey;

    /// Use a regular global static to store this key; the state provided will then be
    /// thread-local.
    pub struct Key<T> {
        // OS-TLS key that we'll use to key off.
        os: OsStaticKey,
        marker: marker::PhantomData<Cell<T>>,
    }

    impl<T> fmt::Debug for Key<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Key").finish_non_exhaustive()
        }
    }

    unsafe impl<T> Sync for Key<T> {}

    struct Value<T: 'static> {
        inner: LazyKeyInner<T>,
        key: &'static Key<T>,
    }

    impl<T: 'static> Key<T> {
        #[rustc_const_unstable(feature = "thread_local_internals", issue = "none")]
        pub const fn new() -> Key<T> {
            Key { os: OsStaticKey::new(Some(destroy_value::<T>)), marker: marker::PhantomData }
        }

        /// It is a requirement for the caller to ensure that no mutable
        /// reference is active when this method is called.
        pub unsafe fn get(&'static self, init: impl FnOnce() -> T) -> Option<&'static T> {
            // SAFETY: See the documentation for this method.
            let ptr = unsafe { self.os.get() as *mut Value<T> };
            if ptr.addr() > 1 {
                // SAFETY: the check ensured the pointer is safe (its destructor
                // is not running) + it is coming from a trusted source (self).
                if let Some(ref value) = unsafe { (*ptr).inner.get() } {
                    return Some(value);
                }
            }
            // SAFETY: At this point we are sure we have no value and so
            // initializing (or trying to) is safe.
            unsafe { self.try_initialize(init) }
        }

        // `try_initialize` is only called once per os thread local variable,
        // except in corner cases where thread_local dtors reference other
        // thread_local's, or it is being recursively initialized.
        unsafe fn try_initialize(&'static self, init: impl FnOnce() -> T) -> Option<&'static T> {
            // SAFETY: No mutable references are ever handed out meaning getting
            // the value is ok.
            let ptr = unsafe { self.os.get() as *mut Value<T> };
            if ptr.addr() == 1 {
                // destructor is running
                return None;
            }

            let ptr = if ptr.is_null() {
                // If the lookup returned null, we haven't initialized our own
                // local copy, so do that now.
                let ptr: Box<Value<T>> = box Value { inner: LazyKeyInner::new(), key: self };
                let ptr = Box::into_raw(ptr);
                // SAFETY: At this point we are sure there is no value inside
                // ptr so setting it will not affect anyone else.
                unsafe {
                    self.os.set(ptr as *mut u8);
                }
                ptr
            } else {
                // recursive initialization
                ptr
            };

            // SAFETY: ptr has been ensured as non-NUL just above an so can be
            // dereferenced safely.
            unsafe { Some((*ptr).inner.initialize(init)) }
        }
    }

    unsafe extern "C" fn destroy_value<T: 'static>(ptr: *mut u8) {
        // SAFETY:
        //
        // The OS TLS ensures that this key contains a null value when this
        // destructor starts to run. We set it back to a sentinel value of 1 to
        // ensure that any future calls to `get` for this thread will return
        // `None`.
        //
        // Note that to prevent an infinite loop we reset it back to null right
        // before we return from the destructor ourselves.
        unsafe {
            let ptr = Box::from_raw(ptr as *mut Value<T>);
            let key = ptr.key;
            key.os.set(ptr::invalid_mut(1));
            drop(ptr);
            key.os.set(ptr::null_mut());
        }
    }
}
