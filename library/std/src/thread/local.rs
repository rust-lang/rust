//! Thread local storage

#![unstable(feature = "thread_local_internals", issue = "none")]

use crate::cell::{Cell, RefCell};
use crate::error::Error;
use crate::fmt;

/// A thread local storage (TLS) key which owns its contents.
///
/// This key uses the fastest implementation available on the target platform.
/// It is instantiated with the [`thread_local!`] macro and the
/// primary method is the [`with`] method, though there are helpers to make
/// working with [`Cell`] types easier.
///
/// The [`with`] method yields a reference to the contained value which cannot
/// outlive the current thread or escape the given closure.
///
/// [`thread_local!`]: crate::thread_local
///
/// # Initialization and Destruction
///
/// Initialization is dynamically performed on the first call to a setter (e.g.
/// [`with`]) within a thread, and values that implement [`Drop`] get
/// destructed when a thread exits. Some platform-specific caveats apply, which
/// are explained below.
/// Note that if the destructor panics, the whole process will be [aborted].
///
/// A `LocalKey`'s initializer cannot recursively depend on itself. Using a
/// `LocalKey` in this way may cause panics, aborts, or infinite recursion on
/// the first call to `with`.
///
/// [aborted]: crate::process::abort
///
/// # Single-thread Synchronization
///
/// Though there is no potential race with other threads, it is still possible to
/// obtain multiple references to the thread-local data in different places on
/// the call stack. For this reason, only shared (`&T`) references may be obtained.
///
/// To allow obtaining an exclusive mutable reference (`&mut T`), typically a
/// [`Cell`] or [`RefCell`] is used (see the [`std::cell`] for more information
/// on how exactly this works). To make this easier there are specialized
/// implementations for [`LocalKey<Cell<T>>`] and [`LocalKey<RefCell<T>>`].
///
/// [`std::cell`]: `crate::cell`
/// [`LocalKey<Cell<T>>`]: struct.LocalKey.html#impl-LocalKey<Cell<T>>
/// [`LocalKey<RefCell<T>>`]: struct.LocalKey.html#impl-LocalKey<RefCell<T>>
///
///
/// # Examples
///
/// ```
/// use std::cell::Cell;
/// use std::thread;
///
/// // explicit `const {}` block enables more efficient initialization
/// thread_local!(static FOO: Cell<u32> = const { Cell::new(1) });
///
/// assert_eq!(FOO.get(), 1);
/// FOO.set(2);
///
/// // each thread starts out with the initial value of 1
/// let t = thread::spawn(move || {
///     assert_eq!(FOO.get(), 1);
///     FOO.set(3);
/// });
///
/// // wait for the thread to complete and bail out on panic
/// t.join().unwrap();
///
/// // we retain our original value of 2 despite the child thread
/// assert_eq!(FOO.get(), 2);
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
#[cfg_attr(not(test), rustc_diagnostic_item = "LocalKey")]
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
    inner: fn(Option<&mut Option<T>>) -> *const T,
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T: 'static> fmt::Debug for LocalKey<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalKey").finish_non_exhaustive()
    }
}

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_process_attrs {

    // Parse `cfg_attr` to figure out whether it's a `rustc_align_static`.
    // Each `cfg_attr` can have zero or more attributes on the RHS, and can be nested.

    // finished parsing the `cfg_attr`, it had no `rustc_align_static`
    (
        [] [$(#[$($prev_other_attrs:tt)*])*];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [] };
        [$($prev_align_attrs_ret:tt)*] [$($prev_other_attrs_ret:tt)*];
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs_ret)*] [$($prev_other_attrs_ret)* #[cfg_attr($($predicate)*, $($($prev_other_attrs)*),*)]];
            $($rest)*
        );
    ),

    // finished parsing the `cfg_attr`, it had nothing but `rustc_align_static`
    (
        [$(#[$($prev_align_attrs:tt)*])+] [];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [] };
        [$($prev_align_attrs_ret:tt)*] [$($prev_other_attrs_ret:tt)*];
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs_ret)*  #[cfg_attr($($predicate)*, $($($prev_align_attrs)*),+)]] [$($prev_other_attrs_ret)*];
            $($rest)*
        );
    ),

    // finished parsing the `cfg_attr`, it had a mix of `rustc_align_static` and other attrs
    (
        [$(#[$($prev_align_attrs:tt)*])+] [$(#[$($prev_other_attrs:tt)*])+];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [] };
        [$($prev_align_attrs_ret:tt)*] [$($prev_other_attrs_ret:tt)*];
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs_ret)*  #[cfg_attr($($predicate)*, $($($prev_align_attrs)*),+)]] [$($prev_other_attrs_ret)* #[cfg_attr($($predicate)*, $($($prev_other_attrs)*),+)]];
            $($rest)*
        );
    ),

    // it's a `rustc_align_static`
    (
        [$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [rustc_align_static($($align_static_args:tt)*) $(, $($attr_rhs:tt)*)?] };
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs)* #[rustc_align_static($($align_static_args)*)]] [$($prev_other_attrs)*];
            @processing_cfg_attr { pred: ($($predicate)*), rhs: [$($($attr_rhs)*)?] };
            $($rest)*
        );
    ),

    // it's a nested `cfg_attr(true, ...)`; recurse into RHS
    (
        [$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [cfg_attr(true, $($cfg_rhs:tt)*) $(, $($attr_rhs:tt)*)?] };
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [] [];
            @processing_cfg_attr { pred: (true), rhs: [$($cfg_rhs)*] };
            [$($prev_align_attrs)*] [$($prev_other_attrs)*];
            @processing_cfg_attr { pred: ($($predicate)*), rhs: [$($($attr_rhs)*)?] };
            $($rest)*
        );
    ),

    // it's a nested `cfg_attr(false, ...)`; recurse into RHS
    (
        [$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [cfg_attr(false, $($cfg_rhs:tt)*) $(, $($attr_rhs:tt)*)?] };
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [] [];
            @processing_cfg_attr { pred: (false), rhs: [$($cfg_rhs)*] };
            [$($prev_align_attrs)*] [$($prev_other_attrs)*];
            @processing_cfg_attr { pred: ($($predicate)*), rhs: [$($($attr_rhs)*)?] };
            $($rest)*
        );
    ),


    // it's a nested `cfg_attr(..., ...)`; recurse into RHS
    (
        [$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [cfg_attr($cfg_lhs:meta, $($cfg_rhs:tt)*) $(, $($attr_rhs:tt)*)?] };
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [] [];
            @processing_cfg_attr { pred: ($cfg_lhs), rhs: [$($cfg_rhs)*] };
            [$($prev_align_attrs)*] [$($prev_other_attrs)*];
            @processing_cfg_attr { pred: ($($predicate)*), rhs: [$($($attr_rhs)*)?] };
            $($rest)*
        );
    ),

    // it's some other attribute
    (
        [$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*];
        @processing_cfg_attr { pred: ($($predicate:tt)*), rhs: [$meta:meta $(, $($attr_rhs:tt)*)?] };
        $($rest:tt)*
    ) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs)*] [$($prev_other_attrs)* #[$meta]];
            @processing_cfg_attr { pred: ($($predicate)*), rhs: [$($($attr_rhs)*)?] };
            $($rest)*
        );
    ),


    // Separate attributes into `rustc_align_static` and everything else:

    // `rustc_align_static` attribute
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*]; #[rustc_align_static $($attr_rest:tt)*] $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs)* #[rustc_align_static $($attr_rest)*]] [$($prev_other_attrs)*];
            $($rest)*
        );
    ),

    // `cfg_attr(true, ...)` attribute; parse it
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*]; #[cfg_attr(true, $($cfg_rhs:tt)*)] $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [] [];
            @processing_cfg_attr { pred: (true), rhs: [$($cfg_rhs)*] };
            [$($prev_align_attrs)*] [$($prev_other_attrs)*];
            $($rest)*
        );
    ),

    // `cfg_attr(false, ...)` attribute; parse it
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*]; #[cfg_attr(false, $($cfg_rhs:tt)*)] $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [] [];
            @processing_cfg_attr { pred: (false), rhs: [$($cfg_rhs)*] };
            [$($prev_align_attrs)*] [$($prev_other_attrs)*];
            $($rest)*
        );
    ),

    // `cfg_attr(..., ...)` attribute; parse it
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*]; #[cfg_attr($cfg_pred:meta, $($cfg_rhs:tt)*)] $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [] [];
            @processing_cfg_attr { pred: ($cfg_pred), rhs: [$($cfg_rhs)*] };
            [$($prev_align_attrs)*] [$($prev_other_attrs)*];
            $($rest)*
        );
    ),

    // doc comment not followed by any other attributes; process it all at once to avoid blowing recursion limit
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*]; $(#[doc $($doc_rhs:tt)*])+ $vis:vis static $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs)*] [$($prev_other_attrs)* $(#[doc $($doc_rhs)*])+];
            $vis static $($rest)*
        );
    ),

    // 8 lines of doc comment; process them all at once to avoid blowing recursion limit
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*];
     #[doc $($doc_rhs_1:tt)*] #[doc $($doc_rhs_2:tt)*] #[doc $($doc_rhs_3:tt)*] #[doc $($doc_rhs_4:tt)*]
     #[doc $($doc_rhs_5:tt)*] #[doc $($doc_rhs_6:tt)*] #[doc $($doc_rhs_7:tt)*] #[doc $($doc_rhs_8:tt)*]
     $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs)*] [$($prev_other_attrs)*
            #[doc $($doc_rhs_1)*] #[doc $($doc_rhs_2)*] #[doc $($doc_rhs_3)*] #[doc $($doc_rhs_4)*]
            #[doc $($doc_rhs_5)*] #[doc $($doc_rhs_6)*] #[doc $($doc_rhs_7)*] #[doc $($doc_rhs_8)*]];
            $($rest)*
        );
    ),

    // other attribute
    ([$($prev_align_attrs:tt)*] [$($prev_other_attrs:tt)*]; #[$($attr:tt)*] $($rest:tt)*) => (
        $crate::thread::local_impl::thread_local_process_attrs!(
            [$($prev_align_attrs)*] [$($prev_other_attrs)* #[$($attr)*]];
            $($rest)*
        );
    ),


    // Delegate to `thread_local_inner` once attributes are fully categorized:

    // process `const` declaration and recurse
    ([$($align_attrs:tt)*] [$($other_attrs:tt)*]; $vis:vis static $name:ident: $t:ty = const $init:block $(; $($($rest:tt)+)?)?) => (
        $($other_attrs)* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::thread::local_impl::thread_local_inner!(@key $t, $($align_attrs)*, const $init);

        $($($crate::thread::local_impl::thread_local_process_attrs!([] []; $($rest)+);)?)?
    ),

    // process non-`const` declaration and recurse
    ([$($align_attrs:tt)*] [$($other_attrs:tt)*]; $vis:vis static $name:ident: $t:ty = $init:expr $(; $($($rest:tt)+)?)?) => (
        $($other_attrs)* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::thread::local_impl::thread_local_inner!(@key $t, $($align_attrs)*, $init);

        $($($crate::thread::local_impl::thread_local_process_attrs!([] []; $($rest)+);)?)?
    ),
}

/// Declare a new thread local storage key of type [`std::thread::LocalKey`].
///
/// # Syntax
///
/// The macro wraps any number of static declarations and makes them thread local.
/// Publicity and attributes for each static are allowed. Example:
///
/// ```
/// use std::cell::{Cell, RefCell};
///
/// thread_local! {
///     pub static FOO: Cell<u32> = const { Cell::new(1) };
///
///     static BAR: RefCell<Vec<f32>> = RefCell::new(vec![1.0, 2.0]);
/// }
///
/// assert_eq!(FOO.get(), 1);
/// BAR.with_borrow(|v| assert_eq!(v[1], 2.0));
/// ```
///
/// Note that only shared references (`&T`) to the inner data may be obtained, so a
/// type such as [`Cell`] or [`RefCell`] is typically used to allow mutating access.
///
/// This macro supports a special `const {}` syntax that can be used
/// when the initialization expression can be evaluated as a constant.
/// This can enable a more efficient thread local implementation that
/// can avoid lazy initialization. For types that do not
/// [need to be dropped][crate::mem::needs_drop], this can enable an
/// even more efficient implementation that does not need to
/// track any additional state.
///
/// ```
/// use std::cell::RefCell;
///
/// thread_local! {
///     pub static FOO: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
/// }
///
/// FOO.with_borrow(|v| assert_eq!(v.len(), 0));
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
    () => {};

    ($($tt:tt)+) => {
        $crate::thread::local_impl::thread_local_process_attrs!([] []; $($tt)+);
    };
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

// This ensures the panicking code is outlined from `with` for `LocalKey`.
#[cfg_attr(not(panic = "immediate-abort"), inline(never))]
#[track_caller]
#[cold]
fn panic_access_error(err: AccessError) -> ! {
    panic!("cannot access a Thread Local Storage value during or after destruction: {err:?}")
}

impl<T: 'static> LocalKey<T> {
    #[doc(hidden)]
    #[unstable(
        feature = "thread_local_internals",
        reason = "recently added to create a key",
        issue = "none"
    )]
    pub const unsafe fn new(inner: fn(Option<&mut Option<T>>) -> *const T) -> LocalKey<T> {
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
    ///
    /// # Examples
    ///
    /// ```
    /// thread_local! {
    ///     pub static STATIC: String = String::from("I am");
    /// }
    ///
    /// assert_eq!(
    ///     STATIC.with(|original_value| format!("{original_value} initialized")),
    ///     "I am initialized",
    /// );
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        match self.try_with(f) {
            Ok(r) => r,
            Err(err) => panic_access_error(err),
        }
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
    ///
    /// # Examples
    ///
    /// ```
    /// thread_local! {
    ///     pub static STATIC: String = String::from("I am");
    /// }
    ///
    /// assert_eq!(
    ///     STATIC.try_with(|original_value| format!("{original_value} initialized")),
    ///     Ok(String::from("I am initialized")),
    /// );
    /// ```
    #[stable(feature = "thread_local_try_with", since = "1.26.0")]
    #[inline]
    pub fn try_with<F, R>(&'static self, f: F) -> Result<R, AccessError>
    where
        F: FnOnce(&T) -> R,
    {
        let thread_local = unsafe { (self.inner)(None).as_ref().ok_or(AccessError)? };
        Ok(f(thread_local))
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
        let mut init = Some(init);

        let reference = unsafe {
            match (self.inner)(Some(&mut init)).as_ref() {
                Some(r) => r,
                None => panic_access_error(AccessError),
            }
        };

        f(init, reference)
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
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
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
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<i32> = const { Cell::new(1) };
    /// }
    ///
    /// assert_eq!(X.get(), 1);
    /// ```
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
    pub fn get(&'static self) -> T
    where
        T: Copy,
    {
        self.with(Cell::get)
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
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<Option<i32>> = const { Cell::new(Some(1)) };
    /// }
    ///
    /// assert_eq!(X.take(), Some(1));
    /// assert_eq!(X.take(), None);
    /// ```
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
    pub fn take(&'static self) -> T
    where
        T: Default,
    {
        self.with(Cell::take)
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
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<i32> = const { Cell::new(1) };
    /// }
    ///
    /// assert_eq!(X.replace(2), 1);
    /// assert_eq!(X.replace(3), 2);
    /// ```
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
    #[rustc_confusables("swap")]
    pub fn replace(&'static self, value: T) -> T {
        self.with(|cell| cell.replace(value))
    }

    /// Updates the contained value using a function.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(local_key_cell_update)]
    /// use std::cell::Cell;
    ///
    /// thread_local! {
    ///     static X: Cell<i32> = const { Cell::new(5) };
    /// }
    ///
    /// X.update(|x| x + 1);
    /// assert_eq!(X.get(), 6);
    /// ```
    #[unstable(feature = "local_key_cell_update", issue = "143989")]
    pub fn update(&'static self, f: impl FnOnce(T) -> T)
    where
        T: Copy,
    {
        self.with(|cell| cell.update(f))
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
    /// # Examples
    ///
    /// ```
    /// use std::cell::RefCell;
    ///
    /// thread_local! {
    ///     static X: RefCell<Vec<i32>> = RefCell::new(Vec::new());
    /// }
    ///
    /// X.with_borrow(|v| assert!(v.is_empty()));
    /// ```
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
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
    /// # Examples
    ///
    /// ```
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
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
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
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
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
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
    pub fn take(&'static self) -> T
    where
        T: Default,
    {
        self.with(RefCell::take)
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
    #[stable(feature = "local_key_cell_methods", since = "1.73.0")]
    #[rustc_confusables("swap")]
    pub fn replace(&'static self, value: T) -> T {
        self.with(|cell| cell.replace(value))
    }
}
