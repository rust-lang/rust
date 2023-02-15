#[doc(hidden)]
#[macro_export]
#[allow_internal_unstable(
    thread_local_internals,
    cfg_target_thread_local,
    thread_local,
    libstd_thread_internals
)]
#[allow_internal_unsafe]
macro_rules! __thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, const $init:expr) => {{
        #[cfg_attr(not(bootstrap), inline)]
        #[deny(unsafe_op_in_unsafe_fn)]
        unsafe fn __getit(
            _init: $crate::option::Option<&mut $crate::option::Option<$t>>,
        ) -> $crate::option::Option<&'static $t> {
            const INIT_EXPR: $t = $init;
            // If the platform has support for `#[thread_local]`, use it.
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
                        $crate::thread::__LocalKeyInner::<$t>::register_dtor(
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

        unsafe {
            $crate::thread::LocalKey::new(__getit)
        }
    }};

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {
        {
            #[inline]
            fn __init() -> $t { $init }

            #[cfg_attr(not(bootstrap), inline)]
            unsafe fn __getit(
                init: $crate::option::Option<&mut $crate::option::Option<$t>>,
            ) -> $crate::option::Option<&'static $t> {
                #[thread_local]
                static __KEY: $crate::thread::__LocalKeyInner<$t> =
                    $crate::thread::__LocalKeyInner::<$t>::new();

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

#[doc(hidden)]
pub mod fast {
    use super::super::lazy::LazyKeyInner;
    use crate::cell::Cell;
    use crate::sys::thread_local_dtor::register_dtor;
    use crate::{fmt, mem, panic};

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
        // free `register_dtor` function defined elsewhere in std.
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
        //
        // Wrap the call in a catch to ensure unwinding is caught in the event
        // a panic takes place in a destructor.
        if let Err(_) = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
            let value = (*ptr).inner.take();
            (*ptr).dtor_state.set(DtorState::RunningOrHasRun);
            drop(value);
        })) {
            rtabort!("thread local panicked on drop");
        }
    }
}
