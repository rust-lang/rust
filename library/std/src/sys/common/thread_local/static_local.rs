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
        #[inline] // see comments below
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
            static mut VAL: $t = INIT_EXPR;
            unsafe { $crate::option::Option::Some(&VAL) }
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
            #[inline]
            unsafe fn __getit(
                init: $crate::option::Option<&mut $crate::option::Option<$t>>,
            ) -> $crate::option::Option<&'static $t> {
                static __KEY: $crate::thread::__LocalKeyInner<$t> =
                    $crate::thread::__LocalKeyInner::new();

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

/// On some targets like wasm there's no threads, so no need to generate
/// thread locals and we can instead just use plain statics!
#[doc(hidden)]
pub mod statik {
    use super::super::lazy::LazyKeyInner;
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
