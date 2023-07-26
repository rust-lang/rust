use super::lazy::LazyKeyInner;
use crate::fmt;

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
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
    }},

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {
        {
            #[inline]
            fn __init() -> $t { $init }
            #[inline]
            unsafe fn __getit(
                init: $crate::option::Option<&mut $crate::option::Option<$t>>,
            ) -> $crate::option::Option<&'static $t> {
                static __KEY: $crate::thread::local_impl::Key<$t> =
                    $crate::thread::local_impl::Key::new();

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
    },
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $($init:tt)*) => {
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::thread::local_impl::thread_local_inner!(@key $t, $($init)*);
    },
}

/// On some targets like wasm there's no threads, so no need to generate
/// thread locals and we can instead just use plain statics!

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
