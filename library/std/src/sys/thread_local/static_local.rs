//! On some targets like wasm there's no threads, so no need to generate
//! thread locals and we can instead just use plain statics!

use crate::cell::UnsafeCell;

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, const $init:expr) => {{
        const __INIT: $t = $init;

        #[inline]
        #[deny(unsafe_op_in_unsafe_fn)]
        unsafe fn __getit(
            _init: $crate::option::Option<&mut $crate::option::Option<$t>>,
        ) -> $crate::option::Option<&'static $t> {
            use $crate::thread::local_impl::EagerStorage;

            static VAL: EagerStorage<$t> = EagerStorage { value: __INIT };
            $crate::option::Option::Some(&VAL.value)
        }

        unsafe {
            $crate::thread::LocalKey::new(__getit)
        }
    }},

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        #[inline]
        #[deny(unsafe_op_in_unsafe_fn)]
        unsafe fn __getit(
            init: $crate::option::Option<&mut $crate::option::Option<$t>>,
        ) -> $crate::option::Option<&'static $t> {
            use $crate::thread::local_impl::LazyStorage;

            static VAL: LazyStorage<$t> = LazyStorage::new();
            unsafe { $crate::option::Option::Some(VAL.get(init, __init)) }
        }

        unsafe {
            $crate::thread::LocalKey::new(__getit)
        }
    }},
    ($(#[$attr:meta])* $vis:vis $name:ident, $t:ty, $($init:tt)*) => {
        $(#[$attr])* $vis const $name: $crate::thread::LocalKey<$t> =
            $crate::thread::local_impl::thread_local_inner!(@key $t, $($init)*);
    },
}

#[allow(missing_debug_implementations)]
pub struct EagerStorage<T> {
    pub value: T,
}

// SAFETY: the target doesn't have threads.
unsafe impl<T> Sync for EagerStorage<T> {}

#[allow(missing_debug_implementations)]
pub struct LazyStorage<T> {
    value: UnsafeCell<Option<T>>,
}

impl<T> LazyStorage<T> {
    pub const fn new() -> LazyStorage<T> {
        LazyStorage { value: UnsafeCell::new(None) }
    }

    /// Gets a reference to the contained value, initializing it if necessary.
    ///
    /// # Safety
    /// The returned reference may not be used after reentrant initialization has occurred.
    #[inline]
    pub unsafe fn get(
        &'static self,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> &'static T {
        let value = unsafe { &*self.value.get() };
        match value {
            Some(v) => v,
            None => self.initialize(i, f),
        }
    }

    #[cold]
    unsafe fn initialize(
        &'static self,
        i: Option<&mut Option<T>>,
        f: impl FnOnce() -> T,
    ) -> &'static T {
        let value = i.and_then(Option::take).unwrap_or_else(f);
        // Destroy the old value, after updating the TLS variable as the
        // destructor might reference it.
        // FIXME(#110897): maybe panic on recursive initialization.
        unsafe {
            self.value.get().replace(Some(value));
        }
        // SAFETY: we just set this to `Some`.
        unsafe { (*self.value.get()).as_ref().unwrap_unchecked() }
    }
}

// SAFETY: the target doesn't have threads.
unsafe impl<T> Sync for LazyStorage<T> {}
