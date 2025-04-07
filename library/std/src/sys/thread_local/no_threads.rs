//! On some targets like wasm there's no threads, so no need to generate
//! thread locals and we can instead just use plain statics!

use crate::cell::{Cell, UnsafeCell};
use crate::ptr;

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semitransparent"]
pub macro thread_local_inner {
    // used to generate the `LocalKey` value for const-initialized thread locals
    (@key $t:ty, const $init:expr) => {{
        const __INIT: $t = $init;

        // NOTE: Please update the shadowing test in `tests/thread.rs` if these types are renamed.
        unsafe {
            $crate::thread::LocalKey::new(|_| {
                static VAL: $crate::thread::local_impl::EagerStorage<$t> =
                    $crate::thread::local_impl::EagerStorage { value: __INIT };
                &VAL.value
            })
        }
    }},

    // used to generate the `LocalKey` value for `thread_local!`
    (@key $t:ty, $init:expr) => {{
        #[inline]
        fn __init() -> $t { $init }

        unsafe {
            use $crate::thread::LocalKey;
            use $crate::thread::local_impl::LazyStorage;

            LocalKey::new(|init| {
                static VAL: LazyStorage<$t> = LazyStorage::new();
                VAL.get(init, __init)
            })
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

    /// Gets a pointer to the TLS value, potentially initializing it with the
    /// provided parameters.
    ///
    /// The resulting pointer may not be used after reentrant inialialization
    /// has occurred.
    #[inline]
    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let value = unsafe { &*self.value.get() };
        match value {
            Some(v) => v,
            None => self.initialize(i, f),
        }
    }

    #[cold]
    fn initialize(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
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

#[rustc_macro_transparency = "semitransparent"]
pub(crate) macro local_pointer {
    () => {},
    ($vis:vis static $name:ident; $($rest:tt)*) => {
        $vis static $name: $crate::sys::thread_local::LocalPointer = $crate::sys::thread_local::LocalPointer::__new();
        $crate::sys::thread_local::local_pointer! { $($rest)* }
    },
}

pub(crate) struct LocalPointer {
    p: Cell<*mut ()>,
}

impl LocalPointer {
    pub const fn __new() -> LocalPointer {
        LocalPointer { p: Cell::new(ptr::null_mut()) }
    }

    pub fn get(&self) -> *mut () {
        self.p.get()
    }

    pub fn set(&self, p: *mut ()) {
        self.p.set(p)
    }
}

// SAFETY: the target doesn't have threads.
unsafe impl Sync for LocalPointer {}
